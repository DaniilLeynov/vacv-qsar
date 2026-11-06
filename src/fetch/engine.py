from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .errors import FetchError
from .manifest import Manifest
from .paginator import make_paginator
from .requester import Response, build_headers_for, http_get, mask_for

MANIFEST_NAME = "MANIFEST.jsonl"


@dataclass
class FetchResult:
    source: str
    skipped: bool = False
    dry_run: bool = False
    pages: int = 0
    files: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def contained_path(base: Path, relative: str) -> Path | None:
    root = base.resolve()
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None
    return target


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_records(out_dir: Path) -> list[dict]:
    path = out_dir / MANIFEST_NAME
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _write_records(out_dir: Path, records: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / MANIFEST_NAME
    text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    path.write_text(text, encoding="utf-8")


def _save_run(out_dir: Path, source_id: str, records: list[dict]) -> None:
    existing = _load_records(out_dir)
    previous = [r for r in existing if r.get("source") == source_id]
    others = [r for r in existing if r.get("source") != source_id]

    new_files = {r["file"] for r in records}
    for record in previous:
        name = record.get("file")
        if name and name not in new_files:
            stale = contained_path(out_dir, name)
            if stale is not None and stale.exists():
                stale.unlink()

    _write_records(out_dir, others + records)


def _is_cached(out_dir: Path, source_id: str, fingerprint: str) -> bool:
    records = [
        r
        for r in _load_records(out_dir)
        if r.get("source") == source_id
        and r.get("manifest_sha256") == fingerprint
    ]
    if not records:
        return False
    for record in records:
        path = contained_path(out_dir, record.get("file", ""))
        if path is None or not path.exists():
            return False
        if sha256_bytes(path.read_bytes()) != record.get("sha256"):
            return False
    return True


def _record(
    source_id: str,
    url: str,
    response: Response,
    relative_path: str,
    page: int | None,
    fingerprint: str,
) -> dict:
    return {
        "source": source_id,
        "page": page,
        "url": url,
        "http_status": response.status,
        "content_type": response.content_type,
        "bytes": len(response.content),
        "sha256": sha256_bytes(response.content),
        "fetched_at": _now(),
        "file": relative_path,
        "manifest_sha256": fingerprint,
    }


def _fetch_file(
    manifest: Manifest,
    out_dir: Path,
    refresh: bool,
    dry_run: bool,
    http_get_fn: Callable,
    on_file_progress: Callable[[int, int | None], None] | None,
    on_retry: Callable[[int, int | None], None] | None,
) -> FetchResult:
    assert manifest.download is not None
    downloaded = manifest.download
    result = FetchResult(source=manifest.id, dry_run=dry_run, urls=[downloaded.url])

    if not dry_run and not refresh and _is_cached(
        out_dir, manifest.id, manifest.fingerprint()
    ):
        result.skipped = True
        return result
    if dry_run:
        return result

    handler = manifest.error_handler_or_default
    extra = {"on_retry": on_retry} if on_retry else {}
    response = http_get_fn(
        downloaded.url, {}, handler, on_download=on_file_progress, **extra
    )
    digest = sha256_bytes(response.content)
    if downloaded.sha256 and downloaded.sha256.lower() != digest:
        raise FetchError(
            f"{manifest.id}: sha256 mismatch: got {digest}, expected {downloaded.sha256}"
        )

    relative_path = f"{manifest.id}/{downloaded.filename}"
    target = contained_path(out_dir, relative_path)
    if target is None:
        raise FetchError(f"{manifest.id}: refusing to write outside {out_dir}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)
    _save_run(
        out_dir,
        manifest.id,
        [
            _record(
                manifest.id,
                downloaded.url,
                response,
                relative_path,
                None,
                manifest.fingerprint(),
            )
        ],
    )
    result.pages = 1
    result.files = [relative_path]
    return result


def fetch_source(
    manifest: Manifest,
    out_dir: str | Path,
    refresh: bool = False,
    dry_run: bool = False,
    http_get_fn: Callable = http_get,
    on_page: Callable[[int, int | None], None] | None = None,
    on_file_progress: Callable[[int, int | None], None] | None = None,
    on_retry: Callable[[int, int | None], None] | None = None,
) -> FetchResult:
    out_dir = Path(out_dir)
    if manifest.type == "FileSource":
        return _fetch_file(
            manifest,
            out_dir,
            refresh,
            dry_run,
            http_get_fn,
            on_file_progress,
            on_retry,
        )

    assert manifest.requester is not None
    requester = manifest.requester
    paginator = make_paginator(manifest.paginator_or_default, requester)
    handler = manifest.error_handler_or_default
    extension = manifest.output_or_default.extension

    result = FetchResult(source=manifest.id, dry_run=dry_run)

    if not dry_run and not refresh and _is_cached(
        out_dir, manifest.id, manifest.fingerprint()
    ):
        result.skipped = True
        return result

    url = paginator.first_url()
    result.urls.append(mask_for(requester, url))
    if dry_run:
        return result

    headers = build_headers_for(requester)
    records: list[dict] = []
    paginator_config = manifest.paginator_or_default
    if paginator_config.type == "NoPagination":
        total_hint = 1
    elif paginator_config.type in {"OffsetIncrement", "PageIncrement"}:
        total_hint = paginator_config.max_pages
    else:
        total_hint = None

    retry_kwargs = {"on_retry": on_retry} if on_retry else {}

    for page in range(1, paginator_config.max_pages + 1):
        response = http_get_fn(url, headers, handler, **retry_kwargs)
        relative_path = f"{manifest.id}/page-{page:04d}.{extension}"
        target = contained_path(out_dir, relative_path)
        if target is None:
            raise FetchError(f"{manifest.id}: refusing to write outside {out_dir}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)

        records.append(
            _record(
                manifest.id,
                mask_for(requester, url),
                response,
                relative_path,
                page,
                manifest.fingerprint(),
            )
        )
        result.pages += 1
        result.files.append(relative_path)
        if on_page:
            on_page(page, total_hint)

        next_url = paginator.next_url(response)
        if next_url is None:
            break
        url = next_url
        result.urls.append(mask_for(requester, url))

    _save_run(out_dir, manifest.id, records)
    return result


def fetch_manifest(
    manifest: Manifest,
    out_dir: str | Path,
    refresh: bool = False,
    dry_run: bool = False,
    http_get_fn: Callable = http_get,
    on_page: Callable[[int, int | None], None] | None = None,
    on_file_progress: Callable[[int, int | None], None] | None = None,
    on_retry: Callable[[int, int | None], None] | None = None,
) -> FetchResult:
    return fetch_source(
        manifest,
        out_dir,
        refresh=refresh,
        dry_run=dry_run,
        http_get_fn=http_get_fn,
        on_page=on_page,
        on_file_progress=on_file_progress,
        on_retry=on_retry,
    )
