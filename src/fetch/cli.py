from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .engine import FetchResult, fetch_manifest
from .errors import FetchError
from .manifest import Manifest, find_manifests, load_manifest, resolve_source
from .progress import Progress

DEFAULT_SOURCES = Path("sources")
DEFAULT_OUT = Path("data/raw")


def _print_manifest(manifest: Manifest, path: Path) -> None:
    description = manifest.description or ""
    print(f"{manifest.id:<24} {manifest.type:<18} {description}")
    print(f"{'':<24} {path}")


def _cmd_list(manifest_dir: Path) -> int:
    paths = find_manifests(manifest_dir)
    if not paths:
        print(f"no manifests in {manifest_dir}")
        return 0
    print(f"{'id':<24} {'type':<18} description")
    exit_code = 0
    for path in paths:
        try:
            _print_manifest(load_manifest(path), path)
        except FetchError as exc:
            print(f"{path}: ERROR: {exc}", file=sys.stderr)
            exit_code = 1
    return exit_code


def _report(result: FetchResult) -> None:
    if result.dry_run:
        for url in result.urls:
            print(f"GET {url}")
        print("(dry-run: no requests were made)")
        return
    if result.skipped:
        print(
            f"{result.source}: already downloaded, sha256 matches "
            "(use --refresh to re-download)"
        )
        return
    print(f"{result.source}: pages {result.pages}, files {len(result.files)}")
    for name in result.files:
        print(f"  {name}")


def _select_manifests(args) -> list[Manifest]:
    selected: list[Manifest] = []
    seen: set[str] = set()

    def add(manifest: Manifest) -> None:
        if manifest.id in seen:
            return
        seen.add(manifest.id)
        selected.append(manifest)

    if args.all:
        for path in find_manifests(args.manifest_dir):
            add(load_manifest(path))

    for spec in args.sources:
        add(resolve_source(spec, args.manifest_dir))

    return selected


def _progress_enabled(args) -> bool | None:
    if args.no_progress:
        return False
    if args.progress:
        return True
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.fetch",
        description="Downloads raw data according to sources/*.yaml manifests.",
    )
    parser.add_argument(
        "sources",
        nargs="*",
        metavar="source",
        help="source id or manifest path; repeatable",
    )
    parser.add_argument("--list", action="store_true", help="list sources")
    parser.add_argument("--all", action="store_true", help="download all sources")
    parser.add_argument(
        "--sources",
        dest="manifest_dir",
        type=Path,
        default=DEFAULT_SOURCES,
        help=f"manifest directory (default: {DEFAULT_SOURCES})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"raw data directory (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="re-download, ignoring the cache"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print URLs without making requests"
    )
    parser.add_argument(
        "--progress", action="store_true", help="force the progress bar on"
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="disable the progress bar"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        return _cmd_list(args.manifest_dir)

    try:
        manifests = _select_manifests(args)
    except FetchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not manifests:
        print("no sources specified", file=sys.stderr)
        build_parser().print_help(sys.stderr)
        return 2

    enabled = _progress_enabled(args)
    if args.dry_run:
        enabled = False
    progress = Progress([manifest.id for manifest in manifests], enabled=enabled)

    results: list[FetchResult] = []
    failures: list[tuple[str, FetchError]] = []

    try:
        for index, manifest in enumerate(manifests):
            progress.start(index)

            def page(number, total, task=index):
                progress.page(task, number, total)

            def file_progress(received, total, task=index):
                progress.file_progress(task, received, total)

            def retry(attempt, status, task=index):
                progress.retry(task, attempt, status)

            try:
                result = fetch_manifest(
                    manifest,
                    out_dir=args.out,
                    refresh=args.refresh,
                    dry_run=args.dry_run,
                    on_page=page,
                    on_file_progress=file_progress,
                    on_retry=retry,
                )
            except FetchError as exc:
                failures.append((manifest.id, exc))
                progress.finish(index, "error")
                continue
            results.append(result)
            progress.finish(index, "skipped" if result.skipped else "done")
    finally:
        progress.close()

    for result in results:
        _report(result)
    for source_id, exc in failures:
        print(f"{source_id}: ERROR: {exc}", file=sys.stderr)

    return 1 if failures else 0
