import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from src.fetch.engine import MANIFEST_NAME, fetch_source, sha256_bytes
from src.fetch.errors import FetchError
from src.fetch.manifest import parse_manifest
from src.fetch.requester import Response


def read_manifest(out_dir: Path):
    path = out_dir / MANIFEST_NAME
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def offset_manifest(**overrides):
    data = {
        "id": "test_source",
        "type": "DeclarativeSource",
        "requester": {
            "url_base": "https://example.org/api/",
            "path": "items.json",
            "request_parameters": {"limit": "100"},
        },
        "paginator": {
            "type": "OffsetIncrement",
            "page_size": 100,
            "max_pages": 5,
            "stop_when_empty": "$.items",
        },
    }
    data.update(overrides)
    return parse_manifest(data)


class FakeHttp:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def __call__(self, url, headers, error_handler):
        self.calls.append(url)
        offset = int(parse_qs(urlsplit(url).query)["offset"][0])
        payload = self.pages[offset]
        return Response(
            url=url,
            status=200,
            headers={"Content-Type": "application/json"},
            content=json.dumps(payload).encode("utf-8"),
        )


def test_fetch_writes_pages_and_manifest(tmp_path):
    fake = FakeHttp({0: {"items": [1, 2]}, 100: {"items": [3]}, 200: {"items": []}})
    result = fetch_source(offset_manifest(), tmp_path, http_get_fn=fake)

    assert result.pages == 3
    assert result.files == [
        "test_source/page-0001.json",
        "test_source/page-0002.json",
        "test_source/page-0003.json",
    ]
    for name in result.files:
        assert (tmp_path / name).exists()

    records = read_manifest(tmp_path)
    assert len(records) == 3
    assert [record["page"] for record in records] == [1, 2, 3]
    for record in records:
        assert record["source"] == "test_source"
        assert record["http_status"] == 200
        assert record["content_type"] == "application/json"
        assert record["sha256"] == sha256_bytes(
            (tmp_path / record["file"]).read_bytes()
        )


def test_second_run_uses_cache(tmp_path):
    fetch_source(
        offset_manifest(),
        tmp_path,
        http_get_fn=FakeHttp({0: {"items": []}}),
    )
    fake = FakeHttp({0: {"items": []}})
    result = fetch_source(offset_manifest(), tmp_path, http_get_fn=fake)

    assert result.skipped is True
    assert fake.calls == []


def test_refresh_ignores_cache(tmp_path):
    fetch_source(
        offset_manifest(), tmp_path, http_get_fn=FakeHttp({0: {"items": []}})
    )
    fake = FakeHttp({0: {"items": []}})
    result = fetch_source(offset_manifest(), tmp_path, refresh=True, http_get_fn=fake)

    assert result.skipped is False
    assert len(fake.calls) == 1


def test_dry_run_makes_no_requests(tmp_path):
    fake = FakeHttp({0: {"items": []}})
    result = fetch_source(offset_manifest(), tmp_path, dry_run=True, http_get_fn=fake)

    assert result.dry_run is True
    assert fake.calls == []
    assert result.urls and "offset=0" in result.urls[0]
    assert not (tmp_path / MANIFEST_NAME).exists()


def test_refresh_removes_stale_pages(tmp_path):
    fetch_source(
        offset_manifest(),
        tmp_path,
        http_get_fn=FakeHttp(
            {0: {"items": [1, 2]}, 100: {"items": [3]}, 200: {"items": []}}
        ),
    )
    assert (tmp_path / "test_source" / "page-0003.json").exists()

    fetch_source(
        offset_manifest(),
        tmp_path,
        refresh=True,
        http_get_fn=FakeHttp({0: {"items": [1]}, 100: {"items": []}}),
    )
    assert (tmp_path / "test_source" / "page-0002.json").exists()
    assert not (tmp_path / "test_source" / "page-0003.json").exists()
    assert len(read_manifest(tmp_path)) == 2


def test_file_source_downloads_one_file(tmp_path):
    manifest = parse_manifest(
        {
            "id": "file_source",
            "type": "FileSource",
            "download": {
                "url": "https://example.org/table.csv",
                "filename": "table.csv",
            },
        }
    )
    body = b"smiles,ic50\nCCO,1.0\n"

    def fake(url, headers, error_handler, on_download=None):
        if on_download:
            on_download(len(body), len(body))
        return Response(url=url, status=200, headers={}, content=body)

    result = fetch_source(manifest, tmp_path, http_get_fn=fake)

    assert result.pages == 1
    assert (tmp_path / "file_source" / "table.csv").read_bytes() == body
    records = read_manifest(tmp_path)
    assert records[0]["file"] == "file_source/table.csv"
    assert records[0]["sha256"] == sha256_bytes(body)


def test_file_source_checks_sha256(tmp_path):
    manifest = parse_manifest(
        {
            "id": "file_source",
            "type": "FileSource",
            "download": {
                "url": "https://example.org/table.csv",
                "filename": "table.csv",
                "sha256": "0" * 64,
            },
        }
    )

    def fake(url, headers, error_handler, on_download=None):
        return Response(url=url, status=200, headers={}, content=b"data")

    with pytest.raises(FetchError, match="sha256"):
        fetch_source(manifest, tmp_path, http_get_fn=fake)
    assert not (tmp_path / "file_source" / "table.csv").exists()


def test_secret_query_parameter_is_masked_in_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_KEY", "supersecret")
    manifest = parse_manifest(
        {
            "id": "secret_source",
            "type": "DeclarativeSource",
            "requester": {
                "url_base": "https://example.org/api/",
                "path": "items.json",
                "authenticator": {
                    "type": "ApiKeyQuery",
                    "env": "MY_KEY",
                    "query_parameter": "api_key",
                },
            },
            "paginator": {"type": "NoPagination"},
        }
    )

    def fake(url, headers, error_handler):
        return Response(url=url, status=200, headers={}, content=b"{}")

    fetch_source(manifest, tmp_path, http_get_fn=fake)
    text = (tmp_path / MANIFEST_NAME).read_text(encoding="utf-8")
    assert "supersecret" not in text
    assert "REDACTED" in text


def test_on_page_reports_pages_and_total_hint(tmp_path):
    calls = []
    fake = FakeHttp({0: {"items": [1]}, 100: {"items": []}})
    fetch_source(
        offset_manifest(),
        tmp_path,
        http_get_fn=fake,
        on_page=lambda number, total: calls.append((number, total)),
    )
    assert calls == [(1, 5), (2, 5)]


def test_on_file_progress_reports_bytes(tmp_path):
    manifest = parse_manifest(
        {
            "id": "file_source",
            "type": "FileSource",
            "download": {
                "url": "https://example.org/table.csv",
                "filename": "table.csv",
            },
        }
    )
    calls = []

    def fake(url, headers, error_handler, on_download=None):
        body = b"data"
        if on_download:
            on_download(len(body), len(body))
        return Response(url=url, status=200, headers={}, content=body)

    fetch_source(
        manifest,
        tmp_path,
        http_get_fn=fake,
        on_file_progress=lambda received, total: calls.append((received, total)),
    )
    assert calls == [(4, 4)]


def test_manifest_change_invalidates_cache(tmp_path):
    fetch_source(
        offset_manifest(), tmp_path, http_get_fn=FakeHttp({0: {"items": []}})
    )
    changed = offset_manifest()
    changed.paginator.page_size = 50
    fake = FakeHttp({0: {"items": []}})
    result = fetch_source(changed, tmp_path, http_get_fn=fake)

    assert result.skipped is False
    assert len(fake.calls) == 1
    assert "offset=0" in fake.calls[0]
