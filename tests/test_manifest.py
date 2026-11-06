from pathlib import Path

import pytest

from src.fetch.errors import ManifestError
from src.fetch.manifest import (
    find_manifests,
    load_manifest,
    parse_manifest,
    resolve_source,
)

REPO = Path(__file__).resolve().parents[1]


def declarative(**overrides):
    data = {
        "version": "0.1",
        "id": "test_source",
        "type": "DeclarativeSource",
        "requester": {
            "type": "HttpRequester",
            "url_base": "https://example.org/api/",
            "path": "items.json",
        },
        "paginator": {"type": "NoPagination"},
    }
    data.update(overrides)
    return data


def test_minimal_declarative_is_valid():
    manifest = parse_manifest(declarative())
    assert manifest.id == "test_source"
    assert manifest.requester is not None
    assert manifest.requester.url_base == "https://example.org/api/"
    assert manifest.paginator.type == "NoPagination"
    assert manifest.error_handler_or_default.max_retries == 3


def test_requester_is_required_for_declarative_source():
    data = declarative()
    data.pop("requester")
    with pytest.raises(ManifestError, match="requester"):
        parse_manifest(data)


def test_unknown_top_level_field_is_rejected():
    with pytest.raises(ManifestError, match="unknown fields"):
        parse_manifest(declarative(unexpected="value"))


def test_unknown_source_type_is_rejected():
    with pytest.raises(ManifestError, match="unknown type"):
        parse_manifest(declarative(type="MagicSource"))


def test_secret_value_is_rejected():
    data = declarative()
    data["requester"]["authenticator"] = {"type": "NoAuth", "token": "abc123"}
    with pytest.raises(ManifestError, match="secrets"):
        parse_manifest(data)


def test_auth_requires_env():
    data = declarative()
    data["requester"]["authenticator"] = {"type": "Bearer"}
    with pytest.raises(ManifestError, match="env"):
        parse_manifest(data)


def test_api_key_header_requires_header_name():
    data = declarative()
    data["requester"]["authenticator"] = {"type": "ApiKeyHeader", "env": "MY_KEY"}
    with pytest.raises(ManifestError, match="header_name"):
        parse_manifest(data)


def test_next_url_requires_path():
    with pytest.raises(ManifestError, match="next_url_path"):
        parse_manifest(declarative(paginator={"type": "NextUrl"}))


def test_offset_paginator_parses_defaults():
    manifest = parse_manifest(
        declarative(
            paginator={"type": "OffsetIncrement", "page_size": 50, "max_pages": 3}
        )
    )
    assert manifest.paginator.page_size == 50
    assert manifest.paginator.offset_parameter == "offset"
    assert manifest.paginator.limit_parameter == "limit"
    assert manifest.paginator.max_pages == 3


def test_file_source_requires_download():
    with pytest.raises(ManifestError, match="download"):
        parse_manifest({"version": "0.1", "id": "f", "type": "FileSource"})


def test_file_source_is_valid():
    manifest = parse_manifest(
        {
            "version": "0.1",
            "id": "f",
            "type": "FileSource",
            "download": {
                "url": "https://example.org/table.csv",
                "filename": "table.csv",
            },
        }
    )
    assert manifest.download is not None
    assert manifest.download.filename == "table.csv"


def test_download_filename_must_not_contain_directories():
    with pytest.raises(ManifestError, match="filename"):
        parse_manifest(
            {
                "id": "f",
                "type": "FileSource",
                "download": {
                    "url": "https://example.org/table.csv",
                    "filename": "../table.csv",
                },
            }
        )


def test_load_manifest_from_disk(tmp_path):
    path = tmp_path / "source.yaml"
    path.write_text(
        """
id: from_disk
type: DeclarativeSource
requester:
  type: HttpRequester
  url_base: https://example.org/api/
paginator:
  type: NoPagination
""",
        encoding="utf-8",
    )
    assert load_manifest(path).id == "from_disk"


def test_find_manifests_skips_underscore_and_non_yaml(tmp_path):
    (tmp_path / "a.yaml").write_text("id: a\n", encoding="utf-8")
    (tmp_path / "_template.yaml").write_text("id: t\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    (tmp_path / "a.mapping.yaml").write_text("id: a\n", encoding="utf-8")
    assert [path.name for path in find_manifests(tmp_path)] == ["a.yaml"]


def test_resolve_source_by_id_and_by_path(tmp_path):
    path = tmp_path / "known.yaml"
    path.write_text(
        """
id: known
type: DeclarativeSource
requester:
  type: HttpRequester
  url_base: https://example.org/api/
""",
        encoding="utf-8",
    )
    assert resolve_source("known", tmp_path).id == "known"
    assert resolve_source(str(path), tmp_path).id == "known"
    with pytest.raises(ManifestError, match="unknown source"):
        resolve_source("missing", tmp_path)


def test_repository_examples_are_valid():
    chembl = load_manifest(REPO / "sources" / "chembl_vacv.yaml")
    pubchem = load_manifest(REPO / "sources" / "pubchem_example.yaml")
    github = load_manifest(REPO / "sources" / "github_molecules_example.yaml")
    n1l = load_manifest(REPO / "sources" / "chembl_vaccinia_n1l.yaml")
    assert chembl.id == "chembl_vacv"
    assert chembl.requester.path == "assay.json"
    assert chembl.paginator.type == "NextUrl"
    assert n1l.id == "chembl_vaccinia_n1l"
    assert n1l.requester.path == "activity.json"
    assert pubchem.id == "pubchem_example"
    assert pubchem.paginator.type == "NoPagination"
    assert github.type == "FileSource"
    assert github.download is not None
    assert github.download.filename == "FreeSolv_SAMPL.csv"


def test_custom_template_is_valid():
    template = load_manifest(REPO / "sources" / "_template_custom.yaml")
    assert template.id == "my_source"
    assert template.type == "DeclarativeSource"


def test_unsafe_manifest_ids_are_rejected():
    for bad in (".", "..", "..."):
        with pytest.raises(ManifestError, match="safe directory"):
            parse_manifest(declarative(id=bad))
