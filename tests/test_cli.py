from pathlib import Path

from src.fetch.cli import main

REPO = Path(__file__).resolve().parents[1]


def test_list_shows_examples(capsys):
    code = main(["--list", "--sources", str(REPO / "sources")])
    output = capsys.readouterr().out
    assert code == 0
    assert "chembl_vacv" in output
    assert "pubchem_example" in output
    assert "_template_custom" not in output


def test_unknown_source_fails(capsys, tmp_path):
    code = main(["missing", "--sources", str(tmp_path), "--out", str(tmp_path)])
    assert code == 1
    assert "unknown source" in capsys.readouterr().err


def test_dry_run_by_path(capsys, tmp_path):
    manifest = tmp_path / "custom.yaml"
    manifest.write_text(
        """
id: custom
type: DeclarativeSource
requester:
  type: HttpRequester
  url_base: https://example.org/api/
  path: items.json
paginator:
  type: NoPagination
""",
        encoding="utf-8",
    )
    code = main(["custom", "--sources", str(tmp_path), "--out", str(tmp_path), "--dry-run"])
    captured = capsys.readouterr()
    assert code == 0
    assert "GET https://example.org/api/items.json" in captured.out
    assert "dry-run" in captured.out


def _write_manifest(directory, name, base):
    (directory / f"{name}.yaml").write_text(
        f"""
id: {name}
type: DeclarativeSource
requester:
  type: HttpRequester
  url_base: {base}
  path: items.json
paginator:
  type: NoPagination
""",
        encoding="utf-8",
    )


def test_multiple_sources_dry_run(capsys, tmp_path):
    _write_manifest(tmp_path, "alpha", "https://alpha.example.org/")
    _write_manifest(tmp_path, "beta", "https://beta.example.org/")
    code = main(
        ["--sources", str(tmp_path), "--out", str(tmp_path), "--dry-run", "alpha", "beta"]
    )
    output = capsys.readouterr().out
    assert code == 0
    assert "https://alpha.example.org/items.json" in output
    assert "https://beta.example.org/items.json" in output


def test_all_dry_run(capsys, tmp_path):
    _write_manifest(tmp_path, "alpha", "https://alpha.example.org/")
    _write_manifest(tmp_path, "beta", "https://beta.example.org/")
    code = main(["--all", "--sources", str(tmp_path), "--out", str(tmp_path), "--dry-run"])
    output = capsys.readouterr().out
    assert code == 0
    assert "alpha.example.org" in output
    assert "beta.example.org" in output


def test_no_sources_returns_usage(capsys, tmp_path):
    code = main(["--sources", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 2
    assert "no sources specified" in captured.err
