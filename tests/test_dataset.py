import pandas as pd
import pytest

from src.dataset import build, main
from src.sources.base import DATASET_COLUMNS
from src.sources.generic_csv import CsvSource


def write_source(tmp_path):
    (tmp_path / "table.csv").write_text(
        "SMILES,IC50_uM,CC50_uM,DOI,Cell\n"
        "CCO,10,100,10.1/x,CV-1\n"
        "CCC,0.5,50,10.1/y,CV-1\n"
        "CCN,20,,10.1/z,CV-1\n"
        "CCCl,-5,50,10.1/w,CV-1\n",
        encoding="utf-8",
    )
    mapping = tmp_path / "lab.mapping.yaml"
    mapping.write_text(
        "\n".join(
            [
                "id: lab",
                f"file: {tmp_path / 'table.csv'}",
                "smiles: SMILES",
                "ic50: IC50_uM",
                "cc50: CC50_uM",
                "doi: DOI",
                "cell_line: Cell",
                "units: uM",
            ]
        ),
        encoding="utf-8",
    )
    return mapping


def test_build_returns_valid_rows_and_dropped(tmp_path):
    source = CsvSource.from_config(write_source(tmp_path))
    dataset, dropped, problems = build([source], tmp_path)

    assert list(dataset.columns) == DATASET_COLUMNS
    assert len(dataset) == 2
    reasons = set(dropped["dropped_reason"])
    assert "non-positive value" in reasons
    assert "missing value" in reasons
    assert problems.empty


def test_cli_writes_outputs(tmp_path, capsys):
    mapping = write_source(tmp_path)
    out = tmp_path / "dataset.csv"
    dropped = tmp_path / "dropped.csv"

    code = main(
        [
            "--csv",
            str(mapping),
            "--raw-dir",
            str(tmp_path),
            "--out",
            str(out),
            "--dropped",
            str(dropped),
        ]
    )

    assert code == 0
    assert out.exists() and dropped.exists()
    frame = pd.read_csv(out)
    assert list(frame.columns) == DATASET_COLUMNS
    assert len(frame) == 2
    assert "rows" in capsys.readouterr().out


def test_cli_without_sources_returns_usage(tmp_path, capsys):
    code = main(["--raw-dir", str(tmp_path)])
    assert code == 2
    assert "no sources specified" in capsys.readouterr().err


def test_cli_missing_mapping_reports_error(tmp_path, capsys):
    code = main(
        [
            "--csv",
            str(tmp_path / "missing.mapping.yaml"),
            "--out",
            str(tmp_path / "out.csv"),
            "--dropped",
            str(tmp_path / "dropped.csv"),
        ]
    )
    assert code == 1
    assert "ERROR" in capsys.readouterr().err


def test_cli_chembl_flag_uses_download(monkeypatch, tmp_path):
    from src.sources.base import MEASUREMENT_COLUMNS

    class FakeChembl:
        id = "chembl"

        def fetch_raw(self, dest, refresh=False):
            return None

        def load(self, raw_dir):
            return pd.DataFrame(
                [
                    {
                        "source": "chembl",
                        "source_id": "M1",
                        "title": "M1",
                        "smiles": "CCO",
                        "type": "ic50",
                        "measure": "IC50",
                        "value": 10.0,
                        "units": "uM",
                        "value_um": 10.0,
                        "relation": "=",
                        "cell_line": "CV-1",
                        "organism": "Vaccinia virus",
                        "doi": "10.1/x",
                        "dropped_reason": "",
                    },
                    {
                        "source": "chembl",
                        "source_id": "M1",
                        "title": "M1",
                        "smiles": "CCO",
                        "type": "cc50",
                        "measure": "CC50",
                        "value": 100.0,
                        "units": "uM",
                        "value_um": 100.0,
                        "relation": "=",
                        "cell_line": "CV-1",
                        "organism": "Vaccinia virus",
                        "doi": "10.1/x",
                        "dropped_reason": "",
                    },
                ],
                columns=MEASUREMENT_COLUMNS,
            )

    monkeypatch.setattr(
        "src.dataset.ChemblSource.from_download", staticmethod(lambda: FakeChembl())
    )
    out = tmp_path / "chembl.csv"
    code = main(["--chembl", "--out", str(out), "--dropped", str(tmp_path / "d.csv")])

    assert code == 0
    frame = pd.read_csv(out)
    assert len(frame) == 1
    assert frame.loc[0, "SI VACV"] == 10.0


def write_simple_source(tmp_path, rows):
    (tmp_path / "table.csv").write_text(rows, encoding="utf-8")
    mapping = tmp_path / "lab.mapping.yaml"
    mapping.write_text(
        "\n".join(
            [
                "id: lab",
                f"file: {tmp_path / 'table.csv'}",
                "smiles: SMILES",
                "ic50: IC50_uM",
                "cc50: CC50_uM",
                "units: uM",
            ]
        ),
        encoding="utf-8",
    )
    return CsvSource.from_config(mapping)


def test_censored_measurement_produces_bounds(tmp_path):
    (tmp_path / "table.csv").write_text(
        "SMILES,IC50_uM,CC50_uM,Culture\nCCO,>50,100,CV-1\n",
        encoding="utf-8",
    )
    mapping = tmp_path / "lab.mapping.yaml"
    mapping.write_text(
        "\n".join(
            [
                "id: lab",
                f"file: {tmp_path / 'table.csv'}",
                "smiles: SMILES",
                "ic50: IC50_uM",
                "cc50: CC50_uM",
                "units: uM",
                "cell_line: Culture",
            ]
        ),
        encoding="utf-8",
    )
    source = CsvSource.from_config(mapping)
    dataset, dropped, problems = build([source], tmp_path)

    assert len(dataset) == 1
    row = dataset.iloc[0]
    assert pd.isna(row["SI VACV"])
    assert row["SI_lower"] == pytest.approx(0.0)
    assert row["SI_upper"] == pytest.approx(2.0)
    assert row["IC50_uM_VACV"] == pytest.approx(50.0)
    assert row["CC50_uM"] == pytest.approx(100.0)
    assert problems.empty


def test_unpaired_measurement_is_reported(tmp_path):
    source = write_simple_source(tmp_path, "SMILES,IC50_uM,CC50_uM\nCCO,10,\n")
    dataset, dropped, problems = build([source], tmp_path)

    assert dataset.empty
    reasons = " ".join(dropped["dropped_reason"].astype(str))
    assert "no matching CC50 on the same cell line" in reasons
    assert problems.empty
