from pathlib import Path

import pandas as pd
import pytest

from src.dataset import build
from src.sources.base import (
    MEASUREMENT_COLUMNS,
    SourceError,
    build_dataset,
    validate_dataset,
)
from src.sources.generic_csv import CsvSource


def write_source(tmp_path: Path) -> Path:
    (tmp_path / "table.csv").write_text(
        "SMILES,IC50_uM,CC50_uM,DOI,Cell\n"
        "CCO,10,100,10.1/x,CV-1\n"
        "CCC,0.5,50,10.1/y,CV-1\n"
        "CCN,,50,10.1/z,CV-1\n",
        encoding="utf-8",
    )
    mapping = tmp_path / "lab.mapping.yaml"
    mapping.write_text(
        "\n".join(
            [
                "id: lab",
                "file: " + str(tmp_path / "table.csv"),
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


def test_csv_source_builds_dataset(tmp_path):
    source = CsvSource.from_config(write_source(tmp_path))
    dataset, _ = build_dataset(source.load(tmp_path))

    assert len(dataset) == 2
    by_title = dataset.set_index("Title")
    assert by_title.loc["row#0", "SI VACV"] == pytest.approx(10.0)
    assert by_title.loc["row#1", "SI VACV"] == pytest.approx(100.0)
    assert by_title.loc["row#0", "doi"] == "10.1/x"


def test_missing_input_file_raises(tmp_path):
    mapping = tmp_path / "lab.mapping.yaml"
    mapping.write_text(
        "\n".join(
            [
                "id: lab",
                "file: " + str(tmp_path / "missing.csv"),
                "smiles: SMILES",
                "ic50: IC50",
                "cc50: CC50",
            ]
        ),
        encoding="utf-8",
    )
    source = CsvSource.from_config(mapping)
    with pytest.raises(SourceError, match="not found"):
        source.fetch_raw(tmp_path)


def test_target_only_source(tmp_path):
    (tmp_path / "table.csv").write_text(
        "Title,SMILES,SI VACV,CC50_uM,Culture\n"
        "A,CCO,10,100,CV-1\n"
        "B,CCC,10000,,Vero\n",
        encoding="utf-8",
    )
    mapping = tmp_path / "lab.mapping.yaml"
    mapping.write_text(
        "\n".join(
            [
                "id: lab",
                f"file: {tmp_path / 'table.csv'}",
                "smiles: SMILES",
                "title: Title",
                "target: SI VACV",
                "cc50: CC50_uM",
                "units: uM",
                "cell_line: Culture",
            ]
        ),
        encoding="utf-8",
    )
    source = CsvSource.from_config(mapping)
    dataset, _ = build_dataset(source.load(tmp_path))

    by_title = dataset.set_index("Title")
    assert by_title.loc["A", "SI VACV"] == pytest.approx(10.0)
    assert by_title.loc["A", "CC50_uM"] == pytest.approx(100.0)
    assert by_title.loc["B", "SI VACV"] == pytest.approx(10000.0)
    assert pd.isna(by_title.loc["B", "IC50_uM_VACV"])


@pytest.mark.parametrize("source_id", ["organizers", "lab"])
def test_organizer_cap_can_differ_from_concentration_ratio(tmp_path, source_id):
    table = tmp_path / "table.csv"
    table.write_text(
        "SMILES,IC50,CC50,SI,Culture\nCCO,0.01,200,10000,Vero\n",
        encoding="utf-8",
    )
    mapping = tmp_path / "source.mapping.yaml"
    mapping.write_text(
        f"id: {source_id}\nfile: {table}\nsmiles: SMILES\n"
        "ic50: IC50\ncc50: CC50\ntarget: SI\ncell_line: Culture\n",
        encoding="utf-8",
    )
    dataset, dropped, problems = build([CsvSource.from_config(mapping)], tmp_path)

    assert dropped.empty
    if source_id == "organizers":
        assert dataset["SI VACV"].tolist() == [10000.0]
        assert problems.empty
    else:
        assert dataset.empty
        assert problems["reason"].tolist() == ["SI does not equal CC50/IC50"]


def test_bad_row_does_not_drop_other_rows_with_the_same_title():
    def measurement(source_id, smiles):
        return {
            "source": "lab",
            "source_id": source_id,
            "title": "1",
            "smiles": smiles,
            "type": "si",
            "measure": "SI",
            "value": 10.0,
            "units": "",
            "value_um": 10.0,
            "relation": "=",
            "cell_line": "CV-1",
            "organism": "Vaccinia virus",
            "doi": "",
            "dropped_reason": "",
        }

    measurements = pd.DataFrame(
        [
            measurement("1#0", "CCO"),
            measurement("1#1", ""),
        ],
        columns=MEASUREMENT_COLUMNS,
    )
    dataset, _ = build_dataset(measurements)
    valid, problems = validate_dataset(dataset)

    assert len(dataset) == 2
    assert len(valid) == 1
    assert valid.iloc[0]["SMILES"] == "CCO"
    assert len(problems) == 1
