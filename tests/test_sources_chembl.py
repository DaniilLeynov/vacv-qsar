import sqlite3
from pathlib import Path

import pytest

from src.sources.base import SourceError, build_dataset
from src.sources.chembl import ChemblSource

SCHEMA = """
CREATE TABLE assays (assay_id INTEGER, cell_id INTEGER, assay_organism TEXT,
                     assay_type TEXT, description TEXT);
CREATE TABLE activities (assay_id INTEGER, molregno INTEGER, doc_id INTEGER,
                         standard_type TEXT, standard_value REAL,
                         standard_units TEXT, standard_relation TEXT);
CREATE TABLE molecule_dictionary (molregno INTEGER, chembl_id TEXT);
CREATE TABLE compound_structures (molregno INTEGER, canonical_smiles TEXT);
CREATE TABLE cell_dictionary (cell_id INTEGER, cell_name TEXT);
CREATE TABLE docs (doc_id INTEGER, doi TEXT);
"""


def make_db(path: Path) -> Path:
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    connection.executemany(
        "INSERT INTO assays VALUES (?,?,?,?,?)",
        [
            (1, 10, "Homo sapiens", "F", "Antiviral activity against vaccinia virus"),
            (2, 10, "Homo sapiens", "T", "Cytotoxicity in CV-1 cells"),
            (3, 10, "Homo sapiens", "F", "Unrelated assay"),
        ],
    )
    connection.executemany(
        "INSERT INTO molecule_dictionary VALUES (?,?)",
        [(100, "M1"), (101, "M2"), (102, "M3")],
    )
    connection.executemany(
        "INSERT INTO compound_structures VALUES (?,?)",
        [(100, "CCO"), (101, "CCC"), (102, "CCN")],
    )
    connection.execute("INSERT INTO cell_dictionary VALUES (10, 'CV-1')")
    connection.execute("INSERT INTO docs VALUES (500, '10.1000/test')")
    connection.executemany(
        "INSERT INTO activities VALUES (?,?,?,?,?,?,?)",
        [
            (1, 100, 500, "IC50", 10.0, "uM", "="),
            (2, 100, 500, "CC50", 100.0, "uM", "="),
            (1, 101, 500, "IC50", 50.0, "nM", "="),
            (2, 101, 500, "CC50", 5000.0, "nM", "="),
            (1, 102, 500, "IC50", 50.0, "%", "="),
            (3, 100, 500, "IC50", 1.0, "uM", "="),
        ],
    )
    connection.commit()
    connection.close()
    return path


def test_missing_database_raises(tmp_path):
    source = ChemblSource(tmp_path / "missing.db")
    with pytest.raises(SourceError, match="not found"):
        source.fetch_raw(tmp_path)


def test_load_and_build_dataset(tmp_path):
    source = ChemblSource(make_db(tmp_path / "chembl.db"))
    measurements = source.load(tmp_path)

    assert set(measurements["type"]) == {"ic50", "cc50"}
    assert set(measurements["cell_line"]) == {"CV-1"}
    assert "percent is not a concentration" in set(measurements["dropped_reason"])

    dataset, _ = build_dataset(measurements)
    assert len(dataset) == 2

    by_title = dataset.set_index("Title")
    assert by_title.loc["M1", "IC50_uM_VACV"] == pytest.approx(10.0)
    assert by_title.loc["M1", "CC50_uM"] == pytest.approx(100.0)
    assert by_title.loc["M1", "SI VACV"] == pytest.approx(10.0)
    assert by_title.loc["M1", "doi"] == "10.1000/test"
    assert by_title.loc["M2", "SI VACV"] == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("sql", "reason", "rejected_count"),
    [
        ("UPDATE assays SET cell_id=NULL", "unknown cell line", 2),
        (
            "UPDATE activities SET doc_id=501 WHERE standard_type='CC50'",
            "no matching",
            2,
        ),
    ],
)
def test_chembl_drops_unconfirmed_pairs(tmp_path, sql, reason, rejected_count):
    db = make_db(tmp_path / "chembl.db")
    with sqlite3.connect(db) as connection:
        connection.execute(sql)

    measurements = ChemblSource(db).load(tmp_path)
    dataset, unpaired = build_dataset(measurements[measurements["source_id"] == "M1"])

    assert dataset.empty
    assert set(unpaired["type"]) == {"ic50", "cc50"}
    assert len(unpaired) == rejected_count
    assert all(reason in item for item in unpaired["reason"])


def test_replicates_are_aggregated_by_median(tmp_path):
    db = make_db(tmp_path / "chembl.db")
    with sqlite3.connect(db) as connection:
        connection.execute(
            "INSERT INTO activities VALUES (1, 100, 500, 'IC50', 20, 'uM', '=')"
        )

    measurements = ChemblSource(db).load(tmp_path)
    dataset, _ = build_dataset(measurements)

    row = dataset.set_index("Title").loc["M1"]
    assert row["IC50_uM_VACV"] == pytest.approx(15.0)
    assert row["SI VACV"] == pytest.approx(100.0 / 15.0)


def test_chembl_keeps_study_pairs_separate_even_without_doi(tmp_path):
    db = make_db(tmp_path / "chembl.db")
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE docs SET doi=NULL")
        connection.execute("INSERT INTO docs VALUES (501, NULL)")
        connection.executemany(
            "INSERT INTO activities VALUES (?,?,?,?,?,?,?)",
            [(1, 100, 501, "IC50", 5, "uM", "="), (2, 100, 501, "CC50", 200, "uM", "=")],
        )

    measurements = ChemblSource(db).load(tmp_path)
    dataset, unpaired = build_dataset(measurements[measurements["source_id"] == "M1"])

    assert sorted(dataset["SI VACV"]) == [10.0, 40.0]
    assert unpaired.empty


def test_mass_units_use_molecular_weight(tmp_path, monkeypatch):
    db = make_db(tmp_path / "chembl.db")
    connection = sqlite3.connect(db)
    connection.execute("UPDATE activities SET standard_units='ug.mL-1' WHERE molregno=102 AND standard_type='IC50'")
    connection.execute("DELETE FROM activities WHERE molregno=102 AND standard_type='IC50'")
    connection.executemany(
        "INSERT INTO activities VALUES (?,?,?,?,?,?,?)",
        [
            (1, 102, 500, "IC50", 5.0, "ug.mL-1", "="),
            (2, 102, 500, "CC50", 50.0, "ug.mL-1", "="),
        ],
    )
    connection.commit()
    connection.close()

    monkeypatch.setattr("src.sources.chembl.molecular_weight", lambda smiles: 100.0)
    source = ChemblSource(db)
    dataset, _ = build_dataset(source.load(tmp_path))
    row = dataset.set_index("Title").loc["M3"]
    assert row["IC50_uM_VACV"] == pytest.approx(50.0)
    assert row["CC50_uM"] == pytest.approx(500.0)
    assert row["SI VACV"] == pytest.approx(10.0)


def test_from_download_uses_chembl_downloader(tmp_path, monkeypatch):
    import sys

    db = make_db(tmp_path / "chembl.db")

    class FakeModule:
        @staticmethod
        def download_extract_sqlite():
            return db

    monkeypatch.setitem(sys.modules, "chembl_downloader", FakeModule)
    source = ChemblSource.from_download()
    assert source.db_path == db


def test_from_download_without_package(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "chembl_downloader", None)
    with pytest.raises(SourceError, match="chembl-downloader"):
        ChemblSource.from_download()
