import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.chembl_pairs import (
    PAIR_COLUMNS,
    TARGET_COLUMNS,
    build_pairs,
    build_targets,
    load_measurements,
    write_outputs,
)

SCHEMA = """
CREATE TABLE assays (
    assay_id INTEGER, doc_id INTEGER, description TEXT, assay_type TEXT,
    assay_organism TEXT, assay_cell_type TEXT, tid INTEGER, cell_id INTEGER
);
CREATE TABLE activities (
    activity_id INTEGER, assay_id INTEGER, doc_id INTEGER, molregno INTEGER,
    standard_type TEXT, standard_value REAL, standard_units TEXT,
    standard_relation TEXT, standard_flag INTEGER,
    data_validity_comment TEXT, potential_duplicate INTEGER
);
CREATE TABLE molecule_dictionary (molregno INTEGER, chembl_id TEXT);
CREATE TABLE compound_structures (molregno INTEGER, canonical_smiles TEXT);
CREATE TABLE cell_dictionary (cell_id INTEGER, cell_name TEXT);
CREATE TABLE docs (doc_id INTEGER, doi TEXT);
CREATE TABLE target_dictionary (tid INTEGER, pref_name TEXT, organism TEXT);
"""


def make_db(path: Path) -> Path:
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    connection.executemany(
        "INSERT INTO target_dictionary VALUES (?,?,?)",
        [
            (1, "Vaccinia virus", "Vaccinia virus"),
            (2, "Herpes simplex virus", "Herpes simplex virus"),
            (3, "EGFR", "Homo sapiens"),
        ],
    )
    connection.executemany(
        "INSERT INTO cell_dictionary VALUES (?,?)",
        [(10, "CV-1"), (11, "Vero")],
    )
    connection.executemany(
        "INSERT INTO docs VALUES (?,?)",
        [(500, "10.1000/a"), (501, "10.1000/b")],
    )
    connection.executemany(
        "INSERT INTO molecule_dictionary VALUES (?,?)",
        [
            (100, "M1"),
            (101, "M2"),
            (102, "M3"),
            (103, "M4"),
            (104, "M5"),
            (105, "M6"),
            (106, "M7"),
            (107, "M8"),
        ],
    )
    connection.executemany(
        "INSERT INTO compound_structures VALUES (?,?)",
        [
            (100, "CCO"),
            (101, "CCC"),
            (102, "CCN"),
            (103, "CCCl"),
            (104, "CCBr"),
            (105, "CCF"),
            (106, "CCI"),
            (107, "CC#N"),
        ],
    )
    connection.executemany(
        "INSERT INTO assays VALUES (?,?,?,?,?,?,?,?)",
        [
            (1, 500, "Antiviral activity against vaccinia virus", "F",
             "Homo sapiens", "CV-1", 1, 10),
            (2, 500, "Cytotoxicity in CV-1 cells", "T",
             "Homo sapiens", "CV-1", None, 10),
            (3, 500, "Antiviral activity against herpes virus", "F",
             "Homo sapiens", "Vero", 2, 11),
            (4, 501, "EGFR binding", "B", "Homo sapiens", "Vero", 3, 11),
            (5, 501, "EGFR functional assay", "F", "Homo sapiens", "Vero", 3, 11),
        ],
    )
    activities = [
        (1, 1, 500, 100, "IC50", 10.0, "uM", "=", 1, None, 0),
        (2, 2, 500, 100, "CC50", 100.0, "uM", "=", 1, None, 0),
        (3, 1, 500, 101, "IC50", 50.0, "nM", "=", 1, None, 0),
        (4, 2, 500, 101, "CC50", 5000.0, "nM", "=", 1, None, 0),
        (5, 3, 500, 102, "IC50", 1.0, "uM", "=", 1, None, 0),
        (6, 2, 500, 102, "CC50", 50.0, "uM", "=", 1, None, 0),
        (7, 1, 500, 103, "IC50", 10.0, "uM", ">", 1, None, 0),
        (8, 2, 500, 103, "CC50", 100.0, "uM", "=", 1, None, 0),
        (9, 1, 500, 104, "IC50", 10.0, "uM", "=", 1, None, 1),
        (10, 2, 500, 104, "CC50", 100.0, "uM", "=", 1, None, 1),
        (11, 1, 500, 105, "IC50", 10.0, "%", "=", 1, None, 0),
        (12, 2, 500, 105, "CC50", 100.0, "uM", "=", 1, None, 0),
        (13, 1, 500, 106, "IC50", 20.0, "uM", "=", 1, None, 0),
        (14, 2, 500, 106, "CC50", 200.0, "ug.mL-1", "=", 1, None, 0),
        (15, 5, 501, 107, "IC50", 2.0, "uM", "=", 1, None, 0),
        (16, 2, 500, 107, "CC50", 80.0, "uM", "=", 1, None, 0),
    ]
    connection.executemany(
        "INSERT INTO activities VALUES (?,?,?,?,?,?,?,?,?,?,?)", activities
    )
    connection.commit()
    connection.close()
    return path


def test_extract_targets_and_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr("src.chembl_pairs.molecular_weight", lambda smiles: 100.0)
    db = make_db(tmp_path / "chembl.db")
    measurements = load_measurements(db, scope="viral")

    dropped = measurements[measurements["dropped_reason"] != ""]
    reasons = set(dropped["dropped_reason"])
    assert "not a viral IC50" in reasons
    assert "non-positive value" not in reasons  # percent fails earlier
    assert "percent is not a concentration" in reasons

    targets = build_targets(measurements)
    assert list(targets.columns) == TARGET_COLUMNS
    pairs = build_pairs(measurements)
    assert list(pairs.columns) == PAIR_COLUMNS

    by_title = pairs.set_index("Title")
    assert by_title.loc["M1", "SI"] == pytest.approx(10.0)
    assert by_title.loc["M2", "SI"] == pytest.approx(100.0)
    assert by_title.loc["M7", "SI"] == pytest.approx(100.0)
    assert "M3" not in by_title.index

    vacv = targets[(targets["endpoint"] == "ic50") & targets["is_vacv"]]
    assert set(vacv["Title"]) == {"M1", "M2", "M7"}


def test_exact_relation_and_duplicates_are_dropped(tmp_path):
    db = make_db(tmp_path / "chembl.db")
    measurements = load_measurements(db, scope="viral")

    # The censored IC50 is excluded by the SQL filter, so only CC50 remains.
    m4 = measurements[measurements["Title"] == "M4"]
    assert set(m4["endpoint"]) == {"cc50"}
    assert (m4["dropped_reason"] == "").all()

    # potential_duplicate = 1 is filtered out before it reaches pandas.
    assert measurements[measurements["Title"] == "M5"].empty

    m6 = measurements[measurements["Title"] == "M6"]
    assert "percent is not a concentration" in set(m6["dropped_reason"])


def test_exclude_smiles(tmp_path):
    db = make_db(tmp_path / "chembl.db")
    exclude = tmp_path / "exclude.csv"
    exclude.write_text("SMILES\nCCO\n", encoding="utf-8")
    measurements = load_measurements(db, scope="viral", exclude_smiles=(exclude,))
    m1 = measurements[measurements["Title"] == "M1"]
    assert (m1["dropped_reason"] == "excluded SMILES").all()


def test_empty_measurements_build_empty_tables():
    empty = pd.DataFrame(columns=["dropped_reason"])
    assert build_targets(empty).empty
    assert build_pairs(empty).empty


def test_write_outputs_append(tmp_path):
    first = pd.DataFrame({"Title": ["a"]})
    second = pd.DataFrame({"Title": ["b"]})
    targets = pd.DataFrame(columns=TARGET_COLUMNS)
    pairs = pd.DataFrame(columns=PAIR_COLUMNS)

    write_outputs(first, targets, pairs, tmp_path)
    write_outputs(second, targets, pairs, tmp_path, append=True)

    result = pd.read_csv(tmp_path / "chembl_measurements.csv")
    assert list(result["Title"]) == ["a", "b"]
