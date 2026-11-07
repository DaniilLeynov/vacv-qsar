import json

import pytest

from src.sources.base import SourceError, build_dataset
from src.sources.pubchem import PubchemSource

HEADER = (
    "PUBCHEM_RESULT_TAG,PUBCHEM_SID,PUBCHEM_CID,PUBCHEM_EXT_DATASOURCE_SMILES,"
    "PUBCHEM_ACTIVITY_OUTCOME,PUBCHEM_ACTIVITY_SCORE,PUBCHEM_ACTIVITY_URL,"
    "PUBCHEM_ASSAYDATA_COMMENT,PubChem Standard Value,Standard Type,"
    "Standard Relation,Standard Value,Standard Text Value,Standard Units,"
    "Data Validity Comment\n"
)


def write_fixture(tmp_path):
    out = tmp_path / "pubchem"
    out.mkdir()
    def row(cid, smiles, standard_value, standard_type, relation, raw_value, units):
        return ",".join(
            ["", "", cid, smiles, "", "", "", "",
             standard_value, standard_type, relation, raw_value, "", units, "Unspecified"]
        )

    rows = [
        row("60613", "CCO", "32.47", "IC50", "=", "32470", "nM"),
        row("60613", "CCO", "100.0", "CC50", "=", "100000", "nM"),
        row("60613", "CCO", "", "IC50", "", "", ""),
    ]
    (out / "aid_1.csv").write_text(HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    summaries = {
        "1": {
            "AID": 1,
            "Name": "Antiviral activity against Vaccinia virus in Vero cells",
            "Description": ["Vero cells, 4 days"],
            "Comment": ["DOI: 10.1234/test"],
            "Target": [{"Name": "Vaccinia virus"}],
        }
    }
    (out / "summaries.json").write_text(json.dumps(summaries), encoding="utf-8")
    return out


def test_pubchem_load_and_pair(tmp_path):
    write_fixture(tmp_path)
    source = PubchemSource()
    measurements = source.load(tmp_path)

    assert len(measurements) == 3
    assert set(measurements["type"]) == {"ic50", "cc50"}
    assert set(measurements["cell_line"]) == {"vero"}
    assert set(measurements["doi"]) == {"10.1234/test"}
    assert "missing standardized value" in set(measurements["dropped_reason"])

    dataset, _ = build_dataset(measurements)
    assert len(dataset) == 1
    row = dataset.iloc[0]
    assert row["IC50_uM_VACV"] == pytest.approx(32.47)
    assert row["CC50_uM"] == pytest.approx(100.0)
    assert row["SI VACV"] == pytest.approx(100.0 / 32.47)
    assert row["doi"] == "10.1234/test"


def test_missing_raw_dir_raises(tmp_path):
    with pytest.raises(SourceError, match="not found"):
        PubchemSource().load(tmp_path)


def test_pubchem_standard_value_defaults_to_micromolar(tmp_path):
    out = tmp_path / "pubchem"
    out.mkdir()
    row = ",,60614,CCO,,,,,32.47,IC50,=,,,,Unspecified"
    (out / "aid_2.csv").write_text(HEADER + row + "\n", encoding="utf-8")

    measurements = PubchemSource().load(tmp_path)
    hit = measurements[(measurements["source_id"] == "60614")].iloc[0]

    assert hit["units"] == "uM"
    assert hit["value_um"] == pytest.approx(32.47)
