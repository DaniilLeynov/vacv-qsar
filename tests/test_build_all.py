import json

import pandas as pd

from src.build_all import build


PUBCHEM_HEADER = (
    "PUBCHEM_RESULT_TAG,PUBCHEM_SID,PUBCHEM_CID,PUBCHEM_EXT_DATASOURCE_SMILES,"
    "PUBCHEM_ACTIVITY_OUTCOME,PUBCHEM_ACTIVITY_SCORE,PUBCHEM_ACTIVITY_URL,"
    "PUBCHEM_ASSAYDATA_COMMENT,PubChem Standard Value,Standard Type,"
    "Standard Relation,Standard Value,Standard Text Value,Standard Units,"
    "Data Validity Comment\n"
)


def write_pubchem(raw):
    out = raw / "pubchem"
    out.mkdir(parents=True)
    rows = [
        ",,60613,CCO,,,,,32.47,IC50,=,,,,Unspecified",
        ",,60613,CCO,,,,,100.0,CC50,=,,,,Unspecified",
    ]
    (out / "aid_1.csv").write_text(
        PUBCHEM_HEADER + "\n".join(rows) + "\n", encoding="utf-8"
    )
    summaries = {
        "1": {
            "AID": 1,
            "Name": "Antiviral activity against Vaccinia virus in Vero cells",
            "Description": ["Vero cells"],
            "Comment": ["DOI: 10.1234/test"],
            "Target": [{"Name": "Vaccinia virus"}],
        }
    }
    (out / "summaries.json").write_text(json.dumps(summaries), encoding="utf-8")


def write_organizers(tmp_path):
    data = tmp_path / "organizers.csv"
    data.write_text(
        "Title,CC50_uM,IC50_uM_VACV,SI VACV,SMILES,Culture,strain\n"
        "org1,100,10,10,CCO,Vero,Vaccinia virus\n",
        encoding="utf-8",
    )
    mapping = tmp_path / "organizers.mapping.yaml"
    mapping.write_text(
        "\n".join(
            [
                "id: organizers",
                f"file: {data}",
                "smiles: SMILES",
                "title: Title",
                "target: SI VACV",
                "ic50: IC50_uM_VACV",
                "cc50: CC50_uM",
                "units: uM",
                "cell_line: Culture",
                "organism: strain",
            ]
        ),
        encoding="utf-8",
    )
    return mapping


def write_chembl(tmp_path):
    path = tmp_path / "chembl_measurements.csv"
    frame = pd.DataFrame(
        [
            {
                "source": "chembl",
                "source_id": "CHEMBL1",
                "Title": "CHEMBL1",
                "SMILES": "CCC",
                "smiles_key": "CCC",
                "endpoint": "ic50",
                "doc_id": "500",
                "target_organism": "Vaccinia virus",
                "assay_organism": "Vaccinia virus",
                "value": 10.0,
                "units": "uM",
                "value_um": 10.0,
                "log10_um": 1.0,
                "relation": "=",
                "cell_line": "Vero",
                "context_key": "vacv:vero",
                "is_vacv": True,
                "is_viral": True,
                "doi": "10.1/x",
                "dropped_reason": "",
            },
            {
                "source": "chembl",
                "source_id": "CHEMBL1",
                "Title": "CHEMBL1",
                "SMILES": "CCC",
                "smiles_key": "CCC",
                "endpoint": "cc50",
                "doc_id": "500",
                "target_organism": "Homo sapiens",
                "assay_organism": "Homo sapiens",
                "value": 100.0,
                "units": "uM",
                "value_um": 100.0,
                "log10_um": 2.0,
                "relation": "=",
                "cell_line": "Vero",
                "context_key": "cell:vero",
                "is_vacv": False,
                "is_viral": False,
                "doi": "10.1/x",
                "dropped_reason": "",
            },
        ]
    )
    path.write_text(frame.to_csv(index=False), encoding="utf-8")
    return path


def test_build_combines_three_sources(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_pubchem(raw)
    organizers = write_organizers(tmp_path)
    chembl = write_chembl(tmp_path)

    dataset, measurements, targets, dropped = build(
        organizers=organizers,
        chembl=chembl,
        pubchem_sets=("pubchem",),
        raw_dir=raw,
        exclude_smiles=(),
    )

    assert set(dataset["source"]) == {"organizers", "pubchem", "chembl"}
    assert len(dataset) == 3
    assert set(targets["source"]) == {"organizers", "pubchem", "chembl"}
    assert set(targets["endpoint"]) == {"ic50", "cc50"}
    assert not measurements.empty
    assert isinstance(dropped, pd.DataFrame)
