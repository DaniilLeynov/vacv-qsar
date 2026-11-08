import pandas as pd

from src.augment_common import augment
from src.sources.base import DATASET_COLUMNS


def _row(title, smiles, si, source):
    data = {column: "" for column in DATASET_COLUMNS}
    data.update(
        {
            "Title": title,
            "SMILES": smiles,
            "SI VACV": si,
            "SI_lower": si,
            "SI_upper": si,
            "source": source,
        }
    )
    return data


def test_augment_keeps_base_and_filters_extra(tmp_path):
    base = pd.DataFrame([_row("base", "CCO", 10.0, "selected")])
    extra = pd.DataFrame(
        [
            _row("dup", "CCO", 5.0, "chembl"),
            _row("zero", "CCC", 0.0, "chembl"),
            _row("huge", "CCN", 20000.0, "chembl"),
            _row("keep1", "CCCl", 50.0, "pubchem"),
            _row("keep2", "CCBr", 20.0, "chembl"),
        ]
    )
    base_path = tmp_path / "base.csv"
    extra_path = tmp_path / "extra.csv"
    out_path = tmp_path / "out.csv"
    base.to_csv(base_path, index=False)
    extra.to_csv(extra_path, index=False)

    result = augment(base_path, extra_path, out_path, max_si=10000)

    assert len(result) == 3
    assert set(result["Title"]) == {"base", "keep1", "keep2"}
    assert result.loc[result["Title"] == "base", "SI VACV"].iloc[0] == 10.0
    assert out_path.exists()
