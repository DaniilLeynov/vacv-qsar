import pandas as pd

from src.merge import merge


def test_merge_dedup_and_test_exclusion(tmp_path):
    base = tmp_path / "base.csv"
    pd.DataFrame(
        {
            "Title": ["a", "b"],
            "SMILES": ["CCO", "CCN"],
            "SI VACV": [1.0, 2.0],
            "CC50_uM": ["", ""],
            "IC50_uM_VACV": ["", ""],
            "SI_lower": [1.0, 2.0],
            "SI_upper": [1.0, 2.0],
            "strain": ["", ""],
            "Culture": ["Vero", "HeLa"],
            "doi": ["", ""],
            "source": ["selected", "selected"],
        }
    ).to_csv(base, index=False)

    extra = tmp_path / "extra.csv"
    pd.DataFrame(
        {
            "Title": ["c", "d"],
            "SMILES": ["CCC", "CCO"],
            "SI VACV": [3.0, 9.0],
            "CC50_uM": ["", ""],
            "IC50_uM_VACV": ["", ""],
            "SI_lower": [3.0, 9.0],
            "SI_upper": [3.0, 9.0],
            "strain": ["", ""],
            "Culture": ["Vero", "Vero"],
            "doi": ["", ""],
            "source": ["pubchem", "pubchem"],
        }
    ).to_csv(extra, index=False)

    test = tmp_path / "test.csv"
    pd.DataFrame({"SMILES": ["CCC"]}).to_csv(test, index=False)

    out = tmp_path / "merged.csv"
    merged = merge([base, extra], test, out)

    # "CCO" duplicate is dropped (selected wins), "CCC" excluded (in test).
    assert len(merged) == 2
    assert set(merged["SMILES"]) == {"CCO", "CCN"}
    assert merged.loc[merged["SMILES"] == "CCO", "SI VACV"].iloc[0] == 1.0
    assert list(merged.columns) == [
        "Title",
        "CC50_uM",
        "IC50_uM_VACV",
        "SI VACV",
        "SI_lower",
        "SI_upper",
        "SMILES",
        "strain",
        "Culture",
        "doi",
        "source",
    ]


def test_merge_without_test(tmp_path):
    base = tmp_path / "base.csv"
    pd.DataFrame(
        {
            "Title": ["a"],
            "SMILES": ["CCO"],
            "SI VACV": [1.0],
            "CC50_uM": [""],
            "IC50_uM_VACV": [""],
            "SI_lower": [1.0],
            "SI_upper": [1.0],
            "strain": [""],
            "Culture": ["Vero"],
            "doi": [""],
            "source": ["selected"],
        }
    ).to_csv(base, index=False)

    merged = merge([base], None, tmp_path / "merged.csv")
    assert len(merged) == 1
