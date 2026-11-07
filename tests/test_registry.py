import pandas as pd

from src.registry import build_registry


def test_registry_merges_and_marks_status(tmp_path):
    organizers = tmp_path / "org.csv"
    pd.DataFrame(
        {
            "Title": ["a"],
            "SI VACV": [10.0],
            "SMILES": ["CCO"],
            "source": ["organizers"],
        }
    ).to_csv(organizers, index=False)

    paper = tmp_path / "paper.csv"
    pd.DataFrame(
        {
            "Title": ["p1"],
            "SI VACV": [5.0],
            "SMILES": [""],
            "source": ["paper"],
            "note": ["IC50>50"],
        }
    ).to_csv(paper, index=False)

    frame = build_registry(organizers, None, paper, tmp_path / "found.csv")

    assert len(frame) == 2
    status = dict(zip(frame["Title"], frame["status"]))
    assert status == {"a": "ready", "p1": "needs_smiles"}
    assert (tmp_path / "found.csv").exists()
