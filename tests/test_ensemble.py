import json

import joblib
import pandas as pd

from src.ensemble import main


def write_tables(tmp_path):
    smiles = ["CCO", "CCC", "CCN", "CCCl", "CCBr", "CC(=O)O", "CCOC", "CC(C)O"]
    cultures = ["Vero", "Vero", "HeLa", "HeLa", "Vero", "A549", "A549", "Vero"]
    train = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "Title": [f"m{i}" for i in range(len(smiles))],
            "SMILES": smiles,
            "SI VACV": [float(i + 1) for i in range(len(smiles))],
            "SI_lower": [float(i + 1) for i in range(len(smiles))],
            "SI_upper": [float(i + 1) for i in range(len(smiles))],
        }
    ).to_csv(train, index=False)

    organizers = tmp_path / "organizers.csv"
    pd.DataFrame(
        {"SMILES": smiles, "Culture": cultures}
    ).to_csv(organizers, index=False)

    test = tmp_path / "test.csv"
    pd.DataFrame(
        {
            "index": [0, 1, 2],
            "SMILES": ["CCO", "CCCl", "CCN"],
            "Culture": ["Vero", "HeLa", "HeLa"],
        }
    ).to_csv(test, index=False)
    return train, organizers, test


def test_ensemble_train_and_predict(tmp_path):
    train, organizers, test = write_tables(tmp_path)
    out = tmp_path / "run"
    submission = tmp_path / "submission.csv"

    assert (
        main(
            [
                "train",
                "--train",
                str(train),
                "--organizers",
                str(organizers),
                "--out",
                str(out),
                "--folds",
                "2",
            ]
        )
        == 0
    )

    report = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert report["model"] == "ensemble"
    assert report["rows"] == 8
    assert "mse_si" in report["metrics"]
    assert "rmse_si" in report["metrics"]

    payload = joblib.load(out / "model.joblib")
    assert payload["culture_encoder"] is not None

    assert (
        main(
            [
                "predict",
                "--model",
                str(out),
                "--test",
                str(test),
                "--out",
                str(submission),
            ]
        )
        == 0
    )
    frame = pd.read_csv(submission)
    assert list(frame.columns) == ["index", "SI VACV"]
    assert len(frame) == 3
    assert not frame["SI VACV"].isna().any()
