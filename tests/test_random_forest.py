import json

import pandas as pd

from src.random_forest import main


def write_tables(tmp_path):
    rows = []
    for index in range(10):
        rows.append(
            {
                "Title": f"c{index}",
                "SMILES": "CCO",
                "SI VACV": float(index + 1),
                "morgan_1": float(index % 2),
                "morgan_2": float(index % 3),
                "culture_clean_BSC-40": float(index % 2),
            }
        )
    train = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(train, index=False)

    test = tmp_path / "test.csv"
    pd.DataFrame(
        {
            "index": [0, 1, 2],
            "morgan_1": [0.0, 1.0, 0.0],
            "morgan_2": [1.0, 0.0, 2.0],
            "culture_clean_BSC-40": [0.0, 1.0, 0.0],
        }
    ).to_csv(test, index=False)
    return train, test


def test_random_forest_train_and_predict(tmp_path):
    train, test = write_tables(tmp_path)
    out = tmp_path / "run"
    submission = tmp_path / "submission.csv"

    assert main(["train", "--train", str(train), "--out", str(out), "--folds", "2"]) == 0
    report = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert report["model"] == "randomforest"
    assert report["rows"] == 10
    assert (out / "model.joblib").exists()

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
