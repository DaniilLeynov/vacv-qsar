import json

import pandas as pd

from src.train import main


def test_train_smoke(tmp_path):
    rows = [
        ("m1", "CCO", 1.0),
        ("m2", "CCC", 2.0),
        ("m3", "CCN", 3.0),
        ("m4", "CCCl", 4.0),
        ("m5", "CCBr", 5.0),
        ("m6", "CC(=O)O", 6.0),
        ("m7", "CCOC", 7.0),
        ("m8", "CC(C)O", 8.0),
    ]
    frame = pd.DataFrame(
        {
            "Title": [row[0] for row in rows],
            "SMILES": [row[1] for row in rows],
            "SI VACV": [row[2] for row in rows],
            "SI_lower": [row[2] for row in rows],
            "SI_upper": [row[2] for row in rows],
            "source": "organizers",
        }
    )
    data = tmp_path / "data.csv"
    frame.to_csv(data, index=False)
    out = tmp_path / "run"

    code = main(
        [
            "--data",
            str(data),
            "--out",
            str(out),
            "--folds",
            "2",
            "--group",
            "smiles",
            "--iterations",
            "5",
            "--jobs",
            "1",
        ]
    )

    assert code == 0
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["rows"] == len(frame)
    assert (out / "oof.csv").exists()
    assert "mse_si" in metrics["catboost"]
