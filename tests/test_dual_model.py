import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.dual_model import (
    _calibrate,
    _si_metrics,
    aggregate_targets,
    main,
    organizer_frame,
    predict_si_frame,
)
from src.dual_predict import main as predict_main


def write_targets(path: Path) -> Path:
    rows = []
    for index, smiles in enumerate(
        ["CCO", "CCC", "CCN", "CCCl", "CCBr", "CCF", "CCI", "CC#N", "CC=O", "CC(C)O"]
    ):
        rows.append(
            {
                "Title": f"T{index}",
                "SMILES": smiles,
                "smiles_key": smiles,
                "endpoint": "ic50",
                "y_log10_um": -1.0 - index * 0.1,
                "n_measurements": 2,
                "cell_line": "vero",
                "organism": "Vaccinia virus",
                "is_vacv": True,
                "is_viral": True,
            }
        )
        rows.append(
            {
                "Title": f"T{index}",
                "SMILES": smiles,
                "smiles_key": smiles,
                "endpoint": "cc50",
                "y_log10_um": 1.0 + index * 0.1,
                "n_measurements": 3,
                "cell_line": "vero",
                "organism": "Vaccinia virus",
                "is_vacv": False,
                "is_viral": False,
            }
        )
    path.write_text(pd.DataFrame(rows).to_csv(index=False), encoding="utf-8")
    return path


def write_organizers(path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "Title": ["O1", "O2", "O3"],
            "SMILES": ["CCS", "CCP", "CCBr"],
            "CC50_uM": [100.0, 200.0, 50.0],
            "IC50_uM_VACV": [10.0, 20.0, 5.0],
            "SI VACV": [10.0, 10.0, 10.0],
            "Culture": ["Vero", "Vero", "HeLa"],
            "strain": ["Vaccinia virus"] * 3,
        }
    )
    path.write_text(frame.to_csv(index=False), encoding="utf-8")
    return path


def test_aggregate_targets_and_organizer_frame(tmp_path):
    frame = pd.DataFrame(
        {
            "Title": ["a", "a", "b"],
            "SMILES": ["CCO", "CCO", "CCC"],
            "smiles_key": ["CCO", "CCO", "CCC"],
            "endpoint": ["ic50", "ic50", "ic50"],
            "y_log10_um": [-1.0, -3.0, -2.0],
            "n_measurements": [1, 1, 1],
            "cell_line": ["vero", "hela", "vero"],
            "organism": ["Vaccinia virus", "Vaccinia virus", "Vaccinia virus"],
            "is_vacv": [True, True, True],
            "is_viral": [True, True, True],
        }
    )
    aggregated = aggregate_targets(frame)
    assert len(aggregated) == 2
    assert aggregated.loc[aggregated["smiles_key"] == "CCO", "y_log10_um"].iloc[0] == -2.0

    eval_frame, long_frame = organizer_frame(
        write_organizers(tmp_path / "org.csv")
    )
    assert len(eval_frame) == 3
    assert set(long_frame["endpoint"]) == {"ic50", "cc50"}
    assert set(long_frame["context_key"]) == {"vacv", "cell:vero", "cell:hela"}


def test_calibration_and_metrics_are_finite():
    y = np.array([1.0, 10.0, 100.0])
    pred = np.array([2.0, 20.0, 50.0])
    calibration = _calibrate(y, pred)
    assert set(calibration) == {"a", "b", "n"}
    metrics = _si_metrics(y, pred)
    assert metrics["n"] == 3
    assert np.isfinite(metrics["rmse_si"])


def test_dual_model_smoke(tmp_path):
    targets = write_targets(tmp_path / "targets.csv")
    organizers = write_organizers(tmp_path / "organizers.csv")
    test = organizers
    out = tmp_path / "runs"

    code = main(
        [
            "--targets",
            str(targets),
            "--organizers",
            str(organizers),
            "--test",
            str(test),
            "--out",
            str(out),
            "--folds",
            "2",
            "--iterations",
            "2",
            "--no-context",
            "--no-finetune",
            "--group",
            "smiles",
        ]
    )

    assert code == 0
    assert (out / "base_ic50.cbm").exists()
    assert (out / "base_cc50.cbm").exists()
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["si"]["calibrated"]["n"] == 3

    frame = pd.read_csv(organizers)
    predicted, si, log_si = predict_si_frame(frame, out)
    assert len(predicted) == 3
    assert len(si) == 3
    assert np.isfinite(si).all()
    assert np.isfinite(log_si).all()

    finetuned_out = tmp_path / "runs-finetuned"
    code = main(
        [
            "--targets",
            str(targets),
            "--organizers",
            str(organizers),
            "--test",
            str(test),
            "--out",
            str(finetuned_out),
            "--folds",
            "2",
            "--iterations",
            "2",
            "--fine-tune-iterations",
            "2",
            "--no-context",
            "--group",
            "smiles",
        ]
    )
    assert code == 0
    assert (finetuned_out / "final_ic50.cbm").exists()
    assert (finetuned_out / "final_cc50.cbm").exists()

    test = pd.DataFrame(
        {
            "index": [0, 1],
            "SMILES": ["CCS", "not-a-smiles"],
            "Culture": ["Vero", "Vero"],
            "strain": ["Vaccinia virus", "Vaccinia virus"],
        }
    )
    test_path = tmp_path / "test.csv"
    test.to_csv(test_path, index=False)
    submission_path = tmp_path / "submission.csv"
    code = predict_main(
        [
            "--model",
            str(out),
            "--test",
            str(test_path),
            "--out",
            str(submission_path),
        ]
    )
    assert code == 0
    submission = pd.read_csv(submission_path)
    assert list(submission.columns) == ["index", "SI VACV"]
    assert len(submission) == 2
    assert np.isfinite(submission["SI VACV"]).all()
