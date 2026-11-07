import numpy as np
import pandas as pd

from src.rf_robust import (
    MORGAN,
    POSITION_COLUMNS,
    RobustForest,
    _metrics,
    prepare_test_frame,
)
from src.rf_robust_best import RF_COLUMNS

TRAIN = "data/rf_robust/train_407.csv"


def test_prepare_test_frame_builds_expected_columns():
    frame = pd.DataFrame(
        {
            "index": [0],
            "SMILES": ["CCO"],
            "Culture": ["Vero"],
            "strain": ["Vaccinia virus"],
        }
    )
    prepared = prepare_test_frame(frame)
    assert prepared.loc[0, "canonical_smiles"] == "CCO"
    assert prepared.loc[0, "culture_clean"] == "Vero"
    assert prepared.loc[0, "strain_clean"] == "Unknown_VACV"
    assert bool(prepared.loc[0, "culture_clean_BSC-40"]) is False
    assert set(MORGAN).issubset(prepared.columns)
    assert set(POSITION_COLUMNS).issubset(prepared.columns)


def test_original_robust_forest_smoke():
    frame = pd.read_csv(TRAIN).head(20)
    model = RobustForest().fit(frame)
    predictions = model.predict(frame.head(5))
    assert predictions.shape == (5,)
    assert np.isfinite(predictions).all()
    assert len(RF_COLUMNS) == 38


def test_metrics_are_finite():
    metrics = _metrics(np.array([1.0, 10.0]), np.array([2.0, 8.0]))
    assert metrics["n"] == 2
    assert metrics["mse_si"] > 0
    assert np.isfinite(metrics["rmse_log1p"])
