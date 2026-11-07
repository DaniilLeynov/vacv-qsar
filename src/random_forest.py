from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

TARGET = "SI VACV"
METADATA = {"index", "Title", "SMILES", "scaffold_smiles", "canonical_smiles", TARGET}
PARAMETERS = {
    "n_estimators": 500,
    "min_samples_leaf": 2,
    "max_features": 1.0,
    "criterion": "squared_error",
    "random_state": 42,
    "n_jobs": 2,
}

DEFAULT_TRAIN = Path("data/rf/train.csv")
DEFAULT_TEST = Path("data/rf/test.csv")
DEFAULT_OUT = Path("runs/randomforest")
DEFAULT_SUBMISSION = Path("submissions/submission.csv")


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def predictor_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in METADATA]


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="raise").astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("missing or infinite numeric values")
    return values


def check_features(columns: list[str]) -> None:
    for column in columns:
        if not (column.startswith("morgan_") or column.startswith("culture_clean_")):
            raise ValueError(f"unexpected predictor: {column}")


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    error = y_pred - y_true
    mse = float(np.mean(error**2))
    return {
        "mse_si": mse,
        "rmse_si": math.sqrt(mse),
        "mae_si": float(np.mean(np.abs(error))),
        "dummy_mean_mse": float(np.mean((y_true.mean() - y_true) ** 2)),
        "dummy_median_mse": float(np.mean((np.median(y_true) - y_true) ** 2)),
    }


def train(args) -> int:
    train_frame = read(args.train)
    features = predictor_columns(train_frame)
    check_features(features)

    x = numeric(train_frame, features)
    y = numeric(train_frame, [TARGET])[TARGET].to_numpy()
    if (y < 0).any():
        raise ValueError("negative SI VACV")

    oof = np.zeros(len(y))
    folds = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for train_index, valid_index in folds.split(x):
        model = RandomForestRegressor(**{**PARAMETERS, "random_state": args.seed})
        model.fit(x.iloc[train_index], y[train_index])
        oof[valid_index] = model.predict(x.iloc[valid_index])

    model = RandomForestRegressor(**{**PARAMETERS, "random_state": args.seed})
    model.fit(x, y)

    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_columns": features, "target": TARGET},
        args.out / "model.joblib",
        compress=3,
    )
    report = {
        "model": "randomforest",
        "train": str(args.train),
        "rows": int(len(train_frame)),
        "features": len(features),
        "folds": args.folds,
        "seed": args.seed,
        "parameters": {**PARAMETERS, "random_state": args.seed},
        "metrics": metrics(y, oof),
    }
    (args.out / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "Title": train_frame["Title"],
            "y_si": y,
            "pred_si": oof,
        }
    ).to_csv(args.out / "oof.csv", index=False)

    print(
        f"rows={len(train_frame)} features={len(features)} folds={args.folds}"
    )
    print(
        "RandomForest OOF  "
        f"MSE(SI)={report['metrics']['mse_si']:,.1f}  "
        f"RMSE(SI)={report['metrics']['rmse_si']:,.1f}  "
        f"MAE(SI)={report['metrics']['mae_si']:,.1f}"
    )
    print(
        "Dummy OOF  "
        f"mean MSE={report['metrics']['dummy_mean_mse']:,.1f}  "
        f"median MSE={report['metrics']['dummy_median_mse']:,.1f}"
    )
    return 0


def predict(args) -> int:
    payload = joblib.load(args.model / "model.joblib")
    model = payload["model"]
    features = payload["feature_columns"]

    test_frame = read(args.test)
    if "index" not in test_frame:
        raise ValueError("test table has no index column")
    if test_frame["index"].eq("").any() or test_frame["index"].duplicated().any():
        raise ValueError("test index must be present and unique")

    missing = [column for column in features if column not in test_frame.columns]
    if missing:
        raise ValueError(f"test table is missing features: {missing}")

    predictions = model.predict(numeric(test_frame, features))
    if not np.isfinite(predictions).all():
        raise ValueError("invalid predictions")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"index": test_frame["index"], TARGET: np.clip(predictions, 0.0, None)}
    ).to_csv(args.out, index=False)
    print(f"{args.out}: {len(test_frame)} rows, features={len(features)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.random_forest",
        description="RandomForest model trained on the selected 408-row table.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    train_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    train_parser.add_argument("--folds", type=int, default=5)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.set_defaults(func=train)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model", type=Path, default=DEFAULT_OUT)
    predict_parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    predict_parser.add_argument("--out", type=Path, default=DEFAULT_SUBMISSION)
    predict_parser.set_defaults(func=predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
