from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

from .features import FEATURE_NAMES, build_matrix

TARGET = "SI VACV"
DEFAULT_TRAIN = Path("data/dataset.csv")
DEFAULT_ORGANIZERS = Path("data/organizers/train.csv")
DEFAULT_TEST = Path("data/organizers/test.csv")
DEFAULT_OUT = Path("runs/ensemble")
DEFAULT_SUBMISSION = Path("submissions/submission.csv")

SEED = 42
FOLDS = 5
BLEND_WEIGHT = 0.5
FP_SIZE = 2048
FP_RADIUS = 2

# Morgan bits kept for the raw-scale RandomForest, selected on the training
# table by feature importance. Chirality is included: stereochemistry carries
# signal for this assay and is lost by the plain count fingerprints.
SELECTED_BITS = [
    72, 195, 262, 497, 519, 526, 545, 718, 872, 888, 973, 996, 1048, 1053,
    1070, 1141, 1151, 1157, 1176, 1202, 1230, 1235, 1292, 1462, 1555, 1591,
    1603, 1649, 1662, 1665, 1697, 1713, 1729, 1804, 1858, 1885, 1950,
]

CULTURE_MAP = {
    "cells e6sm": "e6sm",
    "e6sm cells": "e6sm",
    "e6sm": "e6sm",
    "lung fibroblast cells": "lung fibroblast",
    "a549 cells": "a549",
    "a549": "a549",
    "bsc-40": "bsc-40",
    "bsc-40 cells": "bsc-40",
    "bsc40": "bsc-40",
    "cem cells": "cem",
    "hela cells": "hela",
    "hela": "hela",
    "human foreskin fibroblast": "hff",
    "prk cells": "prk",
    "cells bsc1": "bsc1",
    "cells hel": "hel",
    "vero": "vero",
}

RF_PARAMS = {
    "n_estimators": 500,
    "min_samples_leaf": 2,
    "random_state": SEED,
    "n_jobs": -1,
}

HGB_PARAMS = {
    "max_iter": 400,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "l2_regularization": 1.0,
    "random_state": SEED,
}

_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=FP_RADIUS, fpSize=FP_SIZE, includeChirality=True
)


def clean_culture(value: object) -> str:
    if not isinstance(value, str):
        return "unknown"
    text = " ".join(value.strip().lower().split())
    if not text:
        return "unknown"
    return CULTURE_MAP.get(text, text)


def load_train(path: Path, organizers: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    org = pd.read_csv(organizers)
    culture = org.drop_duplicates("SMILES").set_index("SMILES")["Culture"]
    fallback = frame["SMILES"].map(culture).map(clean_culture)
    if "Culture" not in frame.columns:
        frame["Culture"] = ""
    current = frame["Culture"].map(clean_culture)
    frame["Culture"] = current.where(current.ne("unknown"), fallback)
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    frame = frame[frame["SMILES"].notna() & frame[TARGET].notna()].copy()
    return frame.reset_index(drop=True)


def load_test(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["Culture"] = frame["Culture"].map(clean_culture)
    return frame


def chirality_bits(smiles: list[str]) -> np.ndarray:
    matrix = np.zeros((len(smiles), FP_SIZE), dtype=np.float32)
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            continue
        DataStructs.ConvertToNumpyArray(_GENERATOR.GetFingerprint(molecule), matrix[index])
    return matrix[:, SELECTED_BITS]


def fit_culture_encoder(train: pd.DataFrame) -> OneHotEncoder:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return encoder.fit(train[["Culture"]])


def culture_features(encoder: OneHotEncoder, frame: pd.DataFrame) -> np.ndarray:
    return encoder.transform(frame[["Culture"]])


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred = np.clip(y_pred, 0.0, None)
    error = y_pred - y_true
    mse = float(np.mean(error**2))
    return {
        "mse_si": mse,
        "rmse_si": math.sqrt(mse),
        "mae_si": float(np.mean(np.abs(error))),
        "rmse_log": float(math.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))),
    }


def _fit_fold(payload: tuple) -> tuple[np.ndarray, np.ndarray]:
    (
        x_rf,
        x_hgb,
        tr,
        va,
        y,
        rf_params,
        hgb_params,
    ) = payload

    rf = RandomForestRegressor(**rf_params)
    rf.fit(x_rf[tr], y[tr])
    rf_valid = rf.predict(x_rf[va])

    hgb = HistGradientBoostingRegressor(**hgb_params)
    hgb.fit(x_hgb[tr], np.log1p(y[tr]))
    hgb_valid = np.expm1(hgb.predict(x_hgb[va]))

    return rf_valid, hgb_valid


def train(args) -> int:
    frame = load_train(args.train, args.organizers)
    smiles = frame["SMILES"].tolist()
    y = frame[TARGET].to_numpy(dtype=np.float64)

    x_rf = chirality_bits(smiles)
    full, _, errors = build_matrix(smiles)
    if errors:
        keep = np.ones(len(frame), dtype=bool)
        for entry in errors:
            keep[entry["index"]] = False
        frame = frame[keep].reset_index(drop=True)
        full = full[keep]
        y = frame[TARGET].to_numpy(dtype=np.float64)
        x_rf = x_rf[keep]
        smiles = frame["SMILES"].tolist()

    # Culture joins full features for the log-scale gradient boosting half.
    culture_encoder = fit_culture_encoder(frame)
    x_hgb = np.hstack([full, culture_features(culture_encoder, frame)])

    folds = list(KFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(y))
    oof = np.zeros(len(y))
    rf_params = {**RF_PARAMS, "random_state": args.seed}
    hgb_params = {**HGB_PARAMS, "random_state": args.seed}

    for tr, va in folds:
        rf_valid, hgb_valid = _fit_fold((x_rf, x_hgb, tr, va, y, rf_params, hgb_params))
        oof[va] = args.blend_weight * rf_valid + (1 - args.blend_weight) * hgb_valid

    rf = RandomForestRegressor(**rf_params)
    rf.fit(x_rf, y)
    hgb = HistGradientBoostingRegressor(**hgb_params)
    hgb.fit(x_hgb, np.log1p(y))

    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "rf": rf,
            "hgb": hgb,
            "blend_weight": args.blend_weight,
            "culture_encoder": culture_encoder,
            "rf_features": "chirality_morgan_bits",
            "hgb_features": FEATURE_NAMES,
        },
        args.out / "model.joblib",
        compress=3,
    )

    dummy = np.zeros(len(y))
    for tr, va in folds:
        dummy[va] = np.median(y[tr])
    report = {
        "model": "ensemble",
        "train": str(args.train),
        "rows": int(len(frame)),
        "folds": args.folds,
        "seed": args.seed,
        "blend_weight": args.blend_weight,
        "rf_params": rf_params,
        "hgb_params": hgb_params,
        "metrics": metrics(y, oof),
        "dummy": metrics(y, dummy),
    }
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(
        {
            "Title": frame["Title"],
            "y_si": y,
            "pred_si": oof,
        }
    ).to_csv(args.out / "oof.csv", index=False)

    m = report["metrics"]
    print(f"rows={len(frame)} folds={args.folds} blend_weight={args.blend_weight}")
    print(
        "Ensemble OOF  "
        f"MSE(SI)={m['mse_si']:,.1f}  RMSE(SI)={m['rmse_si']:,.1f}  "
        f"MAE(SI)={m['mae_si']:,.1f}  RMSE(log)={m['rmse_log']:.4f}"
    )
    print(f"Dummy    OOF  MSE(SI)={report['dummy']['mse_si']:,.1f}")
    return 0


def predict(args) -> int:
    payload = joblib.load(args.model / "model.joblib")
    rf = payload["rf"]
    hgb = payload["hgb"]
    blend_weight = payload["blend_weight"]
    culture_encoder = payload["culture_encoder"]

    frame = load_test(args.test)
    smiles = frame["SMILES"].tolist()

    x_rf = chirality_bits(smiles)
    full, _, errors = build_matrix(smiles)
    if errors:
        raise ValueError(f"invalid SMILES in test set: {errors[:3]}")

    x_hgb = np.hstack([full, culture_features(culture_encoder, frame)])

    raw = blend_weight * rf.predict(x_rf) + (1 - blend_weight) * np.expm1(
        hgb.predict(x_hgb)
    )
    raw = np.clip(raw, 0.0, None)

    index = frame["index"] if "index" in frame else pd.Series(range(len(frame)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"index": index, TARGET: raw}).to_csv(args.out, index=False)
    print(f"{args.out}: {len(frame)} rows")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.ensemble",
        description="Blended model: raw RandomForest + log HistGradientBoosting.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    train_parser.add_argument("--organizers", type=Path, default=DEFAULT_ORGANIZERS)
    train_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    train_parser.add_argument("--folds", type=int, default=FOLDS)
    train_parser.add_argument("--seed", type=int, default=SEED)
    train_parser.add_argument("--blend-weight", type=float, default=BLEND_WEIGHT)
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
