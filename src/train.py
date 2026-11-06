from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .features import build_matrix
from .split import make_splits

DEFAULT_DATA = Path("data/dataset.csv")
DEFAULT_OUT = Path("runs/catboost")


def load_dataset(
    path: Path, include_capped: bool, target: str
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["SMILES"].notna()].copy()
    frame["SI VACV"] = pd.to_numeric(frame["SI VACV"], errors="coerce")
    frame = frame[frame["SI VACV"].notna() & (frame["SI VACV"] >= 0)].copy()

    lower = pd.to_numeric(frame.get("SI_lower", frame["SI VACV"]), errors="coerce")
    upper = pd.to_numeric(frame.get("SI_upper", frame["SI VACV"]), errors="coerce")
    exact = lower.notna() & upper.notna() & np.isclose(lower, upper)
    if not include_capped:
        frame = frame[exact].copy()

    frame = frame.reset_index(drop=True)
    frame["target"] = (
        np.log1p(frame["SI VACV"]) if target == "log1p" else frame["SI VACV"]
    )
    return frame


def inverse_target(values: np.ndarray, target: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if target == "log1p":
        values = np.expm1(values)
    return np.clip(values, 0.0, None)


def regression_metrics(
    y_si: np.ndarray, pred_si: np.ndarray, target: str
) -> dict:
    y_si = np.asarray(y_si, dtype=np.float64)
    pred_si = np.asarray(pred_si, dtype=np.float64)
    mse_si = float(np.mean((pred_si - y_si) ** 2))
    mae_si = float(np.mean(np.abs(pred_si - y_si)))
    y_log = np.log1p(y_si)
    pred_log = np.log1p(pred_si)
    mse_log = float(np.mean((pred_log - y_log) ** 2))
    return {
        "mse_si": mse_si,
        "rmse_si": math.sqrt(mse_si),
        "mae_si": mae_si,
        "mse_log": mse_log,
        "rmse_log": math.sqrt(mse_log),
    }


def _train_fold(payload: tuple) -> tuple[np.ndarray, int]:
    (
        x_train,
        y_train,
        x_valid,
        y_valid,
        params,
        model_path,
        thread_count,
    ) = payload
    model = CatBoostRegressor(**params, thread_count=thread_count)
    model.fit(
        x_train,
        y_train,
        eval_set=(x_valid, y_valid),
        use_best_model=True,
        verbose=False,
    )
    model.save_model(model_path)
    return model.predict(x_valid), int(model.get_best_iteration())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.train",
        description="Train CatBoost to predict SI VACV from SMILES.",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group", choices=["scaffold", "smiles"], default="scaffold"
    )
    parser.add_argument(
        "--target", choices=["log1p", "raw"], default="log1p"
    )
    parser.add_argument(
        "--include-capped",
        action="store_true",
        help="include rows whose SI is a lower bound (SI VACV = 10000)",
    )
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--od-wait", type=int, default=100)
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="parallel fold workers (0 = min(folds, cpu_count))",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    frame = load_dataset(args.data, args.include_capped, args.target)
    matrix, valid, errors = build_matrix(frame["SMILES"].tolist())
    if errors:
        print(f"skipped invalid SMILES: {len(errors)}", file=sys.stderr)
    frame = frame.iloc[valid].reset_index(drop=True)
    if len(frame) < args.folds * 2:
        print("not enough valid rows for cross-validation", file=sys.stderr)
        return 1

    folds = make_splits(
        frame,
        n_splits=args.folds,
        seed=args.seed,
        strategy=args.group,
    )
    y = frame["target"].to_numpy(dtype=np.float64)
    y_si = frame["SI VACV"].to_numpy(dtype=np.float64)

    cpu = os.cpu_count() or 1
    jobs = args.jobs or min(args.folds, cpu)
    jobs = max(1, min(jobs, args.folds))
    thread_count = max(1, cpu // jobs)

    params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": args.iterations,
        "depth": args.depth,
        "learning_rate": args.learning_rate,
        "l2_leaf_reg": args.l2_leaf_reg,
        "random_seed": args.seed,
        "od_type": "Iter",
        "od_wait": args.od_wait,
        "allow_writing_files": False,
        "verbose": False,
    }

    payloads = []
    for fold in range(args.folds):
        train_mask = folds.to_numpy() != fold
        valid_mask = ~train_mask
        payloads.append(
            (
                matrix[train_mask],
                y[train_mask],
                matrix[valid_mask],
                y[valid_mask],
                params,
                str(args.out / f"model_fold{fold}.cbm"),
                thread_count,
            )
        )

    if jobs == 1:
        results = [_train_fold(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(_train_fold, payloads))

    oof = np.zeros(len(frame), dtype=np.float64)
    best_iterations = []
    for fold, (predictions, best_iteration) in enumerate(results):
        oof[folds.to_numpy() == fold] = predictions
        best_iterations.append(best_iteration)

    pred_si = inverse_target(oof, args.target)
    metrics = regression_metrics(y_si, pred_si, args.target)

    dummy = np.zeros(len(frame), dtype=np.float64)
    for fold in range(args.folds):
        train_mask = folds.to_numpy() != fold
        dummy[~train_mask] = np.median(y[train_mask])
    dummy_si = inverse_target(dummy, args.target)
    dummy_metrics = regression_metrics(y_si, dummy_si, args.target)

    report = {
        "data": str(args.data),
        "rows": int(len(frame)),
        "folds": args.folds,
        "seed": args.seed,
        "group": args.group,
        "target": args.target,
        "include_capped": bool(args.include_capped),
        "jobs": jobs,
        "thread_count": thread_count,
        "best_iterations": best_iterations,
        "catboost": metrics,
        "dummy": dummy_metrics,
    }
    (args.out / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "Title": frame["Title"],
            "SMILES": frame["SMILES"],
            "fold": folds,
            "y_si": y_si,
            "pred_si": pred_si,
        }
    ).to_csv(args.out / "oof.csv", index=False)

    print(
        f"rows={len(frame)} folds={args.folds} jobs={jobs} "
        f"threads/fold={thread_count} target={args.target}"
    )
    print(
        "CatBoost OOF  "
        f"MSE(SI)={metrics['mse_si']:,.1f}  "
        f"RMSE(SI)={metrics['rmse_si']:,.1f}  "
        f"MAE(SI)={metrics['mae_si']:,.1f}  "
        f"MSE(log)={metrics['mse_log']:.4f}"
    )
    print(
        "Dummy    OOF  "
        f"MSE(SI)={dummy_metrics['mse_si']:,.1f}  "
        f"RMSE(SI)={dummy_metrics['rmse_si']:,.1f}"
    )
    print(f"best iterations: {best_iterations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
