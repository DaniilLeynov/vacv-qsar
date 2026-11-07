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
from catboost import CatBoostClassifier, CatBoostRegressor

from .features import build_matrix
from .split import make_splits, scaffold_of

DEFAULT_DATA = Path("data/dataset.csv")
DEFAULT_OUT = Path("runs/catboost")
SI_CAP = 10000.0


def load_dataset(path: Path, target: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["SMILES"].notna()].copy()
    frame["SI VACV"] = pd.to_numeric(frame["SI VACV"], errors="coerce")
    frame = frame[frame["SI VACV"].notna() & (frame["SI VACV"] >= 0)].copy()

    lower = pd.to_numeric(frame.get("SI_lower", frame["SI VACV"]), errors="coerce")
    upper = pd.to_numeric(frame.get("SI_upper", frame["SI VACV"]), errors="coerce")
    frame["is_capped"] = (lower != upper).fillna(False)
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


def regression_metrics(y_si: np.ndarray, pred_si: np.ndarray) -> dict:
    y_si = np.asarray(y_si, dtype=np.float64)
    pred_si = np.asarray(pred_si, dtype=np.float64)
    mse = float(np.mean((pred_si - y_si) ** 2))
    mae = float(np.mean(np.abs(pred_si - y_si)))
    y_log = np.log1p(y_si)
    pred_log = np.log1p(pred_si)
    mse_log = float(np.mean((pred_log - y_log) ** 2))
    return {
        "mse_si": mse,
        "rmse_si": math.sqrt(mse),
        "mae_si": mae,
        "mse_log": mse_log,
        "rmse_log": math.sqrt(mse_log),
    }


def _classifier_params(seed: int, iterations: int) -> dict:
    return {
        "loss_function": "Logloss",
        "iterations": iterations,
        "depth": 4,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5.0,
        "random_seed": seed,
        "od_type": "Iter",
        "od_wait": 50,
        "allow_writing_files": False,
        "verbose": False,
    }


def _regressor_params(seed: int, iterations: int, depth: int, lr: float, l2: float) -> dict:
    return {
        "loss_function": "RMSE",
        "iterations": iterations,
        "depth": depth,
        "learning_rate": lr,
        "l2_leaf_reg": l2,
        "random_seed": seed,
        "od_type": "Iter",
        "od_wait": 100,
        "allow_writing_files": False,
        "verbose": False,
    }


def _train_two_stage_fold(payload: tuple) -> tuple[np.ndarray, np.ndarray]:
    (
        x_num,
        sc,
        tr,
        va,
        y_si,
        is_capped,
        exact,
        cls_params,
        reg_params,
        thread_count,
        cls_path,
        reg_path,
        use_scaffold,
    ) = payload

    def frame(indices):
        if use_scaffold:
            data = pd.DataFrame(x_num[indices])
            data["sc"] = sc[indices]
            return data, ["sc"]
        return x_num[indices], []

    x_train, cat = frame(tr)
    x_valid, _ = frame(va)

    clf = CatBoostClassifier(**cls_params, thread_count=thread_count)
    clf.fit(
        x_train,
        is_capped[tr],
        eval_set=(x_valid, is_capped[va]),
        cat_features=cat,
        use_best_model=True,
    )
    clf.save_model(cls_path)
    p_valid = clf.predict_proba(x_valid)[:, 1]

    tr_exact = tr & exact
    va_exact = va & exact
    x_train_e, _ = frame(tr_exact)
    x_valid_e, _ = frame(va_exact)

    reg = CatBoostRegressor(**reg_params, thread_count=thread_count)
    reg.fit(
        x_train_e,
        y_si[tr_exact],
        eval_set=(x_valid_e, y_si[va_exact]),
        cat_features=cat,
        use_best_model=True,
    )
    reg.save_model(reg_path)
    re_valid = reg.predict(x_valid)

    return p_valid, re_valid


def _train_reg_fold(payload: tuple) -> np.ndarray:
    (
        x_num,
        sc,
        tr,
        va,
        y_target,
        reg_params,
        thread_count,
        reg_path,
        use_scaffold,
    ) = payload
    if use_scaffold:
        x_train = pd.DataFrame(x_num[tr]); x_train["sc"] = sc[tr]
        x_valid = pd.DataFrame(x_num[va]); x_valid["sc"] = sc[va]
        cat = ["sc"]
    else:
        x_train, x_valid, cat = x_num[tr], x_num[va], []

    reg = CatBoostRegressor(**reg_params, thread_count=thread_count)
    reg.fit(
        x_train,
        y_target[tr],
        eval_set=(x_valid, y_target[va]),
        cat_features=cat,
        use_best_model=True,
    )
    reg.save_model(reg_path)
    return reg.predict(x_valid)


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
        "--group", choices=["scaffold", "smiles"], default="smiles"
    )
    parser.add_argument("--target", choices=["log1p", "raw"], default="raw")
    parser.add_argument(
        "--mode", choices=["twostage", "reg", "exact"], default="twostage"
    )
    parser.add_argument("--blend-c", type=float, default=SI_CAP)
    parser.add_argument("--use-scaffold", action="store_true")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--classifier-iterations", type=int, default=3000)
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

    frame = load_dataset(args.data, args.target)
    matrix, valid, errors = build_matrix(frame["SMILES"].tolist())
    if errors:
        print(f"skipped invalid SMILES: {len(errors)}", file=sys.stderr)
    frame = frame.iloc[valid].reset_index(drop=True)
    if len(frame) < args.folds * 2:
        print("not enough valid rows for cross-validation", file=sys.stderr)
        return 1

    sc = frame["SMILES"].map(scaffold_of).to_numpy()

    folds = make_splits(
        frame, n_splits=args.folds, seed=args.seed, strategy=args.group
    ).to_numpy()
    y_si = frame["SI VACV"].to_numpy(dtype=np.float64)
    y_target = frame["target"].to_numpy(dtype=np.float64)
    is_capped = frame["is_capped"].to_numpy(dtype=np.int32)
    exact = (is_capped == 0).astype(bool)

    cpu = os.cpu_count() or 1
    jobs = args.jobs or min(args.folds, cpu)
    jobs = max(1, min(jobs, args.folds))
    thread_count = max(1, cpu // jobs)

    oof_p = np.zeros(len(frame))
    oof_re = np.zeros(len(frame))
    oof_reg = np.zeros(len(frame))

    if args.mode == "twostage":
        cls_params = _classifier_params(args.seed, args.classifier_iterations)
        reg_params = _regressor_params(
            args.seed, args.iterations, args.depth, args.learning_rate, args.l2_leaf_reg
        )
        payloads = [
            (
                matrix,
                sc,
                folds != fold,
                folds == fold,
                y_si,
                is_capped,
                exact,
                cls_params,
                reg_params,
                thread_count,
                str(args.out / f"classifier_fold{fold}.cbm"),
                str(args.out / f"regressor_fold{fold}.cbm"),
                args.use_scaffold,
            )
            for fold in range(args.folds)
        ]
        if jobs == 1:
            results = [_train_two_stage_fold(payload) for payload in payloads]
        else:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                results = list(pool.map(_train_two_stage_fold, payloads))
        for fold, (p_valid, re_valid) in enumerate(results):
            mask = folds == fold
            oof_p[mask] = p_valid
            oof_re[mask] = re_valid
    else:
        reg_params = _regressor_params(
            args.seed, args.iterations, args.depth, args.learning_rate, args.l2_leaf_reg
        )
        exact_only = args.mode == "exact"
        payloads = []
        for fold in range(args.folds):
            tr = (folds != fold) & (exact if exact_only else True)
            va = (folds == fold) & (exact if exact_only else True)
            payloads.append(
                (
                    matrix,
                    sc,
                    tr,
                    va,
                    y_target,
                    reg_params,
                    thread_count,
                    str(args.out / f"model_fold{fold}.cbm"),
                    args.use_scaffold,
                )
            )
        if jobs == 1:
            results = [_train_reg_fold(payload) for payload in payloads]
        else:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                results = list(pool.map(_train_reg_fold, payloads))
        for fold, predictions in enumerate(results):
            if exact_only:
                oof_reg[(folds == fold) & exact] = predictions
            else:
                oof_reg[folds == fold] = predictions

    if args.mode == "twostage":
        pred_si = inverse_target(
            oof_p * args.blend_c + (1 - oof_p) * oof_re, "raw"
        )
    else:
        pred_si = inverse_target(oof_reg, args.target)

    eval_mask = exact.astype(bool) if args.mode == "exact" else np.ones(len(frame), bool)
    metrics = regression_metrics(y_si[eval_mask], pred_si[eval_mask])
    capped_mask = is_capped == 1
    exact_mask = is_capped == 0
    per_group = (
        {
            "capped": regression_metrics(y_si[capped_mask], pred_si[capped_mask]),
            "exact": regression_metrics(y_si[exact_mask], pred_si[exact_mask]),
        }
        if args.mode == "twostage"
        else {}
    )

    dummy = np.zeros(len(frame))
    for fold in range(args.folds):
        dummy[folds == fold] = np.median(y_target[folds != fold])
    dummy_metrics = regression_metrics(y_si[eval_mask], inverse_target(dummy, args.target)[eval_mask])

    report = {
        "data": str(args.data),
        "rows": int(len(frame)),
        "folds": args.folds,
        "seed": args.seed,
        "group": args.group,
        "target": args.target,
        "mode": args.mode,
        "blend_c": args.blend_c,
        "use_scaffold": bool(args.use_scaffold),
        "jobs": jobs,
        "thread_count": thread_count,
        "catboost": metrics,
        "dummy": dummy_metrics,
        "per_group": per_group,
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
            "is_capped": is_capped,
            "pred_si": pred_si,
        }
    ).to_csv(args.out / "oof.csv", index=False)

    print(
        f"rows={len(frame)} folds={args.folds} jobs={jobs} "
        f"threads/fold={thread_count} mode={args.mode}"
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
    if per_group:
        print(
            "capped   MSE(SI)=%s  exact MSE(SI)=%s"
            % (
                f"{per_group['capped']['mse_si']:,.1f}",
                f"{per_group['exact']['mse_si']:,.1f}",
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
