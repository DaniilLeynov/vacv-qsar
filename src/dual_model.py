"""Train two regressors (IC50 and CC50) and derive SI as their ratio.

Instead of fitting SI directly, this model learns the two quantities that
define it:

    log10(SI) = log10(CC50) - log10(IC50)

The IC50 head can be pre-trained on the large ChEMBL collection of antiviral
measurements, and the CC50 head on ChEMBL cytotoxicity measurements.  Both
heads are then optionally fine-tuned on the small VACV table.  At inference
time the two predicted log-concentrations are subtracted, so the model sees a
much larger and more balanced training signal than the few hundred SI values.

Only structure and assay context (cell line / organism) are used as features.
The measured ``CC50_uM`` / ``IC50_uM_VACV`` columns are labels, never inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from .features import (
    FEATURE_SETS,
    build_matrix,
    feature_names,
    process_pool_context,
)
from .normalize.common import (
    canonical_organism,
    normalize_cell_line,
    standardize_smiles,
)
from .split import make_splits

DEFAULT_TARGETS = Path("data/dual/all_targets.csv")
DEFAULT_IC50_TARGETS = Path("data/dual/ic50_targets.csv")
DEFAULT_CC50_TARGETS = Path("data/dual/cc50_targets.csv")
DEFAULT_ORGANIZERS = Path("data/organizers/train.csv")
DEFAULT_TEST = Path("data/organizers/test.csv")
DEFAULT_OUT = Path("runs/dual")

ENDPOINTS = ("ic50", "cc50")
SI_CAP = 10000.0


def _log10(value: float) -> float:
    return math.log10(value)


def _finite_positive(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where((numeric > 0) & np.isfinite(numeric))


def _safe_folds(
    frame: pd.DataFrame,
    n_splits: int,
    seed: int,
    strategy: str,
) -> pd.Series:
    if len(frame) < n_splits:
        raise ValueError("not enough rows for the requested number of folds")
    try:
        return make_splits(
            frame.reset_index(drop=True),
            n_splits=n_splits,
            seed=seed,
            strategy=strategy,
            smiles_column="SMILES",
        )
    except ValueError:
        if strategy != "smiles":
            return _safe_folds(frame, n_splits, seed, "smiles")
        rng = random.Random(seed)
        order = list(range(len(frame)))
        rng.shuffle(order)
        folds = np.zeros(len(frame), dtype=int)
        for position, index in enumerate(order):
            folds[index] = position % n_splits
        return pd.Series(folds, index=frame.index)


def _smiles_keys(values: pd.Series) -> pd.Series:
    unique = {value for value in values.dropna().astype(str) if value.strip()}
    mapping = {value: standardize_smiles(value) or "" for value in unique}
    return values.map(mapping).fillna("")


def read_targets(path: Path, endpoints: tuple[str, ...] = ENDPOINTS) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"SMILES", "endpoint", "y_log10_um"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame["endpoint"] = frame["endpoint"].astype(str).str.lower()
    frame = frame[frame["endpoint"].isin(endpoints)].copy()
    frame["y_log10_um"] = pd.to_numeric(frame["y_log10_um"], errors="coerce")
    frame = frame[np.isfinite(frame["y_log10_um"])].copy()
    if "smiles_key" not in frame:
        frame["smiles_key"] = _smiles_keys(frame["SMILES"])
    frame["smiles_key"] = frame["smiles_key"].fillna("").astype(str)
    frame = frame[frame["smiles_key"].ne("")].copy()
    for column in ("cell_line", "organism", "is_vacv", "is_viral", "Title"):
        if column not in frame:
            frame[column] = ""
    frame["is_vacv"] = (
        frame["is_vacv"].astype(str).str.lower().isin(["true", "1", "yes"])
    )
    frame["is_viral"] = (
        frame["is_viral"].astype(str).str.lower().isin(["true", "1", "yes"])
    )
    return frame.reset_index(drop=True)


def _exclude_keys(paths: list[Path]) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        if path is None or not Path(path).exists():
            continue
        frame = pd.read_csv(path)
        column = next(
            (item for item in ("SMILES", "smiles", "canonical_smiles") if item in frame),
            None,
        )
        if column is None:
            continue
        for value in frame[column].dropna().astype(str):
            if not value.strip():
                continue
            keys.add(value.strip())
            canonical = standardize_smiles(value)
            if canonical:
                keys.add(canonical)
    return keys


def aggregate_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse context groups to one row per molecule and endpoint."""
    work = frame.copy()
    if "source" not in work:
        work["source"] = ""
    if "n_measurements" not in work:
        work["n_measurements"] = 1
    grouped = (
        work.groupby(["smiles_key", "endpoint"], sort=False)
        .agg(
            Title=("Title", "first"),
            SMILES=("SMILES", "first"),
            y_log10_um=("y_log10_um", "median"),
            n_measurements=("n_measurements", "sum"),
            cell_line=("cell_line", "first"),
            organism=("organism", "first"),
            is_vacv=("is_vacv", "max"),
            is_viral=("is_viral", "max"),
            source=("source", "first"),
        )
        .reset_index()
    )
    grouped["context_key"] = "mixed"
    return grouped


def organizer_frame(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the SI evaluation table and the same rows in long endpoint form."""
    frame = pd.read_csv(path)
    required = {"SMILES", "CC50_uM", "IC50_uM_VACV", "SI VACV"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    frame = frame.copy()
    frame["SMILES"] = frame["SMILES"].fillna("").astype(str).str.strip()
    frame = frame[frame["SMILES"].ne("")].copy()
    frame["smiles_key"] = _smiles_keys(frame["SMILES"])
    frame = frame[frame["smiles_key"].ne("")].reset_index(drop=True)
    culture = frame["Culture"] if "Culture" in frame else pd.Series("", index=frame.index)
    strain = frame["strain"] if "strain" in frame else pd.Series("", index=frame.index)
    frame["cell_line"] = culture.map(normalize_cell_line)
    frame["organism"] = strain.map(canonical_organism)
    frame["SI VACV"] = pd.to_numeric(frame["SI VACV"], errors="coerce")
    frame["CC50_uM"] = _finite_positive(frame["CC50_uM"])
    frame["IC50_uM_VACV"] = _finite_positive(frame["IC50_uM_VACV"])
    # In the organizers' table a missing IC50 (or CC50) is encoded as SI = 0.
    # There is no true "SI = 0" molecule, so this is a separate zero class.
    frame["is_zero"] = frame["SI VACV"] <= 0
    frame["is_capped"] = frame["is_zero"] | (frame["SI VACV"] >= SI_CAP)
    frame = frame[np.isfinite(frame["SI VACV"])].copy()

    rows: list[dict] = []
    for _, row in frame.iterrows():
        base = {
            "Title": row.get("Title", ""),
            "SMILES": row["SMILES"],
            "smiles_key": row["smiles_key"],
            "cell_line": row["cell_line"],
            "organism": row["organism"],
            "is_vacv": True,
            "is_viral": True,
            "doi": "",
            "source": "organizers",
            "n_measurements": 1,
        }
        if pd.notna(row["IC50_uM_VACV"]):
            rows.append(
                {
                    **base,
                    "endpoint": "ic50",
                    "context_key": "vacv",
                    "y_log10_um": _log10(row["IC50_uM_VACV"]),
                }
            )
        if pd.notna(row["CC50_uM"]):
            rows.append(
                {
                    **base,
                    "endpoint": "cc50",
                    "context_key": "cell:" + (row["cell_line"] or "unknown"),
                    "y_log10_um": _log10(row["CC50_uM"]),
                }
            )
    return frame, pd.DataFrame(rows)


def _feature_map(
    smiles: list[str], feature_set: str = "all", jobs: int = 1
) -> dict[str, np.ndarray]:
    unique = list(dict.fromkeys(value for value in smiles if isinstance(value, str)))
    matrix, valid, errors = build_matrix(unique, feature_set, jobs)
    mapping = {
        unique[index]: matrix[position] for position, index in enumerate(valid)
    }
    if errors:
        print(f"skipped invalid SMILES: {len(errors)}", file=sys.stderr)
    return mapping


def _features(
    frame: pd.DataFrame,
    mapping: dict[str, np.ndarray],
    context: bool,
) -> tuple[object, list[str]]:
    rows = [mapping[value] for value in frame["SMILES"] if value in mapping]
    if len(rows) != len(frame):
        missing = len(frame) - len(rows)
        raise ValueError(f"{missing} rows have no valid features")
    matrix = np.vstack(rows).astype(np.float32)
    if not context:
        return matrix, []
    data = pd.DataFrame(matrix, columns=[f"f{index}" for index in range(matrix.shape[1])])
    data["cell_line"] = frame["cell_line"].fillna("unknown").astype(str)
    data["organism"] = frame["organism"].fillna("unknown").astype(str)
    return data, ["cell_line", "organism"]


def _params(
    seed: int,
    iterations: int,
    depth: int,
    learning_rate: float,
    l2_leaf_reg: float,
    od_wait: int = 100,
) -> dict:
    return {
        "loss_function": "RMSE",
        "iterations": iterations,
        "depth": depth,
        "learning_rate": learning_rate,
        "l2_leaf_reg": l2_leaf_reg,
        "random_seed": seed,
        "od_type": "Iter",
        "od_wait": od_wait,
        "allow_writing_files": False,
        "verbose": False,
    }


def _classifier_params(
    seed: int,
    iterations: int,
    depth: int,
    learning_rate: float,
    l2_leaf_reg: float,
) -> dict:
    return {
        "loss_function": "Logloss",
        "iterations": iterations,
        "depth": depth,
        "learning_rate": learning_rate,
        "l2_leaf_reg": l2_leaf_reg,
        "random_seed": seed,
        "od_type": "Iter",
        "od_wait": 50,
        "allow_writing_files": False,
        "verbose": False,
    }


def _fit_classifier(
    frame: pd.DataFrame,
    mapping: dict[str, np.ndarray],
    context: bool,
    params: dict,
    target: np.ndarray,
    eval_frame: pd.DataFrame | None = None,
    eval_target: np.ndarray | None = None,
    init_model: str | None = None,
    thread_count: int | None = None,
) -> CatBoostClassifier:
    x_train, cat_features = _features(frame, mapping, context)
    extra = {} if thread_count is None else {"thread_count": thread_count}
    model = CatBoostClassifier(**params, **extra)
    fit_kwargs = {"cat_features": cat_features, "verbose": False}
    if init_model is not None:
        fit_kwargs["init_model"] = init_model
    if eval_frame is not None and len(eval_frame) > 0 and eval_target is not None:
        x_valid, _ = _features(eval_frame, mapping, context)
        fit_kwargs["eval_set"] = (x_valid, eval_target)
        fit_kwargs["use_best_model"] = True
    model.fit(x_train, target, **fit_kwargs)
    return model


def _fit(
    frame: pd.DataFrame,
    mapping: dict[str, np.ndarray],
    context: bool,
    params: dict,
    eval_frame: pd.DataFrame | None = None,
    init_model: str | None = None,
    thread_count: int | None = None,
    target: np.ndarray | None = None,
) -> CatBoostRegressor:
    x_train, cat_features = _features(frame, mapping, context)
    extra = {} if thread_count is None else {"thread_count": thread_count}
    model = CatBoostRegressor(**params, **extra)
    fit_kwargs = {"cat_features": cat_features, "verbose": False}
    if init_model is not None:
        fit_kwargs["init_model"] = init_model
    if eval_frame is not None and len(eval_frame) > 0:
        x_valid, _ = _features(eval_frame, mapping, context)
        fit_kwargs["eval_set"] = (x_valid, eval_frame["y_log10_um"].to_numpy())
        fit_kwargs["use_best_model"] = True
    y_train = frame["y_log10_um"].to_numpy() if target is None else target
    model.fit(x_train, y_train, **fit_kwargs)
    return model


def _train_fold_worker(payload: tuple) -> np.ndarray:
    (
        x_train,
        y_train,
        x_valid,
        y_valid,
        params,
        cat_features,
        thread_count,
    ) = payload
    model = CatBoostRegressor(**params, thread_count=thread_count)
    fit_kwargs = {"cat_features": cat_features, "verbose": False}
    if x_valid is not None:
        fit_kwargs["eval_set"] = (x_valid, y_valid)
        fit_kwargs["use_best_model"] = True
    model.fit(x_train, y_train, **fit_kwargs)
    return model.predict(x_valid)


def _regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if len(y) == 0:
        return {"n": 0}
    residuals = pred - y
    mse = float(np.mean(residuals**2))
    mae = float(np.mean(np.abs(residuals)))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residuals**2)) / ss_tot if ss_tot > 0 else float("nan")
    return {
        "n": int(len(y)),
        "mse_log10": mse,
        "rmse_log10": math.sqrt(mse),
        "mae_log10": mae,
        "r2": r2,
    }


def _si_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), 0.0, 1e12)
    if len(y) == 0:
        return {"n": 0}
    mse = float(np.mean((pred - y) ** 2))
    mae = float(np.mean(np.abs(pred - y)))
    y_log = np.log1p(y)
    pred_log = np.log1p(pred)
    mse_log = float(np.mean((pred_log - y_log) ** 2))
    return {
        "n": int(len(y)),
        "mse_si": mse,
        "rmse_si": math.sqrt(mse),
        "mae_si": mae,
        "mse_log1p": mse_log,
        "rmse_log1p": math.sqrt(mse_log),
        "mae_log1p": float(np.mean(np.abs(pred_log - y_log))),
    }


def _train_base_endpoint(
    frame: pd.DataFrame,
    endpoint: str,
    mapping: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[Path | None, dict]:
    rows = frame[frame["endpoint"] == endpoint].reset_index(drop=True)
    rows = rows[rows["SMILES"].isin(mapping)].reset_index(drop=True)
    if len(rows) < 2:
        return None, {"n": int(len(rows)), "skipped": "not enough rows"}

    folds = _safe_folds(rows, args.folds, args.seed, args.group)
    oof = np.full(len(rows), np.nan)
    params = _params(
        args.seed,
        args.iterations,
        args.depth,
        args.learning_rate,
        args.l2_leaf_reg,
    )
    x_all, cat_features = _features(rows, mapping, args.context)
    y_all = rows["y_log10_um"].to_numpy()
    if args.cap_targets:
        cap = float(y_all.mean() + 3.0 * y_all.std())
        y_fit = np.minimum(y_all, cap)
    else:
        cap = None
        y_fit = y_all
    cpu = os.cpu_count() or 1
    jobs = args.jobs or min(args.folds, cpu)
    if len(rows) < 200:
        jobs = 1
    jobs = max(1, min(jobs, args.folds))
    thread_count = max(1, cpu // jobs)

    payloads: list[tuple] = []
    fold_ids: list[int] = []
    for fold in range(args.folds):
        train_mask = (folds != fold).to_numpy()
        valid_mask = (folds == fold).to_numpy()
        if not valid_mask.any() or train_mask.sum() < 2:
            continue
        if args.context:
            x_train = x_all.iloc[np.flatnonzero(train_mask)].reset_index(drop=True)
            x_valid = x_all.iloc[np.flatnonzero(valid_mask)].reset_index(drop=True)
        else:
            x_train = x_all[train_mask]
            x_valid = x_all[valid_mask]
        payloads.append(
            (
                x_train,
                y_fit[train_mask],
                x_valid,
                y_all[valid_mask],
                params,
                cat_features,
                thread_count,
            )
        )
        fold_ids.append(fold)

    if jobs == 1 or len(payloads) <= 1:
        results = [_train_fold_worker(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(
            max_workers=jobs, mp_context=process_pool_context()
        ) as pool:
            results = list(pool.map(_train_fold_worker, payloads))
    for fold, predictions in zip(fold_ids, results):
        oof[(folds == fold).to_numpy()] = predictions

    valid = np.isfinite(oof)
    metrics = _regression_metrics(
        rows.loc[valid, "y_log10_um"].to_numpy(), oof[valid]
    )
    metrics["fold_count"] = int(valid.sum())
    metrics["jobs"] = jobs
    metrics["thread_count"] = thread_count
    metrics["capped"] = bool(args.cap_targets)
    metrics["cap"] = cap

    base = _fit(
        rows,
        mapping,
        args.context,
        params,
        thread_count=cpu,
        target=y_fit,
    )
    path = args.out / f"base_{endpoint}.cbm"
    base.save_model(str(path))
    return path, metrics


def _predict_by_fold(
    models: dict[int, CatBoostRegressor | None],
    frame: pd.DataFrame,
    mapping: dict[str, np.ndarray],
    context: bool,
    folds: pd.Series,
    fallback: CatBoostRegressor | None,
) -> np.ndarray:
    predictions = np.full(len(frame), np.nan, dtype=float)
    for fold, model in models.items():
        mask = (folds == fold).to_numpy()
        if not mask.any():
            continue
        current = model if model is not None else fallback
        if current is None:
            continue
        x, _ = _features(frame[mask].reset_index(drop=True), mapping, context)
        predictions[mask] = current.predict(x)
    return predictions


def _calibrate(y: np.ndarray, pred: np.ndarray) -> dict:
    mask = np.isfinite(y) & np.isfinite(pred) & (y > 0) & (pred > 0)
    if mask.sum() < 2:
        return {"a": 1.0, "b": 0.0, "n": int(mask.sum())}
    a, b = np.polyfit(np.log10(pred[mask]), np.log10(y[mask]), 1)
    return {"a": float(a), "b": float(b), "n": int(mask.sum())}


def _apply_calibration(pred: np.ndarray, calibration: dict) -> np.ndarray:
    pred = np.clip(np.asarray(pred, dtype=float), 1e-12, None)
    return np.power(10.0, calibration["a"] * np.log10(pred) + calibration["b"])


def add_context_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    culture = frame["Culture"] if "Culture" in frame else pd.Series("", index=frame.index)
    strain = frame["strain"] if "strain" in frame else pd.Series("", index=frame.index)
    frame["cell_line"] = culture.map(normalize_cell_line)
    frame["organism"] = strain.map(canonical_organism)
    return frame


def load_prediction_models(
    model_dir: Path,
) -> tuple[dict, dict[str, list[CatBoostRegressor]], list[CatBoostClassifier]]:
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path}: train the dual model first")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    models: dict[str, list[CatBoostRegressor]] = {}
    for endpoint in ENDPOINTS:
        paths = sorted(model_dir.glob(f"{endpoint}_fold*.cbm"))
        final = model_dir / f"final_{endpoint}.cbm"
        if final.exists():
            paths.append(final)
        if not paths:
            base = model_dir / f"base_{endpoint}.cbm"
            if not base.exists():
                raise FileNotFoundError(
                    f"{model_dir}: no {endpoint} models (base, fold or final)"
                )
            paths = [base]
        loaded = []
        for path in paths:
            model = CatBoostRegressor()
            model.load_model(str(path))
            loaded.append(model)
        models[endpoint] = loaded

    zero_models: list[CatBoostClassifier] = []
    zero_paths = sorted(model_dir.glob("zero_fold*.cbm"))
    final_zero = model_dir / "final_zero.cbm"
    if final_zero.exists():
        zero_paths.append(final_zero)
    for path in zero_paths:
        model = CatBoostClassifier()
        model.load_model(str(path))
        zero_models.append(model)
    return config, models, zero_models


def predict_si_frame(
    frame: pd.DataFrame,
    model_dir: Path,
    calibrate: bool = True,
    zero_blend: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Predict SI for a table with a ``SMILES`` column."""
    config, models, zero_models = load_prediction_models(model_dir)
    frame = frame.copy()
    frame["_dual_row"] = np.arange(len(frame))
    frame["SMILES"] = frame["SMILES"].fillna("").astype(str).str.strip()
    frame = frame[frame["SMILES"].ne("")].reset_index(drop=True)
    mapping = _feature_map(
        frame["SMILES"].tolist(), config.get("feature_set", "all")
    )
    frame = frame[frame["SMILES"].isin(mapping)].reset_index(drop=True)
    if frame.empty:
        empty = np.zeros(0, dtype=float)
        return frame, empty, empty
    if config.get("context"):
        frame = add_context_columns(frame)

    predictions: dict[str, np.ndarray] = {}
    for endpoint in ENDPOINTS:
        x, _ = _features(frame, mapping, config.get("context", False))
        predictions[endpoint] = np.mean(
            [model.predict(x) for model in models[endpoint]], axis=0
        )
    log_si = predictions["cc50"] - predictions["ic50"]
    pred_si = np.power(10.0, log_si)
    if calibrate:
        calibration = config.get("calibration") or {"a": 1.0, "b": 0.0}
        pred_si = _apply_calibration(pred_si, calibration)
    if zero_blend and zero_models:
        x_zero, _ = _features(frame, mapping, config.get("context", False))
        zero_prob = np.mean(
            [model.predict_proba(x_zero)[:, 1] for model in zero_models], axis=0
        )
        pred_si = pred_si * (1.0 - zero_prob)
    return frame, np.clip(pred_si, 0.0, None), log_si


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.dual_model",
        description="Train separate IC50/CC50 heads and derive SI as a ratio.",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=None,
        help="combined target table; when omitted, --ic50-targets and "
        "--cc50-targets are read separately",
    )
    parser.add_argument(
        "--ic50-targets", type=Path, default=DEFAULT_IC50_TARGETS
    )
    parser.add_argument(
        "--cc50-targets", type=Path, default=DEFAULT_CC50_TARGETS
    )
    parser.add_argument("--organizers", type=Path, default=DEFAULT_ORGANIZERS)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group", choices=["scaffold", "smiles"], default="scaffold"
    )
    parser.add_argument(
        "--context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="add cell line and organism as categorical features",
    )
    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SETS,
        default="all",
        help="molecular features for both heads",
    )
    parser.add_argument(
        "--cap-targets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="cap each endpoint above mean + 3*std before training",
    )
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="parallel fold/feature workers (0 = auto)",
    )
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--fine-tune-iterations", type=int, default=300)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=0.02)
    parser.add_argument(
        "--finetune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fine-tune both heads on the organizers' VACV table",
    )
    parser.add_argument(
        "--zero-blend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="train a classifier for SI=0 (missing IC50/CC50) and blend it in",
    )
    parser.add_argument("--zero-iterations", type=int, default=500)
    parser.add_argument("--zero-learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=0,
        help="debug: keep at most N unique molecules (0 = all)",
    )
    parser.add_argument(
        "--exclude-smiles",
        action="append",
        type=Path,
        default=[],
        metavar="CSV",
        help="additional SMILES to remove from the ChEMBL pre-training set",
    )
    parser.add_argument(
        "--allow-test-overlap",
        action="store_true",
        help="do not automatically exclude the test SMILES from pre-training",
    )
    parser.add_argument(
        "--allow-organizer-overlap",
        action="store_true",
        help="do not automatically exclude organizer-train SMILES from pre-training",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.folds < 2:
        print("ERROR: --folds must be at least 2", file=sys.stderr)
        return 2
    target_paths = (
        [args.targets]
        if args.targets is not None
        else [args.ic50_targets, args.cc50_targets]
    )
    missing_targets = [path for path in target_paths if not path.exists()]
    if missing_targets:
        print(
            "ERROR: target table not found: "
            + ", ".join(str(path) for path in missing_targets)
            + ". Run `make data-all` first (see README.md).",
            file=sys.stderr,
        )
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    base = pd.concat(
        [read_targets(path) for path in target_paths], ignore_index=True
    )
    organizer_targets = (
        base[base["source"].astype(str) == "organizers"].copy()
        if "source" in base
        else pd.DataFrame()
    )
    if base.empty:
        print("no usable target rows", file=sys.stderr)
        return 1

    exclude_paths = list(args.exclude_smiles)
    if args.organizers.exists() and not args.allow_organizer_overlap:
        exclude_paths.append(args.organizers)
    if args.test.exists() and not args.allow_test_overlap:
        exclude_paths.append(args.test)
    excluded = _exclude_keys(exclude_paths)
    if excluded:
        before = len(base)
        base = base[~base["smiles_key"].isin(excluded)].copy()
        print(f"excluded {before - len(base)} pre-training rows by SMILES")

    if not args.context:
        base = aggregate_targets(base)

    if args.max_molecules and base["smiles_key"].nunique() > args.max_molecules:
        keys = base["smiles_key"].drop_duplicates().to_numpy()
        rng = np.random.default_rng(args.seed)
        chosen = set(rng.choice(keys, args.max_molecules, replace=False))
        base = base[base["smiles_key"].isin(chosen)].copy()

    organizer_eval, organizer_long = organizer_frame(args.organizers)
    if not organizer_targets.empty:
        organizer_long = (
            organizer_targets.copy()
            if args.context
            else aggregate_targets(organizer_targets)
        )
    all_smiles = list(
        dict.fromkeys(
            base["SMILES"].astype(str).tolist()
            + organizer_eval["SMILES"].astype(str).tolist()
        )
    )
    cpu = os.cpu_count() or 1
    feature_jobs = args.jobs or cpu
    mapping = _feature_map(all_smiles, args.feature_set, feature_jobs)
    base = base[base["SMILES"].isin(mapping)].reset_index(drop=True)
    organizer_eval = organizer_eval[organizer_eval["SMILES"].isin(mapping)].reset_index(
        drop=True
    )
    if base.empty:
        print("no pre-training rows left after SMILES filtering", file=sys.stderr)
        return 1
    if organizer_eval.empty:
        print("no valid organizer rows left", file=sys.stderr)
        return 1

    print(
        f"pre-training: {len(base)} rows, {base['smiles_key'].nunique()} molecules; "
        f"organizers: {len(organizer_eval)} rows"
    )
    endpoint_metrics: dict[str, dict] = {}
    base_models: dict[str, CatBoostRegressor] = {}
    base_paths: dict[str, Path] = {}
    for endpoint in ENDPOINTS:
        path, metrics = _train_base_endpoint(base, endpoint, mapping, args)
        endpoint_metrics[endpoint] = metrics
        if path is None:
            print(f"{endpoint}: skipped ChEMBL pre-training ({metrics['skipped']})")
            continue
        base_paths[endpoint] = path
        base_models[endpoint] = CatBoostRegressor()
        base_models[endpoint].load_model(str(path))
        print(
            f"{endpoint}: {metrics['n']} rows, "
            f"RMSE(log10)={metrics['rmse_log10']:.3f}, R2={metrics['r2']:.3f}"
        )

    organizer_folds = _safe_folds(
        organizer_eval, args.folds, args.seed, args.group
    )
    organizer_long = organizer_long[organizer_long["SMILES"].isin(mapping)].copy()
    organizer_long["fold"] = organizer_long["smiles_key"].map(
        dict(zip(organizer_eval["smiles_key"], organizer_folds))
    )
    organizer_long = organizer_long[organizer_long["fold"].notna()].copy()
    organizer_long["fold"] = organizer_long["fold"].astype(int)

    fold_models: dict[str, dict[int, CatBoostRegressor | None]] = {
        endpoint: {} for endpoint in ENDPOINTS
    }
    if args.finetune:
        fine_params = _params(
            args.seed,
            args.fine_tune_iterations,
            args.depth,
            args.fine_tune_learning_rate,
            args.l2_leaf_reg,
        )
        for endpoint in ENDPOINTS:
            rows = organizer_long[organizer_long["endpoint"] == endpoint].reset_index(
                drop=True
            )
            init_model = (
                str(base_paths[endpoint]) if endpoint in base_paths else None
            )
            for fold in range(args.folds):
                train = rows[rows["fold"] != fold].reset_index(drop=True)
                if len(train) < 2:
                    fold_models[endpoint][fold] = None
                    continue
                model = _fit(
                    train,
                    mapping,
                    args.context,
                    fine_params,
                    init_model=init_model,
                )
                model.save_model(str(args.out / f"{endpoint}_fold{fold}.cbm"))
                fold_models[endpoint][fold] = model
            if len(rows) >= 2:
                final = _fit(
                    rows,
                    mapping,
                    args.context,
                    fine_params,
                    init_model=init_model,
                )
                final.save_model(str(args.out / f"final_{endpoint}.cbm"))

    zero_prob = np.zeros(len(organizer_eval), dtype=float)
    zero_available = False
    if args.zero_blend:
        y_zero = organizer_eval["is_zero"].to_numpy(dtype=int)
        if len(np.unique(y_zero)) < 2:
            print("zero classifier skipped: all organizer rows have the same class")
        else:
            zero_params = _classifier_params(
                args.seed,
                args.zero_iterations,
                args.depth,
                args.zero_learning_rate,
                args.l2_leaf_reg,
            )
            for fold in range(args.folds):
                train_mask = (organizer_folds != fold).to_numpy()
                valid_mask = (organizer_folds == fold).to_numpy()
                if not valid_mask.any() or len(np.unique(y_zero[train_mask])) < 2:
                    continue
                model = _fit_classifier(
                    organizer_eval[train_mask].reset_index(drop=True),
                    mapping,
                    args.context,
                    zero_params,
                    y_zero[train_mask],
                    eval_frame=organizer_eval[valid_mask].reset_index(drop=True),
                    eval_target=y_zero[valid_mask],
                )
                model.save_model(str(args.out / f"zero_fold{fold}.cbm"))
                x_valid, _ = _features(
                    organizer_eval[valid_mask].reset_index(drop=True),
                    mapping,
                    args.context,
                )
                zero_prob[valid_mask] = model.predict_proba(x_valid)[:, 1]
            if len(np.unique(y_zero)) == 2:
                final_zero = _fit_classifier(
                    organizer_eval,
                    mapping,
                    args.context,
                    zero_params,
                    y_zero,
                )
                final_zero.save_model(str(args.out / "final_zero.cbm"))
                zero_available = True

    # OOF predictions on the organizers' table.
    predictions: dict[str, np.ndarray] = {}
    for endpoint in ENDPOINTS:
        if args.finetune:
            predictions[endpoint] = _predict_by_fold(
                fold_models[endpoint],
                organizer_eval,
                mapping,
                args.context,
                organizer_folds,
                base_models.get(endpoint),
            )
        elif endpoint in base_models:
            x, _ = _features(organizer_eval, mapping, args.context)
            predictions[endpoint] = base_models[endpoint].predict(x)
        else:
            predictions[endpoint] = np.full(len(organizer_eval), np.nan)

    if not all(np.isfinite(predictions[endpoint]).all() for endpoint in ENDPOINTS):
        print(
            "ERROR: one of the heads has no model; SI cannot be computed. "
            "Add a CC50 export or allow fine-tuning on organizer labels.",
            file=sys.stderr,
        )
        return 1

    log_si = predictions["cc50"] - predictions["ic50"]
    pred_si = np.clip(np.power(10.0, log_si), 1e-12, 1e12)
    y_si = organizer_eval["SI VACV"].to_numpy(dtype=float)
    calibration = _calibrate(y_si, pred_si)
    pred_si_cal = _apply_calibration(pred_si, calibration)
    pred_si_blend = pred_si_cal * (1.0 - zero_prob)

    zero = organizer_eval["is_zero"].to_numpy(dtype=bool)
    positive = ~zero
    metrics = {
        "data": {
            "targets": [str(path) for path in target_paths],
            "pretrain_rows": int(len(base)),
            "pretrain_molecules": int(base["smiles_key"].nunique()),
            "organizer_rows": int(len(organizer_eval)),
        },
        "config": {
            "folds": args.folds,
            "seed": args.seed,
            "group": args.group,
            "context": bool(args.context),
            "finetune": bool(args.finetune),
            "iterations": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "l2_leaf_reg": args.l2_leaf_reg,
            "fine_tune_iterations": args.fine_tune_iterations,
            "fine_tune_learning_rate": args.fine_tune_learning_rate,
        },
        "endpoints": endpoint_metrics,
        "si": {
            "raw": _si_metrics(y_si, pred_si),
            "calibrated": _si_metrics(y_si, pred_si_cal),
            "blended": _si_metrics(y_si, pred_si_blend),
            "zero": _si_metrics(y_si[zero], pred_si_blend[zero]),
            "positive": _si_metrics(y_si[positive], pred_si_blend[positive]),
        },
        "zero_classifier": {
            "enabled": bool(args.zero_blend),
            "trained": zero_available,
            "positives": int(zero.sum()),
        },
        "calibration": calibration,
    }
    (args.out / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    config = {
        "context": bool(args.context),
        "feature_set": args.feature_set,
        "cap_targets": bool(args.cap_targets),
        "endpoints": list(ENDPOINTS),
        "calibration": calibration,
        "zero_blend": bool(args.zero_blend),
        "zero_model": zero_available,
        "feature_names": feature_names(args.feature_set)
        + (["cell_line", "organism"] if args.context else []),
    }
    (args.out / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "Title": organizer_eval.get("Title", ""),
            "SMILES": organizer_eval["SMILES"],
            "fold": organizer_folds,
            "y_si": y_si,
            "is_zero": zero,
            "pred_zero_prob": zero_prob,
            "pred_ic50_log10": predictions["ic50"],
            "pred_cc50_log10": predictions["cc50"],
            "pred_si": pred_si,
            "pred_si_calibrated": pred_si_cal,
            "pred_si_blended": pred_si_blend,
        }
    ).to_csv(args.out / "oof_si.csv", index=False)

    print(
        "SI OOF (blended): "
        f"RMSE={metrics['si']['blended']['rmse_si']:,.1f}  "
        f"MSE={metrics['si']['blended']['mse_si']:,.1f}  "
        f"RMSE(log1p)={metrics['si']['blended']['rmse_log1p']:.3f}"
    )
    print(f"calibration: {calibration}")
    print(f"{args.out}/metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
