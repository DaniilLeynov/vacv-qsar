from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from .features import featurize
from .split import scaffold_of

DEFAULT_TEST = Path("data/organizers/test.csv")
DEFAULT_MODEL = Path("runs/catboost")
DEFAULT_OUT = Path("submissions/submission.csv")


def _load(path: Path, model_type: str):
    models = []
    for p in sorted(path.glob(f"{model_type}_fold*.cbm")):
        model = (
            CatBoostClassifier() if model_type == "classifier" else CatBoostRegressor()
        )
        model.load_model(str(p))
        models.append(model)
    return models


def load_models(model_dir: Path) -> dict:
    metrics = {}
    metrics_path = model_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    mode = metrics.get("mode", "twostage")
    if mode == "exact":
        mode = "reg"
    use_scaffold = metrics.get("use_scaffold", False)
    blend_c = float(metrics.get("blend_c", 10000.0))
    target = metrics.get("target", "raw")

    if mode == "twostage":
        classifiers = _load(model_dir, "classifier")
        regressors = _load(model_dir, "regressor")
        if not classifiers or not regressors:
            raise FileNotFoundError(f"missing fold models in {model_dir}")
        return {
            "mode": mode,
            "classifiers": classifiers,
            "regressors": regressors,
            "blend_c": blend_c,
            "use_scaffold": use_scaffold,
            "target": target,
        }

    models = _load(model_dir, "model")
    if not models:
        raise FileNotFoundError(f"no model_fold*.cbm in {model_dir}")
    return {"mode": mode, "models": models, "use_scaffold": use_scaffold, "target": target}


def _features(frame: pd.DataFrame, use_scaffold: bool):
    matrix = np.vstack([featurize(value) for value in frame["SMILES"]])
    if use_scaffold:
        data = pd.DataFrame(matrix)
        data["sc"] = frame["SMILES"].map(scaffold_of).to_numpy()
        return data
    return matrix


def predict_frame(frame: pd.DataFrame, spec: dict) -> np.ndarray:
    features = _features(frame, spec["use_scaffold"])

    if spec["mode"] == "twostage":
        probs = np.mean(
            [clf.predict_proba(features)[:, 1] for clf in spec["classifiers"]],
            axis=0,
        )
        regs = np.mean(
            [reg.predict(features) for reg in spec["regressors"]], axis=0
        )
        predictions = probs * spec["blend_c"] + (1 - probs) * regs
    else:
        predictions = np.mean(
            [model.predict(features) for model in spec["models"]], axis=0
        )
        if spec["target"] == "log1p":
            predictions = np.expm1(predictions)

    return np.clip(predictions, 0.0, None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.predict",
        description="Predict SI VACV and write a Kaggle submission.",
    )
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = load_models(args.model)

    frame = pd.read_csv(args.test)
    predictions = predict_frame(frame, spec)

    index = frame["index"] if "index" in frame else pd.Series(range(len(frame)))
    submission = pd.DataFrame({"index": index, "SI VACV": predictions})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)

    model_count = len(spec.get("models", spec.get("classifiers", [])))
    print(
        f"{args.out}: {len(submission)} rows, mode={spec['mode']}, "
        f"models={model_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
