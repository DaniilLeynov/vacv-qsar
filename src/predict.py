from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .features import featurize

DEFAULT_TEST = Path("data/organizers/test.csv")
DEFAULT_MODEL = Path("runs/catboost")
DEFAULT_OUT = Path("submissions/submission.csv")


def load_models(model_dir: Path) -> tuple[list[CatBoostRegressor], str]:
    paths = sorted(model_dir.glob("model_fold*.cbm"))
    if not paths:
        raise FileNotFoundError(f"no model_fold*.cbm in {model_dir}")
    models = []
    for path in paths:
        model = CatBoostRegressor()
        model.load_model(str(path))
        models.append(model)

    target = "log1p"
    metrics_path = model_dir / "metrics.json"
    if metrics_path.exists():
        target = json.loads(metrics_path.read_text(encoding="utf-8")).get(
            "target", target
        )
    return models, target


def predict_frame(
    frame: pd.DataFrame, models: list[CatBoostRegressor], target: str
) -> np.ndarray:
    matrix = np.vstack([featurize(value) for value in frame["SMILES"]])
    predictions = np.mean([model.predict(matrix) for model in models], axis=0)
    if target == "log1p":
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
    models, target = load_models(args.model)

    frame = pd.read_csv(args.test)
    predictions = predict_frame(frame, models, target)

    index = frame["index"] if "index" in frame else pd.Series(range(len(frame)))
    submission = pd.DataFrame({"index": index, "SI VACV": predictions})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)

    print(f"{args.out}: {len(submission)} rows, models={len(models)}, target={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
