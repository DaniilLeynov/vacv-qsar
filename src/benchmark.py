from __future__ import annotations

import contextlib
import io
import json
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

warnings.filterwarnings("ignore")

from . import predict as catboost_predict
from . import random_forest
from . import train as catboost_train

RUNS = Path("runs")
DEFAULT_SUBMISSION = Path("submissions/submission.csv")


@dataclass
class ModelSpec:
    name: str
    train: Callable[[], int]
    predict: Callable[[Path], int]
    metrics: Callable[[], dict]


def _catboost_metrics() -> dict:
    payload = json.loads((RUNS / "catboost" / "metrics.json").read_text(encoding="utf-8"))
    return payload["catboost"]


def _random_forest_metrics() -> dict:
    payload = json.loads(
        (RUNS / "randomforest" / "metrics.json").read_text(encoding="utf-8")
    )
    return payload["metrics"]


def build_models() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="catboost",
            train=lambda: catboost_train.main(
                [
                    "--data",
                    "data/dataset.csv",
                    "--out",
                    str(RUNS / "catboost"),
                    "--model",
                    "catboost",
                    "--mode",
                    "exact",
                    "--use-scaffold",
                ]
            ),
            predict=lambda out: catboost_predict.main(
                [
                    "--test",
                    "data/organizers/test.csv",
                    "--model",
                    str(RUNS / "catboost"),
                    "--out",
                    str(out),
                ]
            ),
            metrics=_catboost_metrics,
        ),
        ModelSpec(
            name="randomforest",
            train=lambda: random_forest.main(
                [
                    "train",
                    "--train",
                    "data/rf/train.csv",
                    "--out",
                    str(RUNS / "randomforest"),
                ]
            ),
            predict=lambda out: random_forest.main(
                [
                    "predict",
                    "--model",
                    str(RUNS / "randomforest"),
                    "--test",
                    "data/rf/test.csv",
                    "--out",
                    str(out),
                ]
            ),
            metrics=_random_forest_metrics,
        ),
    ]


def _quiet(call) -> tuple[int, str]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = call()
    return code, buffer.getvalue()


def _check(result, log: str, name: str) -> None:
    if result not in (None, 0):
        if log.strip():
            print(log)
        raise RuntimeError(f"{name} failed with exit code {result}")


def run(
    models: list[ModelSpec],
    submission: Path = DEFAULT_SUBMISSION,
    submissions_dir: Path | None = None,
) -> str:
    submissions_dir = Path(submissions_dir or submission.parent)
    submissions_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[ModelSpec, dict]] = []
    for spec in models:
        print(f"training {spec.name} ...")
        code, log = _quiet(spec.train)
        _check(code, log, spec.name)
        results.append((spec, spec.metrics()))

    best_spec, best_metrics = min(results, key=lambda item: item[1]["mse_si"])
    per_model = submissions_dir / f"submission_{best_spec.name}.csv"
    code, log = _quiet(lambda: best_spec.predict(per_model))
    _check(code, log, best_spec.name)
    shutil.copyfile(per_model, submission)

    print("\nmodel         OOF MSE(SI)      RMSE      MAE")
    for spec, metrics in sorted(results, key=lambda item: item[1]["mse_si"]):
        print(
            f"{spec.name:<13} {metrics['mse_si']:>13,.1f} "
            f"{metrics['rmse_si']:>9,.1f} {metrics['mae_si']:>8,.1f}"
        )
    print(f"\nbest: {best_spec.name} -> {submission}")
    return best_spec.name


def main() -> int:
    run(build_models())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
