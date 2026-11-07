from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .dual_model import DEFAULT_OUT, DEFAULT_TEST, predict_si_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.dual_predict",
        description="Predict SI with the dual IC50/CC50 model.",
    )
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out", type=Path, default=Path("submissions/submission.csv"))
    parser.add_argument(
        "--calibrate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply the log-space calibration fitted on organizer OOF predictions",
    )
    parser.add_argument(
        "--zero-blend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="multiply by 1 - P(SI = 0), where SI = 0 means missing IC50/CC50",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = pd.read_csv(args.test)
    predicted, predictions, _ = predict_si_frame(
        frame, args.model, args.calibrate, args.zero_blend
    )

    aligned = np.full(len(frame), np.nan, dtype=float)
    aligned[predicted["_dual_row"].to_numpy()] = predictions
    if np.isnan(aligned).any():
        fallback = float(np.nanmedian(aligned)) if np.isfinite(aligned).any() else 1.0
        aligned = np.where(np.isfinite(aligned), aligned, fallback)

    index = frame["index"] if "index" in frame else pd.Series(range(len(frame)))
    submission = pd.DataFrame({"index": index, "SI VACV": aligned})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)
    print(f"{args.out}: {len(submission)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
