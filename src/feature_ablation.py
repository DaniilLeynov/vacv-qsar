"""Run feature-set ablation for the dual IC50/CC50 model.

Each variant is trained in a separate process (so CatBoost memory is released)
and writes to ``runs/ablation/<name>``.  The final table is sorted by the raw
SI OOF MSE, as requested, but also reports MAE and log1p RMSE because a single
large SI outlier dominates raw MSE.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

RUNS = Path("runs/ablation")

VARIANTS: dict[str, list[str]] = {
    "morgan": ["--feature-set", "morgan"],
    "maccs": ["--feature-set", "maccs"],
    "descriptors": ["--feature-set", "descriptors"],
    "morgan_maccs": ["--feature-set", "morgan_maccs"],
    "rf_bits": ["--feature-set", "rf_bits"],
    "rf_bits_maccs": ["--feature-set", "rf_bits_maccs"],
    "rf_bits_all": ["--feature-set", "rf_bits_all"],
    "morgan_descriptors": ["--feature-set", "morgan_descriptors"],
    "maccs_descriptors": ["--feature-set", "maccs_descriptors"],
    "all": ["--feature-set", "all"],
    "morgan3": ["--feature-set", "morgan3"],
    "rdkit": ["--feature-set", "rdkit"],
    "atom_pair": ["--feature-set", "atom_pair"],
    "torsion": ["--feature-set", "torsion"],
    "extended_descriptors": ["--feature-set", "extended_descriptors"],
    "all_fp": ["--feature-set", "all_fp"],
    "extended_all": ["--feature-set", "extended_all"],
    "all_context": ["--feature-set", "all", "--context"],
}


def _metrics(out: Path, name: str) -> dict:
    payload = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    si = payload["si"]["blended"]
    oof = pd.read_csv(out / "oof_si.csv")
    y = oof["y_si"].to_numpy()
    pred = oof["pred_si_blended"].to_numpy()
    keep = y < y.max()
    mse_excl_max = float(((pred[keep] - y[keep]) ** 2).mean())
    return {
        "variant": name,
        "ic50_rmse_log10": payload["endpoints"]["ic50"]["rmse_log10"],
        "ic50_r2": payload["endpoints"]["ic50"]["r2"],
        "cc50_rmse_log10": payload["endpoints"]["cc50"]["rmse_log10"],
        "cc50_r2": payload["endpoints"]["cc50"]["r2"],
        "si_mse": si["mse_si"],
        "si_rmse": si["rmse_si"],
        "si_mae": si["mae_si"],
        "si_rmse_log1p": si["rmse_log1p"],
        "si_mse_excl_max": mse_excl_max,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.feature_ablation",
        description="Compare feature sets by dual-model SI OOF MSE.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(VARIANTS),
        default=list(VARIANTS),
    )
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--fine-tune-iterations", type=int, default=100)
    parser.add_argument("--cap-targets", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group", choices=["smiles", "scaffold"], default="smiles")
    parser.add_argument("--out", type=Path, default=RUNS / "results.csv")
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("data/processed/all_targets.csv"),
    )
    parser.add_argument("--organizers", type=Path, default=Path("data/organizers/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/organizers/test.csv"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    RUNS.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for name in args.variants:
        out = RUNS / name
        print(f"\n=== {name} ===", flush=True)
        command = [
            sys.executable,
            "-m",
            "src.dual_model",
            "--targets",
            str(args.targets),
            "--organizers",
            str(args.organizers),
            "--test",
            str(args.test),
            "--out",
            str(out),
            "--folds",
            str(args.folds),
            "--seed",
            str(args.seed),
            "--group",
            args.group,
            "--iterations",
            str(args.iterations),
            "--fine-tune-iterations",
            str(args.fine_tune_iterations),
            *VARIANTS[name],
        ]
        if args.cap_targets:
            command.append("--cap-targets")
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print(f"{name}: failed with {result.returncode}", file=sys.stderr)
            continue
        rows.append(_metrics(out, name))
        frame = pd.DataFrame(rows).sort_values("si_mse")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.out, index=False)
        print(frame.tail(3).to_string(index=False))

    if not rows:
        print("no successful variants", file=sys.stderr)
        return 1
    frame = pd.DataFrame(rows).sort_values("si_mse")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print("\nRanking by SI OOF MSE:")
    print(frame.to_string(index=False))
    print(f"\n{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
