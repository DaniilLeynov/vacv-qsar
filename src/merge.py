from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .normalize.common import standardize_smiles

COLUMNS = [
    "Title",
    "CC50_uM",
    "IC50_uM_VACV",
    "SI VACV",
    "SI_lower",
    "SI_upper",
    "SMILES",
    "strain",
    "Culture",
    "doi",
    "source",
]

# Lower wins: on a duplicate SMILES the curated source is kept over the raw
# scraped one.
SOURCE_PRIORITY = {
    "selected": 0,
    "organizers": 0,
    "chembl": 1,
    "paper_cidofovir": 2,
    "paper_bmcl2022": 2,
    "paper_adamantane2021": 2,
    "pubchem": 3,
}


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in COLUMNS:
        if column not in frame:
            frame[column] = ""
    return frame[COLUMNS]


def merge(
    datasets: list[Path],
    test: Path | None,
    out: Path,
) -> pd.DataFrame:
    frames = [_read(path) for path in datasets if path.exists()]
    if not frames:
        raise FileNotFoundError("no input datasets found")

    combined = pd.concat(frames, ignore_index=True)
    combined["_canonical"] = combined["SMILES"].map(standardize_smiles)
    combined = combined[combined["_canonical"].notna()].copy()
    combined["_priority"] = combined["source"].map(
        lambda value: SOURCE_PRIORITY.get(value, 99)
    )
    combined = combined.sort_values("_priority", kind="stable")

    dropped_test = 0
    if test is not None and test.exists():
        test_smiles = set(
            pd.read_csv(test)["SMILES"].map(standardize_smiles).dropna()
        )
        keep = ~combined["_canonical"].isin(test_smiles)
        dropped_test = int((~keep).sum())
        combined = combined[keep]

    before = len(combined)
    combined = combined.drop_duplicates("_canonical", keep="first")
    duplicates = before - len(combined)

    result = combined.drop(columns=["_canonical", "_priority"]).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)

    print(
        f"{out}: {len(result)} rows "
        f"(removed {duplicates} duplicate SMILES, {dropped_test} test overlaps)"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.merge",
        description="Concatenate dataset CSVs, drop test overlap and duplicate SMILES.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        action="append",
        default=[],
        help="input dataset CSV; repeatable, earlier wins on duplicate SMILES",
    )
    parser.add_argument("--test", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("data/common/dataset.csv"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dataset:
        raise SystemExit("no --dataset given")
    merge(args.dataset, args.test, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
