"""Build an opt-in augmented dataset for non-dual models.

The base ``data/common/dataset.csv`` is left untouched.  This script appends
new rows from the combined database only when they pass simple filters:

* valid SMILES;
* positive SI (``SI = 0`` means missing IC50/CC50 and is not useful as a
  regression target for CatBoost/ensemble);
* SI not above ``--max-si`` (default 10000) to avoid the extreme tail that
  dominates raw MSE;
* no duplicate canonical SMILES already present in the base table.

The result is ``data/common/dataset_augmented.csv``.  It is used only when the
training command explicitly asks for it:

    make train MODEL=catboost TRAIN_DATA=data/common/dataset_augmented.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .normalize.common import standardize_smiles
from .sources.base import DATASET_COLUMNS

DEFAULT_BASE = Path("data/common/dataset.csv")
DEFAULT_EXTRA = Path("data/common/dataset_all.csv")
DEFAULT_OUT = Path("data/common/dataset_augmented.csv")

SOURCE_PRIORITY = {
    "selected": 0,
    "organizers": 0,
    "chembl": 1,
    "paper_cidofovir": 2,
    "paper_bmcl2022": 2,
    "paper_adamantane2021": 2,
    "pubchem": 3,
}


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path)
    for column in columns:
        if column not in frame:
            frame[column] = ""
    return frame[columns].copy()


def _canonical(values: pd.Series) -> pd.Series:
    return values.map(standardize_smiles)


def augment(
    base_path: Path = DEFAULT_BASE,
    extra_path: Path = DEFAULT_EXTRA,
    out_path: Path = DEFAULT_OUT,
    max_si: float = 10000.0,
) -> pd.DataFrame:
    base = _read(Path(base_path), DATASET_COLUMNS)
    extra = _read(Path(extra_path), DATASET_COLUMNS)
    if extra.empty:
        result = base.reset_index(drop=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        return result

    base_keys = set(_canonical(base["SMILES"]).dropna())
    extra = extra.copy()
    extra["_canonical"] = _canonical(extra["SMILES"])
    si = pd.to_numeric(extra["SI VACV"], errors="coerce")
    extra["_si"] = si
    keep = (
        extra["_canonical"].notna()
        & extra["_canonical"].ne("")
        & ~extra["_canonical"].isin(base_keys)
        & extra["_si"].notna()
        & (extra["_si"] > 0)
        & (extra["_si"] <= max_si)
    )
    extra = extra[keep].copy()
    extra["_priority"] = extra["source"].map(
        lambda value: SOURCE_PRIORITY.get(value, 99)
    )
    extra = extra.sort_values("_priority", kind="stable")
    extra = extra.drop_duplicates("_canonical", keep="first")
    extra = extra.drop(columns=["_canonical", "_si", "_priority"])

    result = pd.concat([base, extra], ignore_index=True)
    result = result[DATASET_COLUMNS]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(
        f"{out_path}: {len(result)} rows "
        f"(base {len(base)}, added {len(extra)}, "
        f"max SI {max_si:g})"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.augment_common",
        description="Build opt-in augmented data/common/dataset_augmented.csv.",
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--extra", type=Path, default=DEFAULT_EXTRA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-si", type=float, default=10000.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    augment(args.base, args.extra, args.out, args.max_si)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
