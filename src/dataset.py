from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .sources.base import (
    SourceAdapter,
    SourceError,
    build_dataset,
    validate_dataset,
    write_dataset,
    write_problems,
)
from .sources.chembl import ChemblSource
from .sources.generic_csv import CsvSource
from .sources.pubchem import PubchemSource

DEFAULT_RAW = Path("data/raw")
DEFAULT_OUT = Path("data/dataset.csv")
DEFAULT_DROPPED = Path("data/processed/dropped.csv")


def collect(adapters: list[SourceAdapter], raw_dir: Path) -> pd.DataFrame:
    frames = []
    for adapter in adapters:
        adapter.fetch_raw(raw_dir)
        frames.append(adapter.load(raw_dir))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build(adapters: list[SourceAdapter], raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    measurements = collect(adapters, raw_dir)
    if measurements.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    parsed = measurements.copy()
    parsed["dropped_reason"] = parsed["dropped_reason"].fillna("").astype(str)

    numeric = pd.to_numeric(parsed["value_um"], errors="coerce")
    invalid_value = numeric.notna() & (numeric < 0)
    invalid_value |= (parsed["type"] != "si") & numeric.notna() & (numeric == 0)
    parsed.loc[invalid_value, "dropped_reason"] = "non-positive value"
    unconverted = numeric.isna() & (parsed["dropped_reason"] == "")
    parsed.loc[unconverted, "dropped_reason"] = "value could not be converted"

    parsed_dropped = parsed[parsed["dropped_reason"] != ""].copy()
    parsed_dropped = parsed_dropped.assign(stage="parse")

    dataset, unpaired = build_dataset(parsed)
    pair_dropped = unpaired.rename(columns={"reason": "dropped_reason"})
    pair_dropped = pair_dropped.assign(stage="pair")

    dropped = pd.concat([parsed_dropped, pair_dropped], ignore_index=True)

    valid, problems = validate_dataset(dataset)
    return valid, dropped, problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.dataset",
        description="Builds data/dataset.csv from configured source adapters.",
    )
    parser.add_argument(
        "--chembl",
        action="store_true",
        help="use ChEMBL via chembl-downloader (downloads the dump if needed)",
    )
    parser.add_argument(
        "--chembl-db",
        type=Path,
        help="path to an existing ChEMBL SQLite dump",
    )
    parser.add_argument(
        "--pubchem",
        action="store_true",
        help="use PubChem bioassays for vaccinia + IC50 (downloads if needed)",
    )
    parser.add_argument(
        "--csv",
        action="append",
        default=[],
        metavar="MAPPING",
        help="path to a *.mapping.yaml for a CSV source; repeatable",
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--dropped",
        type=Path,
        default=DEFAULT_DROPPED,
        help="where to write refused rows",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    adapters: list[SourceAdapter] = []
    try:
        if args.chembl_db:
            adapters.append(ChemblSource(args.chembl_db))
        elif args.chembl:
            adapters.append(ChemblSource.from_download())
        for mapping in args.csv:
            adapters.append(CsvSource.from_config(mapping))

        if not adapters:
            print("no sources specified", file=sys.stderr)
            build_parser().print_help(sys.stderr)
            return 2

        dataset, dropped_measurements, problems = build(adapters, args.raw_dir)
    except SourceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    write_dataset(dataset, args.out)

    dropped_frames = []
    if not dropped_measurements.empty:
        dropped_frames.append(dropped_measurements)
    if not problems.empty:
        dropped_frames.append(
            problems.rename(
                columns={"Title": "source_id", "reason": "dropped_reason"}
            ).assign(stage="validate")
        )

    if dropped_frames:
        write_problems(pd.concat(dropped_frames, ignore_index=True), args.dropped)
    else:
        write_problems(
            pd.DataFrame(columns=["stage", "source_id", "dropped_reason"]),
            args.dropped,
        )

    print(f"{args.out}: {len(dataset)} rows")
    print(f"{args.dropped}: {sum(len(item) for item in dropped_frames)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
