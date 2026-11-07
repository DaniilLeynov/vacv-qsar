"""Parse ChEMBL web-export CSV files into the dual-model target table.

The ChEMBL web interface can export an activity table as CSV.  These files
contain almost the same information as the SQLite dump and are much easier to
share than a 5 GB database.  This module accepts one or more such exports and
writes the same three tables as :mod:`src.chembl_pairs`.

Typical use with separate IC50 and EC50 exports::

    python -m src.chembl_export \
        --csv "/path/IC_data.csv" \
        --csv "/path/EC_data.csv" \
        --ec50-as-ic50 \
        --exclude-smiles data/organizers/test.csv \
        --out-dir data/processed/chembl_export

The ``--ec50-as-ic50`` switch is on by default: in antiviral assays EC50 and
IC50 are both measurements of the same potency and are pooled into one
regression target.  A CC50 export can be added with ``--csv`` as well; without
it the CC50 head has nothing to train on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .chembl_pairs import (
    build_pairs,
    build_targets,
    finalize_measurements,
    write_outputs,
)

DEFAULT_OUT_DIR = Path("data/processed/chembl_export")

COLUMN_MAP = {
    "Molecule ChEMBL ID": "source_id",
    "Molecule Name": "molecule_name",
    "Smiles": "SMILES",
    "Standard Type": "endpoint",
    "Standard Relation": "relation",
    "Standard Value": "value",
    "Standard Units": "units",
    "Data Validity Comment": "validity",
    "Potential Duplicate": "duplicate",
    "Assay ChEMBL ID": "assay_id",
    "Assay Description": "description",
    "Assay Type": "assay_type",
    "Assay Organism": "assay_organism",
    "Assay Cell Type": "cell_line",
    "Target Name": "target_name",
    "Target Organism": "target_organism",
    "Document ChEMBL ID": "doc_id",
}

def _clean_relation(value: object) -> str:
    text = str(value).strip()
    return text.replace("'", "").replace('"', "")


def _read_one(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(COLUMN_MAP) - set(frame.columns)
    # Export files vary slightly between ChEMBL releases.  Require at least the
    # columns needed for the target table; optional context columns become "".
    required_columns = {
        "Molecule ChEMBL ID",
        "Smiles",
        "Standard Type",
        "Standard Value",
        "Standard Units",
    }
    missing_required = required_columns - set(frame.columns)
    if missing_required:
        raise ValueError(f"{path}: missing columns {sorted(missing_required)}")
    for column in missing:
        frame[column] = ""
    frame = frame.rename(columns=COLUMN_MAP)
    frame["export_file"] = str(path)
    return frame


def read_exports(
    paths: list[Path],
    ec50_as_ic50: bool = True,
    exact_only: bool = True,
    scope: str = "viral",
    ic50_assay_types: tuple[str, ...] = ("F",),
    exclude_smiles: tuple[Path, ...] = (),
    min_um: float = 1e-6,
    max_um: float = 1e6,
    limit: int | None = None,
) -> pd.DataFrame:
    frames = [_read_one(Path(path)) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    if limit:
        frame = frame.iloc[: int(limit)].reset_index(drop=True)

    frame["source"] = "chembl"
    frame["Title"] = frame["source_id"].astype(str)
    frame["SMILES"] = frame["SMILES"].fillna("").astype(str).str.strip()
    frame["endpoint"] = frame["endpoint"].astype(str).str.strip().str.lower()
    frame["relation"] = frame["relation"].map(_clean_relation)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["units"] = frame["units"].fillna("").astype(str)

    reasons = pd.Series("", index=frame.index, dtype=object)

    unsupported = ~frame["endpoint"].isin(["ic50", "ec50", "cc50"])
    reasons.loc[unsupported] = "unsupported endpoint"
    if ec50_as_ic50:
        frame.loc[frame["endpoint"] == "ec50", "endpoint"] = "ic50"

    duplicate = pd.to_numeric(frame["duplicate"], errors="coerce").fillna(0)
    reasons.loc[(reasons == "") & (duplicate != 0)] = "potential duplicate"

    validity = frame["validity"].fillna("").astype(str).str.strip()
    bad_validity = (reasons == "") & (~validity.isin(["", "Manually validated"]))
    reasons.loc[bad_validity] = "invalid data validity: " + validity[bad_validity]

    if exact_only:
        bad_relation = (reasons == "") & (frame["relation"] != "=")
        reasons.loc[bad_relation] = "censored relation"

    assay_types = set(ic50_assay_types)
    wrong_type = (
        (reasons == "")
        & (frame["endpoint"] == "ic50")
        & (~frame["assay_type"].astype(str).isin(assay_types))
    )
    reasons.loc[wrong_type] = "unsupported assay type"

    return finalize_measurements(
        frame, reasons, scope, exclude_smiles, min_um, max_um
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.chembl_export",
        description="Build IC50/CC50 targets from ChEMBL web-export CSV files.",
    )
    parser.add_argument(
        "--csv",
        action="append",
        type=Path,
        required=True,
        metavar="PATH",
        help="ChEMBL activity export; repeatable (IC50, EC50, CC50, ...)",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument(
        "--scope",
        choices=["all", "viral", "vacv"],
        default="viral",
        help="which potency rows to keep (CC50 rows are always kept)",
    )
    parser.add_argument(
        "--ic50-assay-types",
        default="F",
        help="comma-separated assay types allowed for IC50/EC50",
    )
    parser.add_argument(
        "--ec50-as-ic50",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pool EC50 with IC50 (default: yes)",
    )
    parser.add_argument(
        "--exact-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--exclude-smiles",
        action="append",
        type=Path,
        default=[],
        metavar="CSV",
    )
    parser.add_argument("--min-um", type=float, default=1e-6)
    parser.add_argument("--max-um", type=float, default=1e6)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    assay_types = tuple(
        item.strip() for item in str(args.ic50_assay_types).split(",") if item.strip()
    )
    measurements = read_exports(
        args.csv,
        ec50_as_ic50=args.ec50_as_ic50,
        exact_only=args.exact_only,
        scope=args.scope,
        ic50_assay_types=assay_types,
        exclude_smiles=tuple(args.exclude_smiles),
        min_um=args.min_um,
        max_um=args.max_um,
        limit=args.limit,
    )
    targets = build_targets(measurements)
    pairs = build_pairs(measurements)
    paths = write_outputs(
        measurements,
        targets,
        pairs,
        args.out_dir,
        args.measurements,
        args.targets,
        args.pairs,
    )
    dropped = measurements[measurements["dropped_reason"] != ""]
    print(f"measurements: {len(measurements)} rows ({len(dropped)} dropped)")
    if not measurements.empty:
        print(measurements.groupby("endpoint").size().to_string())
    print(f"targets: {len(targets)} rows")
    print(f"pairs: {len(pairs)} rows")
    for path in paths:
        print(path)
    if not dropped.empty:
        print("dropped reasons:")
        print(dropped["dropped_reason"].value_counts().to_string())
    if "cc50" not in set(targets.get("endpoint", [])):
        print(
            "WARNING: no CC50 rows found; the CC50 head cannot be trained",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
