"""Materialise ChEMBL IC50/CC50 measurements for the dual-target model.

The official VACV table is tiny.  ChEMBL contains many more measurements of
potency (IC50) and cytotoxicity (CC50), but they live in different assays and
often use different cell lines.  This module pulls those measurements out of a
local ChEMBL SQLite dump, normalises concentrations to uM, and writes three
tables:

* ``chembl_measurements.csv`` - one row per activity, including rejected rows
  with a ``dropped_reason``;
* ``chembl_targets.csv``      - one row per molecule/endpoint/context, ready
  for training separate IC50 and CC50 regressors;
* ``chembl_pairs.csv``        - observed SI values where an IC50 and a CC50
  were measured for the same molecule and cell line.

The scripts never download anything: point ``--db`` at an existing dump.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .normalize.common import (
    UnitError,
    canonical_unit,
    molecular_weight,
    normalize_cell_line,
    standardize_smiles,
    unit_factor,
    unit_needs_molecular_weight,
)

DEFAULT_DB = Path("data/raw/chembl/chembl.db")
DEFAULT_OUT_DIR = Path("data/processed/chembl")

ENDPOINTS = ("IC50", "EC50", "CC50")
POTENCY_ENDPOINTS = ("IC50", "EC50")

VACV_RE = re.compile(r"vaccinia|vacv|\bvv\b|smallpox|variola", re.IGNORECASE)

VIRAL_RE = re.compile(
    r"virus|viral|vaccinia|variola|cowpox|ectromelia|monkeypox|mpox|"
    r"herpes|hsv|cmv|hiv|hepatitis|hcv|hbv|influenza|sars|covid|"
    r"coronavirus|respiratory syncytial|\brsv\b|dengue|zika|ebola|"
    r"enterovirus|rhinovirus|rotavirus|norovirus|measles|polio|"
    r"adenovirus|papillomavirus|\bhpv\b|cytomegalovirus|varicella|"
    r"epstein|\bebv\b",
    re.IGNORECASE,
)

MEASUREMENT_COLUMNS = [
    "source",
    "source_id",
    "Title",
    "SMILES",
    "smiles_key",
    "endpoint",
    "relation",
    "value",
    "units",
    "value_um",
    "log10_um",
    "cell_line",
    "cell_key",
    "assay_organism",
    "target_name",
    "target_organism",
    "context_key",
    "is_vacv",
    "is_viral",
    "assay_id",
    "assay_type",
    "description",
    "doc_id",
    "doi",
    "dropped_reason",
]

TARGET_COLUMNS = [
    "Title",
    "SMILES",
    "smiles_key",
    "endpoint",
    "context_key",
    "y_log10_um",
    "n_measurements",
    "value_min_um",
    "value_max_um",
    "cell_line",
    "organism",
    "target_name",
    "is_vacv",
    "is_viral",
    "doi",
    "source",
]

PAIR_COLUMNS = [
    "Title",
    "SMILES",
    "smiles_key",
    "IC50_uM",
    "CC50_uM",
    "SI",
    "log10_SI",
    "cell_line",
    "organism",
    "doi",
    "n_ic50",
    "n_cc50",
    "pair_type",
    "source",
]

SQL = """
SELECT act.activity_id AS activity_id,
       act.molregno AS molregno,
       act.standard_type AS endpoint,
       act.standard_value AS value,
       act.standard_units AS units,
       COALESCE(act.standard_relation, '=') AS relation,
       COALESCE(act.data_validity_comment, '') AS validity,
       COALESCE(act.potential_duplicate, 0) AS duplicate,
       act.doc_id AS doc_id,
       md.chembl_id AS chembl_id,
       cs.canonical_smiles AS smiles,
       a.assay_id AS assay_id,
       a.description AS description,
       a.assay_type AS assay_type,
       COALESCE(a.assay_organism, '') AS assay_organism,
       COALESCE(NULLIF(cd.cell_name, ''), NULLIF(a.assay_cell_type, ''), '')
           AS cell_line,
       COALESCE(td.pref_name, '') AS target_name,
       COALESCE(td.organism, '') AS target_organism,
       COALESCE(d.doi, '') AS doi
FROM activities act
JOIN assays a ON a.assay_id = act.assay_id
JOIN molecule_dictionary md ON md.molregno = act.molregno
JOIN compound_structures cs ON cs.molregno = act.molregno
LEFT JOIN cell_dictionary cd ON cd.cell_id = a.cell_id
LEFT JOIN target_dictionary td ON td.tid = a.tid
LEFT JOIN docs d ON d.doc_id = act.doc_id
WHERE act.standard_type IN ({endpoints})
  AND act.standard_value IS NOT NULL
  AND cs.canonical_smiles IS NOT NULL
  AND COALESCE(act.potential_duplicate, 0) = 0
  AND (act.data_validity_comment IS NULL
       OR act.data_validity_comment = 'Manually validated')
  AND ({assay_filter})
  AND ({relation_filter})
{limit}
"""


def _placeholders(values: tuple[str, ...]) -> str:
    return ",".join("?" for _ in values)


def _build_sql(
    endpoints: tuple[str, ...],
    ic50_assay_types: tuple[str, ...],
    exact_only: bool,
    limit: int | None,
) -> tuple[str, list]:
    params: list = list(endpoints)
    assay_filter = "1 = 1"
    potency = [item for item in endpoints if item in POTENCY_ENDPOINTS]
    if potency:
        if ic50_assay_types:
            potency_keys = ",".join(f"'{item}'" for item in potency)
            assay_filter = (
                f"(act.standard_type NOT IN ({potency_keys}) "
                f"OR a.assay_type IN ({_placeholders(ic50_assay_types)}))"
            )
            params.extend(ic50_assay_types)
        else:
            potency_keys = ",".join(f"'{item}'" for item in potency)
            assay_filter = f"act.standard_type NOT IN ({potency_keys})"

    relation_filter = "act.standard_relation = '='" if exact_only else "1 = 1"
    limit_sql = "LIMIT ?" if limit else ""
    sql = SQL.format(
        endpoints=_placeholders(endpoints),
        assay_filter=assay_filter,
        relation_filter=relation_filter,
        limit=limit_sql,
    )
    if limit:
        params.append(int(limit))
    return sql, params


def _read_smiles_keys(path: Path) -> set[str]:
    frame = pd.read_csv(path)
    column = None
    for candidate in ("SMILES", "smiles", "canonical_smiles"):
        if candidate in frame:
            column = candidate
            break
    if column is None:
        raise ValueError(f"{path}: no SMILES column")
    keys: set[str] = set()
    for value in frame[column].dropna().astype(str):
        if not value.strip():
            continue
        keys.add(value.strip())
        canonical = standardize_smiles(value)
        if canonical:
            keys.add(canonical)
    return keys


def _context_key(endpoint: str, is_vacv: bool, cell_key: str, target_organism: str,
                 target_name: str) -> str:
    if endpoint == "cc50":
        return "cell:" + (cell_key or "unknown")
    if is_vacv:
        return "vacv:" + (cell_key or "unknown")
    target = (target_organism or target_name or "").strip().lower()
    return "target:" + (target or "unknown") + ":" + (cell_key or "unknown")


def _classify(text: pd.Series, pattern: re.Pattern[str]) -> pd.Series:
    return text.fillna("").astype(str).str.contains(pattern, regex=True)


def _canonical_map(values: pd.Series) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values.dropna().astype(str).unique():
        if not value.strip():
            continue
        mapping[value] = standardize_smiles(value) or ""
    return mapping


def _convert_values(
    frame: pd.DataFrame, reasons: pd.Series, mw_cache: dict[str, float | None]
) -> pd.Series:
    value_um = pd.Series(np.nan, index=frame.index, dtype=float)
    unit_keys = frame["units"].map(canonical_unit)
    for unit, indices in unit_keys.groupby(unit_keys).groups.items():
        indices = list(indices)
        try:
            factor = unit_factor(unit)
        except UnitError as exc:
            if not unit_needs_molecular_weight(unit):
                reasons.loc[indices] = str(exc)
                continue
            for index in indices:
                if reasons.at[index] != "":
                    continue
                smiles = frame.at[index, "SMILES"]
                if smiles not in mw_cache:
                    mw_cache[smiles] = molecular_weight(smiles)
                weight = mw_cache[smiles]
                try:
                    factor = unit_factor(unit, weight)
                except UnitError as inner:
                    reasons.at[index] = str(inner)
                    continue
                value_um.at[index] = float(frame.at[index, "value"]) * factor
            continue
        values = pd.to_numeric(frame.loc[indices, "value"], errors="coerce")
        value_um.loc[indices] = values.to_numpy(dtype=float) * factor
    return value_um


def load_measurements(
    db_path: str | Path,
    endpoints: tuple[str, ...] = ENDPOINTS,
    ic50_assay_types: tuple[str, ...] = ("F",),
    exact_only: bool = True,
    ec50_as_ic50: bool = True,
    scope: str = "all",
    exclude_smiles: tuple[Path, ...] = (),
    min_um: float = 1e-6,
    max_um: float = 1e6,
    limit: int | None = None,
) -> pd.DataFrame:
    """Read and normalise IC50/CC50 activities from a ChEMBL SQLite dump."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"ChEMBL SQLite dump not found at {db_path}. Download it first; "
            "see doc/chembl.md"
        )

    sql, params = _build_sql(endpoints, ic50_assay_types, exact_only, limit)
    with sqlite3.connect(db_path) as connection:
        frame = pd.read_sql_query(sql, connection, params=params)

    if frame.empty:
        return pd.DataFrame(columns=MEASUREMENT_COLUMNS)

    frame = frame.rename(columns={"smiles": "SMILES", "chembl_id": "source_id"})
    frame["source"] = "chembl"
    frame["Title"] = frame["source_id"]
    frame["endpoint"] = frame["endpoint"].astype(str).str.lower()
    if ec50_as_ic50:
        frame.loc[frame["endpoint"] == "ec50", "endpoint"] = "ic50"
    reasons = pd.Series("", index=frame.index, dtype=object)
    return finalize_measurements(
        frame, reasons, scope, exclude_smiles, min_um, max_um
    )


def finalize_measurements(
    frame: pd.DataFrame,
    reasons: pd.Series,
    scope: str,
    exclude_smiles: tuple[Path, ...],
    min_um: float,
    max_um: float,
) -> pd.DataFrame:
    """Shared post-processing for SQLite and CSV exports."""
    frame = frame.copy()
    reasons = reasons.copy()
    frame["dropped_reason"] = ""
    for column in MEASUREMENT_COLUMNS:
        if column not in frame:
            frame[column] = ""

    missing = frame["SMILES"].isna() | frame["SMILES"].astype(str).str.strip().eq("")
    reasons.loc[missing] = "missing SMILES"

    frame["SMILES"] = frame["SMILES"].fillna("").astype(str).str.strip()
    frame["smiles_key"] = ""
    key_map = _canonical_map(frame.loc[~missing, "SMILES"])
    frame.loc[~missing, "smiles_key"] = frame.loc[~missing, "SMILES"].map(key_map)
    reasons.loc[(reasons == "") & (frame["smiles_key"] == "")] = "invalid SMILES"

    mw_cache: dict[str, float | None] = {}
    frame["value_um"] = _convert_values(frame, reasons, mw_cache)

    numeric = pd.to_numeric(frame["value_um"], errors="coerce")
    bad_value = numeric.isna() & (reasons == "")
    reasons.loc[bad_value] = "value could not be converted"
    non_positive = (reasons == "") & (numeric <= 0)
    reasons.loc[non_positive] = "non-positive value"
    outside = (reasons == "") & ((numeric < min_um) | (numeric > max_um))
    reasons.loc[outside] = "outside value range"
    frame["value_um"] = numeric
    frame["log10_um"] = np.log10(numeric)

    text = (
        frame["description"].fillna("").astype(str)
        + " "
        + frame["assay_organism"].fillna("").astype(str)
        + " "
        + frame["target_name"].fillna("").astype(str)
        + " "
        + frame["target_organism"].fillna("").astype(str)
    )
    frame["is_vacv"] = _classify(text, VACV_RE)
    frame["is_viral"] = _classify(text, VIRAL_RE)

    frame["cell_line"] = frame["cell_line"].fillna("").astype(str).str.strip()
    frame["cell_key"] = frame["cell_line"].map(normalize_cell_line)
    frame["context_key"] = [
        _context_key(endpoint, bool(vacv), cell, organism, target)
        for endpoint, vacv, cell, organism, target in zip(
            frame["endpoint"],
            frame["is_vacv"],
            frame["cell_key"],
            frame["target_organism"],
            frame["target_name"],
        )
    ]

    if scope == "viral":
        drop = (frame["endpoint"] == "ic50") & ~frame["is_viral"]
        reasons.loc[(reasons == "") & drop] = "not a viral IC50"
    elif scope == "vacv":
        drop = (frame["endpoint"] == "ic50") & ~frame["is_vacv"]
        reasons.loc[(reasons == "") & drop] = "not a VACV IC50"

    excluded: set[str] = set()
    for path in exclude_smiles:
        excluded |= _read_smiles_keys(Path(path))
    if excluded:
        hit = (reasons == "") & frame["smiles_key"].isin(excluded)
        reasons.loc[hit] = "excluded SMILES"

    frame["dropped_reason"] = reasons
    return frame[MEASUREMENT_COLUMNS]


def build_targets(measurements: pd.DataFrame) -> pd.DataFrame:
    """Aggregate valid measurements to one row per molecule/endpoint/context."""
    if measurements.empty:
        return pd.DataFrame(columns=TARGET_COLUMNS)

    valid = measurements[measurements["dropped_reason"].fillna("") == ""].copy()
    if valid.empty:
        return pd.DataFrame(columns=TARGET_COLUMNS)

    if "source" not in valid:
        valid["source"] = ""
    valid["organism"] = valid["target_organism"].where(
        valid["target_organism"].astype(str).str.strip().ne(""),
        valid["assay_organism"],
    )
    grouped = (
        valid.groupby(["smiles_key", "endpoint", "context_key"], sort=False)
        .agg(
            Title=("Title", "first"),
            SMILES=("SMILES", "first"),
            y_log10_um=("log10_um", "median"),
            n_measurements=("log10_um", "size"),
            value_min_um=("value_um", "min"),
            value_max_um=("value_um", "max"),
            cell_line=("cell_line", "first"),
            organism=("organism", "first"),
            target_name=("target_name", "first"),
            is_vacv=("is_vacv", "max"),
            is_viral=("is_viral", "max"),
            doi=("doi", "first"),
            source=("source", "first"),
        )
        .reset_index()
    )
    return grouped[TARGET_COLUMNS]


def _pair_block(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    columns = keys + ["Title", "SMILES", "cell_line", "organism", "doi", "source"]
    grouped = (
        frame.groupby(keys, sort=False)
        .agg(
            Title=("Title", "first"),
            SMILES=("SMILES", "first"),
            cell_line=("cell_line", "first"),
            organism=("target_organism", "first"),
            doi=("doi", "first"),
            source=("source", "first"),
            n_ic50=("log10_um", "size"),
            ic50_log=("log10_um", "median"),
        )
        .reset_index()
    )
    return grouped[columns + ["n_ic50", "ic50_log"]]


def build_pairs(measurements: pd.DataFrame) -> pd.DataFrame:
    """Pair VACV IC50 with CC50 on the same molecule and cell line."""
    if measurements.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)

    valid = measurements[measurements["dropped_reason"].fillna("") == ""].copy()
    ic50 = valid[(valid["endpoint"] == "ic50") & valid["is_vacv"].astype(bool)]
    cc50 = valid[valid["endpoint"] == "cc50"]
    ic50 = ic50[ic50["cell_key"].ne("")]
    cc50 = cc50[cc50["cell_key"].ne("")]
    if ic50.empty or cc50.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)

    strict_ic = _pair_block(ic50, ["smiles_key", "cell_key", "doc_id"])
    strict_cc = _pair_block(cc50, ["smiles_key", "cell_key", "doc_id"])
    strict = strict_ic.merge(
        strict_cc,
        on=["smiles_key", "cell_key", "doc_id"],
        suffixes=("_ic", "_cc"),
    )
    strict["pair_type"] = "same_doc"

    loose_ic = _pair_block(ic50, ["smiles_key", "cell_key"])
    loose_cc = _pair_block(cc50, ["smiles_key", "cell_key"])
    loose = loose_ic.merge(
        loose_cc,
        on=["smiles_key", "cell_key"],
        suffixes=("_ic", "_cc"),
    )
    loose["pair_type"] = "same_cell"

    pairs = pd.concat([strict, loose], ignore_index=True)
    pairs = pairs.drop_duplicates(["smiles_key", "cell_key"], keep="first")
    pairs = pairs.rename(
        columns={
            "Title_ic": "Title",
            "SMILES_ic": "SMILES",
            "cell_line_ic": "cell_line",
            "organism_ic": "organism",
            "doi_ic": "doi",
            "n_ic50_ic": "n_ic50",
            "n_ic50_cc": "n_cc50",
            "ic50_log_ic": "ic50_log",
            "ic50_log_cc": "cc50_log",
            "source_ic": "source",
        }
    )
    pairs["IC50_uM"] = np.power(10.0, pairs["ic50_log"])
    pairs["CC50_uM"] = np.power(10.0, pairs["cc50_log"])
    pairs["log10_SI"] = pairs["cc50_log"] - pairs["ic50_log"]
    pairs["SI"] = np.power(10.0, pairs["log10_SI"])
    for column in PAIR_COLUMNS:
        if column not in pairs:
            pairs[column] = ""
    return pairs[PAIR_COLUMNS]


def write_outputs(
    measurements: pd.DataFrame,
    targets: pd.DataFrame,
    pairs: pd.DataFrame,
    out_dir: Path,
    measurements_path: Path | None = None,
    targets_path: Path | None = None,
    pairs_path: Path | None = None,
    append: bool = False,
) -> tuple[Path, Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    measurements_path = measurements_path or out_dir / "chembl_measurements.csv"
    targets_path = targets_path or out_dir / "chembl_targets.csv"
    pairs_path = pairs_path or out_dir / "chembl_pairs.csv"
    for path, frame in (
        (measurements_path, measurements),
        (targets_path, targets),
        (pairs_path, pairs),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        if append and path.exists():
            old = pd.read_csv(path)
            frame = pd.concat([old, frame], ignore_index=True)
            frame = frame.drop_duplicates().reset_index(drop=True)
        frame.to_csv(path, index=False)
    return measurements_path, targets_path, pairs_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.chembl_pairs",
        description="Extract IC50/CC50 pairs and targets from a ChEMBL dump.",
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument(
        "--scope",
        choices=["all", "viral", "vacv"],
        default="viral",
        help="which IC50 rows to keep (CC50 rows are always kept)",
    )
    parser.add_argument(
        "--ic50-assay-types",
        default="F",
        help="comma-separated ChEMBL assay_type values allowed for IC50 "
        "(F = functional/cellular, B = binding)",
    )
    parser.add_argument(
        "--exact-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep only standard_relation '=' (default: yes)",
    )
    parser.add_argument(
        "--include-ec50",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pool EC50 with IC50 (default: yes)",
    )
    parser.add_argument(
        "--exclude-smiles",
        action="append",
        type=Path,
        default=[],
        metavar="CSV",
        help="CSV with SMILES to exclude (e.g. the organizers' test set); repeatable",
    )
    parser.add_argument("--min-um", type=float, default=1e-6)
    parser.add_argument("--max-um", type=float, default=1e6)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="debug: stop after N activities",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="append new parsed rows to existing output files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.db.exists():
        print(
            f"ERROR: ChEMBL dump not found at {args.db}. "
            "Download it first; see doc/chembl.md",
            file=sys.stderr,
        )
        return 1

    endpoints = tuple(
        item
        for item in ENDPOINTS
        if item != "EC50" or args.include_ec50
    )
    ic50_types = tuple(
        item.strip() for item in str(args.ic50_assay_types).split(",") if item.strip()
    )
    measurements = load_measurements(
        args.db,
        endpoints=endpoints,
        ic50_assay_types=ic50_types,
        exact_only=args.exact_only,
        ec50_as_ic50=args.include_ec50,
        scope=args.scope,
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
        append=args.append,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
