"""Build the combined organizers + ChEMBL + PubChem dataset.

The wide table keeps the organizers' contract (one row per molecule/assay
with SI and provenance).  The long table is what the dual IC50/CC50 model
consumes:

    SMILES, endpoint, y_log10_um, context, source, ...

ChEMBL measurements are produced by :mod:`src.chembl_pairs`; PubChem is
downloaded by the existing :class:`src.sources.pubchem.PubchemSource` adapter.
Nothing in this module goes to the network.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .chembl_pairs import (
    MEASUREMENT_COLUMNS,
    _read_smiles_keys,
    build_targets,
    finalize_measurements,
)
from .normalize.common import standardize_smiles
from .sources.base import build_dataset, validate_dataset
from .sources.generic_csv import CsvSource
from .sources.pubchem import PubchemSource

DEFAULT_OUT = Path("data/common/dataset_all.csv")
DEFAULT_MEASUREMENTS = Path("data/dual/all_measurements.csv")
DEFAULT_TARGETS = Path("data/dual/all_targets.csv")
DEFAULT_IC50_TARGETS = Path("data/dual/ic50_targets.csv")
DEFAULT_CC50_TARGETS = Path("data/dual/cc50_targets.csv")
DEFAULT_DROPPED = Path("data/dual/all_dropped.csv")
DEFAULT_CHEMBL = Path("data/processed/chembl/chembl_measurements.csv")
DEFAULT_ORGANIZERS = Path("sources/organizers.mapping.yaml")
DEFAULT_PCHEM_SETS = (
    "pubchem",
    "pubchem_ec50",
    "pubchem_cc50",
    "pubchem_vacv_ic50",
    "pubchem_vacv_ec50",
    "pubchem_vacv_cc50",
    "pubchem_orthopox_ic50",
)

COMMON_COLUMNS = MEASUREMENT_COLUMNS + ["study_id", "organism"]

ASCII_TERMS = {
    "pubchem": "vaccinia IC50",
    "pubchem_ec50": "vaccinia EC50",
    "pubchem_cc50": "vaccinia CC50",
    "pubchem_vacv_ic50": "VACV IC50",
    "pubchem_vacv_ec50": "VACV EC50",
    "pubchem_vacv_cc50": "VACV CC50",
    "pubchem_orthopox_ic50": "orthopoxvirus IC50",
}


def _smiles_keys(values: pd.Series) -> pd.Series:
    unique = {value for value in values.dropna().astype(str) if value.strip()}
    mapping = {value: standardize_smiles(value) or "" for value in unique}
    return values.map(mapping).fillna("")


def _column(frame: pd.DataFrame, *names: str, default="") -> pd.Series:
    for name in names:
        if name in frame:
            return frame[name]
    return pd.Series(default, index=frame.index)


def _prepare_source_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["source"] = _column(frame, "source", default=source).fillna(source)
    frame["Title"] = _column(frame, "title", "Title").fillna("")
    frame["SMILES"] = _column(frame, "smiles", "SMILES").fillna("")
    frame["source_id"] = _column(frame, "source_id").fillna(frame["Title"])
    frame["type"] = _column(frame, "type", "endpoint").astype(str).str.lower()
    frame["endpoint"] = frame["type"]
    frame["organism"] = _column(frame, "organism").fillna("")
    frame["assay_organism"] = _column(frame, "assay_organism").fillna(
        frame["organism"]
    )
    frame["target_name"] = _column(frame, "target_name").fillna(frame["organism"])
    frame["target_organism"] = _column(frame, "target_organism").fillna(
        frame["organism"]
    )
    frame["description"] = _column(frame, "description").fillna("")
    frame["cell_line"] = _column(frame, "cell_line").fillna("")
    frame["doc_id"] = _column(frame, "doc_id", "study_id").fillna("")
    frame["study_id"] = _column(frame, "study_id", "doc_id").fillna("")
    frame["doi"] = _column(frame, "doi").fillna("")
    frame["relation"] = _column(frame, "relation", default="=").fillna("=").astype(str)
    frame["value"] = _column(frame, "value", "value_um")
    frame["units"] = _column(frame, "units").fillna("")
    return frame


def _endpoint_rows(
    frame: pd.DataFrame, source: str, scope: str = "vacv"
) -> pd.DataFrame:
    prepared = _prepare_source_frame(frame, source)
    endpoint = prepared[prepared["endpoint"].isin(["ic50", "cc50"])].copy()
    endpoint = endpoint[endpoint["SMILES"].astype(str).str.strip().ne("")].copy()
    if "dropped_reason" in endpoint:
        reasons = endpoint["dropped_reason"].fillna("").astype(str)
    else:
        reasons = pd.Series("", index=endpoint.index, dtype=object)
    finalized = finalize_measurements(
        endpoint, reasons, scope, (), 1e-6, 1e6
    )
    finalized["study_id"] = prepared.loc[finalized.index, "study_id"]
    finalized["organism"] = prepared.loc[finalized.index, "organism"]
    return finalized


def _si_rows(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    prepared = _prepare_source_frame(frame, source)
    si = prepared[prepared["endpoint"].eq("si")].copy()
    if si.empty:
        return pd.DataFrame(columns=COMMON_COLUMNS)
    si["value_um"] = pd.to_numeric(si["value_um"], errors="coerce")
    si["log10_um"] = None
    si["smiles_key"] = _smiles_keys(si["SMILES"])
    si["cell_key"] = si["cell_line"].fillna("").astype(str)
    si["context_key"] = "si"
    text = (
        si["description"].fillna("").astype(str)
        + " "
        + si["organism"].fillna("").astype(str)
    )
    si["is_vacv"] = text.str.contains("vaccinia|vacv|orthopox", case=False, regex=True)
    si["is_viral"] = True
    for column in COMMON_COLUMNS:
        if column not in si:
            si[column] = ""
    return si[COMMON_COLUMNS]


def _clean_organizer_ic50(
    frame: pd.DataFrame, chembl: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop IC50 labels that are really censored/inactive values.

    The organizers' table has no relation column.  A repeated high value such
    as 1200 uM together with SI < 1 usually means "no inhibition up to the
    maximum tested concentration", not an exact IC50.  ChEMBL measurements for
    the same canonical SMILES provide a cross-check: if they disagree by more
    than 10x and the organizer SI is below 1, the organizer IC50 is not safe as
    a regression label.
    """
    frame = frame.copy()
    frame["smiles_key"] = _smiles_keys(frame["smiles"])
    si_map = (
        frame[frame["type"] == "si"]
        .groupby("source_id")["value_um"]
        .median()
        .to_dict()
    )

    chembl_ic = chembl[
        (chembl["endpoint"] == "ic50") & chembl["is_vacv"].astype(bool)
    ].copy()
    if chembl_ic.empty:
        chembl_map: dict[str, float] = {}
    else:
        chembl_map = (
            chembl_ic.groupby("smiles_key")["value_um"].median().to_dict()
        )

    reports: list[dict] = []
    ic50 = frame["type"] == "ic50"
    for index, row in frame[ic50].iterrows():
        si = si_map.get(row["source_id"])
        chembl_value = chembl_map.get(row["smiles_key"])
        ratio = None
        if chembl_value and row["value_um"]:
            ratio = float(row["value_um"]) / float(chembl_value)
        conflict = (
            si is not None
            and si < 1
            and ratio is not None
            and (ratio > 10 or ratio < 0.1)
        )
        suspicious_cap = (
            si is not None
            and si < 1
            and float(row["value_um"] or 0) >= 500
        )
        if conflict or suspicious_cap:
            reason = "censored/conflicting IC50"
            frame.at[index, "dropped_reason"] = reason
            reports.append(
                {
                    "Title": row.get("title", ""),
                    "source_id": row["source_id"],
                    "SI": si,
                    "IC50_uM": row["value_um"],
                    "chembl_IC50_uM": chembl_value,
                    "ratio": ratio,
                    "reason": reason,
                }
            )
    return frame, pd.DataFrame(reports)


def _read_chembl(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["dropped_reason"] = frame["dropped_reason"].fillna("").astype(str)
    frame = frame[frame["dropped_reason"] == ""].copy()
    frame["endpoint"] = frame["endpoint"].astype(str).str.lower()
    frame = frame[frame["endpoint"].isin(["ic50", "cc50"])].copy()
    frame["type"] = frame["endpoint"]
    frame["study_id"] = frame["doc_id"].fillna("").astype(str)
    frame["organism"] = frame["target_organism"].where(
        frame["target_organism"].fillna("").astype(str).str.strip().ne(""),
        frame["assay_organism"],
    )
    for column in COMMON_COLUMNS:
        if column not in frame:
            frame[column] = ""
    return frame[COMMON_COLUMNS]


def _read_pubchem(raw_dir: Path, names: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for name in names:
        path = Path(raw_dir) / name
        if not path.exists():
            continue
        source = PubchemSource(name=name)
        frame = source.load(Path(raw_dir))
        if frame.empty:
            continue
        frame["source"] = "pubchem"
        frame["description"] = (
            _column(frame, "description").fillna("").astype(str)
            + " "
            + ASCII_TERMS.get(name, name)
        ).str.strip()
        frames.append(frame)
        print(f"pubchem {name}: {len(frame)} measurements")
    if not frames:
        print(
            f"WARNING: no PubChem raw sets found under {raw_dir}; "
            "dataset_all will use local sources only."
        )
        return pd.DataFrame(columns=MEASUREMENT_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def build(
    organizers: Path = DEFAULT_ORGANIZERS,
    chembl: Path = DEFAULT_CHEMBL,
    pubchem_sets: tuple[str, ...] = DEFAULT_PCHEM_SETS,
    raw_dir: Path = Path("data/raw"),
    exclude_smiles: tuple[Path, ...] = (Path("data/organizers/test.csv"),),
    clean_ic50: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    excluded = set()
    for path in exclude_smiles:
        if Path(path).exists():
            excluded |= _read_smiles_keys(Path(path))

    endpoint_parts: list[pd.DataFrame] = []
    si_parts: list[pd.DataFrame] = []
    conflict_parts: list[pd.DataFrame] = []

    chembl_frame = (
        _read_chembl(Path(chembl)) if Path(chembl).exists() else pd.DataFrame()
    )
    if chembl_frame.empty:
        print(
            f"WARNING: no ChEMBL rows from {chembl}; "
            "dataset_all will use local sources only. "
            "Run `make chembl-pairs` after downloading the dump."
        )

    local_sources: list[tuple[str, Path]] = []
    if Path(organizers).exists():
        local_sources.append(("organizers", Path(organizers)))
    for source_id, mapping in local_sources:
        source = CsvSource.from_config(mapping)
        frame = source.load(Path(raw_dir))
        if clean_ic50 and source_id == "organizers" and not chembl_frame.empty:
            frame, conflicts = _clean_organizer_ic50(frame, chembl_frame)
            if not conflicts.empty:
                conflict_parts.append(conflicts)
                print(f"{source_id}: flagged {len(conflicts)} censored IC50 rows")
        endpoint = _endpoint_rows(frame, source_id, scope="all")
        endpoint["is_vacv"] = True
        si = _si_rows(frame, source_id)
        si["is_vacv"] = True
        endpoint_parts.append(endpoint)
        if not si.empty:
            si_parts.append(si)
        print(f"{source_id}: {len(endpoint)} endpoint rows, {len(si)} SI rows")

    pubchem = _read_pubchem(Path(raw_dir), pubchem_sets)
    if not pubchem.empty:
        endpoint = _endpoint_rows(pubchem, "pubchem")
        si = _si_rows(pubchem, "pubchem")
        endpoint_parts.append(endpoint)
        if not si.empty:
            si_parts.append(si)
        print(f"pubchem: {len(endpoint)} endpoint rows, {len(si)} SI rows")

    if not chembl_frame.empty:
        endpoint_parts.append(chembl_frame)
        print(f"chembl: {len(chembl_frame)} endpoint rows")

    endpoints = (
        pd.concat(endpoint_parts, ignore_index=True)
        if endpoint_parts
        else pd.DataFrame(columns=COMMON_COLUMNS)
    )
    if excluded and not endpoints.empty:
        endpoints = endpoints[~endpoints["smiles_key"].isin(excluded)].copy()

    si_rows = (
        pd.concat(si_parts, ignore_index=True)
        if si_parts
        else pd.DataFrame(columns=COMMON_COLUMNS)
    )
    if excluded and not si_rows.empty:
        si_rows = si_rows[~si_rows["smiles_key"].isin(excluded)].copy()

    measurements = pd.concat([endpoints, si_rows], ignore_index=True)
    for column in COMMON_COLUMNS:
        if column not in measurements:
            measurements[column] = ""

    dataset, targets, dropped = assemble(measurements, conflict_parts)
    return dataset, measurements, targets, dropped


def assemble(
    measurements: pd.DataFrame,
    conflict_parts: list[pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build wide SI rows and endpoint targets from measurement-level rows."""
    pair_frame = measurements.copy()
    pair_frame["type"] = pair_frame["endpoint"]
    pair_frame["title"] = pair_frame["Title"]
    pair_frame["smiles"] = pair_frame["SMILES"]
    pair_frame["organism"] = pair_frame["organism"].where(
        pair_frame["organism"].fillna("").astype(str).str.strip().ne(""),
        pair_frame["target_organism"],
    )
    dataset, unpaired = build_dataset(pair_frame)
    valid, problems = validate_dataset(dataset)
    unpaired = unpaired.assign(stage="pair")
    problems = problems.rename(
        columns={"Title": "source_id", "reason": "reason"}
    ).assign(stage="validate")
    parsed = measurements[
        measurements["dropped_reason"].fillna("").astype(str).ne("")
    ].copy()
    parsed = parsed.rename(columns={"Title": "source_id"}).assign(stage="parse")
    quality = [
        item.assign(stage="quality")
        for item in (conflict_parts or [])
        if not item.empty
    ]
    dropped_frames = [
        item.reset_index(drop=True).loc[:, ~item.columns.duplicated()]
        for item in [unpaired, problems, parsed, *quality]
    ]
    endpoints = measurements[measurements["endpoint"].isin(["ic50", "cc50"])]
    return (
        valid,
        build_targets(endpoints),
        pd.concat(dropped_frames, ignore_index=True, sort=False),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.build_all",
        description="Combine organizers, ChEMBL and PubChem into one dataset.",
    )
    parser.add_argument("--organizers", type=Path, default=DEFAULT_ORGANIZERS)
    parser.add_argument("--chembl", type=Path, default=DEFAULT_CHEMBL)
    parser.add_argument("--pubchem-set", action="append", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--measurements", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--ic50-targets", type=Path, default=DEFAULT_IC50_TARGETS)
    parser.add_argument("--cc50-targets", type=Path, default=DEFAULT_CC50_TARGETS)
    parser.add_argument("--dropped", type=Path, default=DEFAULT_DROPPED)
    parser.add_argument(
        "--exclude-smiles",
        action="append",
        type=Path,
        default=None,
        metavar="CSV",
    )
    parser.add_argument(
        "--clean-ic50",
        action="store_true",
        help="drop censored/conflicting organizer IC50 labels (experimental)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="append new parsed rows to existing measurement/target files",
    )
    parser.add_argument(
        "--allow-local-only",
        action="store_true",
        help="allow overwriting a larger dataset_all with local-only sources",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exclude = (
        tuple(args.exclude_smiles)
        if args.exclude_smiles is not None
        else (Path("data/organizers/test.csv"),)
    )
    pubchem_sets = tuple(args.pubchem_set or DEFAULT_PCHEM_SETS)
    dataset, measurements, targets, dropped = build(
        args.organizers,
        args.chembl,
        pubchem_sets,
        Path("data/raw"),
        exclude,
        clean_ic50=args.clean_ic50,
    )
    if args.append and args.measurements.exists():
        old = pd.read_csv(args.measurements)
        measurements = pd.concat([old, measurements], ignore_index=True)
        measurements = measurements.drop_duplicates().reset_index(drop=True)
        dataset, targets, dropped = assemble(measurements)
        print(f"append: {len(measurements)} measurements after merge")

    external_sources = {"chembl", "pubchem"}
    has_external = measurements["source"].isin(external_sources).any()
    if not has_external and not args.allow_local_only and args.out.exists():
        old_rows = len(pd.read_csv(args.out))
        if len(dataset) < old_rows:
            print(
                f"ERROR: refusing to overwrite {args.out} ({old_rows} rows) "
                f"with local-only data ({len(dataset)} rows). "
                "Download ChEMBL/PubChem first or pass --allow-local-only.",
                file=sys.stderr,
            )
            return 1

    ic50 = targets[targets["endpoint"] == "ic50"].copy()
    cc50 = targets[targets["endpoint"] == "cc50"].copy()
    for path, frame in (
        (args.out, dataset),
        (args.measurements, measurements),
        (args.targets, targets),
        (args.ic50_targets, ic50),
        (args.cc50_targets, cc50),
        (args.dropped, dropped),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    print(f"{args.out}: {len(dataset)} SI/pair rows")
    print(f"{args.measurements}: {len(measurements)} measurements")
    print(f"{args.targets}: {len(targets)} endpoint targets")
    print(f"{args.ic50_targets}: {len(ic50)} IC50 targets")
    print(f"{args.cc50_targets}: {len(cc50)} CC50 targets")
    print(f"dropped/unpaired: {len(dropped)}")
    if not targets.empty:
        print(targets.groupby(["source", "endpoint"]).size().to_string())
    return 0 if not dataset.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
