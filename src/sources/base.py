from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

from ..normalize.common import selectivity_index

MEASUREMENT_COLUMNS = [
    "source",
    "source_id",
    "title",
    "smiles",
    "type",
    "measure",
    "value",
    "units",
    "value_um",
    "relation",
    "cell_line",
    "organism",
    "doi",
    "dropped_reason",
]

# Canonical format: identical to the organizers' train.csv plus provenance.
UNPAIRED_COLUMNS = [
    "source",
    "source_id",
    "type",
    "cell_line",
    "doi",
    "reason",
]

DATASET_COLUMNS = [
    "Title",
    "CC50_uM",
    "IC50_uM_VACV",
    "SI VACV",
    "SMILES",
    "strain",
    "Culture",
    "doi",
    "source",
]

SI_COLUMN = "SI VACV"


class SourceError(Exception):
    pass


class SourceAdapter(Protocol):
    id: str

    def fetch_raw(self, dest: Path, refresh: bool = False) -> Path | None:
        ...

    def load(self, raw_dir: Path) -> pd.DataFrame:
        ...


def empty_measurements() -> pd.DataFrame:
    return pd.DataFrame(columns=MEASUREMENT_COLUMNS)


def _first_nonempty(series: pd.Series) -> str:
    for value in series:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _aggregate(chunk: pd.DataFrame) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for cell, group in chunk.groupby("cell_key", sort=False):
        values = pd.to_numeric(group["value_um"], errors="coerce").dropna()
        if values.empty:
            continue
        relations = [
            str(item) if isinstance(item, str) and item else "="
            for item in group["relation"]
        ]
        relation = relations[0] if len(set(relations)) == 1 else "~"
        result[str(cell)] = {
            "value_um": float(values.median()),
            "relation": relation,
            "smiles": _first_nonempty(group["smiles"]),
            "title": _first_nonempty(group["title"]),
            "cell_line": _first_nonempty(group["cell_line"]),
            "organism": _first_nonempty(group["organism"]),
            "doi": _first_nonempty(group["doi"]),
        }
    return result


def _cell_key(frame: pd.DataFrame) -> pd.Series:
    return frame["cell_line"].fillna("").astype(str).str.strip().str.lower()


def build_dataset(
    measurements: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if measurements.empty:
        return (
            pd.DataFrame(columns=DATASET_COLUMNS),
            pd.DataFrame(columns=UNPAIRED_COLUMNS),
        )

    frame = measurements.copy()
    for column in MEASUREMENT_COLUMNS:
        if column not in frame:
            frame[column] = ""

    dropped = frame["dropped_reason"].fillna("").astype(str) != ""
    frame = frame[~dropped]
    frame["value_um"] = pd.to_numeric(frame["value_um"], errors="coerce")
    positive = frame["value_um"] > 0
    non_negative_si = (frame["type"] == "si") & (frame["value_um"] >= 0)
    frame = frame[frame["value_um"].notna() & (positive | non_negative_si)]
    frame["cell_key"] = _cell_key(frame)

    rows: list[dict] = []
    unpaired: list[dict] = []
    for (source, source_id), group in frame.groupby(
        ["source", "source_id"], sort=False
    ):
        ic50_map = _aggregate(group[group["type"] == "ic50"])
        cc50_map = _aggregate(group[group["type"] == "cc50"])
        si_map = _aggregate(group[group["type"] == "si"])

        for cell in sorted(set(ic50_map) | set(cc50_map) | set(si_map)):
            ic50 = ic50_map.get(cell)
            cc50 = cc50_map.get(cell)
            si_record = si_map.get(cell)

            if si_record is not None:
                si_value = si_record["value_um"]
            elif ic50 is not None and cc50 is not None:
                si_value = selectivity_index(ic50["value_um"], cc50["value_um"])
            else:
                present = ic50 or cc50
                missing = "CC50" if ic50 is not None else "IC50"
                unpaired.append(
                    {
                        "source": source,
                        "source_id": source_id,
                        "type": "ic50" if ic50 is not None else "cc50",
                        "cell_line": present["cell_line"],
                        "doi": present["doi"],
                        "reason": f"no matching {missing} on the same cell line",
                    }
                )
                continue
            if si_value is None:
                continue

            records = [item for item in (si_record, ic50, cc50) if item]
            rows.append(
                {
                    "Title": _first_of(records, "title") or source_id,
                    "CC50_uM": cc50["value_um"] if cc50 else None,
                    "IC50_uM_VACV": ic50["value_um"] if ic50 else None,
                    SI_COLUMN: si_value,
                    "SMILES": _first_of(records, "smiles"),
                    "strain": _first_of(records, "organism"),
                    "Culture": _first_of(records, "cell_line"),
                    "doi": _first_of(records, "doi"),
                    "source": source,
                }
            )

    dataset = pd.DataFrame(rows, columns=DATASET_COLUMNS)
    return dataset, pd.DataFrame(unpaired, columns=UNPAIRED_COLUMNS)


def _first_of(records: list[dict], key: str) -> str:
    for record in records:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def validate_dataset(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    problems: list[dict] = []
    if dataset.empty:
        return dataset, pd.DataFrame(columns=["Title", "reason"])

    frame = dataset.copy()
    for column in ("CC50_uM", "IC50_uM_VACV", SI_COLUMN):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    bad_rows: set = set()
    for index, row in frame.iterrows():
        title = row.get("Title", index)
        smiles = row.get("SMILES")
        if not isinstance(smiles, str) or not smiles.strip():
            problems.append({"Title": title, "reason": "missing SMILES"})
            bad_rows.add(index)
            continue
        if pd.isna(row[SI_COLUMN]) or row[SI_COLUMN] < 0:
            problems.append({"Title": title, "reason": "SI must be non-negative"})
            bad_rows.add(index)
            continue
        if pd.notna(row["CC50_uM"]) and pd.notna(row["IC50_uM_VACV"]):
            if row["CC50_uM"] <= 0 or row["IC50_uM_VACV"] <= 0:
                problems.append(
                    {"Title": title, "reason": "concentrations must be positive"}
                )
                bad_rows.add(index)
                continue
            expected = row["CC50_uM"] / row["IC50_uM_VACV"]
            tolerance = max(1e-3, abs(expected) * 1e-3)
            if abs(expected - row[SI_COLUMN]) > tolerance:
                problems.append(
                    {"Title": title, "reason": "SI does not equal CC50/IC50"}
                )
                bad_rows.add(index)

    valid = frame[~frame.index.isin(bad_rows)].copy()
    return valid, pd.DataFrame(problems, columns=["Title", "reason"])


def write_dataset(dataset: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(path, index=False)


def write_problems(problems: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    problems.to_csv(path, index=False)
