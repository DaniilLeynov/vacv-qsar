from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from ..normalize.common import (
    UnitError,
    molecular_weight,
    normalize_relation,
    parse_value,
    to_um,
)
from .base import MEASUREMENT_COLUMNS, SourceError


@dataclass
class CsvMapping:
    id: str
    file: Path
    smiles: str
    ic50: str | None = None
    cc50: str | None = None
    target: str | None = None
    units: str = "uM"
    ic50_units: str | None = None
    cc50_units: str | None = None
    doi: str | None = None
    title: str | None = None
    cell_line: str | None = None
    organism: str | None = None
    relation: str | None = None

    @classmethod
    def from_config(cls, path: str | Path) -> "CsvMapping":
        path = Path(path)
        if not path.exists():
            raise SourceError(f"mapping file not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if "id" not in data or "file" not in data or "smiles" not in data:
            raise SourceError(f"{path}: required fields are id, file, smiles")
        has_pair = "ic50" in data and "cc50" in data
        if "target" not in data and not has_pair:
            raise SourceError(
                f"{path}: provide either target or both ic50 and cc50 columns"
            )
        return cls(
            id=str(data["id"]),
            file=Path(data["file"]),
            smiles=str(data["smiles"]),
            ic50=str(data["ic50"]) if "ic50" in data else None,
            cc50=str(data["cc50"]) if "cc50" in data else None,
            target=str(data["target"]) if "target" in data else None,
            units=str(data.get("units", "uM")),
            ic50_units=data.get("ic50_units"),
            cc50_units=data.get("cc50_units"),
            doi=data.get("doi"),
            title=data.get("title"),
            cell_line=data.get("cell_line"),
            organism=data.get("organism"),
            relation=data.get("relation"),
        )


class CsvSource:
    def __init__(self, mapping: CsvMapping):
        self.mapping = mapping
        self.id = mapping.id

    @classmethod
    def from_config(cls, path: str | Path) -> "CsvSource":
        return cls(CsvMapping.from_config(path))

    def fetch_raw(self, dest: Path, refresh: bool = False) -> Path:
        if not self.mapping.file.exists():
            raise SourceError(f"{self.id}: input file not found: {self.mapping.file}")
        return self.mapping.file

    def _value(self, row: pd.Series, column: str | None) -> str:
        if not column:
            return ""
        value = row.get(column)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        text = str(value).strip()
        if text.lower() in {"nan", "none"}:
            return ""
        return text

    def _base_entry(self, row, index, title, source_id, kind) -> dict:
        return {
            "source": self.id,
            "source_id": source_id,
            "title": title,
            "smiles": self._value(row, self.mapping.smiles),
            "type": kind,
            "measure": kind.upper(),
            "value": None,
            "units": self.mapping.units,
            "value_um": None,
            "relation": "=",
            "cell_line": self._value(row, self.mapping.cell_line),
            "organism": self._value(row, self.mapping.organism),
            "doi": self._value(row, self.mapping.doi),
            "dropped_reason": "",
        }

    def _parse(
        self, entry: dict, row, text: str, units: str | None, convert: bool
    ) -> None:
        if not text:
            entry["dropped_reason"] = "missing value"
            return
        try:
            relation, numeric = parse_value(text)
            if self.mapping.relation:
                relation = normalize_relation(
                    self._value(row, self.mapping.relation)
                )
            entry["relation"] = relation
            entry["value"] = numeric
            if convert:
                entry["units"] = units or self.mapping.units
                entry["value_um"] = to_um(
                    numeric,
                    entry["units"],
                    molecular_weight(entry["smiles"]),
                )
            else:
                entry["units"] = ""
                entry["value_um"] = numeric
        except (ValueError, UnitError) as exc:
            entry["dropped_reason"] = str(exc)

    def load(self, raw_dir: Path) -> pd.DataFrame:
        frame = pd.read_csv(self.mapping.file)
        rows: list[dict] = []

        for index, row in frame.iterrows():
            title = self._value(row, self.mapping.title)
            source_id = f"{title or 'row'}#{index}"

            if self.mapping.target:
                text = self._value(row, self.mapping.target)
                entry = self._base_entry(row, index, title, source_id, "si")
                self._parse(entry, row, text, None, convert=False)
                rows.append(entry)

            for kind, column, default_units in (
                ("ic50", self.mapping.ic50, self.mapping.ic50_units),
                ("cc50", self.mapping.cc50, self.mapping.cc50_units),
            ):
                if not column:
                    continue
                text = self._value(row, column)
                entry = self._base_entry(row, index, title, source_id, kind)
                self._parse(entry, row, text, default_units, convert=True)
                rows.append(entry)

        if not rows:
            return pd.DataFrame(columns=MEASUREMENT_COLUMNS)
        return pd.DataFrame(rows, columns=MEASUREMENT_COLUMNS)
