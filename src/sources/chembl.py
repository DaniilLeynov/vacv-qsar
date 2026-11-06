from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd

from ..normalize.common import (
    UnitError,
    molecular_weight,
    normalize_relation,
    to_um,
)
from .base import MEASUREMENT_COLUMNS, SourceError

DEFAULT_DB = Path(os.environ.get("CHEMBL_SQLITE", "data/raw/chembl/chembl.db"))

SQL = """
WITH vacv_assays AS (
    SELECT assay_id, cell_id, assay_organism
    FROM assays
    WHERE assay_type = 'F' AND lower(description) LIKE ?
),
potency AS (
    SELECT act.molregno AS molregno,
           act.standard_type AS measure,
           act.standard_value AS value,
           act.standard_units AS units,
           COALESCE(act.standard_relation, '=') AS relation,
           cd.cell_name AS cell_line,
           v.assay_organism AS organism,
           act.doc_id AS study_id,
           d.doi AS doi
    FROM activities act
    JOIN vacv_assays v ON v.assay_id = act.assay_id
    LEFT JOIN cell_dictionary cd ON cd.cell_id = v.cell_id
    LEFT JOIN docs d ON d.doc_id = act.doc_id
    WHERE act.standard_type IN ({ic50})
      AND act.standard_value IS NOT NULL
),
cytotox AS (
    SELECT act.molregno AS molregno,
           act.standard_type AS measure,
           act.standard_value AS value,
           act.standard_units AS units,
           COALESCE(act.standard_relation, '=') AS relation,
           cd.cell_name AS cell_line,
           s.assay_organism AS organism,
           act.doc_id AS study_id,
           d.doi AS doi
    FROM activities act
    LEFT JOIN assays s ON s.assay_id = act.assay_id
    LEFT JOIN cell_dictionary cd ON cd.cell_id = s.cell_id
    LEFT JOIN docs d ON d.doc_id = act.doc_id
    WHERE act.standard_type IN ({cc50})
      AND act.standard_value IS NOT NULL
      AND act.molregno IN (SELECT molregno FROM potency)
)
SELECT p.molregno,
       p.measure,
       p.value,
       p.units,
       p.relation,
       p.cell_line,
       p.organism,
       p.study_id,
       p.doi,
       md.chembl_id,
       cs.canonical_smiles
FROM (SELECT * FROM potency UNION ALL SELECT * FROM cytotox) p
JOIN molecule_dictionary md ON md.molregno = p.molregno
JOIN compound_structures cs ON cs.molregno = p.molregno
"""


class ChemblSource:
    id = "chembl"

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB,
        ic50_measures: tuple[str, ...] = ("IC50",),
        cc50_measures: tuple[str, ...] = ("CC50",),
    ):
        self.db_path = Path(db_path)
        self.ic50_measures = tuple(ic50_measures)
        self.cc50_measures = tuple(cc50_measures)

    @classmethod
    def from_download(cls, **kwargs) -> "ChemblSource":
        try:
            import chembl_downloader
        except ImportError as exc:
            raise SourceError(
                "chembl-downloader is not installed; run "
                "`pip install chembl-downloader`"
            ) from exc
        db_path = Path(chembl_downloader.download_extract_sqlite())
        return cls(db_path, **kwargs)

    def fetch_raw(self, dest: Path, refresh: bool = False) -> Path:
        if not self.db_path.exists():
            raise SourceError(
                f"ChEMBL SQLite dump not found at {self.db_path}. "
                "Run `python -m chembl_downloader download -v`, then pass the "
                "printed .db path via --chembl-db or CHEMBL_SQLITE."
            )
        return self.db_path

    def _query(self) -> list[tuple]:
        ic50 = ",".join("?" for _ in self.ic50_measures)
        cc50 = ",".join("?" for _ in self.cc50_measures)
        sql = SQL.format(ic50=ic50, cc50=cc50)
        params = ["%vaccinia%", *self.ic50_measures, *self.cc50_measures]
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(sql, params)
            return cursor.fetchall()

    def load(self, raw_dir: Path) -> pd.DataFrame:
        rows: list[dict] = []
        for (
            _molregno,
            measure,
            value,
            units,
            relation,
            cell_line,
            organism,
            study_id,
            doi,
            source_id,
            smiles,
        ) in self._query():
            measure = str(measure or "")
            if measure in self.ic50_measures:
                kind = "ic50"
            elif measure in self.cc50_measures:
                kind = "cc50"
            else:
                continue

            row = {
                "source": self.id,
                "source_id": source_id,
                "title": source_id,
                "smiles": smiles,
                "type": kind,
                "measure": measure,
                "value": value,
                "units": units,
                "value_um": None,
                "relation": "=",
                "cell_line": cell_line or "",
                "organism": organism or "",
                "study_id": str(study_id) if study_id is not None else "",
                "doi": doi or "",
                "dropped_reason": "",
            }

            try:
                row["relation"] = normalize_relation(relation)
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                row["dropped_reason"] = f"invalid value: {exc}"
                rows.append(row)
                continue

            weight = molecular_weight(smiles)
            try:
                row["value_um"] = to_um(numeric, units, weight)
            except UnitError as exc:
                row["dropped_reason"] = str(exc)
            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=MEASUREMENT_COLUMNS)
        return pd.DataFrame(rows, columns=MEASUREMENT_COLUMNS)
