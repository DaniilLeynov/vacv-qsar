from __future__ import annotations

import csv
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from ..normalize.common import UnitError, molecular_weight, normalize_relation, to_um
from .base import MEASUREMENT_COLUMNS, SourceError

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
DEFAULT_TERM = "vaccinia AND IC50"
DOI_RE = re.compile(r"DOI:\s*([^,\s]+)", re.IGNORECASE)

CELL_LINES = [
    "vero",
    "hela",
    "a549",
    "a-549",
    "hff",
    "huh7",
    "bhk",
    "cv-1",
    "bsc-1",
    "e6sm",
    "hel",
    "prk",
    "mrc-5",
    "caco-2",
    "hek293",
    "mdck",
    "llc-mk2",
    "mdbk",
]

POTENCY_MEASURES = {"IC50", "EC50"}
CYTOTOXICITY_MEASURES = {"CC50", "TC50"}
# Direct selectivity-index measures are already ratios (dimensionless); their
# Standard Value is the SI itself, not a concentration.
SELECTIVITY_MEASURES = {
    "Ratio CC50/IC50",
    "Selectivity Index",
    "Selectivity index",
    "Therapeutic index",
    "Therapeutic Index",
    "TI",
    "TI50",
}


class PubchemSource:
    id = "pubchem"

    def __init__(self, term: str = DEFAULT_TERM, delay: float = 0.2, name: str = "pubchem"):
        self.term = term
        self.delay = delay
        self.name = name

    def _get(self, url: str, timeout: int = 30) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": "vacv-qsar/0.1"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    def _aids(self) -> list[int]:
        query = urllib.parse.urlencode(
            {"db": "pcassay", "term": self.term, "retmax": "1000", "retmode": "json"}
        )
        payload = json.loads(self._get(f"{EUTILS}?{query}"))
        return [int(value) for value in payload["esearchresult"]["idlist"]]

    def fetch_raw(self, dest: Path, refresh: bool = False) -> Path:
        out = Path(dest) / self.name
        out.mkdir(parents=True, exist_ok=True)

        aids = self._aids()
        (out / "aids.json").write_text(json.dumps(aids), encoding="utf-8")

        for aid in aids:
            path = out / f"aid_{aid}.csv"
            if path.exists() and not refresh:
                continue
            try:
                path.write_bytes(self._get(f"{PUG}/assay/aid/{aid}/CSV"))
            except Exception as exc:  # keep going on a single failed assay
                (out / f"aid_{aid}.error").write_text(str(exc), encoding="utf-8")
            time.sleep(self.delay)

        summaries = out / "summaries.json"
        if refresh or not summaries.exists():
            collected: dict[str, dict] = {}
            for start in range(0, len(aids), 50):
                batch = aids[start : start + 50]
                url = f"{PUG}/assay/aid/{','.join(map(str, batch))}/summary/JSON"
                try:
                    payload = json.loads(self._get(url))
                    for item in payload.get("AssaySummaries", {}).get(
                        "AssaySummary", []
                    ):
                        collected[str(item["AID"])] = item
                except Exception:
                    pass
                time.sleep(self.delay)
            summaries.write_text(json.dumps(collected), encoding="utf-8")

        return out

    def load(self, raw_dir: Path) -> pd.DataFrame:
        out = Path(raw_dir) / self.name
        if not out.exists():
            raise SourceError(f"PubChem raw data not found: {out}")

        summaries: dict[str, dict] = {}
        summaries_path = out / "summaries.json"
        if summaries_path.exists():
            summaries = json.loads(summaries_path.read_text(encoding="utf-8"))

        rows: list[dict] = []
        for path in sorted(out.glob("aid_*.csv")):
            aid = int(path.stem.split("_")[1])
            summary = summaries.get(str(aid), {})

            doi = ""
            for comment in summary.get("Comment", []) or []:
                match = DOI_RE.search(str(comment))
                if match:
                    doi = match.group(1)
                    break

            name = summary.get("Name", "")
            if isinstance(name, list):
                name = " ".join(str(item) for item in name)
            description = " ".join(str(item) for item in summary.get("Description", []) or [])
            text = f"{name} {description}".lower()
            cell_line = next((cell for cell in CELL_LINES if cell in text), "")

            organism = ""
            targets = summary.get("Target") or []
            if isinstance(targets, list) and targets:
                organism = str(targets[0].get("Name", ""))

            study_id = doi or f"aid:{aid}"

            with path.open(newline="", encoding="utf-8", errors="replace") as handle:
                for record in csv.DictReader(handle):
                    cid = record.get("PUBCHEM_CID") or ""
                    if not cid:
                        continue

                    measure = (record.get("Standard Type") or "").strip()
                    if measure in POTENCY_MEASURES:
                        kind = "ic50"
                    elif measure in CYTOTOXICITY_MEASURES:
                        kind = "cc50"
                    elif measure in SELECTIVITY_MEASURES:
                        kind = "si"
                    else:
                        continue

                    standardized = (
                        record.get("Standard Value")
                        or record.get("PubChem Standard Value")
                        or ""
                    )

                    entry = {
                        "source": self.id,
                        "source_id": cid,
                        "title": cid,
                        "smiles": record.get("PUBCHEM_EXT_DATASOURCE_SMILES") or "",
                        "type": kind,
                        "measure": measure,
                        "value": standardized,
                        "units": record.get("Standard Units") or "",
                        "value_um": None,
                        "relation": "=",
                        "cell_line": cell_line,
                        "organism": organism,
                        "study_id": study_id,
                        "doi": doi,
                        "dropped_reason": "",
                    }

                    try:
                        entry["relation"] = normalize_relation(
                            record.get("Standard Relation") or "="
                        )
                    except ValueError:
                        entry["dropped_reason"] = "invalid relation"

                    if standardized == "":
                        entry["dropped_reason"] = (
                            entry["dropped_reason"] or "missing standardized value"
                        )
                    else:
                        try:
                            if kind == "si":
                                entry["value_um"] = float(standardized)
                            else:
                                entry["value_um"] = to_um(
                                    float(standardized),
                                    entry["units"],
                                    molecular_weight(entry["smiles"]),
                                )
                        except (ValueError, UnitError) as exc:
                            entry["dropped_reason"] = (
                                entry["dropped_reason"] or str(exc)
                            )

                    rows.append(entry)

        return pd.DataFrame(rows, columns=MEASUREMENT_COLUMNS)
