from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

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
    "status",
    "note",
]


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in COLUMNS:
        if column not in frame:
            frame[column] = ""
    frame["status"] = (
        frame["SMILES"].fillna("").astype(str).str.strip().ne("").map(
            {True: "ready", False: "needs_smiles"}
        )
    )
    return frame[COLUMNS]


def build_registry(
    organizers: Path,
    chembl: Path | None,
    papers: list[Path],
    out: Path,
) -> pd.DataFrame:
    parts = []
    if organizers.exists():
        parts.append(_read(organizers))
    if chembl is not None and chembl.exists():
        parts.append(_read(chembl))
    for paper in papers:
        if paper is not None and Path(paper).exists():
            parts.append(_read(paper))
    if not parts:
        raise FileNotFoundError("no input tables found")

    frame = pd.concat(parts, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.registry",
        description="Merge every found source into one common table.",
    )
    parser.add_argument("--organizers", type=Path, default=Path("data/dataset.csv"))
    parser.add_argument("--chembl", type=Path, default=Path("data/dataset_chembl.csv"))
    parser.add_argument(
        "--paper",
        type=Path,
        action="append",
        default=None,
        help="extra paper table; repeatable",
    )
    parser.add_argument("--out", type=Path, default=Path("data/found.csv"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    papers = args.paper or [
        Path("data/paper_cidofovir.csv"),
        Path("data/paper_bmcl2022.csv"),
    ]
    frame = build_registry(args.organizers, args.chembl, papers, args.out)

    print(f"{args.out}: {len(frame)} rows")
    print(frame.groupby(["source", "status"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
