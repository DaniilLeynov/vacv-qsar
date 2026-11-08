"""Run the colleague's original RobustForest model.

The algorithm is **not** reimplemented here: this module imports
``src/rf_robust_best.py`` and uses its ``RobustForest`` class unchanged.
It only supplies the missing test-feature preparation (the original archive
imported ``model_training.rf_robust.run.data``) and a small CLI for
training, OOF metrics and prediction.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.model_selection import GroupKFold

from .normalize.common import standardize_smiles_clean

from .rf_robust_best import (
    DESCRIPTOR_NAMES,
    DESCRIPTORS,
    GROUP,
    HALOGENS,
    MORGAN,
    POSITION_COLUMNS,
    TARGET,
    TYPES,
    FIELDS,
    RobustForest,
    SOURCE,
    neighbors,
    recover,
)

DEFAULT_TRAIN = Path("data/rf_robust/train_407.csv")
DEFAULT_MEASUREMENTS = Path("data/dual/all_measurements.csv")
DEFAULT_OUT = Path("runs/rfrobust")
DEFAULT_TEST = Path("data/organizers/test.csv")

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048, includeChirality=True
)


def _pool_context():
    if sys.platform.startswith("linux"):
        return multiprocessing.get_context("fork")
    return None


def substituent(atom) -> str:
    atomic_number = atom.GetAtomicNum()
    if atomic_number in HALOGENS:
        return "halo"
    if atomic_number == 6:
        if any(
            bond.GetBondType() == Chem.BondType.DOUBLE
            and bond.GetOtherAtom(atom).GetAtomicNum() == 8
            for bond in atom.GetBonds()
        ):
            return "acyl"
        if sum(neighbor.GetAtomicNum() in HALOGENS for neighbor in atom.GetNeighbors()) >= 2:
            return "halo"
        if not atom.GetIsAromatic():
            return "alkyl"
    if atomic_number == 7:
        return "amine"
    if atomic_number == 8:
        return "oxygen"
    return "other"


def group_type(atom) -> str:
    kind = substituent(atom)
    if kind == "amine" and atom.GetFormalCharge() > 0:
        if any(neighbor.GetAtomicNum() == 8 for neighbor in atom.GetNeighbors()):
            return "nitro"
    if kind == "alkyl" and any(
        bond.GetBondType() == Chem.BondType.TRIPLE
        and bond.GetOtherAtom(atom).GetAtomicNum() == 7
        for bond in atom.GetBonds()
    ):
        return "cyano"
    return kind


def aromatic_context(smiles: str) -> dict:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    features: Counter = Counter()
    for ring in molecule.GetRingInfo().AtomRings():
        if len(ring) != 6:
            continue
        atoms = [molecule.GetAtomWithIdx(i) for i in ring]
        if not all(atom.GetIsAromatic() for atom in atoms):
            continue
        ring_kind = f"ring_N{sum(atom.GetAtomicNum() == 7 for atom in atoms)}"
        outside = [
            (j, neighbor, group_type(neighbor))
            for j, atom in enumerate(atoms)
            for neighbor in atom.GetNeighbors()
            if neighbor.GetIdx() not in ring
        ]
        for j, anchor, kind in outside:
            if kind != "acyl":
                continue
            for bond in anchor.GetBonds():
                linker = bond.GetOtherAtom(anchor)
                if bond.GetBondType() != Chem.BondType.SINGLE or linker.GetIdx() in ring:
                    continue
                element = linker.GetSymbol()
                features[f"acyl_{element}"] += 1
                roles = [element]
                if element == "N":
                    attached = [
                        a for a in linker.GetNeighbors() if a.GetIdx() != anchor.GetIdx()
                    ]
                    roles += [
                        f"N_{a.GetSymbol()}_r{int(a.IsInRing())}_d{a.GetDegree()}"
                        for a in attached
                    ]
                for k, _, other in outside:
                    distance = min(abs(j - k), len(ring) - abs(j - k))
                    if not distance:
                        continue
                    groups = [other]
                    if other in ("halo", "nitro", "cyano", "acyl"):
                        groups.append("electron_withdrawing")
                    for group in groups:
                        features[f"acyl_{group}_{distance}"] += 1
                        features[f"{ring_kind}_acyl_{group}_{distance}"] += 1
                        for role in roles:
                            features[f"acyl_{role}_{group}_{distance}"] += 1
    return dict(features)


def position_features(smiles: str) -> dict:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return {}
    features: Counter = Counter()
    ring_nitrogens = 0
    acyl_halo_nonadjacent = 0
    order = {kind: index for index, kind in enumerate(TYPES)}
    for ring in molecule.GetRingInfo().AtomRings():
        if len(ring) != 6:
            continue
        atoms = [molecule.GetAtomWithIdx(i) for i in ring]
        if not all(atom.GetIsAromatic() for atom in atoms):
            continue
        ring_nitrogens += sum(atom.GetAtomicNum() == 7 for atom in atoms)
        substituents: list[tuple[int, str]] = []
        for position, atom in enumerate(atoms):
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() not in ring:
                    kind = group_type(neighbor)
                    if kind not in order:
                        kind = "other"
                    substituents.append((position, kind))
        for first in range(len(substituents)):
            for second in range(first + 1, len(substituents)):
                i, kind_i = substituents[first]
                j, kind_j = substituents[second]
                distance = min(abs(i - j), 6 - abs(i - j))
                if distance == 0 or distance > 3:
                    continue
                if order[kind_i] <= order[kind_j]:
                    key = f"aromatic_{kind_i}_{kind_j}_distance{distance}"
                else:
                    key = f"aromatic_{kind_j}_{kind_i}_distance{distance}"
                features[key] += 1
                if {kind_i, kind_j} == {"acyl", "halo"} and distance > 1:
                    acyl_halo_nonadjacent += 1
    features["aromatic_ring_nitrogens"] = ring_nitrogens
    features["aromatic_acyl_halo_nonadjacent"] = acyl_halo_nonadjacent
    return dict(features)


def clean_culture(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return "Unknown_Culture"
    text = value.strip().lower().replace("-", " ")
    text = " ".join(text.replace("cells", " ").replace("cell", " ").split())
    mapping = {
        "vero": "Vero",
        "vero 76": "Vero 76",
        "human foreskin fibroblast": "HFF",
        "hff": "HFF",
        "hela": "HeLa",
        "hel": "HEL",
        "prk": "PRK",
        "lung fibroblast": "Lung fibroblasts",
        "lung fibroblasts": "Lung fibroblasts",
        "bsc 40": "BSC-40",
        "bsc40": "BSC-40",
        "bsc 1": "BSC-1",
        "bsc1": "BSC-1",
        "mt 4": "MT-4",
        "mt4": "MT-4",
        "cem": "CEM",
        "a549": "A549",
        "bhk 21": "BHK-21",
        "e6sm": "E6SM",
        "cv 1": "CV-1",
    }
    return mapping.get(text, "Unknown_Culture")


def clean_strain(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    text = value.lower()
    if "copenhagen" in text:
        return "Copenhagen"
    if "western reserve" in text or " wr" in text:
        return "Western Reserve"
    if "tiantan" in text or "tian tan" in text:
        return "Tian Tan"
    if "ihd" in text:
        return "IHD"
    if "lederle" in text:
        return "Lederle"
    if "lister" in text:
        return "LIVP" if "livp" in text else "Lister"
    if "variola" in text or "cowpox" in text or "ectromelia" in text:
        return "Non_VACV"
    if "vaccinia" in text:
        return "Unknown_VACV"
    return "unknown"


def _descriptors(molecule) -> dict:
    values = {}
    for name in DESCRIPTOR_NAMES.split():
        if name == "FormalCharge":
            values[name] = Chem.GetFormalCharge(molecule)
        elif name == "AtomCountWithH":
            values[name] = Chem.AddHs(molecule).GetNumAtoms()
        elif name.startswith("Count_"):
            element = name.split("_", 1)[1]
            values[name] = sum(
                1 for atom in molecule.GetAtoms() if atom.GetSymbol() == element
            )
        else:
            values[name] = float(getattr(Descriptors, name)(molecule))
    return values


def build_source_records(path: Path = DEFAULT_MEASUREMENTS) -> pd.DataFrame:
    """Build SI+condition records from the combined measurement table."""
    columns = [GROUP, TARGET, "culture_clean", "strain_clean", "reference"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path)
    frame["dropped_reason"] = frame["dropped_reason"].fillna("").astype(str)
    frame["relation"] = frame["relation"].fillna("=").astype(str)
    frame["endpoint"] = frame["endpoint"].astype(str).str.lower()
    is_vacv = frame["is_vacv"].astype(str).str.lower().isin(["true", "1"])
    frame = frame[
        frame["endpoint"].eq("si")
        & frame["relation"].eq("=")
        & frame["dropped_reason"].eq("")
        & is_vacv
    ].copy()
    rows: list[dict] = []
    for _, row in frame.iterrows():
        smiles = row.get("SMILES", "")
        molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if molecule is None:
            continue
        value = pd.to_numeric(row.get("value_um"), errors="coerce")
        if pd.isna(value) or value <= 0:
            continue
        organism = row.get("organism", "")
        if not isinstance(organism, str) or not organism.strip():
            organism = row.get("target_organism", "")
        rows.append(
            {
                GROUP: Chem.MolToSmiles(molecule),
                TARGET: float(value),
                "culture_clean": clean_culture(row.get("cell_line", "")),
                "strain_clean": clean_strain(organism),
                "reference": f"raw:{row.get('source', '')}:{row.get('source_id', '')}",
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).drop_duplicates()


def _feature_row(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return np.zeros(2048, dtype=np.float32), {}, {}
    return (
        _MORGAN.GetFingerprintAsNumPy(molecule).astype(np.float32),
        _descriptors(molecule),
        position_features(smiles),
    )


def prepare_test_frame(frame: pd.DataFrame, jobs: int = 1) -> pd.DataFrame:
    """Build the columns expected by the original RobustForest from CSVs."""
    frame = frame.copy()
    frame["SMILES"] = frame["SMILES"].fillna("").astype(str).str.strip()
    canonical = []
    for smiles in frame["SMILES"]:
        molecule = Chem.MolFromSmiles(smiles)
        canonical.append(Chem.MolToSmiles(molecule) if molecule is not None else "")
    frame["canonical_smiles"] = canonical
    culture = frame["Culture"] if "Culture" in frame else pd.Series("", index=frame.index)
    strain = frame["strain"] if "strain" in frame else pd.Series("", index=frame.index)
    frame["culture_clean"] = culture.map(clean_culture)
    frame["strain_clean"] = strain.map(clean_strain)
    frame["culture_clean_BSC-40"] = frame["culture_clean"].eq("BSC-40")
    frame["official_train"] = False

    smiles = frame["SMILES"].tolist()
    if jobs > 1 and len(smiles) >= 200:
        chunksize = max(1, len(smiles) // (jobs * 4))
        with ProcessPoolExecutor(
            max_workers=jobs, mp_context=_pool_context()
        ) as pool:
            values = list(pool.map(_feature_row, smiles, chunksize=chunksize))
    else:
        values = [_feature_row(value) for value in smiles]
    morgan = [value[0] for value in values]
    descriptors = [value[1] for value in values]
    positions = [value[2] for value in values]

    morgan_frame = pd.DataFrame(np.vstack(morgan), columns=MORGAN, index=frame.index)
    descriptor_frame = pd.DataFrame(
        [
            {f"rdkit_{name}": item.get(name, 0.0) for name in DESCRIPTOR_NAMES.split()}
            for item in descriptors
        ],
        columns=DESCRIPTORS,
        index=frame.index,
    )
    position_frame = pd.DataFrame(
        [
            {column: item.get(column, 0) for column in POSITION_COLUMNS}
            for item in positions
        ],
        columns=POSITION_COLUMNS,
        index=frame.index,
    )
    generated = list(MORGAN) + list(DESCRIPTORS) + list(POSITION_COLUMNS)
    frame = frame.drop(columns=[column for column in generated if column in frame])
    return pd.concat(
        [frame, morgan_frame, descriptor_frame, position_frame], axis=1
    )


def clean_training_table(frame: pd.DataFrame, jobs: int = 1) -> pd.DataFrame:
    """Salt-strip SMILES and median-aggregate replicates per molecule/cell."""
    raw = pd.DataFrame(
        {
            "SMILES": frame["SMILES"].map(standardize_smiles_clean),
            "SI VACV": pd.to_numeric(frame.get("SI VACV"), errors="coerce"),
            "Culture": frame.get("culture_clean", ""),
            "strain": frame.get("strain_clean", ""),
            "Title": frame.get("Title", ""),
            "official_train": frame.get(SOURCE, True),
        }
    )
    raw = raw[raw["SMILES"].notna() & raw["SMILES"].ne("")].copy()
    raw = raw[raw["SI VACV"].notna()].copy()
    raw["_cell"] = raw["Culture"].fillna("unknown").astype(str)
    grouped = (
        raw.groupby(["SMILES", "_cell"], sort=False)
        .agg(
            SI=("SI VACV", "median"),
            official_train=("official_train", "max"),
            Culture=("Culture", "first"),
            strain=("strain", "first"),
            Title=("Title", "first"),
        )
        .reset_index()
    )
    grouped = grouped.rename(
        columns={"SMILES": "SMILES", "SI": "SI VACV"}
    )
    prepared = prepare_test_frame(
        grouped[["SMILES", "Culture", "strain", "Title"]], jobs=jobs
    )
    prepared[SOURCE] = grouped["official_train"].astype(bool).to_numpy()
    prepared[TARGET] = grouped["SI VACV"].to_numpy()
    return prepared


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), 0.0, None)
    mse = float(np.mean((pred - y) ** 2))
    mae = float(np.mean(np.abs(pred - y)))
    mse_log = float(np.mean((np.log1p(pred) - np.log1p(y)) ** 2))
    return {
        "n": int(len(y)),
        "mse_si": mse,
        "rmse_si": math.sqrt(mse),
        "mae_si": mae,
        "mse_log1p": mse_log,
        "rmse_log1p": math.sqrt(mse_log),
    }


def _fit_fold_worker(payload: tuple) -> np.ndarray:
    train_frame, valid_frame = payload
    model = RobustForest().fit(train_frame)
    return model.predict(valid_frame)


def cross_validate(
    frame: pd.DataFrame, folds: int = 5, jobs: int = 1
) -> tuple[dict, pd.DataFrame]:
    groups = frame[GROUP].to_numpy()
    oof = np.full(len(frame), np.nan)
    splits = list(GroupKFold(n_splits=folds).split(frame, groups=groups))
    payloads = [
        (frame.iloc[train_index], frame.iloc[valid_index])
        for train_index, valid_index in splits
    ]
    if jobs > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(
            max_workers=min(jobs, len(payloads)), mp_context=_pool_context()
        ) as pool:
            predictions = list(pool.map(_fit_fold_worker, payloads))
    else:
        predictions = [_fit_fold_worker(payload) for payload in payloads]
    for (_, valid_index), values in zip(splits, predictions):
        oof[valid_index] = values
    metrics = _metrics(frame[TARGET].to_numpy(), oof)
    output = frame[["Title", GROUP, TARGET]].copy()
    output["pred_si"] = oof
    return metrics, output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.rf_robust",
        description="Original RobustForest model from the colleague's archive.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    train.add_argument("--out", type=Path, default=DEFAULT_OUT)
    train.add_argument("--folds", type=int, default=5)
    train.add_argument(
        "--clean",
        action="store_true",
        help="salt-strip SMILES and median-aggregate replicates before training",
    )
    train.add_argument(
        "--jobs", type=int, default=0, help="parallel fold workers (0 = auto)"
    )
    train.add_argument(
        "--neighbor-imputation",
        action="store_true",
        help="also fill conditions by Morgan-similarity voting",
    )
    train.add_argument("--neighbor-threshold", type=float, default=0.7)
    train.add_argument("--neighbor-agreement", type=float, default=0.8)
    predict = sub.add_parser("predict")
    predict.add_argument("--model", type=Path, default=DEFAULT_OUT)
    predict.add_argument("--test", type=Path, default=DEFAULT_TEST)
    predict.add_argument("--out", type=Path, default=Path("submissions/submission.csv"))
    predict.add_argument(
        "--jobs", type=int, default=0, help="parallel feature workers (0 = auto)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        args.out.mkdir(parents=True, exist_ok=True)
        records = build_source_records()
        cpu = os.cpu_count() or 1
        if args.clean:
            frame = clean_training_table(
                pd.read_csv(args.train), jobs=args.jobs or cpu
            )
        else:
            frame = pd.read_csv(args.train)
            if TARGET not in frame:
                raise ValueError(f"{args.train}: missing {TARGET}")
        frame = frame.copy()
        if "source_row" not in frame:
            frame["source_row"] = np.arange(len(frame)) + 1
        restored, audit = recover(frame, records)
        restored["culture_clean_BSC-40"] = restored["culture_clean"].eq("BSC-40")
        if args.neighbor_imputation:
            donors = pd.concat(
                [records[[GROUP, *FIELDS]], restored[[GROUP, *FIELDS]]],
                ignore_index=True,
            ).drop_duplicates()
            restored, neighbor_audit = neighbors(
                restored,
                donors,
                threshold=args.neighbor_threshold,
                agreement=args.neighbor_agreement,
            )
            audit = audit + neighbor_audit
        model = RobustForest().fit(restored)
        joblib.dump(model, args.out / "model.joblib")
        cpu = os.cpu_count() or 1
        jobs = args.jobs or min(args.folds, cpu)
        if len(frame) < 100:
            jobs = 1
        metrics, oof = cross_validate(restored, folds=args.folds, jobs=jobs)
        (args.out / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        pd.DataFrame(audit).to_csv(args.out / "condition_audit.csv", index=False)
        oof.to_csv(args.out / "oof.csv", index=False)
        print(
            f"{args.out}: rows={metrics['n']} "
            f"MSE={metrics['mse_si']:,.1f} RMSE={metrics['rmse_si']:,.1f} "
            f"MAE={metrics['mae_si']:,.1f} "
            f"conditions_filled={len(audit)}"
        )
        return 0

    model = joblib.load(args.model / "model.joblib")
    original = pd.read_csv(args.test)
    test = original.copy()
    test["_row"] = np.arange(len(test))
    cpu = os.cpu_count() or 1
    jobs = args.jobs or cpu
    prepared = prepare_test_frame(test, jobs=jobs)
    valid = prepared["canonical_smiles"].ne("")
    prepared = prepared[valid].reset_index(drop=True)
    predictions = model.predict(prepared)
    aligned = np.full(len(original), np.nan, dtype=float)
    aligned[prepared["_row"].to_numpy()] = predictions
    if np.isnan(aligned).any():
        fallback = float(np.nanmedian(aligned)) if np.isfinite(aligned).any() else 0.0
        aligned = np.where(np.isfinite(aligned), aligned, fallback)
    index = original["index"] if "index" in original else pd.Series(range(len(original)))
    submission = pd.DataFrame({"index": index, "SI VACV": aligned})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)
    print(f"{args.out}: {len(submission)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
