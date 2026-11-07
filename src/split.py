from __future__ import annotations

import random
from collections import Counter

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def scaffold_of(smiles: str) -> str:
    if not isinstance(smiles, str) or not smiles.strip():
        return ""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ""
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
    return scaffold or "_acyclic"


def group_key(smiles: str, strategy: str = "scaffold") -> str:
    if strategy == "smiles":
        return str(smiles).strip()
    return scaffold_of(smiles)


def assign_folds(groups: list[str], n_splits: int = 5, seed: int = 42) -> list[int]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    unique = list(dict.fromkeys(groups))
    if len(unique) < n_splits:
        raise ValueError(
            f"cannot build {n_splits} folds from {len(unique)} groups"
        )

    sizes = Counter(groups)
    rng = random.Random(seed)
    order = sorted(unique, key=lambda group: (-sizes[group], rng.random()))

    fold_sizes = [0] * n_splits
    mapping: dict[str, int] = {}
    for group in order:
        fold = min(range(n_splits), key=lambda index: (fold_sizes[index], index))
        mapping[group] = fold
        fold_sizes[fold] += sizes[group]

    return [mapping[group] for group in groups]


def make_splits(
    frame: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    strategy: str = "scaffold",
    smiles_column: str = "SMILES",
) -> pd.Series:
    groups = [group_key(value, strategy) for value in frame[smiles_column]]
    folds = assign_folds(groups, n_splits=n_splits, seed=seed)
    return pd.Series(folds, index=frame.index, name="fold")
