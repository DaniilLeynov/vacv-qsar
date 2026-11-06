from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors

FP_SIZE = 2048
FP_RADIUS = 2

DESCRIPTORS = [
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA),
    ("NumHDonors", Lipinski.NumHDonors),
    ("NumHAcceptors", Lipinski.NumHAcceptors),
    ("NumRotatableBonds", Lipinski.NumRotatableBonds),
    ("NumAromaticRings", rdMolDescriptors.CalcNumAromaticRings),
    ("RingCount", rdMolDescriptors.CalcNumRings),
    ("FractionCSP3", rdMolDescriptors.CalcFractionCSP3),
]

FEATURE_NAMES = [f"morgan_{index}" for index in range(FP_SIZE)] + [
    name for name, _ in DESCRIPTORS
]

_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=FP_RADIUS, fpSize=FP_SIZE
)


class FeatureError(ValueError):
    pass


def featurize(smiles: str) -> np.ndarray:
    if not isinstance(smiles, str) or not smiles.strip():
        raise FeatureError("empty SMILES")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise FeatureError(f"invalid SMILES: {smiles!r}")
    fingerprint = _GENERATOR.GetCountFingerprintAsNumPy(molecule).astype(np.float32)
    values = np.array([function(molecule) for _, function in DESCRIPTORS], dtype=np.float32)
    return np.concatenate([fingerprint, values])


def build_matrix(smiles: list[str]) -> tuple[np.ndarray, list[int], list[dict]]:
    rows: list[np.ndarray] = []
    valid: list[int] = []
    errors: list[dict] = []
    for index, value in enumerate(smiles):
        try:
            rows.append(featurize(value))
            valid.append(index)
        except FeatureError as exc:
            errors.append({"index": index, "smiles": value, "reason": str(exc)})
    if not rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), valid, errors
    return np.vstack(rows), valid, errors
