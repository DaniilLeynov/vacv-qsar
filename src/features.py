from __future__ import annotations

import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors

FP_SIZE = 2048
FP_RADIUS = 2
MACCS_SIZE = 167

# Bits selected by the colleague's RobustForest on the 407-row table.
RF_BITS = [
    72, 195, 262, 497, 519, 526, 545, 718, 872, 888, 973, 996, 1048,
    1053, 1070, 1141, 1151, 1157, 1176, 1202, 1230, 1235, 1292, 1462,
    1555, 1591, 1603, 1649, 1662, 1665, 1697, 1713, 1729, 1804, 1858,
    1885, 1950,
]

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

# Each feature set is an ordered list of components.  The default ``all``
# must stay byte-for-byte compatible with the original 2224 features.
FEATURE_SET_COMPONENTS: dict[str, tuple[str, ...]] = {
    "morgan": ("morgan",),
    "morgan_chiral": ("morgan_chiral",),
    "rf_bits": ("rf_bits",),
    "morgan3": ("morgan3",),
    "maccs": ("maccs",),
    "descriptors": ("descriptors",),
    "extended_descriptors": ("extended_descriptors",),
    "rdkit": ("rdkit",),
    "atom_pair": ("atom_pair",),
    "torsion": ("torsion",),
    "morgan_maccs": ("morgan", "maccs"),
    "morgan_descriptors": ("morgan", "descriptors"),
    "maccs_descriptors": ("maccs", "descriptors"),
    "rf_bits_maccs": ("rf_bits", "maccs"),
    "rf_bits_all": ("rf_bits", "maccs", "descriptors"),
    "all": ("morgan", "maccs", "descriptors"),
    "all_fp": ("morgan", "maccs", "atom_pair", "torsion", "descriptors"),
    "extended_all": ("morgan", "maccs", "extended_descriptors"),
}

FEATURE_SETS = tuple(FEATURE_SET_COMPONENTS)

_GENERATORS: dict[str, object] = {}
_EXTENDED_DESCRIPTOR_NAMES: list[str] | None = None


class FeatureError(ValueError):
    pass


def process_pool_context():
    """Use fork on Linux to avoid the forkserver socket in restricted tests."""
    if sys.platform.startswith("linux"):
        return multiprocessing.get_context("fork")
    return None


def _generator(kind: str):
    if kind in _GENERATORS:
        return _GENERATORS[kind]
    if kind == "morgan":
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=FP_RADIUS, fpSize=FP_SIZE
        )
    elif kind == "morgan_chiral":
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=FP_RADIUS, fpSize=FP_SIZE, includeChirality=True
        )
    elif kind == "morgan3":
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=3, fpSize=FP_SIZE
        )
    elif kind == "rdkit":
        generator = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=FP_SIZE)
    elif kind == "atom_pair":
        generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=FP_SIZE)
    elif kind == "torsion":
        generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
            fpSize=FP_SIZE
        )
    else:
        raise FeatureError(f"unknown fingerprint {kind!r}")
    _GENERATORS[kind] = generator
    return generator


def extended_descriptor_names() -> list[str]:
    global _EXTENDED_DESCRIPTOR_NAMES
    if _EXTENDED_DESCRIPTOR_NAMES is None:
        molecule = Chem.MolFromSmiles("CCO")
        _EXTENDED_DESCRIPTOR_NAMES = sorted(
            Descriptors.CalcMolDescriptors(molecule).keys()
        )
    return _EXTENDED_DESCRIPTOR_NAMES


def _component_names(component: str) -> list[str]:
    if component in {
        "morgan",
        "morgan_chiral",
        "morgan3",
        "rdkit",
        "atom_pair",
        "torsion",
    }:
        prefix = {
            "morgan": "morgan",
            "morgan_chiral": "morgan_chiral",
            "morgan3": "morgan3",
            "rdkit": "rdkit",
            "atom_pair": "atompair",
            "torsion": "torsion",
        }[component]
        return [f"{prefix}_{index}" for index in range(FP_SIZE)]
    if component == "rf_bits":
        return [f"morgan_{index}" for index in RF_BITS]
    if component == "maccs":
        return [f"maccs_{index}" for index in range(MACCS_SIZE)]
    if component == "descriptors":
        return [name for name, _ in DESCRIPTORS]
    if component == "extended_descriptors":
        return [f"desc_{name}" for name in extended_descriptor_names()]
    raise FeatureError(f"unknown feature component {component!r}")


def feature_names(feature_set: str = "all") -> list[str]:
    if feature_set not in FEATURE_SET_COMPONENTS:
        raise FeatureError(f"unknown feature set {feature_set!r}")
    names: list[str] = []
    for component in FEATURE_SET_COMPONENTS[feature_set]:
        names.extend(_component_names(component))
    return names


FEATURE_NAMES = feature_names("all")


def _component_values(molecule: Chem.Mol, component: str) -> np.ndarray:
    if component in {"morgan", "morgan3", "rdkit", "atom_pair", "torsion"}:
        return _generator(component).GetCountFingerprintAsNumPy(molecule)
    if component == "morgan_chiral":
        return _generator(component).GetFingerprintAsNumPy(molecule)
    if component == "rf_bits":
        fingerprint = _generator("morgan_chiral").GetFingerprintAsNumPy(molecule)
        return fingerprint[RF_BITS]
    if component == "maccs":
        values = np.zeros((MACCS_SIZE,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(
            rdMolDescriptors.GetMACCSKeysFingerprint(molecule), values
        )
        return values
    if component == "descriptors":
        return np.array([function(molecule) for _, function in DESCRIPTORS])
    if component == "extended_descriptors":
        values = Descriptors.CalcMolDescriptors(molecule)
        array = np.array(
            [values[name] for name in extended_descriptor_names()], dtype=float
        )
        return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    raise FeatureError(f"unknown feature component {component!r}")


def featurize(smiles: str, feature_set: str = "all") -> np.ndarray:
    if not isinstance(smiles, str) or not smiles.strip():
        raise FeatureError("empty SMILES")
    if feature_set not in FEATURE_SET_COMPONENTS:
        raise FeatureError(f"unknown feature set {feature_set!r}")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise FeatureError(f"invalid SMILES: {smiles!r}")

    parts = [
        _component_values(molecule, component).astype(np.float32)
        for component in FEATURE_SET_COMPONENTS[feature_set]
    ]
    return np.concatenate(parts)


def _featurize_one(payload: tuple[int, str, str]):
    index, smiles, feature_set = payload
    try:
        return index, featurize(smiles, feature_set), None
    except FeatureError as exc:
        return index, None, {"index": index, "smiles": smiles, "reason": str(exc)}


def build_matrix(
    smiles: list[str], feature_set: str = "all", jobs: int = 1
) -> tuple[np.ndarray, list[int], list[dict]]:
    if jobs > 1 and len(smiles) >= 200:
        payloads = [
            (index, value, feature_set) for index, value in enumerate(smiles)
        ]
        chunksize = max(1, len(payloads) // (jobs * 4))
        with ProcessPoolExecutor(
            max_workers=jobs, mp_context=process_pool_context()
        ) as pool:
            results = list(pool.map(_featurize_one, payloads, chunksize=chunksize))
        results.sort(key=lambda item: item[0])
    else:
        results = [
            _featurize_one((index, value, feature_set))
            for index, value in enumerate(smiles)
        ]

    rows: list[np.ndarray] = []
    valid: list[int] = []
    errors: list[dict] = []
    for index, vector, error in results:
        if error is not None:
            errors.append(error)
            continue
        rows.append(vector)
        valid.append(index)
    if not rows:
        return (
            np.empty((0, len(feature_names(feature_set))), dtype=np.float32),
            valid,
            errors,
        )
    return np.vstack(rows), valid, errors
