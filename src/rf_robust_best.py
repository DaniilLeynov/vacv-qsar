"""Winning Random Forest + Ridge algorithm for SI VACV.

The default training file is train_407.csv next to this script.
Run with the project test data:
    python rf_robust_best.py --project /path/to/project --output submission.csv

Use --test with a prepared test CSV to run outside the project.
RobustForest.fit and predict also accept prepared DataFrames.
Dependencies: numpy, pandas, rdkit, scikit-learn, threadpoolctl.
"""

import argparse
from collections import Counter
from functools import lru_cache
from itertools import combinations_with_replacement
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits


TARGET = "SI VACV"
GROUP = "canonical_smiles"
SOURCE = "official_train"
CONDITIONS = ["culture_clean", "strain_clean"]
MORGAN = [f"morgan_{i}" for i in range(2048)]
RF_BITS = [
    72, 195, 262, 497, 519, 526, 545, 718, 872, 888, 973, 996, 1048,
    1053, 1070, 1141, 1151, 1157, 1176, 1202, 1230, 1235, 1292, 1462, 1555, 1591,
    1603, 1649, 1662, 1665, 1697, 1713, 1729, 1804, 1858, 1885, 1950,
]
RF_COLUMNS = [f"morgan_{i}" for i in RF_BITS] + ["culture_clean_BSC-40"]
DESCRIPTOR_NAMES = (
    "MolWt ExactMolWt HeavyAtomMolWt MolLogP MolMR TPSA LabuteASA "
    "HeavyAtomCount NumHeteroatoms NumHDonors NumHAcceptors NumRotatableBonds "
    "RingCount NumAromaticRings NumAliphaticRings NumSaturatedRings FractionCSP3 "
    "NumValenceElectrons BertzCT BalabanJ Kappa1 Kappa2 Kappa3 FormalCharge AtomCountWithH "
    "Count_C Count_N Count_O Count_S Count_P Count_F Count_Cl Count_Br Count_I"
)
DESCRIPTORS = [f"rdkit_{name}" for name in DESCRIPTOR_NAMES.split()]
TYPES = ('acyl', 'halo', 'alkyl', 'amine', 'oxygen', 'other')
HALOGENS = {9, 35, 53, 17}
POSITION_COLUMNS = [
    f"aromatic_{a}_{b}_distance{distance}"
    for a, b in combinations_with_replacement(TYPES, 2)
    for distance in (1, 2, 3)
] + ["aromatic_acyl_halo_nonadjacent", "aromatic_ring_nitrogens"]


def substituent(atom):
    atomic_number = atom.GetAtomicNum()
    if atomic_number in HALOGENS:
        return "halo"
    if atomic_number == 6:
        if any(bond.GetBondType() == Chem.BondType.DOUBLE
               and bond.GetOtherAtom(atom).GetAtomicNum() == 8 for bond in atom.GetBonds()):
            return "acyl"
        if sum(n.GetAtomicNum() in HALOGENS for n in atom.GetNeighbors()) >= 2:
            return "halo"
        if not atom.GetIsAromatic():
            return "alkyl"
    if atomic_number == 7:
        return "amine"
    if atomic_number == 8:
        return "oxygen"
    return "other"

def group_type(atom):
    kind = substituent(atom)
    if kind == "amine" and atom.GetFormalCharge() > 0:
        if any(neighbor.GetAtomicNum() == 8 for neighbor in atom.GetNeighbors()):
            return "nitro"
    if kind == "alkyl" and any(
        bond.GetBondType() == Chem.BondType.TRIPLE and bond.GetOtherAtom(atom).GetAtomicNum() == 7
        for bond in atom.GetBonds()
    ):
        return "cyano"
    return kind

@lru_cache(maxsize=2048)
def aromatic_context(smiles):
    """Encode a ring, its acyl linkage, and the distance to each substituent."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid canonical SMILES: {smiles}")
    features = Counter()
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
                    attached = [a for a in linker.GetNeighbors() if a.GetIdx() != anchor.GetIdx()]
                    roles += [f"N_{a.GetSymbol()}_r{int(a.IsInRing())}_d{a.GetDegree()}" for a in attached]
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


class RobustForest:
    """Use the capped forest in the body and log-scale Ridge in the upper tail."""

    def fit(self, train):
        y = train[TARGET].to_numpy(dtype=float)
        threshold = y.mean() + 3 * y.std()
        self.forest_ = RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, max_features=1.0, random_state=42, n_jobs=1,
        ).fit(train[RF_COLUMNS].to_numpy(), np.minimum(y, threshold))
        self.tail_cutoff_ = float(np.quantile(y, 0.95))
        self.log_limit_ = float(np.log1p(4 * y.max()))
        repeats = train.groupby(GROUP, sort=False)[GROUP].transform("size").to_numpy(dtype=float)
        weights = (1 + y / max(self.tail_cutoff_, 1)) ** 0.4
        weights /= np.sqrt(repeats)
        weights *= np.where(train[SOURCE], 1, 2)
        weights /= weights.mean()

        self.morgan_ = StandardScaler().fit(train[MORGAN])
        self.structure_ = StandardScaler().fit(train[DESCRIPTORS + POSITION_COLUMNS])
        self.conditions_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train[CONDITIONS])
        self.context_ = DictVectorizer(sparse=False).fit([aromatic_context(smiles) for smiles in train[GROUP]])
        self.ridge_ = Ridge(alpha=10).fit(self._design(train), np.log1p(y), sample_weight=weights)
        return self

    def _design(self, frame):
        return np.column_stack([
            self.morgan_.transform(frame[MORGAN]),
            0.5 * self.structure_.transform(frame[DESCRIPTORS + POSITION_COLUMNS]),
            3 * self.conditions_.transform(frame[CONDITIONS]),
            self.context_.transform([aromatic_context(smiles) for smiles in frame[GROUP]]),
        ])

    def predict(self, frame):
        core = self.forest_.predict(frame[RF_COLUMNS].to_numpy())
        log_si = self.ridge_.predict(self._design(frame))
        tail = np.expm1(np.clip(log_si, 0, self.log_limit_))
        return np.where(tail > self.tail_cutoff_, tail, core)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path(__file__).with_name("train_407.csv"))
    parser.add_argument("--project", type=Path, default=Path.cwd(), help="Project supplying the default test")
    parser.add_argument("--test", type=Path, help="Prepared test CSV with the structural feature columns")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()
    train = pd.read_csv(args.train)
    if args.test is None:
        sys.path.insert(0, str(args.project.resolve()))
        from model_training.rf_robust.run import data

        _, test, order = data()
    else:
        test = pd.read_csv(args.test)
        order = pd.Index(test["index"])
    if not order.is_unique or order.hasnans:
        raise ValueError("Expected unique, nonmissing test indices.")
    with threadpool_limits(limits=1):
        prediction = RobustForest().fit(train).predict(test)
    submission = pd.Series(prediction, index=test["index"], name=TARGET).reindex(order)
    submission.to_csv(args.output, index_label="index")


if __name__ == "__main__":
    main()
