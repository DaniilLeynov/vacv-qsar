"""RF + Ridge with source recovery and similarity-based condition imputation.

Run with the source project and its existing Python environment:
    python rf_robust_restored.py --project /path/to/project --output submission.csv

The project supplies the original 407-row training data, prepared test,
normalization rules and merge audit. --raw-measurements supplies the raw
measurement CSV. Prepared inputs can be supplied with --train, --test and
--source-records. Dependencies: numpy, pandas, rdkit, scikit-learn,
threadpoolctl. Only the fixed winning configuration is implemented.
"""

import argparse
from collections import Counter
from functools import lru_cache
from itertools import combinations_with_replacement
from pathlib import Path
import sys
import json
import re

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
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



FIELDS = {'culture_clean': {'', 'unknown', 'Unknown_Culture'}, 'strain_clean': {'', 'unknown'}}
CELL_NAMES = {'vero': 'Vero', 'hff': 'HFF', 'humanforeskinfibroblast': 'HFF', 'hel': 'HEL', 'cellshel': 'HEL', 'hela': 'HeLa', 'helacells': 'HeLa', 'huh7': 'Huh7', 'bsc40': 'BSC-40', 'bsc40cells': 'BSC-40', 'bsc1': 'BSC-1', 'cellsbsc1': 'BSC-1', 'prk': 'PRK', 'prkcells': 'PRK', 'vero76': 'Vero 76', 'cv1': 'CV-1', 'a549': 'A549', 'e6sm': 'E6SM', 'mt4': 'MT-4', 'cem': 'CEM', 'bhk21': 'BHK-21', 'lungfibroblast': 'Lung fibroblasts', 'lungfibroblastcells': 'Lung fibroblasts'}
GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)


@lru_cache(maxsize=4096)
def structure(smiles):
    """Cache only deterministic calculations, never fitted or target-dependent data."""
    values = np.full(len(MORGAN) + len(DESCRIPTORS), np.nan)
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is not None:
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        values[:2048] = GENERATOR.GetFingerprintAsNumPy(mol)
        for j, name in enumerate(DESCRIPTORS):
            short = name.removeprefix("rdkit_")
            try:
                if short == "FormalCharge":
                    value = Chem.GetFormalCharge(mol)
                elif short == "AtomCountWithH":
                    value = Chem.AddHs(mol).GetNumAtoms()
                elif short.startswith("Count_"):
                    value = sum(a.GetSymbol() == short.removeprefix("Count_") for a in mol.GetAtoms())
                else:
                    value = getattr(Descriptors, short)(mol)
                values[2048+j] = value if np.isfinite(value) else np.nan
            except (ValueError, RuntimeError, OverflowError, ZeroDivisionError):
                pass
        group = "canonical:" + canonical
    else:
        canonical = None
        group = "invalid:" + (smiles.strip() if smiles else "<missing>")
    return canonical, group, np.r_[values, float(np.isnan(values).any())]


def clean_conditions(strain, culture, rules):
    strain = "unknown" if pd.isna(strain) else str(strain)
    culture = "unknown" if pd.isna(culture) else str(culture)
    if strain in rules["strain_exact"]:
        s = rules["strain_exact"][strain]
    else:
        text = strain.lower()
        if strain == "unknown":
            s = "unknown"
        elif "copenhagen" in text:
            s = "Copenhagen"
        elif "western reserve" in text or re.search(r"\bwr\b", text):
            s = "Western Reserve"
        elif "vaccinia" in text or re.search(r"\bvv\b", text):
            s = "Unknown_VACV"
        else:
            s = "unknown"
    c = rules["culture_exact"].get(culture, "unknown")
    if culture.strip().lower() == "vero" and re.search(r"vero\s*76", strain.lower()):
        c = "Vero 76"
    return s, c


def normalize(culture, strain, rules):
    s, c = clean_conditions(strain, culture, rules)
    key = re.sub(r"[^a-z0-9]", "", str(culture).lower())
    if key in CELL_NAMES:
        c = CELL_NAMES[key]
    if c in FIELDS["culture_clean"]:
        c = "Unknown_Culture"
    return c, s


def source_records(previous, project, raw_measurements):
    rules = json.loads((project / "model_training/data/category_rules.json").read_text())
    audit = pd.read_csv(project / "reports/smiles_si_merge_audit.csv").fillna("")
    audit = audit.loc[audit["status"].isin(["included", "duplicate_molecule_si"])]
    rows = []
    for _, row in audit.iterrows():
        culture, strain = normalize(row["Culture"], row["strain"], rules)
        rows.append({GROUP: row[GROUP], TARGET: row[TARGET], "culture_clean": culture,
                     "strain_clean": strain, "reference": f"merge_audit:{row['input_source']}:{row['input_csv_row']}"})
    raw = pd.read_csv(raw_measurements, dtype=str, keep_default_na=False)
    exact = raw.loc[raw["is_vacv"].str.lower().eq("true") & raw["endpoint"].eq("si")
                    & raw["relation"].eq("=") & raw["dropped_reason"].eq("")]
    for _, row in exact.iterrows():
        canonical = structure(row["SMILES"])[0]
        culture, strain = normalize(row["cell_line"], row["organism"], rules)
        rows.append({GROUP: canonical, TARGET: float(row["value_um"]), "culture_clean": culture,
                     "strain_clean": strain, "reference": f"raw:{row['source']}:{row['source_id']}"})
    records = pd.DataFrame(rows)
    records = records.loc[records[GROUP].notna()].copy()
    prior = previous.loc[:, [GROUP, TARGET, *FIELDS]].assign(reference="previous_train")
    return pd.concat([records, prior], ignore_index=True).drop_duplicates()


def recover(frame, records):
    result = frame.copy()
    groups = dict(tuple(records.groupby(GROUP, sort=False)))
    log = []
    for idx, row in result.iterrows():
        records_for_molecule = groups.get(row[GROUP], records.iloc[:0])
        matched = records_for_molecule.loc[np.isclose(records_for_molecule[TARGET].astype(float), row[TARGET],
                                                     rtol=.001, atol=.001)]
        if row["culture_clean"] not in FIELDS["culture_clean"]:
            matched = matched.loc[matched["culture_clean"].eq(row["culture_clean"])
                                  | matched["culture_clean"].isin(FIELDS["culture_clean"])]
        for field, missing in FIELDS.items():
            if row[field] not in missing:
                continue
            observed = matched.loc[~matched[field].isin(missing)]
            values = observed[field].unique()
            if len(values) == 1:
                result.at[idx, field] = values[0]
                log.append(dict(source_row=row["source_row"], field=field, before=row[field], after=values[0],
                                method="source_record", similarity=1.0, agreement=1.0,
                                donor_smiles=row[GROUP], reference="|".join(observed["reference"].unique()[:5])))
    return result, log


def neighbors(frame, donors, threshold, agreement=.8):
    """Vote once per distinct donor molecule, using only recorded conditions."""
    result = frame.copy()
    log = []
    for field, missing in FIELDS.items():
        labels = donors.loc[~donors[field].isin(missing), [GROUP, field]].drop_duplicates()
        count = labels.groupby(GROUP)[field].transform("nunique")
        labels = labels.loc[count.eq(1)].reset_index(drop=True)
        if field == "strain_clean":
            labels = labels.loc[~labels[field].isin(["Unknown_VACV", "Non_VACV"])].reset_index(drop=True)
        if labels.empty:
            continue
        bits = np.stack([structure(smiles)[2][:2048] for smiles in labels[GROUP]]).astype(np.float32)
        bit_sum = bits.sum(axis=1)
        for idx, row in result.loc[result[field].isin(missing)].iterrows():
            query = row[MORGAN].to_numpy(dtype=np.float32)
            intersection = bits @ query
            union = bit_sum + query.sum() - intersection
            similarity = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
            keep = np.flatnonzero(similarity >= threshold)
            if not len(keep):
                continue
            keep = keep[np.argsort(-similarity[keep], kind="stable")[:5]]
            vote = pd.Series(similarity[keep] ** 4).groupby(labels.loc[keep, field].to_numpy()).sum()
            winner = vote.idxmax()
            support = float(vote.max() / vote.sum())
            if support < agreement:
                continue
            same_class = keep[labels.loc[keep, field].eq(winner).to_numpy()]
            nearest = same_class[0]
            result.at[idx, field] = winner
            log.append(dict(source_row=row["source_row"], field=field, before=row[field], after=winner,
                            method="similar_molecules", similarity=float(similarity[nearest]), agreement=support,
                            donor_smiles=labels.loc[nearest, GROUP], reference=f"{len(keep)}_neighbors"))
    result["culture_clean_BSC-40"] = result["culture_clean"].eq("BSC-40")
    return result, log


def restore_training_conditions(train, records):
    train = train.copy()
    if "source_row" not in train:
        train["source_row"] = np.arange(len(train)) + 1
    recovered, source_audit = recover(train, records)
    recovered["culture_clean_BSC-40"] = recovered["culture_clean"].eq("BSC-40")
    donors = pd.concat([records[[GROUP, *FIELDS]], recovered[[GROUP, *FIELDS]]], ignore_index=True)
    restored, neighbor_audit = neighbors(recovered, donors.drop_duplicates(), threshold=0.7, agreement=0.8)
    return restored, source_audit + neighbor_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--train", type=Path, help="Prepared training CSV with original SI and known conditions")
    parser.add_argument("--test", type=Path, help="Prepared test CSV with unique index")
    parser.add_argument("--source-records", type=Path, help="Normalized source records with SI and conditions")
    parser.add_argument("--raw-measurements", type=Path,
                        default=Path.home() / "Downloads/Telegram Desktop/ic_cc_raw/combined/all_measurements.csv")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()
    project = args.project.resolve()
    default_train = default_test = default_order = None
    if args.train is None or args.test is None:
        sys.path.insert(0, str(project))
        from model_training.rf_robust.run import data
        default_train, default_test, default_order = data()
    train = pd.read_csv(args.train, low_memory=False) if args.train else default_train
    test = pd.read_csv(args.test, low_memory=False) if args.test else default_test
    order = pd.Index(test["index"]) if args.test else default_order
    if not order.is_unique or order.hasnans:
        raise ValueError("Expected unique, nonmissing test indices.")
    records = (pd.read_csv(args.source_records, low_memory=False) if args.source_records else
               source_records(train, project, args.raw_measurements))
    restored, audit = restore_training_conditions(train, records)
    with threadpool_limits(limits=1):
        model = RobustForest().fit(restored)
        prediction = model.predict(test)
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("Expected finite nonnegative SI predictions.")
    pd.Series(prediction, index=test["index"], name=TARGET).reindex(order).to_csv(args.output, index_label="index")
    print(f"Trained on {len(restored)} rows; filled {len(audit)} condition values; saved {len(prediction)} predictions.")


if __name__ == "__main__":
    main()
