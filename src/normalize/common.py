from __future__ import annotations

import re

INF = float("inf")

VALID_RELATIONS = {"=", ">", "<", ">=", "<=", "~"}
CENSORED_RELATIONS = {">", "<", ">=", "<=", "~"}
EXACT_RELATIONS = {"="}

_MOLAR_TO_UM = {
    "m": 1e6,
    "mol/l": 1e6,
    "mm": 1e3,
    "mmol/l": 1e3,
    "um": 1.0,
    "umol/l": 1.0,
    "nm": 1e-3,
    "nmol/l": 1e-3,
    "pm": 1e-6,
    "pmol/l": 1e-6,
    "fm": 1e-9,
    "fmol/l": 1e-9,
}

_MASS_TO_MG_L = {
    "ug/ml": 1.0,
    "mg/l": 1.0,
    "ug/l": 1e-3,
    "ng/ml": 1e-3,
    "mg/ml": 1e3,
}

_VALUE_RE = re.compile(
    r"^\s*([<>]=?|=|~)?\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)


class UnitError(ValueError):
    pass


def canonical_unit(unit: str | None) -> str:
    if unit is None:
        return ""
    text = str(unit).strip().lower()
    text = text.replace("μ", "u").replace("µ", "u")
    text = text.replace(" ", "")
    text = text.replace(".ml-1", "/ml").replace("ml-1", "/ml")
    text = text.replace(".l-1", "/l").replace("l-1", "/l")
    return text


def to_um(value: float, unit: str | None, molecular_weight: float | None = None) -> float:
    canonical = canonical_unit(unit)
    if canonical in _MOLAR_TO_UM:
        return float(value) * _MOLAR_TO_UM[canonical]
    if canonical in _MASS_TO_MG_L:
        if not molecular_weight or molecular_weight <= 0:
            raise UnitError(
                f"unit {unit!r} needs a molecular weight for conversion to uM"
            )
        return float(value) * _MASS_TO_MG_L[canonical] * 1000.0 / molecular_weight
    if canonical in {"%", "percent"}:
        raise UnitError("percent is not a concentration")
    raise UnitError(f"unsupported concentration unit {unit!r}")


def parse_value(text) -> tuple[str, float]:
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return "=", float(text)
    match = _VALUE_RE.match(str(text))
    if not match:
        raise ValueError(f"cannot parse value {text!r}")
    relation = match.group(1) or "="
    return relation, float(match.group(2))


def relation_bounds(relation: str, value: float) -> tuple[float, float]:
    rel = normalize_relation(relation)
    if rel in {">", ">="}:
        return value, INF
    if rel in {"<", "<="}:
        return 0.0, value
    return value, value


def normalize_relation(relation: str | None) -> str:
    if relation is None:
        return "="
    text = str(relation).strip()
    if not text:
        return "="
    if text in VALID_RELATIONS:
        return text
    raise ValueError(f"unsupported relation {relation!r}")


def is_exact(relation: str) -> bool:
    return relation in EXACT_RELATIONS


def selectivity_index(ic50_um: float, cc50_um: float) -> float | None:
    if ic50_um is None or cc50_um is None:
        return None
    if ic50_um <= 0 or cc50_um <= 0:
        return None
    return cc50_um / ic50_um


def selectivity_is_exact(ic50_relation: str, cc50_relation: str) -> bool:
    return is_exact(ic50_relation) and is_exact(cc50_relation)


def rdkit_available() -> bool:
    try:
        import rdkit  # noqa: F401
    except ImportError:
        return False
    return True


def standardize_smiles(smiles: str | None) -> str | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    if not rdkit_available():
        return smiles.strip()
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule)


def molecular_weight(smiles: str | None) -> float | None:
    if not isinstance(smiles, str) or not smiles.strip() or not rdkit_available():
        return None
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return float(Descriptors.MolWt(molecule))
