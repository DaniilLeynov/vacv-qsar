import pytest

from src.normalize.common import (
    UnitError,
    canonical_organism,
    canonical_unit,
    is_exact,
    normalize_cell_line,
    normalize_relation,
    parse_value,
    selectivity_index,
    selectivity_is_exact,
    standardize_smiles,
    to_um,
    unit_factor,
    unit_needs_molecular_weight,
)


def test_molar_units_to_um():
    assert to_um(1, "nM") == pytest.approx(0.001)
    assert to_um(1, "uM") == pytest.approx(1.0)
    assert to_um(1, "µM") == pytest.approx(1.0)
    assert to_um(1, "mM") == pytest.approx(1000.0)
    assert to_um(1, "nmol/l") == pytest.approx(0.001)


def test_mass_units_require_molecular_weight():
    assert to_um(1, "ug/ml", molecular_weight=100) == pytest.approx(10.0)
    assert to_um(1, "ug.mL-1", molecular_weight=100) == pytest.approx(10.0)
    with pytest.raises(UnitError, match="molecular weight"):
        to_um(1, "ug/ml")


def test_percent_and_unknown_units_are_rejected():
    with pytest.raises(UnitError, match="percent"):
        to_um(50, "%")
    with pytest.raises(UnitError, match="unsupported"):
        to_um(50, "furlongs")


def test_parse_value_with_relation():
    assert parse_value(">100") == (">", 100.0)
    assert parse_value("<0.1") == ("<", 0.1)
    assert parse_value("5") == ("=", 5.0)
    assert parse_value(5) == ("=", 5.0)
    with pytest.raises(ValueError):
        parse_value("not-a-number")


def test_normalize_relation():
    assert normalize_relation(None) == "="
    assert normalize_relation(">") == ">"
    with pytest.raises(ValueError):
        normalize_relation("~=")


def test_selectivity_index():
    assert selectivity_index(10, 100) == pytest.approx(10.0)
    assert selectivity_index(0, 100) is None
    assert selectivity_index(10, None) is None


def test_exactness():
    assert is_exact("=") is True
    assert is_exact(">") is False
    assert selectivity_is_exact("=", "=") is True
    assert selectivity_is_exact("=", ">") is False


def test_standardize_smiles_keeps_text_without_rdkit():
    assert standardize_smiles(" CCO ") == "CCO"
    assert standardize_smiles("") is None
    assert canonical_unit("ug.mL-1") == "ug/ml"


def test_unit_factor_and_mass_detection():
    assert unit_factor("nM") == pytest.approx(0.001)
    assert unit_factor("mg/l", molecular_weight=100) == pytest.approx(10.0)
    assert unit_needs_molecular_weight("ug/ml") is True
    assert unit_needs_molecular_weight("nM") is False


def test_context_normalisation():
    assert normalize_cell_line("HeLa cells") == "hela"
    assert normalize_cell_line("Vero") == "vero"
    assert canonical_organism("Vaccinia virus (strain Copenhagen), Vero") == (
        "Vaccinia virus"
    )
    assert canonical_organism("Cowpox virus") == "Cowpox virus"
    assert canonical_organism("") == ""
