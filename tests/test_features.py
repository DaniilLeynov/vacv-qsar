import numpy as np
import pytest

from src.features import (
    FEATURE_NAMES,
    FEATURE_SETS,
    FeatureError,
    build_matrix,
    feature_names,
    featurize,
)


def test_featurize_returns_fixed_length_vector():
    vector = featurize("CCO")
    assert vector.shape == (len(FEATURE_NAMES),)
    assert vector.dtype == np.float32
    assert vector.sum() > 0


def test_featurize_rejects_invalid_smiles():
    with pytest.raises(FeatureError):
        featurize("not a smiles")
    with pytest.raises(FeatureError):
        featurize("")


def test_every_feature_set_has_stable_length_and_finite_values():
    for name in FEATURE_SETS:
        names = feature_names(name)
        vector = featurize("CCO", feature_set=name)
        assert vector.shape == (len(names),)
        assert np.isfinite(vector).all()


def test_parallel_feature_build_matches_sequential():
    smiles = ["CCO", "CCC"] * 150
    sequential, _, _ = build_matrix(smiles, feature_set="maccs", jobs=1)
    parallel, _, errors = build_matrix(smiles, feature_set="maccs", jobs=2)
    assert not errors
    assert np.array_equal(sequential, parallel)
