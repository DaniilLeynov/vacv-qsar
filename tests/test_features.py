import numpy as np
import pytest

from src.features import FEATURE_NAMES, FeatureError, featurize


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
