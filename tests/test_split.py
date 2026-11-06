import pandas as pd

from src.split import assign_folds, group_key, make_splits, scaffold_of


def test_same_group_stays_in_one_fold():
    groups = ["a", "a", "b", "b", "c", "c", "d", "d"]
    folds = assign_folds(groups, n_splits=2, seed=1)
    for group in set(groups):
        assert len({fold for g, fold in zip(groups, folds) if g == group}) == 1


def test_assign_folds_is_deterministic():
    assert assign_folds(list("abcdefgh"), 3, 7) == assign_folds(
        list("abcdefgh"), 3, 7
    )


def test_make_splits_adds_fold_column():
    frame = pd.DataFrame({"SMILES": ["CCO", "CCC", "CCN", "CCCl"]})
    folds = make_splits(frame, n_splits=2, seed=0, strategy="smiles")
    assert len(folds) == len(frame)
    assert set(folds) <= {0, 1}


def test_scaffold_of_distinguishes_rings():
    assert scaffold_of("c1ccccc1CCO")
    assert group_key("CCO", "smiles") == "CCO"
