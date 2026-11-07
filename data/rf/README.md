# Random Forest — public score 1,106,361.40883

Model and data ported from the colleague's standalone repository.

- `train.csv` — 408 rows, the selected table the model was trained on.
- `test.csv` — 597 rows, the matching feature table.
- `submission_reference.csv` — the exact scored submission (public score 1,106,361.40883).
- 38 predictors: 37 selected Morgan bits + `culture_clean_BSC-40`.
- Target: `SI VACV` on the raw scale, including the 36066.667 outlier.
- Titles, SMILES, scaffolds and index are not used as features.

Model parameters: `n_estimators=500`, `min_samples_leaf=2`, `max_features=1.0`,
`criterion='squared_error'`, `random_state=42`, `n_jobs=2`.

In this repository the model is retrained with the local scikit-learn version
by `make train MODEL=randomforest` (artifacts under `runs/randomforest/`) and
predicts with `make submit MODEL=randomforest`. Retrained predictions match
`submission_reference.csv` within a small numerical difference caused by the
scikit-learn version.
