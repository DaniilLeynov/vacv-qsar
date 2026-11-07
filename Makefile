.DEFAULT_GOAL := benchmark

PY ?= .venv/bin/python
SOURCES ?= sources
OUT ?= data/raw
MODEL ?= catboost
JOBS ?= 0
CHEMBL_DB ?= data/raw/chembl/chembl.db
DUAL_OUT ?= runs/dual
FEATURE_SET ?= rf_bits_maccs
IC50_TARGETS ?= data/dual/ic50_targets.csv
CC50_TARGETS ?= data/dual/cc50_targets.csv
SUBMISSION ?= submissions/submission.csv
VENV_FLAGS ?= --system-site-packages

export PYTHONWARNINGS := ignore

.PHONY: benchmark venv chembl chembl-pairs data data-all data-selected data-augmented data-organizers data-chembl datasets registry list fetch train train-catboost train-randomforest train-ensemble train-dual train-rfrobust ablation submit submit-catboost submit-randomforest submit-ensemble submit-dual submit-rfrobust test lint

benchmark:
	@$(PY) -m src.benchmark

venv:
	@python3 -m venv $(VENV_FLAGS) .venv
	@.venv/bin/pip install -r requirements.txt -r requirements-dev.txt

chembl:
	@$(PY) -m chembl_downloader download -v

chembl-pairs:
	@$(PY) -m src.chembl_pairs \
		--db $(CHEMBL_DB) \
		--scope viral \
		--exclude-smiles data/organizers/test.csv \
		--out-dir data/processed/chembl

data: data-augmented

data-selected:
	@$(PY) -m src.dataset --csv sources/selected.mapping.yaml --out data/common/dataset.csv --dropped data/processed/dropped.csv

data-augmented:
	@$(PY) -m src.dataset \
		--csv sources/selected.mapping.yaml \
		--csv sources/chembl_static.mapping.yaml \
		--csv sources/paper_cidofovir.mapping.yaml \
		--csv sources/paper_bmcl2022.mapping.yaml \
		--csv sources/paper_adamantane2021.mapping.yaml \
		--out data/processed/dataset_augmented.csv --dropped data/processed/dropped_augmented.csv
	@$(PY) -m src.merge --dataset data/processed/dataset_augmented.csv --out data/common/dataset.csv

data-all:
	@$(PY) -m src.build_all \
		--out data/common/dataset_all.csv \
		--measurements data/dual/all_measurements.csv \
		--targets data/dual/all_targets.csv \
		--ic50-targets data/dual/ic50_targets.csv \
		--cc50-targets data/dual/cc50_targets.csv \
		--dropped data/dual/all_dropped.csv

data-organizers:
	@$(PY) -m src.dataset --csv sources/organizers.mapping.yaml --out data/dataset_organizers.csv --dropped data/processed/dropped_organizers.csv

data-chembl:
	@$(PY) -m src.dataset --chembl --out data/dataset_chembl.csv --dropped data/processed/dropped_chembl.csv

datasets: data-selected data-chembl

registry:
	@$(PY) -m src.registry --out data/found.csv

list:
	@$(PY) -m src.fetch --list

fetch:
	@$(PY) -m src.fetch --all --out $(OUT)

train:
	@$(MAKE) --no-print-directory train-$(MODEL)

train-catboost:
	@$(PY) -m src.train --data data/common/dataset.csv --out runs/catboost --model catboost --mode exact --use-scaffold

train-randomforest:
	@$(PY) -m src.random_forest train --train data/rf/train.csv --out runs/randomforest

train-ensemble:
	@$(PY) -m src.ensemble train --train data/common/dataset.csv --organizers data/organizers/train.csv --out runs/ensemble

train-dual:
	@test -f $(IC50_TARGETS) -a -f $(CC50_TARGETS) || $(MAKE) --no-print-directory data-all
	@$(PY) -m src.dual_model \
		--ic50-targets $(IC50_TARGETS) \
		--cc50-targets $(CC50_TARGETS) \
		--organizers data/organizers/train.csv \
		--test data/organizers/test.csv \
		--out $(DUAL_OUT) \
		--feature-set $(FEATURE_SET) \
		--jobs $(JOBS) \
		--group smiles --seed 42 \
		--iterations 300 --fine-tune-iterations 100

train-rfrobust:
	@$(PY) -m src.rf_robust train \
		--train data/rf_robust/train_407.csv \
		--out runs/rfrobust --jobs $(JOBS)

ablation:
	@$(PY) -m src.feature_ablation \
		--out runs/ablation/results.csv \
		--iterations 200 --fine-tune-iterations 80 \
		--folds 5 --seed 42 --group smiles

submit: train
	@$(MAKE) --no-print-directory submit-$(MODEL)

submit-catboost:
	@$(PY) -m src.predict --test data/organizers/test.csv --model runs/catboost --out $(SUBMISSION)

submit-randomforest:
	@$(PY) -m src.random_forest predict --model runs/randomforest --test data/rf/test.csv --out $(SUBMISSION)

submit-ensemble:
	@$(PY) -m src.ensemble predict --model runs/ensemble --test data/organizers/test.csv --out $(SUBMISSION)

submit-dual:
	@$(PY) -m src.dual_predict --model $(DUAL_OUT) --test data/organizers/test.csv --out $(SUBMISSION)

submit-rfrobust:
	@$(PY) -m src.rf_robust predict \
		--model runs/rfrobust \
		--test data/organizers/test.csv \
		--out $(SUBMISSION) \
		--jobs $(JOBS)

test:
	@$(PY) -m pytest -q

lint:
	@$(PY) -m ruff check --isolated --select E4,E7,E9,F src tests
