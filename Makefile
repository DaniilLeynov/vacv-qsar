.DEFAULT_GOAL := benchmark

PY ?= .venv/bin/python
SOURCES ?= sources
OUT ?= data/raw
MODEL ?= catboost
TRAIN_OUT ?= runs/$(MODEL)
JOBS ?= 0
VENV_FLAGS ?= --system-site-packages

export PYTHONWARNINGS := ignore

.PHONY: benchmark venv chembl data data-selected data-augmented data-organizers data-chembl datasets registry list fetch train train-catboost train-randomforest train-ensemble train-rfrobust submit submit-catboost submit-randomforest submit-ensemble submit-rfrobust test lint

benchmark:
	@$(PY) -m src.benchmark

venv:
	@python3 -m venv $(VENV_FLAGS) .venv
	@.venv/bin/pip install -r requirements.txt -r requirements-dev.txt

chembl:
	@$(PY) -m chembl_downloader download -v

data: data-augmented

data-selected:
	@$(PY) -m src.dataset --csv sources/selected.mapping.yaml --out data/dataset.csv --dropped data/processed/dropped.csv

data-augmented:
	@$(PY) -m src.dataset \
		--csv sources/selected.mapping.yaml \
		--csv sources/chembl_static.mapping.yaml \
		--csv sources/paper_cidofovir.mapping.yaml \
		--csv sources/paper_bmcl2022.mapping.yaml \
		--csv sources/paper_adamantane2021.mapping.yaml \
		--out data/processed/dataset_augmented.csv --dropped data/processed/dropped_augmented.csv
	@$(PY) -m src.merge --dataset data/processed/dataset_augmented.csv --out data/dataset.csv

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
	@$(PY) -m src.train --data data/dataset.csv --out $(TRAIN_OUT) --model catboost --mode exact --use-scaffold

train-randomforest:
	@$(PY) -m src.random_forest train --train data/rf/train.csv --out $(TRAIN_OUT)

train-ensemble:
	@$(PY) -m src.ensemble train --train data/dataset.csv --organizers data/organizers/train.csv --out $(TRAIN_OUT)

train-rfrobust:
	@$(PY) -m src.rf_robust train \
		--train data/rf_robust/train_407.csv \
		--out runs/rfrobust --jobs $(JOBS)

submit: train
	@$(MAKE) --no-print-directory submit-$(MODEL)

submit-catboost:
	@$(PY) -m src.predict --test data/organizers/test.csv --model $(TRAIN_OUT) --out submissions/submission.csv

submit-randomforest:
	@$(PY) -m src.random_forest predict --model $(TRAIN_OUT) --test data/rf/test.csv --out submissions/submission.csv

submit-ensemble:
	@$(PY) -m src.ensemble predict --model $(TRAIN_OUT) --test data/organizers/test.csv --out submissions/submission.csv

submit-rfrobust:
	@$(PY) -m src.rf_robust predict \
		--model runs/rfrobust \
		--test data/organizers/test.csv \
		--out submissions/submission.csv \
		--jobs $(JOBS)

test:
	@$(PY) -m pytest -q

lint:
	@$(PY) -m ruff check --isolated --select E4,E7,E9,F src tests
