PY ?= .venv/bin/python
SOURCES ?= sources
OUT ?= data/raw
MODEL ?= catboost
TRAIN_OUT ?= runs/$(MODEL)
VENV_FLAGS ?= --system-site-packages

.PHONY: venv chembl data data-organizers data-chembl datasets registry list fetch train train-catboost train-randomforest submit submit-catboost submit-randomforest test lint

venv:
	python3 -m venv $(VENV_FLAGS) .venv
	.venv/bin/pip install -r requirements.txt -r requirements-dev.txt

chembl:
	$(PY) -m chembl_downloader download -v

data: data-organizers

data-organizers:
	$(PY) -m src.dataset --csv sources/organizers.mapping.yaml --out data/dataset.csv --dropped data/processed/dropped.csv

data-chembl:
	$(PY) -m src.dataset --chembl --out data/dataset_chembl.csv --dropped data/processed/dropped_chembl.csv

datasets: data-organizers data-chembl

registry:
	$(PY) -m src.registry --out data/found.csv

list:
	$(PY) -m src.fetch --list

fetch:
	$(PY) -m src.fetch --all --out $(OUT)

train:
	@$(MAKE) --no-print-directory train-$(MODEL)

train-catboost:
	$(PY) -m src.train --data data/dataset.csv --out $(TRAIN_OUT) --model catboost --mode exact --use-scaffold

train-randomforest:
	$(PY) -m src.random_forest train --train data/rf/train.csv --out $(TRAIN_OUT)

submit: train
	@$(MAKE) --no-print-directory submit-$(MODEL)

submit-catboost:
	$(PY) -m src.predict --test data/organizers/test.csv --model $(TRAIN_OUT) --out submissions/submission.csv

submit-randomforest:
	$(PY) -m src.random_forest predict --model $(TRAIN_OUT) --test data/rf/test.csv --out submissions/submission.csv

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check --isolated --select E4,E7,E9,F src tests
