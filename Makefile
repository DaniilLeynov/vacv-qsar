PY ?= .venv/bin/python
SOURCES ?= sources
OUT ?= data/raw
VENV_FLAGS ?= --system-site-packages

.PHONY: venv chembl data data-organizers data-chembl datasets list fetch test lint

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

list:
	$(PY) -m src.fetch --list

fetch:
	$(PY) -m src.fetch --all --out $(OUT)

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check --isolated --select E4,E7,E9,F src tests
