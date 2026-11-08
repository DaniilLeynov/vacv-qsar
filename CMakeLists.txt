cmake_minimum_required(VERSION 3.16)
project(vacv_qsar NONE)

set(MODEL "catboost" CACHE STRING "Model for train/submit: catboost, randomforest, ensemble, dual or rfrobust")

find_package(Python3 REQUIRED COMPONENTS Interpreter)

if(WIN32)
  set(VENV_PYTHON "${CMAKE_SOURCE_DIR}/.venv/Scripts/python.exe")
else()
  set(VENV_PYTHON "${CMAKE_SOURCE_DIR}/.venv/bin/python")
endif()

if(EXISTS "${VENV_PYTHON}")
  set(_default_python "${VENV_PYTHON}")
else()
  set(_default_python "${Python3_EXECUTABLE}")
endif()
set(PYTHON "${_default_python}" CACHE FILEPATH "Python interpreter")

function(py_target name)
  add_custom_target(${name}
    COMMAND "${CMAKE_COMMAND}" -E env "PYTHONWARNINGS=ignore" "${PYTHON}" ${ARGN}
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    USES_TERMINAL)
endfunction()

add_custom_target(venv
  COMMAND "${Python3_EXECUTABLE}" -m venv --system-site-packages .venv
  COMMAND "${VENV_PYTHON}" -m pip install -r requirements.txt -r requirements-dev.txt
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  USES_TERMINAL)

add_custom_target(setup
  COMMAND "${CMAKE_COMMAND}" -E env "PYTHONWARNINGS=ignore" "${PYTHON}" -m pip install -r requirements.txt -r requirements-dev.txt
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  USES_TERMINAL)

py_target(benchmark -m src.benchmark)
py_target(test -m pytest -q)
py_target(list -m src.fetch --list)
py_target(data -m src.dataset --csv sources/selected.mapping.yaml --out data/common/dataset.csv --dropped data/processed/dropped.csv)
py_target(data-all -m src.build_all --out data/common/dataset_all.csv --measurements data/dual/all_measurements.csv --targets data/dual/all_targets.csv --ic50-targets data/dual/ic50_targets.csv --cc50-targets data/dual/cc50_targets.csv --dropped data/dual/all_dropped.csv)
py_target(chembl-pairs -m src.chembl_pairs --db data/raw/chembl/chembl.db --scope viral --exclude-smiles data/organizers/test.csv --out-dir data/processed/chembl)

if(MODEL STREQUAL "catboost")
  py_target(train -m src.train --data data/common/dataset.csv --out runs/${MODEL} --model catboost --mode exact --use-scaffold)
  py_target(submit -m src.predict --test data/organizers/test.csv --model runs/${MODEL} --out submissions/submission.csv)
elseif(MODEL STREQUAL "randomforest")
  py_target(train -m src.random_forest train --train data/rf/train.csv --out runs/${MODEL})
  py_target(submit -m src.random_forest predict --model runs/${MODEL} --test data/rf/test.csv --out submissions/submission.csv)
elseif(MODEL STREQUAL "ensemble")
  py_target(train -m src.ensemble train --train data/common/dataset.csv --organizers data/organizers/train.csv --out runs/${MODEL})
  py_target(submit -m src.ensemble predict --model runs/${MODEL} --test data/organizers/test.csv --out submissions/submission.csv)
elseif(MODEL STREQUAL "dual")
  py_target(train -m src.dual_model --ic50-targets data/dual/ic50_targets.csv --cc50-targets data/dual/cc50_targets.csv --organizers data/organizers/train.csv --test data/organizers/test.csv --out runs/${MODEL} --feature-set rf_bits_maccs --jobs 0 --group smiles --seed 42 --iterations 300 --fine-tune-iterations 100)
  py_target(submit -m src.dual_predict --model runs/${MODEL} --test data/organizers/test.csv --out submissions/submission.csv)
elseif(MODEL STREQUAL "rfrobust")
  py_target(train -m src.rf_robust train --train data/rf_robust/train_407.csv --out runs/${MODEL} --jobs 0 --clean)
  py_target(submit -m src.rf_robust predict --model runs/${MODEL} --test data/organizers/test.csv --out submissions/submission.csv --jobs 0)
else()
  message(FATAL_ERROR "Unknown MODEL '${MODEL}'.")
endif()
add_dependencies(submit train)
