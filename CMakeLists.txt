cmake_minimum_required(VERSION 3.16)
project(vacv_qsar NONE)

set(MODEL "catboost" CACHE STRING "Model for train/submit: catboost or randomforest")
set(PYTHON "${CMAKE_SOURCE_DIR}/.venv/bin/python" CACHE FILEPATH "Python interpreter")

function(py_target name)
  add_custom_target(${name}
    COMMAND ${CMAKE_COMMAND} -E env PYTHONWARNINGS=ignore ${PYTHON} ${ARGN}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    USES_TERMINAL)
endfunction()

py_target(benchmark -m src.benchmark)
py_target(test -m pytest -q)
py_target(list -m src.fetch --list)
py_target(data -m src.dataset --csv sources/selected.mapping.yaml --out data/dataset.csv --dropped data/processed/dropped.csv)
py_target(datasets -m src.dataset --csv sources/selected.mapping.yaml --out data/dataset.csv --dropped data/processed/dropped.csv)

if(MODEL STREQUAL "catboost")
  py_target(train -m src.train --data data/dataset.csv --out runs/${MODEL} --model catboost --mode exact --use-scaffold)
  py_target(submit -m src.predict --test data/organizers/test.csv --model runs/${MODEL} --out submissions/submission.csv)
elseif(MODEL STREQUAL "randomforest")
  py_target(train -m src.random_forest train --train data/rf/train.csv --out runs/${MODEL})
  py_target(submit -m src.random_forest predict --model runs/${MODEL} --test data/rf/test.csv --out submissions/submission.csv)
else()
  message(FATAL_ERROR "Unknown MODEL '${MODEL}'. Use catboost or randomforest.")
endif()
add_dependencies(submit train)
