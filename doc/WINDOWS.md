# Запуск на Windows

Проект на Python, поэтому нужен Python 3.10+ и CMake 3.16+. Компилятор не
нужен: CMake-проект объявлен без языка (`project(... NONE)`).

## Первый запуск

Из корня репозитория (PowerShell):

```powershell
py -3 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt

cmake -S . -B build
cmake --build build --target benchmark
```

Либо через цели CMake:

```powershell
cmake -S . -B build
cmake --build build --target venv
cmake -S . -B build            # перечитать созданный .venv
cmake --build build --target benchmark
```

## Модели

Модель выбирается при конфигурации:

```powershell
cmake -S . -B build -DMODEL=randomforest
cmake --build build --target submit
```

По умолчанию `MODEL=catboost`. Цели: `benchmark`, `test`, `data`,
`train`, `submit`, `list`, `venv`, `setup`.

`cmake --build build --target benchmark` обучает все модели, печатает
таблицу MSE и оставляет сабмит лучшей в `submissions/submission.csv`.
