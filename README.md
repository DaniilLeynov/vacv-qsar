# VACV QSAR

**Команда:** три кота

Модель предсказывает индекс селективности молекулы против вируса оспы
(штамм VACV) по SMILES:

```
SI VACV = CC50 / IC50
```

Задача — регрессия `SI VACV`, метрика — сырой MSE. `CC50_uM` и
`IC50_uM_VACV` используются только для расчёта SI/метки и не подаются в
признаки.

## 1. Структура репозитория

```
Makefile                 команды обучения, бенчмарка и submission
requirements.txt         зависимости Python
src/
  features.py            Morgan, MACCS, RDKit-дескрипторы, наборы признаков
  normalize/common.py    единицы, цензурирование, стандартизация SMILES
  sources/               адаптеры источников (ChEMBL, PubChem, CSV)
  train.py               CatBoost-регрессия SI
  predict.py             инференс CatBoost
  random_forest.py       RandomForest на готовых признаках
  ensemble.py            бленд RandomForest (raw) + HistGB (log1p)
  dual_model.py          две головы IC50/CC50 → SI
  dual_predict.py        инференс dual-модели
  rf_robust.py           RobustForest + Ridge, CLI и подготовка данных
  rf_robust_best.py      реализация RobustForest
  chembl_pairs.py        парсинг ChEMBL SQLite
  chembl_export.py       парсинг CSV-выгрузок ChEMBL
  build_all.py           сборка organizers + ChEMBL + PubChem
  augment_common.py      opt-in augmented-датасет для не-dual моделей
  benchmark.py           сравнение моделей и выбор лучшей
  fetch/, merge.py       инфраструктура выгрузки и склейки
data/
  common/                общие и opt-in базы
  rf/                    данные randomforest
  rf_robust/             данные RobustForest
  dual/                  таргеты IC50/CC50 (генерируется)
  organizers/            train/test/sample_submit
  raw/, processed/       скачанные и промежуточные данные (gitignore)
sources/                 маппинги источников
tests/                   pytest
```

## 2. Данные и источники

Используются:

- таблица организаторов (`data/organizers/`);
- ChEMBL 37 — локальный SQLite-дамп, парсится в `data/processed/chembl/`;
- PubChem — assay через REST API, сохраняются в `data/raw/pubchem*/`.

Загрузка и парсинг:

```bash
make chembl                              # скачать ChEMBL SQLite
make chembl-pairs CHEMBL_DB=chembl_37.db # распарсить дамп
make data                                # курируемая база data/common/dataset.csv
make data-all                            # dataset_all.csv + data/dual/*
make data-nondual                        # opt-in augmented-датасет
```

Сеть используется только на этапе выгрузки. Обучение и инференс работают с
локальными файлами.

Правила:

- `CC50_uM` / `IC50_uM_VACV` и их производные — только метки, не признаки;
- `SI = 0` означает отсутствие IC50/CC50, а не настоящий ноль;
- молекулы теста исключаются из pretraining внешними данными;
- единицы концентрации — мкМ.

## 3. Подходы и модели

| модель | идея |
|---|---|
| `catboost` | CatBoost на Morgan + MACCS + RDKit-дескрипторах |
| `randomforest` | RandomForest на готовых признаках |
| `ensemble` | RandomForest на сыром SI + HistGB на `log1p(SI)` |
| `dual` | отдельные модели `log10(IC50)` и `log10(CC50)`, затем `SI = 10**(CC50−IC50)` |
| `rfrobust` | RandomForest с капом SI + Ridge на `log1p(SI)` для хвоста |

Для `rfrobust` дополнительно применяются:

- восстановление `culture_clean`/`strain_clean` из source records;
- снятие солей/гидратов и медианная агрегация реплик по
  `(канонический SMILES, культура)`;
- опционально neighbor-imputation условий и tail-признаки
  (atom-pair / topological torsion).

## 4. Эксперименты

Проверялись:

- наборы признаков для `dual`: Morgan, MACCS, RDKit-дескрипторы, их
  комбинации, `rf_bits` и хиральные Morgan; выбран `rf_bits_maccs`;
- очистка данных для `rfrobust`: снятие солей и медианная агрегация
  реплик; включена в финальную модель;
- альтернативные варианты `rfrobust`: neighbor-imputation условий,
  отдельный классификатор `SI = 0`, atom-pair/topological-torsion в хвосте.
  Эти варианты не вошли в финальную модель, так как ухудшали результат на
  тесте;
- полное расширение train ChEMBL/PubChem: отклонено из-за сдвига домена.

## 5. Финальный пайплайн

```bash
make venv
make train MODEL=rfrobust
make submit MODEL=rfrobust
# -> submissions/submission.csv
```

По умолчанию `train-rfrobust` использует `--clean` (снятие солей +
медианная агрегация). Отключить:

```bash
make train MODEL=rfrobust ROBUST_CLEAN=
```

Параллелизм: `JOBS=0` — авто, `JOBS=1` — последовательно.

## 6. Воспроизведение

Из чистого клона:

```bash
make venv
make train MODEL=rfrobust
make submit MODEL=rfrobust
make test
make lint
```

Данные `data/rf_robust/train_407.csv` закоммичены, поэтому обучение финальной
модели не требует сети. Для dual и расширенных экспериментов нужны
скачанные ChEMBL/PubChem (`make chembl`, `make chembl-pairs`, `make data-all`).

## 7. Локальные метрики

`make` (бенчмарк, OOF, 5 фолдов):

| модель | OOF MSE(SI) | RMSE | MAE |
|---|---:|---:|---:|
| **rfrobust** | **181 968.9** | **426.6** | **132.0** |
| ensemble | 3 168 204.5 | 1 779.9 | 195.5 |
| catboost | 3 231 454.8 | 1 797.6 | 258.3 |
| randomforest | 3 301 680.7 | 1 817.1 | 205.3 |
| dual | 4 043 985.5 | 2 011.0 | 196.5 |

OOF-метрики считаются на разных наборах, поэтому сравнение ориентировочное.
Сырой MSE сильно зависит от редких больших SI.

## 8. Ограничения

- Данных мало; хвост `SI > 1000` даёт основную часть ошибки.
- Внешние ChEMBL/PubChem данные имеют другой домен и шкалу SI.
- Публичный лидерборд чувствителен к хвосту; OOF-MSE не всегда отражает
  качество на тесте.
