# VACV QSAR

Модель машинного обучения, которая по SMILES-записи молекулы предсказывает
индекс селективности против вируса оспы (штамм VACV):

```
SI VACV = CC50 / IC50
```

Задача — регрессия `SI VACV`; метрика — сырой MSE. `CC50_uM` и
`IC50_uM_VACV` используются только для расчёта SI/метки и никогда не
подаются в признаки.

## Быстрый старт

```bash
make venv
make train MODEL=rfrobust
make submit MODEL=rfrobust   # submissions/submission.csv
```

Финальная модель — `rfrobust` (RobustForest + Ridge). Обучение и инференс
не требуют сети: `data/rf_robust/train_407.csv` закоммичен.

## Бенчмарк

```bash
make   # == make benchmark
```

Обучает и сравнивает по OOF `mse_si`: `catboost`, `randomforest`,
`ensemble`, `dual`, `rfrobust`; лучшая пишется в `submissions/submission.csv`.
OOF-метрики считаются на разных наборах, поэтому сравнение ориентировочное.

## Данные

```
data/
  common/dataset.csv             # курируемая база catboost + ensemble (424 строки)
  common/dataset_all.csv         # organizers + ChEMBL + PubChem (623 строки)
  common/dataset_augmented.csv   # opt-in: база + отфильтрованные новые строки
  rf/                            # randomforest, готовые признаки
  rf_robust/train_407.csv        # RobustForest, таблица коллеги
  dual/                          # таргеты IC50/CC50 (генерируется make data-all)
  organizers/                    # train/test/sample_submit
  raw/                           # скачанные дампы (gitignore)
  processed/                     # промежуточные выгрузки (gitignore)
```

### data/rf_robust/train_407.csv

Подготовленная таблица из архива коллеги (`rf_robust_best.zip`): 407 строк —
319 официальных наблюдений организаторов и 88 дополнительных. Колонки:
`SI VACV`, `SMILES`, канонический SMILES, готовые признаки (Morgan 2048 бит с
хиральностью, RDKit-дескрипторы, признаки ароматических колец), а также
`culture_clean`, `strain_clean`, `official_train`. `SI VACV = 0` означает
отсутствие IC50/CC50.

## Выгрузка и парсинг внешних данных

Сеть нужна только на этапе выгрузки.

```bash
make chembl                              # скачать ChEMBL SQLite (~30 ГБ)
make chembl-pairs CHEMBL_DB=chembl_37.db # распарсить в data/processed/chembl
make data-all                            # собрать dataset_all.csv и data/dual/*
make data-nondual                        # собрать data/common/dataset_augmented.csv
```

PubChem assay скачиваются адаптером `src/sources/pubchem.py`
(`PubchemSource.fetch_raw`), повторный запуск продолжает с места остановки.

## Модели

```bash
make train MODEL=catboost
make train MODEL=randomforest
make train MODEL=ensemble
make train MODEL=dual
make train MODEL=rfrobust
make submit MODEL=<model>
make submit MODEL=<model> SUBMISSION=submissions/out.csv
```

### rfrobust

```bash
make train MODEL=rfrobust              # по умолчанию --clean
make train MODEL=rfrobust ROBUST_CLEAN=  # без очистки
```

`--clean`: снятие солей/гидратов и медианная агрегация реплик по
`(канонический SMILES, культура)`. Опционально `--neighbor-imputation`
(заполнение culture/strain по сходству Morgan) и `--tail-features`
(atom-pair + topological-torsion в Ridge-хвост).

### dual

```bash
make train MODEL=dual FEATURE_SET=rf_bits_maccs
```

Обучает `log10(IC50)` и `log10(CC50)` отдельно, затем
`SI = 10 ** (pred_cc50 - pred_ic50)`.

## Параллелизм

```bash
make train MODEL=dual JOBS=0
make train MODEL=rfrobust JOBS=0
```

`JOBS=0` — авто по числу ядер, `JOBS=1` — последовательно.

## Правила

- `CC50_uM` / `IC50_uM_VACV` и их производные — только для расчёта SI/метки,
  не в признаки (утечка таргета).
- `SI = 0` — «нет IC50/CC50», а не настоящий ноль.
- Тестовые молекулы исключаются из pretraining внешними данными.
- Единицы концентрации — мкМ.

## Проверки

```bash
make test
make lint
```

## Структура

```
AGENTS.md            # правила для контрибьюторов
README.md            # этот файл
Makefile             # команды
requirements.txt     # зависимости
src/                 # data, features, models, predict
data/                # данные по моделям
sources/             # маппинги источников
tests/               # pytest
```
