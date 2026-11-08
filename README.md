# VACV QSAR

Предсказание индекса селективности `SI VACV = CC50 / IC50` по SMILES-записи
молекулы. Соревнование: <https://www.kaggle.com/t/9150c85d954848faa7d298978c7a3a33>.

Основная метрика — сырой MSE на `SI VACV`. Данных мало, поэтому половина
решения — собранная из открытых источников база и честная валидация.

## Быстрый старт

```bash
# 1. Виртуальное окружение + зависимости
make venv
# или в уже созданное окружение:
make deps

# 2. Лучшая модель (RobustForest, salt-strip) — обучение и submission
make train MODEL=rfrobust
make submit MODEL=rfrobust
# -> submissions/submission.csv
```

Это воспроизводимо из чистого клона: `data/rf_robust/train_407.csv`
закоммичен, сеть на шагах обучения/инференса не нужна. Текущий лучший
Kaggle-скор этой модели — **816 071**.

## Бенчмарк всех моделей

```bash
make            # == make benchmark
```

Обучает и сравнивает по OOF `mse_si`:

- `catboost`
- `randomforest`
- `ensemble`
- `dual` (IC50/CC50 по отдельности)
- `rfrobust`

Лучшая модель копируется в `submissions/submission.csv`.

Важно: OOF-метрики моделей считаются на **разных** наборах (см. ниже), поэтому
сравнение — ориентир, а не строгий бенчмарк. Итоговый выбор — `rfrobust`.

## Данные

```
data/
  common/
    dataset.csv            # курируемая база для catboost + ensemble (424 строки)
    dataset_all.csv        # organizers + ChEMBL + PubChem (623 строки)
    dataset_augmented.csv  # opt-in: dataset.csv + отфильтрованные новые строки
  rf/                      # randomforest (готовые признаки)
  rf_robust/train_407.csv  # RobustForest (таблица коллеги)
  dual/                    # две головы IC50/CC50 (генерируется make data-all)
  organizers/              # train/test/sample_submit
  raw/                     # скачанные дампы (ChEMBL/PubChem) — gitignore
  processed/               # промежуточные выгрузки — gitignore
```

Какая модель что читает:

| модель | файл обучения |
|---|---|
| catboost | `data/common/dataset.csv` |
| ensemble | `data/common/dataset.csv` |
| randomforest | `data/rf/train.csv` |
| rfrobust | `data/rf_robust/train_407.csv` |
| dual | `data/dual/ic50_targets.csv` + `cc50_targets.csv` |

## Выгрузка и парсинг внешних данных

Только на этапе выгрузки нужна сеть. ChEMBL качается одним большим дампом,
PubChem — пачкой assay.

```bash
# ChEMBL: скачать и распаковать SQLite (около 30 ГБ)
make chembl

# распарсить дамп в data/processed/chembl
make chembl-pairs CHEMBL_DB=chembl_37.db

# собрать расширенную базу и dual-таргеты
make data-all

# только opt-in датасет для не-dual моделей
make data-nondual
```

PubChem assay скачиваются адаптером `src/sources/pubchem.py`
(`PubchemSource.fetch_raw`). Повторный запуск продолжает с места остановки.

## Модели

```bash
make train MODEL=catboost
make train MODEL=randomforest
make train MODEL=ensemble
make train MODEL=dual
make train MODEL=rfrobust
```

Submission:

```bash
make submit MODEL=<model>            # пишет в submissions/submission.csv
make submit MODEL=<model> SUBMISSION=submissions/out.csv
```

### RobustForest (финальная)

- `--clean` включён по умолчанию (`ROBUST_CLEAN ?= --clean`): снятие
  солей/гидратов + медианная агрегация реплик по `(SMILES, культура)`.
- `--neighbor-imputation` — заполнение пропущенных `culture/strain` по
  сходству Morgan (по умолчанию выключено).
- `--tail-features` — atom-pair + topological-torsion в Ridge-хвост
  (эксперимент, на Kaggle не помог).

```bash
make train MODEL=rfrobust ROBUST_CLEAN=
```

### Dual

Обучает `log10(IC50)` и `log10(CC50)` отдельно, затем
`SI = 10 ** (pred_cc50 - pred_ic50)`.

```bash
make train MODEL=dual FEATURE_SET=rf_bits_maccs
```

## Параллелизм

```bash
make train MODEL=dual JOBS=0
make train MODEL=rfrobust JOBS=0
```

`JOBS=0` — авто по числу ядер, `JOBS=1` — последовательно.

## Правила и ограничения

- `CC50_uM` и `IC50_uM_VACV` (и их производные) **нельзя** использовать в
  признаках — только для расчёта SI/метки. Это утечка таргета.
- `SI = 0` в таблице организаторов означает «нет IC50/CC50», а не настоящий
  ноль.
- Тестовые молекулы исключаются из pretraining внешними данными.
- Единицы концентрации — мкМ.

## Проверки

```bash
make test     # pytest
make lint     # ruff
```

## Структура репозитория

```
AGENTS.md            # правила для контрибьюторов
Makefile             # основные команды
requirements.txt     # зависимости
src/                 # код: data, features, models, predict
doc/                 # постановка и рабочие материалы
data/                # данные по моделям
sources/             # маппинги источников
tests/               # pytest
```

Подробнее: `doc/chembl.md`, `doc/rf_robust.md`, `doc/data-layout.md`,
`doc/vacv-qsar.md`.
