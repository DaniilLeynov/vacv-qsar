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

`make data-all` использует локально скачанные ChEMBL/PubChem. Если их нет,
скрипт предупреждает и не перезаписывает существующую полную
`data/common/dataset_all.csv` меньшим набором; для сборки только из
organizers есть `--allow-local-only`. `make` пропускает `dual`, если
`data/dual/*.csv` отсутствуют, и не падает без внешних данных.

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

### 4.1 Сравнение моделей

Проверялись пять семейств: CatBoost, RandomForest, бленд RandomForest +
HistGB, dual-модель (IC50/CC50) и RobustForest. CatBoost и RandomForest
показали худшие OOF-метрики и не использовались как финальные. Бленд
улучшил их, но уступил RobustForest. Dual-модель проверялась как способ
увеличить объём обучения за счёт раздельных IC50/CC50; на organizer-OOF она
проиграла из-за хвоста SI. Финальной выбрана `rfrobust` — лучшие локальные
MSE/RMSE/MAE (раздел 7).

### 4.2 Признаки для dual-модели

Перебирались наборы: Morgan (counts), Morgan radius 3, MACCS, RDKit-
дескрипторы, расширенные дескрипторы, RDKit/AtomPair/Torsion-отпечатки,
хиральные Morgan и комбинации, включая `rf_bits`, `rf_bits_maccs`,
`rf_bits_all`, `all_fp` и контекст `cell_line`/`organism`.

Результаты:

- лучшим по среднему SI OOF-MSE оказался `rf_bits_maccs` — 37 Morgan-бит,
  отобранных в RobustForest, плюс MACCS;
- чистые отпечатки (Morgan, AtomPair, Torsion) и полный набор дескрипторов
  выступали хуже;
- контекст `cell_line`/`organism` не дал устойчивого улучшения;
- `rf_bits_all` был близок, но уступал по среднему MSE.

### 4.3 Данные для RobustForest

- **Восстановление условий из source records** — точное совпадение молекулы
  и SI. Оставлено в финальной модели.
- **Neighbor-imputation culture/strain** по Morgan-сходству (до 5 соседей,
  сходство ≥ 0.7, согласие ≥ 80%). Заполняет больше пропусков, но на наших
  source records ухудшает OOF; оставлено opt-in.
- **Снятие солей/гидратов и медианная агрегация реплик** по
  `(канонический SMILES, культура)`. Включено в финальную модель: локальный
  OOF-MSE при этом хуже, но результат на тесте лучше.
- **Отдельный классификатор `SI = 0`** с умножением регрессии на
  `1 − P(0)`. Проверялся, но заметно ухудшил результат на тесте; отклонён.
- **Atom-pair + topological-torsion в Ridge-хвосте** улучшили OOF на
  высоких SI, но ухудшили результат на тесте; отклонены.
- **Полное расширение train ChEMBL/PubChem** (`dataset_all.csv`, 623 строки).
  Отклонено: другой домен и шкала SI (до 157 026 против organizer-шкалы),
  ухудшение результата.
- **Кап/винзоризация хвоста** сверх уже имеющегося `mean + 3σ` в
  RandomForest не помогли; дополнительные ограничения ухудшали прогноз.
- **Строгие same_doc-пары ChEMBL** — пригодных пар мало, заметного эффекта
  нет.
- **Конфликтующие дубликаты** в `train_407.csv` (25 дублей canonical SMILES,
  12 молекул с разным SI) частично снимаются медианной агрегацией;
  отдельное удаление дало маргинальный эффект.

### 4.4 Вывод

Сырой OOF-MSE плохо предсказывает качество на тесте: основную часть ошибки
даёт редкий хвост `SI > 1000`. Поэтому решения принимались по совокупности
локальных метрик и результата на тесте, а не по одному OOF-MSE. В финальную
модель вошла `rfrobust` с очисткой данных; `CC50_uM`/`IC50_uM_VACV` теста не
использовались ни как признаки, ни как метки.

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
