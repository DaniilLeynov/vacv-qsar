# ChEMBL + PubChem и модель «IC50 + CC50 → SI»

Основной комбинированный датасет этого эксперимента:

- `data/dataset_all.csv` — широкий формат организаторов: `Title`, `CC50_uM`,
  `IC50_uM_VACV`, `SI VACV`, `SMILES`, `Culture`, `doi`, `source`;
- `data/processed/all_targets.csv` — длинный формат для dual-модели:
  `SMILES, endpoint, y_log10_um, source, ...`.

`data/dataset.csv` остаётся старым курируемым SI-набором для обратной
совместимости (`make data-selected` / `make data-augmented`).

Этот документ описывает эксперимент: обучать не сам SI, а две отдельные
величины, из которых он получается:

```
SI = CC50 / IC50          =>      log10(SI) = log10(CC50) - log10(IC50)
```

Так у модели появляется намного больше обучающих примеров: в ChEMBL есть
десятки тысяч измерений IC50 и CC50, тогда как готовых SI в таблице
организаторов всего несколько сотен.

Важно: `CC50_uM` и `IC50_uM_VACV` остаются **метками**, а не признаками.
Признаки — только структура (Morgan, MACCS, дескрипторы) и, опционально,
контекст assay (клеточная линия и организм). Это не нарушает правило о
недопустимости утечки целевой переменной.

## 1. Скачать ChEMBL

Нужен SQLite-дамп ChEMBL. На ноябрь 2026 актуальна версия 37; каталог
`latest` всегда указывает на текущую.

Прямая ссылка:

```bash
mkdir -p data/raw/chembl
cd data/raw/chembl

wget https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_37_sqlite.tar.gz
tar -xzf chembl_37_sqlite.tar.gz

# внутри будет chembl_37/chembl_37_sqlite/chembl_37.db
find . -name 'chembl_37.db'
```

Сжатый архив занимает около 5.4 ГБ, распакованная база — заметно больше.
Свободного места нужно порядка 20–30 ГБ.

Альтернатива через `chembl-downloader` (сам скачает и распакует):

```bash
make venv
make chembl            # python -m chembl_downloader download -v
```

`make chembl` печатает путь к `.db`. Его можно передать напрямую:

```bash
make chembl-pairs CHEMBL_DB=/home/user/.cache/chembl_downloader/chembl_37.db
```

Если файл лежит по умолчанию (`data/raw/chembl/chembl.db`), аргумент не нужен.

## 1a. Если уже есть CSV-выгрузки ChEMBL

Экспорт из веб-интерфейса ChEMBL (колонки `Smiles`, `Standard Type`,
`Standard Value`, `Standard Units`, `Assay Cell Type`, ...) можно разобрать
без SQLite-дампа:

```bash
make chembl-export \
    CHEMBL_IC="/path/IC_data.csv" \
    CHEMBL_EC="/path/EC_data.csv" \
    CHEMBL_CC="/path/CC_data.csv"
```

Или напрямую:

```bash
python -m src.chembl_export \
    --csv "/path/IC_data.csv" \
    --csv "/path/EC_data.csv" \
    --csv "/path/CC_data.csv" \
    --exclude-smiles data/organizers/test.csv \
    --out-dir data/processed/chembl_export
```

По умолчанию EC50 объединяется с IC50 в один endpoint `ic50`: в
противовирусных assay это две записи одной и той же потенции. Отключить:
`--no-ec50-as-ic50`. CC50 остаётся отдельным endpoint.

Важно: файлы `IC_data.csv` и `EC_data.csv` — это **потенция** (IC50/EC50),
а не цитотоксичность. Из них нельзя получить SI без CC50. Если CC50-выгрузки
нет, `src.dual_model` обучит CC50-голову только на CC50 из таблицы
организаторов (fallback); это уже не перенос с ChEMBL, но pipeline остаётся
рабочим и позволяет проверить саму идею.

Чтобы получить CC50 в веб-интерфейсе ChEMBL, ищите активности с
`Standard Type = CC50` (цитотоксичность; обычно `Assay Type = T` или `F`)
и выгружайте CSV с теми же колонками. Для VACV-модели полезны клеточные
линии, которые встречаются у организаторов: Vero, BSC-1, BSC-40, HEL, HFF,
HeLa, CV-1.

## 1b. PubChem через существующий адаптер

За скачивание отвечает `src/sources/pubchem.py` (`PubchemSource`). Он умеет
искать assay в PubChem, скачивать CSV и `summaries.json`, а затем разбирать
`IC50`/`EC50`, `CC50`/`TC50` и готовый `Ratio CC50/IC50`/`Selectivity Index`.
Данные кладутся в `data/raw/<имя-набора>/`.

Чтобы собрать не только VACV IC50, но и EC50/CC50, запускаем несколько
поисковых запросов:

```bash
python - <<'PY'
from pathlib import Path
from src.sources.pubchem import PubchemSource

terms = {
    "pubchem": "vaccinia AND IC50",
    "pubchem_ec50": "vaccinia AND EC50",
    "pubchem_cc50": "vaccinia AND CC50",
    "pubchem_vacv_ic50": "VACV AND IC50",
    "pubchem_vacv_ec50": "VACV AND EC50",
    "pubchem_vacv_cc50": "VACV AND CC50",
    "pubchem_orthopox_ic50": "orthopoxvirus AND IC50",
}
for name, term in terms.items():
    PubchemSource(term=term, delay=0.05, name=name).fetch_raw(Path("data/raw"))
PY
```

На ноябрь 2026 это дало 777 assay-файлов. После разбора:

- 2 550 endpoint-строк PubChem (IC50/EC50/CC50);
- 927 строк с готовым SI;
- в target-таблицу после агрегации и исключения теста попали
  735 IC50 и 138 CC50;
- в широкий датасет — 222 строки PubChem.

Важно: в PubChem часть assay отдаёт только `PubChem Standard Value` (всегда
в мкМ) без колонки `Standard Units`; адаптер теперь подставляет `uM`
автоматически. Иначе такие строки молча отбрасывались.

## 2. Разобрать дамп

```bash
make chembl-pairs CHEMBL_DB=/path/to/chembl_37.db
```

Команда `python -m src.chembl_pairs` пишет три таблицы в
`data/processed/chembl/`:

| файл | что внутри |
|---|---|
| `chembl_measurements.csv` | по одной строке на активность; сконвертировано в мкМ; отклонённые строки здесь же, с `dropped_reason` |
| `chembl_targets.csv` | по одной строке на молекулу × endpoint × контекст; готово для обучения двух регрессоров |
| `chembl_pairs.csv` | пары «VACV IC50 + CC50 на той же клеточной линии»; наблюдаемый SI |

Что делается при разборе:

- концентрации переводятся в мкМ (`nM`, `uM`, `mM`, `ug/mL` с молекулярной
  массой);
- по умолчанию берутся только точные значения (`relation = '='`);
  `--no-exact-only` отключает это ограничение;
- отбрасываются `potential_duplicate = 1`, записи с `data_validity_comment`,
  отличным от `Manually validated`, неположительные и выходящие за диапазон
  значения;
- VACV определяется по описанию assay, `assay_organism` и организму мишени
  (`vaccinia`, `vacv`, `variola`, `smallpox`);
- `--scope viral` (по умолчанию) оставляет IC50 только для вирусных assay,
  `--scope vacv` — только для VACV, `--scope all` — вообще все IC50;
- CC50 берутся всегда: это цитотоксичность, она нужна отдельной голове.

`make chembl-pairs` исключает молекулы теста организаторов из выгрузки
(`--exclude-smiles data/organizers/test.csv`). Это защита от утечки: иначе
модель могла бы увидеть в ChEMBL измерения для тех же молекул, что и в тесте.
При прямом вызове `python -m src.chembl_pairs` флаг нужно передать явно.

Полезные флаги:

```bash
python -m src.chembl_pairs --help

# только VACV IC50, плюс отладочное ограничение
python -m src.chembl_pairs --db ... --scope vacv --limit 100000

# исключить молекулы теста вручную (make chembl-pairs делает это по умолчанию)
python -m src.chembl_pairs --db ... \
    --exclude-smiles data/organizers/test.csv
```

## 3. Обучить две головы

```bash
make train MODEL=dual
# эквивалентно: две головы читают РАЗНЫЕ target-таблицы
python -m src.dual_model \
    --ic50-targets data/processed/ic50_targets.csv \
    --cc50-targets data/processed/cc50_targets.csv \
    --organizers data/organizers/train.csv \
    --test data/organizers/test.csv \
    --out runs/dual

# если данные пришли из CSV-экспорта и лежат одной таблицей:
python -m src.dual_model \
    --targets data/processed/chembl_export/chembl_targets.csv \
    --organizers data/organizers/train.csv
```

Внутри `src/dual_model.py` это две независимые модели: для `ic50`
берутся только строки `endpoint == "ic50"`, для `cc50` — только
`endpoint == "cc50"`. Отдельно сохраняются `base_ic50.cbm`,
`base_cc50.cbm`, `final_ic50.cbm`, `final_cc50.cbm` и фолдовые модели.
Общими у них остаются только признаки из SMILES и финальная формула
`SI = 10 ** (pred_cc50 - pred_ic50)`.

Что происходит:

1. Загружаются `chembl_targets.csv` и таблица организаторов.
2. Из pre-training автоматически удаляются SMILES из train и test
   организаторов. Иначе кросс-валидация на train была бы нечестной.
3. Для `ic50` и `cc50` отдельно обучается `CatBoostRegressor` на
   `log10(концентрация в мкМ)`. Считаются OOF-метрики на scaffold-фолдах.
4. Обе головы дообучаются на таблице организаторов (IC50 — 239 строк,
   CC50 — 318 строк). Для честной OOF-оценки дообучение делается по фолдам.
5. SI получается как `10 ** (pred_cc50 - pred_ic50)`.
6. На OOF-предсказаниях подбирается линейная калибровка в лог-пространстве:
   `log10(SI) = a * log10(pred_SI) + b`.

Артефакты в `runs/dual/`:

- `base_ic50.cbm`, `base_cc50.cbm` — модели, обученные только на ChEMBL;
- `ic50_fold*.cbm`, `cc50_fold*.cbm` — дообученные на 4/5 организаторов;
- `final_ic50.cbm`, `final_cc50.cbm` — дообученные на всех организаторах;
- `zero_fold*.cbm`, `final_zero.cbm` — классификатор «SI = 0» (IC50/CC50
  отсутствует); при `--no-zero-blend` не создаётся;
- `metrics.json` — OOF-метрики по endpoint'ам и по SI (raw, calibrated,
  capped, exact);
- `oof_si.csv` — предсказания по строкам train;
- `config.json` — контекст, калибровка, список признаков.

Основные флаги:

```bash
# без дообучения на организаторах: только перенос с ChEMBL
python -m src.dual_model ... --no-finetune

# добавить клеточную линию и организм как категориальные признаки
python -m src.dual_model ... --context

# отладка на маленькой подвыборке
python -m src.dual_model ... --max-molecules 5000 --iterations 200

# отключить классификатор SI = 0
python -m src.dual_model ... --no-zero-blend
```

## 4. Получить submission

```bash
make submit MODEL=dual
# или
python -m src.dual_predict --model runs/dual \
    --test data/organizers/test.csv \
    --out submissions/submission.csv
```

Предсказание усредняет все доступные дообученные модели
(`final_*` и `*_fold*.cbm`) и применяет калибровку из `config.json`.
Отключить калибровку: `--no-calibrate`.

После того как `chembl_targets.csv` появился, `make benchmark` автоматически
добавляет `dual` к списку сравниваемых моделей.

## 4a. Собрать общий датасет

```bash
make data-all
# эквивалентно:
python -m src.build_all \
    --out data/dataset_all.csv \
    --measurements data/processed/all_measurements.csv \
    --targets data/processed/all_targets.csv
```

`src.build_all` объединяет:

- `data/organizers/train.csv` — SI и IC50/CC50 организаторов;
- `data/processed/chembl/chembl_measurements.csv` — ChEMBL;
- `data/raw/pubchem*` — PubChem через `PubchemSource`.

Из теста организаторов молекулы исключаются по SMILES. Результат:

| таблица | строк |
|---|---:|
| `data/dataset_all.csv` | 623 |
| `data/processed/all_measurements.csv` | 55 248 |
| `data/processed/all_targets.csv` | 48 531 |
| `data/processed/ic50_targets.csv` | 1 594 |
| `data/processed/cc50_targets.csv` | 46 937 |
| `data/processed/all_dropped.csv` | 52 789 |

`all_dropped.csv` — это отброшенные строки **только для широкой SI-таблицы**
(стадия `pair`/`validate`). Они **не** выброшены из обучения голов:
например, 50 357 CC50 без IC50-пары остаются в `cc50_targets.csv`, а IC50
без CC50 — в `ic50_targets.csv`. Для обучения IC-головы CC50 вообще не нужен,
и наоборот.

В `dataset_all.csv`: 291 строка организаторов, 110 ChEMBL, 222 PubChem.
В `all_targets.csv`: 1 033 IC50 и 37 472 CC50 после исключения теста и
агрегации по молекулам.

После этого основная команда обучения:

```bash
make train MODEL=dual      # зависит от data-all, targets=all_targets.csv
make submit MODEL=dual

# модель коллеги RobustForest + Ridge
make train MODEL=rfrobust
make submit MODEL=rfrobust
```

`make` (benchmark) теперь сравнивает `catboost`, `randomforest`,
`ensemble`, `dual` и `rfrobust`. Метрики `rfrobust` считаются на 407 строках
из архива коллеги, а не на 320 строках организаторов, поэтому его лидерство
в benchmark — не строгое сравнение один в один.

## 5. Как это оценивать

Главная метрика — SI на OOF организаторов. В их таблице `SI VACV = 0`
означает не настоящий нулевой индекс, а отсутствие IC50 или CC50. Поэтому
`src/dual_model.py` обучает третий, вспомогательный классификатор «SI = 0» и
умножает отношение голов на `1 - P(SI = 0)`. В `metrics.json` есть группы:

- `zero` — строки с SI = 0;
- `positive` — строки с SI > 0;
- `blended` — все строки после классификатора нулей;
- `calibrated` — только калибровка в лог-пространстве, без нулей.

Один выброс `SI = 36066` в train даёт почти весь вклад в `MSE(SI)` группы
`positive`. Поэтому рядом с MSE всегда смотрите `RMSE(log1p)` и медианные
ошибки: по ним модель заметно устойчивее.

## 5a. Что получилось на двух файлах IC50/EC50

Проверка на `IC_data.csv` (417 строк) и `EC_data.csv` (640 строк):

- 1057 measurements, 780 валидных, 627 targets;
- EC50 объединён с IC50;
- 94 строки отброшены как `Outside typical range`;
- 181 строка исключена, потому что молекулы встречаются в тесте
  организаторов;
- CC50 в файлах нет, поэтому CC50-голова обучена только на 318 значениях
  организаторов; это предварительный результат.

OOF, 5 фолдов, seed 42, CatBoost:

| split | IC50 RMSE(log10) | SI MSE (blended) | SI RMSE | SI RMSE(log1p) |
|---|---|---|---|---|
| scaffold | 0.810 | 4 169 106 | 2 042 | 1.645 |
| smiles | 0.727 | 4 088 443 | 2 022 | 1.489 |

Отдельно по группам (split smiles):

| группа | n | MSE(SI) | RMSE(log1p) |
|---|---:|---:|---:|
| SI = 0 | 83 | 292 | 1.540 |
| SI > 0 | 237 | 5 520 159 | 1.471 |

Классификатор нулей на тесте организаторов даёт ROC-AUC ≈ 0.936 против
факта «IC50 не измерен» (это диагностика, в обучении тест не участвует).

Для сравнения, прямой CatBoost из `src/train.py` на `data/dataset.csv`
(exact-режим, split smiles) даёт OOF MSE ≈ 3.23 млн на 424 точных строках.
Это другая подвыборка, поэтому цифры нельзя сравнивать один в один; но
видно, что dual-модель уже конкурентоспособна, а её ошибки хорошо
разложены по группам.

## 5b. Полный ChEMBL 37 (SQLite)

После полной выгрузки `chembl_37.db` (30.5 ГБ) и команды

```bash
python -m src.chembl_pairs --db chembl_37.db --scope vacv \
    --exclude-smiles data/organizers/test.csv \
    --out-dir data/processed/chembl
```

получилось:

- 1 214 412 measurements, 51 352 валидных;
- 725 VACV IC50/EC50 targets (634 уникальные молекулы);
- 46 550 CC50 targets (37 477 молекул);
- 105 наблюдаемых пар VACV IC50 + CC50 (100 в одной публикации).

Dual-модель, 5 фолдов, split smiles, seed 42:

| показатель | значение |
|---|---|
| IC50 OOF RMSE(log10) | 0.713 |
| CC50 OOF RMSE(log10) | 0.608 |
| SI OOF MSE (blended) | 4 138 786 |
| SI OOF RMSE | 2 034 |
| SI OOF RMSE(log1p) | 1.571 |
| SI = 0, MSE | 163 |
| SI > 0, MSE | 5 588 177 |
| SI > 0 без выброса 36066, MSE | 23 755 |

Классификатор нулей на тесте организаторов: ROC-AUC ≈ 0.936 против факта
«IC50 не измерен» (диагностика; тест в обучении не участвует).

Прямой CatBoost на тех же 320 строках и том же split smiles
(`python -m src.train --data data/organizers/train.csv --mode reg
--target raw --group smiles`) даёт MSE 4 248 326, MAE 264 и MSE(log1p) 10.17
(RMSE(log1p) ≈ 3.19). Dual-модель чуть лучше по сырому MSE, заметно лучше по
MAE и в разы лучше по лог-ошибке. Основной вклад в сырой MSE даёт один
выброс `SI = 36066`; без него ошибка падает до 23 755.

## 5c. Организаторы + ChEMBL + PubChem

Dual-модель на `data/processed/all_targets.csv`, набор `all`, 5 фолдов,
split smiles (для сравнения с ablation ниже):

| показатель | значение |
|---|---|
| IC50 OOF, строк | 1 033 |
| IC50 RMSE(log10) | 0.710 |
| IC50 R² | 0.613 |
| CC50 OOF, строк | 37 472 |
| CC50 RMSE(log10) | 0.610 |
| CC50 R² | 0.299 |
| SI OOF MSE (blended) | 4 202 990 |
| SI OOF RMSE | 2 050 |
| SI OOF RMSE(log1p) | 1.553 |

По сравнению с ChEMBL-only IC50-голова стала заметно лучше по R²
(0.613 против 0.548) и получила больше строк (1 033 против 595). SI-метрика
почти не изменилась: основной вклад по-прежнему даёт один выброс
`SI = 36066`. Зато в широком датасете стало 623 SI-строки вместо 320, и
PubChem добавил 222 из них.

## 5d. Ablation признаков

Сравнить наборы признаков по SI OOF MSE:

```bash
make ablation
```

Скрипт `src/feature_ablation.py` по очереди обучает dual-модель на разных
наборах и пишет `runs/ablation/results.csv`. Доступные наборы
(`--feature-set`):

- `morgan` — Morgan radius 2, 2048 counts;
- `morgan3` — Morgan radius 3;
- `maccs` — MACCS 167;
- `descriptors` — 9 коротких RDKit-дескрипторов;
- `extended_descriptors` — все ~217 дескрипторов RDKit;
- `rdkit`, `atom_pair`, `torsion` — соответствующие отпечатки;
- `morgan_chiral` — Morgan radius 2 с хиральностью, бинарный;
- `rf_bits` — 37 Morgan-бит, отобранных в RobustForest коллеги;
- комбинации `morgan_maccs`, `morgan_descriptors`,
  `maccs_descriptors`, `rf_bits_maccs`, `rf_bits_all`;
- `all` — текущий набор (Morgan + MACCS + 9 дескрипторов);
- `all_fp` — Morgan + MACCS + AtomPair + Torsion + дескрипторы;
- `extended_all` — Morgan + MACCS + все дескрипторы;
- `all_context` — `all` плюс клеточная линия и организм.

Финальную модель можно обучить с выбранным набором:

```bash
make train MODEL=dual FEATURE_SET=morgan_maccs
```

### Результат ablation (300 итераций, 5 фолдов, split smiles)

| набор | IC50 RMSE | CC50 RMSE | SI MSE | SI MAE | SI RMSE(log1p) |
|---|---:|---:|---:|---:|---:|
| **rf_bits_maccs** | 0.704 | 0.627 | **4 043 986** | 196.5 | **1.465** |
| maccs | 0.713 | 0.625 | 4 125 450 | 201.0 | 1.488 |
| maccs_descriptors | 0.714 | 0.620 | 4 162 066 | 204.2 | 1.528 |
| extended_all | 0.704 | 0.603 | 4 170 486 | 200.5 | 1.529 |
| all_fp | 0.696 | 0.603 | 4 172 996 | 200.4 | 1.489 |
| atom_pair | 0.697 | 0.618 | 4 193 051 | 200.3 | 1.533 |
| all (было) | 0.710 | 0.610 | 4 202 990 | 203.3 | 1.553 |
| rdkit | 0.695 | 0.585 | 4 214 812 | 206.3 | 1.561 |

Проверка на seed 42/43/44 для трёх лучших:

| набор | MSE seed 42 | seed 43 | seed 44 | среднее |
|---|---:|---:|---:|---:|
| **rf_bits_maccs** | 4 043 985 | 4 122 099 | 4 127 356 | **4 097 813** |
| rf_bits_all | 4 131 960 | 4 097 979 | 4 111 541 | 4 113 827 |
| maccs | 4 125 450 | 4 149 459 | 4 195 109 | 4 156 673 |

Вывод: лучший набор по SI OOF MSE — **`rf_bits_maccs`**: 37 Morgan-бит,
отобранных в RobustForest коллеги, плюс MACCS. Он же лучший по
`RMSE(log1p)`; `rf_bits_all` даёт лучший MSE без главного выброса.
По умолчанию `make train MODEL=dual` теперь использует
`FEATURE_SET=rf_bits_maccs`.

## 5e. Параллелизм

`dual_model` умеет параллелить две вещи:

- генерацию признаков из SMILES (`ProcessPoolExecutor`);
- обучение CV-фолдов (каждый фолд — отдельный процесс CatBoost).

```bash
make train MODEL=dual              # JOBS=0 -> авто (все ядра)
make train MODEL=dual JOBS=1       # последовательно, для отладки
make train MODEL=dual JOBS=5       # 5 фолдов параллельно
```

При `--jobs 0` число процессов = `cpu_count`, а `thread_count` каждого
CatBoost = `cpu_count / jobs`. На 14 ядрах полный прогон MACCS-модели
(38 505 строк, 2 головы, 5 фолдов, 300 итераций) занимает ~8 секунд вместо
~26 секунд при `--jobs 1`. Метрики при этом идентичны.

Если признаки тяжёлые (`all_fp`, 6 320 признаков), матрица занимает больше
памяти; при нехватке RAM уменьшите `JOBS` (например, `JOBS=2`).

`rfrobust` тоже принимает `--jobs`: CV-фолды RobustForest обучаются
параллельно. Полный `make train MODEL=rfrobust` занимает несколько секунд
на 14 ядрах.

## 5e. Что с выбросом SI = 36066

Это не случайная флуктуация, а конкретная молекула `8e-as-555-15-16`:

```
CC50 = 1082 µM, IC50 = 0.03 µM, SI = 36066
```

В ChEMBL у того же SMILES стоит `IC50 = 0.03 nM` (в 1000 раз меньше), то есть
значение пришло из другого assay/единиц. Это ультра-селективное соединение,
и модель не экстраполирует в пикомолярную potency (предсказывает SI ≈ 670).

Проверка данных: если пометить 12 «подозрительных» IC50 организаторов
(цензурированные `1200 µM`, конфликты с ChEMBL) как выброшенные
(`--clean-ic50`), OOF MSE **ухудшается**: 4.20 млн против 4.13 млн. Поэтому
по умолчанию очистка выключена.

Проверка распределения теста: 99-й перцентиль предсказаний ≈ 500,
максимум ≈ 1 500, только 2 молекулы > 1000. Экстремального SI уровня 36 000
в тесте нет. Поэтому Kaggle-скор (~2 млн) ниже OOF (4.13 млн): выброс
раздувает только train OOF и почти не влияет на Kaggle.

Вывод: «брить» этот выброс не нужно — он валиден и на тест не переносится.
Для выбора моделей лучше смотреть `RMSE(log1p)` / winsorized MSE, а не
сырой OOF MSE.

## 6. Ограничения

- CC50 и IC50 могут быть измерены на разных клеточных линиях и в разных
  публикациях. Для `chembl_pairs.csv` пары строятся по клеточной линии;
  строгие пары (та же публикация) помечены `pair_type = same_doc`.
- IC50 в ChEMBL — это разные мишени и вирусы. `--scope vacv` даёт самый
  близкий к задаче набор, но он маленький; `--scope viral` — компромисс;
  `--scope all` — максимальный объём и максимальный сдвиг домена.
- Клеточные линии в ChEMBL и у организаторов называются по-разному
  (`HeLa`, `HELA`, `HeLa cells`). Контекст нормализуется, но не идеально;
  без `--context` модель вообще не использует эту информацию.
- Калибровка подбирается на OOF train. Если distribution теста заметно
  отличается, её лучше пересчитать или отключить.
- Сетевых запросов в обучении и инференсе нет: ChEMBL читается только на
  этапе `src.chembl_pairs`.

## 7. Быстрая проверка без полного дампа

Для дымового теста можно собрать маленькую SQLite-базу с теми же таблицами
(см. `tests/test_chembl_pairs.py`) и запустить:

```bash
python -m src.chembl_pairs --db /tmp/mini_chembl.db --out-dir /tmp/chembl
python -m src.dual_model --targets /tmp/chembl/chembl_targets.csv \
    --max-molecules 100 --iterations 20 --folds 3
```
