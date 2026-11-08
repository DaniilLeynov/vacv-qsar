# Данные по моделям

Данные разложены по папкам, чтобы было видно, что читает каждая модель.
Общие файлы лежат в `data/common/`.

## Структура

```
data/
  common/
    dataset.csv              # курируемая база для catboost + ensemble
    dataset_all.csv          # объединённая база organizers+ChEMBL+PubChem
    dataset_augmented.csv    # opt-in: dataset.csv + отфильтрованные новые строки
  rf/
    train.csv, test.csv      # randomforest (готовые признаки)
  rf_robust/
    train_407.csv            # RobustForest
  dual/
    ic50_targets.csv         # dual: IC50/EC50
    cc50_targets.csv         # dual: CC50
    all_targets.csv
    all_measurements.csv
    all_dropped.csv
  organizers/
    train.csv, test.csv, sample_submit.csv
  raw/                       # скачанные ChEMBL/PubChem, gitignore
  processed/                 # промежуточные выгрузки, gitignore
```

`data/dual/` генерируется командой `make data-all`, в git не попадает.

## Какая модель что читает

| модель | файл |
|---|---|
| catboost | `data/common/dataset.csv` |
| ensemble | `data/common/dataset.csv` |
| randomforest | `data/rf/train.csv` |
| rfrobust | `data/rf_robust/train_407.csv` |
| dual | `data/dual/ic50_targets.csv` + `data/dual/cc50_targets.csv` |

По умолчанию `make` не использует augmented-датасет: CatBoost и ensemble
учатся на исходном `data/common/dataset.csv`.

## Скачивание и парсинг

1. Пользователь скачивает ChEMBL SQLite (или PubChem assay) в `data/raw/`.
2. `make chembl-pairs` парсит ChEMBL в `data/processed/chembl/`.
3. `make data-all` собирает:
   - `data/common/dataset_all.csv` — полная база;
   - `data/dual/*` — таргеты для dual;
   - `data/common/dataset_augmented.csv` — opt-in для не-dual.

### Инкрементальный парсинг

Если появился новый raw-дамп, парсеры можно запустить с `--append`, чтобы
дописать новые строки к существующим CSV, а не перезаписывать их:

```bash
python -m src.chembl_pairs --db new.db --append
python -m src.chembl_export --csv new.csv --append
python -m src.build_all --append
```

`append` объединяет новые строки со старыми и делает `drop_duplicates`.
Для `build_all --append` таргеты и широкая база пересобираются из
объединённой measurement-таблицы.

## Augmented-датасет для не-dual

```bash
make data-nondual
# то же: python -m src.augment_common
```

Фильтры:

- SMILES валиден;
- `SI > 0` (`SI = 0` — отсутствие IC50/CC50, для регрессии не берём);
- `SI <= 10000` (отсекаем экстремальный хвост, который доминирует в MSE);
- нет дубля по каноническому SMILES с исходной базой;
- при дублях приоритет: `selected/organizers > chembl > paper > pubchem`.

Использование только через явный флаг:

```bash
make train MODEL=catboost TRAIN_DATA=data/common/dataset_augmented.csv
make train MODEL=ensemble TRAIN_DATA=data/common/dataset_augmented.csv
make TRAIN_DATA=data/common/dataset_augmented.csv
```

`make` без `TRAIN_DATA` работает как раньше — на `data/common/dataset.csv`.
