# RobustForest + Ridge (модель коллеги)

Модель коллеги теперь лежит в репозитории как обычная часть проекта, без
отдельного `vendor/`-каталога.

## Файлы

- `src/rf_robust_best.py` — обновлённый алгоритм `RobustForest` из
  `rf_robust_restored.py`, включая восстановление условий;
- `src/rf_robust.py` — обёртка: подготовка тестовых признаков, source
  records из `data/dual/all_measurements.csv`, CLI, OOF-метрики и
  параллельный запуск;
- `data/rf_robust/train_407.csv` — исходная обучающая таблица;
- `data/rf_robust/README.md` — исходное описание коллеги.

## Как работает

- RandomForest на 37 отобранных Morgan-битах и индикаторе BSC-40, метки SI
  ограничены `mean + 3*std`;
- Ridge на `log1p(SI)` по Morgan, RDKit-дескрипторам, признакам положения
  заместителей, культуре и штамму;
- если Ridge предсказывает SI выше 95-го перцентиля, берётся Ridge, иначе
  RandomForest;
- **новое:** перед обучением недостающие `culture_clean`/`strain_clean`
  восстанавливаются из source records (точное совпадение молекулы и SI).
  Дополнительно можно включить голосование ближайших молекул по
  Morgan-сходству (`--neighbor-imputation`, threshold 0.7, agreement 0.8).

Гиперпараметры самого `RobustForest` не менялись.

## Запуск

```bash
make train MODEL=rfrobust
make submit MODEL=rfrobust
```

Прямой вызов:

```bash
python -m src.rf_robust train \
    --train data/rf_robust/train_407.csv \
    --out runs/rfrobust --jobs 0

python -m src.rf_robust predict \
    --model runs/rfrobust \
    --test data/organizers/test.csv \
    --out submissions/submission.csv --jobs 0
```

`--jobs 0` — авто: CV-фолды и подготовка тестовых признаков идут
параллельно. Оригинальный `RandomForest` внутри класса оставлен с
`n_jobs=1`; параллелизм вынесен на уровень фолдов.

## Метрики

5-фолдовая групповая CV по `canonical_smiles` на 407 строках:

| режим | MSE(SI) | RMSE | MAE |
|---|---:|---:|---:|
| без восстановления | 169 861 | 412.1 | 135.6 |
| **точное восстановление (default)** | **167 851** | **409.7** | **133.4** |
| + neighbor imputation 0.7/0.8 | 176 848 | 420.5 | 139.9 |

По умолчанию включено только точное восстановление из source records:
на наших данных neighbor-голосование ухудшает OOF. Вариант с соседями
воспроизводит подход коллеги и включается флагом
`--neighbor-imputation`.

Файл коллеги `submission_neighbors_070.csv` сохранён как
`submissions/submission_neighbors_070.csv` для сравнения. Наша
воспроизводимая версия даёт `submissions/submission.csv`; отличия от файла
коллеги — в 44 строках из 597, из-за другого набора source records и
нормализации условий (в архиве не было `category_rules.json` и
`smiles_si_merge_audit.csv`).
