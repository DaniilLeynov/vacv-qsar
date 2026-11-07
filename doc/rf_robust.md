# RobustForest + Ridge (модель коллеги)

Модель коллеги теперь лежит в репозитории как обычная часть проекта, без
отдельного `vendor/`-каталога.

## Файлы

- `src/rf_robust_best.py` — исходный алгоритм `RobustForest` из архива,
  без изменений;
- `src/rf_robust.py` — обёртка: подготовка тестовых признаков, CLI,
  OOF-метрики и параллельный запуск;
- `data/rf_robust/train_407.csv` — исходная обучающая таблица;
- `data/rf_robust/README.md` — исходное описание коллеги.

## Как работает

- RandomForest на 37 отобранных Morgan-битах и индикаторе BSC-40, метки SI
  ограничены `mean + 3*std`;
- Ridge на `log1p(SI)` по Morgan, RDKit-дескрипторам, признакам положения
  заместителей, культуре и штамму;
- если Ridge предсказывает SI выше 95-го перцентиля, берётся Ridge, иначе
  RandomForest.

Алгоритм и гиперпараметры не менялись: класс `RobustForest` импортируется
из `src/rf_robust_best.py` как есть.

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

| метрика | значение |
|---|---:|
| MSE(SI) | 169 861 |
| RMSE(SI) | 412 |
| MAE(SI) | 135.6 |
