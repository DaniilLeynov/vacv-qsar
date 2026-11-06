# Хакатон VACV — постановка задачи и рабочие материалы

Сводный документ по проекту `vacv-qsar`: официальная постановка задачи хакатона и полное содержимое рабочей страницы команды (план, задачи, ресурсы, выжимки и материалы по ролям).

- Соревнование: <https://www.kaggle.com/t/9150c85d954848faa7d298978c7a3a33>
- Исходное описание: страница «Хакатон VACV» (вкладки «План», «Выжимки», «По ролям»)
- Команда: 4 человека, все пишут на Python
- Стек: RDKit, pandas, scikit-learn, CatBoost / LightGBM, Optuna, Chemprop, SHAP

## Официальная постановка задачи

### Введение

Вирус оспы — один из наиболее опасных вирусов для человека. Смертность среди заражённых достигает 40%, а в связи с его большой вирулентностью его определяют как вирусный агент с наибольшей угрозой для человека. При этом считается, что последний известный случай заражения человека вирусом оспы произошёл в 1978 году. Для борьбы с распространением вируса оспы проводилась вакцинация с использованием vaccinia virus (VACV, VV), представителя семейства вирусов оспы, который был гораздо менее заразным и эффективно содействовал выработке иммунитета, что связано с генетической близостью всех существующих вариантов вируса оспы: иммунитет, который сформирован против одного из представителей семейства вирусов оспы, действует по отношению ко всем другим представителям этого семейства.

Несмотря на то, что в 1980 году было заявлено о победе над вирусом оспы, работы, направленные на борьбу с ним, вновь становятся актуальными по ряду причин. Во-первых, в силу наличия опасения о том, что вирус оспы может каким бы то ни было образом использоваться в качестве биологического оружия. Более того, для подобной биологической атаки не обязательно иметь образец настоящего вируса: достаточно, используя современные генные технологии, воссоздать его в лаборатории. Во-вторых, потому что новые методы лечения вируса оспы были бы полезны для лечения родственных вирусов, например вируса оспы обезьян. Эти вирусы становятся опасными для человека и в силу падения коллективного иммунитета против вируса оспы человека в связи с ранее объявленной победой над ним. В-третьих, потому что вирус оспы различных животных является опасным с точки зрения биологической безопасности человечества.

При этом разработки одной лишь только вакцины будет недостаточно для борьбы с последствиями возможного распространения вируса оспы, необходимо разрабатывать и лекарства, используя современные научные инструменты и методы. Кроме того, в современной практике борьбы с инфекционными заболеваниями подчёркивается необходимость сочетания вакцин и противовирусных препаратов.

### Постановка задачи

Вам необходимо разработать и обучить модель машинного обучения, которая будет на основании SMILES-записи молекулы определять значения её индекса селективности против вируса оспы (штамм VACV) — задача регрессии. Индекс селективности — это отношение полумаксимальной концентрации цитотоксичности к полумаксимальной концентрации ингибирования (SI = CC50 / IC50).

Для этого вам предоставлен набор данных об активности малых молекул против вируса оспы (штамма VACV или VV). Однако этот набор достаточно мал.

Потому в качестве решения задачи необходимо не только обучить модель, но и представить:

- скрипт загрузки данных из открытых баз, который будет содержать SMILES-запись, её значение IC50, CC50, SI и DOI работы (при наличии), в которой данные были опубликованы;
- сформированный набор данных, используемый для обучения.

> **Уточнение организаторов.** Для построения моделей и предсказания целевой
> переменной использовать колонки `CC50_uM` и `IC50_uM_VACV` нельзя: целевая
> колонка — это результат, который вычисляется математически из этих двух
> параметров. Они оставлены только для общего кругозора. В признаки идёт
> структура молекулы, а не измерения активности или токсичности.

Проверка моделей будет осуществляться с использованием Kaggle.

### Условия допуска к финальной защите

К финальной защите будет допущено не более (но возможно менее) пяти команд, результаты работы которых удовлетворяют следующим условиям:

- имеют лучший результат на Kaggle;
- представили исходный код в виде ссылки на GitHub;
- код содержит в том числе (но не только) скрипт создания модели и её проверки, скрипт выгрузки данных и их валидации с указанием ссылок, из которых происходит загрузка;
- собрали базу данных в указанном выше формате;
- полученный результат на Kaggle должен быть воспроизводимым: git-репозиторий должен содержать pipeline, который можно запустить, получить `.csv` и загрузить в Kaggle, и порядок метрики хотя бы будет таким же. В ином случае к питчу приглашается следующая команда по рейтингу.

### Бонус для участников

- Возможность принять участие в исследованиях на базе Лаборатории хемоинформатики и молекулярного моделирования ИИКС НИЯУ МИФИ.
- Поступление в профильную магистратуру НИЯУ МИФИ на базе соответствующей лаборатории.

### Ссылка на соревнование

<https://www.kaggle.com/t/9150c85d954848faa7d298978c7a3a33>

---

## Материалы страницы «Хакатон VACV»

*Ниже — полное содержимое рабочей страницы команды: чтение условия, пайплайн, девять групп задач, ресурсы и порядок изучения, выжимки по источникам и материалы по ролям. Оценки времени, пороги и толкования условия на этой странице — предложения команды, а не требования организаторов.*

*Хакатон · команда из 4*

### Бьёт по вирусу. Щадит клетку. Считаем заранее.

Модель машинного обучения по формуле соединения оценивает, подействует ли оно на вирус оспы (VACV) и не повредит ли клеткам человека.

*Задача*

#### Что просят сделать

Разработать модель машинного обучения, которая оценивает способность химических соединений воздействовать на вирус оспы (VACV) без поражения иных объектов в организме человека.

**Вход:** Не указан. Вероятно, таблица: SMILES и измерения активности и токсичности.

**Выход:** Оценка способности соединения действовать на VACV без вреда для клеток: число или класс.

**Метрика:** Не указана.

**Ограничения:** Не указаны: ни внешние данные, ни предобученные модели.

**Как мы читаем условие**

##### Скорее всего, это селективность

Слова «без поражения иных объектов» похожи на требование низкой токсичности для клеток. В лабораторных данных это пара IC50 (доза, подавляющая вирус) и CC50 (доза, убивающая клетки), а их отношение SI = CC50 / IC50 называют индексом селективности. Это наше допущение, его нужно подтвердить у организаторов.

**Тип**

##### Хемоинформатика

QSAR: зависимость «структура и свойство». Классификация или регрессия по молекулам.

**Темы**

##### Что нужно знать

SMILES, отпечатки и дескрипторы, scaffold split, градиентный бустинг, графовые сети, метрики при дисбалансе, SHAP.

**Стек**

##### Python у всех

RDKit, pandas, scikit-learn, CatBoost или LightGBM, Optuna, Chemprop, SHAP.

*Бейзлайн за 1–2 часа*

#### Пять шагов от таблицы до оценки

##### Данные и таргет

SMILES, IC50, CC50. Считаем SI и бинарную метку.

##### Очистка

Соли, заряды, дубликаты, единицы измерения.

##### Признаки

Отпечаток Моргана и дескрипторы RDKit.

##### Модели

Градиентный бустинг, затем графовая сеть Chemprop.

##### Проверка

Scaffold split, PR-AUC и ROC-AUC, SHAP.

Каркас бейзлайна: RDKit и CatBoost на scaffold-группах (не запускался)

```
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from catboost import CatBoostClassifier

df = pd.read_csv("data/raw.csv")            # smiles, ic50_uM, cc50_uM (названия колонок уточнить)
df["si"] = df["cc50_uM"] / df["ic50_uM"]
df["label"] = (df["si"] >= 10).astype(int)  # порог 10: наше допущение, подтвердить

chooser, unch = rdMolStandardize.LargestFragmentChooser(), rdMolStandardize.Uncharger()
def standardize(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = chooser.choose(rdMolStandardize.Cleanup(mol))
    return Chem.MolToSmiles(unch.uncharge(mol))

df["smiles_std"] = df["smiles"].map(standardize)
df = df.dropna(subset=["smiles_std"]).drop_duplicates("smiles_std")

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
X = np.stack(df["smiles_std"].map(lambda s: gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s))))
y = df["label"].values
groups = df["smiles_std"].map(lambda s: MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or "acyclic")

oof = np.zeros(len(df))                     # финальные разбиения выдаст роль 4, это временная схема
for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
    model = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05,
                               auto_class_weights="Balanced", verbose=0)
    model.fit(X[tr], y[tr])
    oof[va] = model.predict_proba(X[va])[:, 1]

print("ROC-AUC", roc_auc_score(y, oof), "PR-AUC", average_precision_score(y, oof))
```

Chemprop из командной строки (флаги сверены с документацией, не запускалось)

```
pip install chemprop

# SMILES и метка из нашего CSV, scaffold-разбиение внутри Chemprop
chemprop train --data-path data/clean.csv --task-type classification \
    --smiles-columns smiles_std --target-columns label \
    --split-type scaffold_balanced --epochs 50 --ensemble-size 3 \
    --output-dir runs/chemprop_cls

# когда роль 4 выдаст splits.csv, вместо --split-type:
#   --splits-column split
```

*Задачи*

#### Девять групп задач

У каждой группы есть подгруппы и ответственный. Подробный список подгрупп с источниками на вкладке «По ролям».

**Группа 1**

##### Постановка и данные

Роль 1

- формат данных и метрика
- таргеты: pIC50, SI, бинарная метка
- цензурированные значения и повторы

**Группа 2**

##### Подготовка и очистка

Роль 1

- валидность SMILES, соли, заряды
- дубликаты и противоречия
- единицы измерения, EDA, дисбаланс

**Группа 3**

##### Признаки

Роль 2

Роль 3

- отпечатки Моргана и MACCS
- дескрипторы RDKit
- предобученные эмбеддинги (MoLFormer)

**Группа 4**

##### Схема валидации

Роль 4

- scaffold и кластерные разбиения
- тест на далёких молекулах
- seed и файл splits.csv

**Группа 5**

##### Бейзлайны

Роль 2

- случайный лес, CatBoost, LightGBM, SVM
- отдельные модели активности и токсичности
- таблица результатов

**Группа 6**

##### Улучшение моделей

Роль 2

Роль 3

- Chemprop и multitask
- Optuna, веса классов, порог
- ансамбль и стекинг

**Группа 7**

##### Оценка качества

Роль 4

- ROC-AUC, PR-AUC, F1, MCC, RMSE
- повторы и доверительные интервалы
- область применимости

**Группа 8**

##### Интерпретируемость

Роль 4

- SHAP по бустингу
- подсветка подструктур
- химические выводы и разбор ошибок

**Группа 9**

##### Оформление решения

Роль 4

Роль 1

Роль 2

Роль 3

- один скрипт и функция predict(SMILES)
- слайды и демо
- ограничения и дальнейшие шаги

*Команда*

#### Четыре потока работы

Первые два часа все читают общий минимум. Потом расходимся по ролям и обмениваемся файлами в общем формате. Если у кого-то больше опыта с нейросетями, роли 2 и 3 можно поменять местами.

**Роль 1 · материалы роли**

##### Данные и химия

Превращают таблицу организаторов в чистые данные и понятный таргет.

- Постановка и таргет
- Очистка и стандартизация структур
- Анализ данных и внешние источники

**Роль 2 · материалы роли**

##### Признаки и классические модели

Строят признаки и быстрый сильный бейзлайн, с которым сравнивают всё остальное.

- Признаки
- Бейзлайны
- Подбор гиперпараметров
- Дисбаланс и порог

**Роль 3 · материалы роли**

##### Нейросети и представления

Графовая сеть Chemprop, предобученные модели и итоговый ансамбль.

- Chemprop (D-MPNN)
- Multitask и предобучение
- Ансамбль и неопределённость

**Роль 4 · материалы роли**

##### Валидация, интерпретация и питч

Честная схема проверки, метрики, объяснения и всё, что увидит жюри.

- Схема валидации
- Метрики и калибровка
- Интерпретация и область применимости
- Сборка, predict(SMILES) и питч

*Договорённости*

#### Что кто кому передаёт

**Роль 1 → все:** `data/clean.csv`: id, smiles_std, ic50, cc50, si, label. Отдать как можно раньше, даже в сыром виде.

**Роль 4 → все:** `splits.csv`: id и колонка split (train, val, test). В течение первых двух часов, дальше не меняется.

**Роль 2 → роль 4:** `preds/<модель>_val.csv` и `_test.csv`: id и вероятность или значение.

**Роль 3 → роль 4:** Предсказания Chemprop в том же формате. Ансамбль собираем на val.

**Все → роль 4:** Номер версии признаков и параметры модели: чтобы воспроизвести результат и объяснить его на защите.

*Защита*

#### Что показать жюри

##### Схема пайплайна

Одна картинка с пятью шагами и тем, что в них своё.

##### Честная валидация

Метрики на случайном и на scaffold-разбиении рядом: видно, насколько оценка падает на новых каркасах.

##### Таблица метрик

PR-AUC и ROC-AUC на val и test. Абляция: бейзлайн, затем Chemprop, затем ансамбль.

##### Объяснения

SHAP по признакам и подсветка подструктур у активных и безопасных молекул.

##### Область применимости

Расстояние до ближайшей обучающей молекулы против ошибки, и предупреждение в predict.

##### Воспроизводимость

Один скрипт запуска от CSV до файла результата и функция `predict(smiles)`.

*Ресурсы*

#### Всё найдено и открыто

Русские и английские материалы, с упором на код и работу с библиотеками. Метка «код» значит, что на странице есть команды или фрагменты кода. Время на изучение оценено нами. «Минимум» отмечает то, что нужно сделать в первые два часа.

##### Код и библиотеки

Документация, руководства и репозитории с кодом. Здесь больше всего того, что можно запустить.

- [RDKit. Getting Started with the RDKit in Python](https://rdkit.readthedocs.io/en/latest/GettingStartedInPython.html) — Официальное введение: чтение молекул, подструктуры, отпечатки и сходство, дескрипторы.
- [MolSSI. Introduction to RDKit](https://education.molssi.org/python-data-science-chemistry/rdkit_descriptors/rdkit.html) — Урок из курса Python for Data Science in Chemistry: молекулы из SMILES, дескрипторы, подструктуры.
- [useful_rdkit_utils. Fingerprints and Descriptors](https://useful-rdkit-utils.readthedocs.io/en/latest/descriptors.html) — Готовые функции Pat Walters: отпечатки из SMILES, дескрипторы в pandas, сравнение train и test по Tanimoto.
- [useful_rdkit_utils. Cross-validation and data splitting](https://useful-rdkit-utils.readthedocs.io/en/latest/split_utils.html) — Группы для разбиения: scaffold, кластеры Butina, KMeans, GroupKFold и вложенная кросс-валидация.
- [scikit-fingerprints. scaffold_train_test_split](https://scikit-fingerprints.readthedocs.io/latest/modules/generated/skfp.model_selection.scaffold_train_test_split.html) — Разбиение по scaffold Бемиса и Мерко одной функцией, с примером кода.
- [Pat Walters. practical_cheminformatics_tutorials (GitHub)](https://github.com/PatWalters/practical_cheminformatics_tutorials) — 41 ноутбук: QSAR за 8 строк, классификация и регрессия, кросс-валидация, кластеризация, Chemprop.
- [MolVS: Molecule Validation and Standardization](https://molvs.readthedocs.io/en/latest/) — Проверка и стандартизация молекул: соли, заряды, таутомеры.
- [CatBoost. Python quick start](https://catboost.ai/docs/en/concepts/python-quickstart) — Классификатор и регрессор CatBoost в нескольких строках, объект Pool.
- [Optuna Examples (README на GitHub)](https://github.com/optuna/optuna-examples/blob/main/README.md) — Индекс примеров Optuna для LightGBM, CatBoost, XGBoost и scikit-learn.
- [SHAP. TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) — TreeExplainer: объяснения для XGBoost, LightGBM, CatBoost и деревьев scikit-learn.
- [scikit-learn. average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score) — Средняя точность: метрика для редкого положительного класса.
- [PyPI. chemprop](https://pypi.org/project/chemprop/) — Установка Chemprop: версия, Python, лицензия.
- [Chemprop. Документация](https://chemprop.readthedocs.io/en/main/) — Руководства по CLI и Python, ноутбуки: multitask, дообучение, неопределённость, интерпретация.
- [Chemprop. CLI: обучение](https://chemprop.readthedocs.io/en/latest/tutorial/cli/train.html) — Команда chemprop train: типы задач, разбиения, эпохи, ансамбль.
- [Chemprop. Разбиение данных (Python)](https://chemprop.readthedocs.io/en/latest/tutorial/python/data/splitting.html) — Типы разбиений: scaffold_balanced, kmeans, kennard_stone и защита от дубликатов.

##### Данные и базы

Как получить и подготовить данные об активности, если организаторы разрешат внешние источники.

- [ChEMBL. Web services](https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services) — REST API ChEMBL и Python-клиент: как достать активности и структуры.
- [TeachOpenCADD T001. Данные из ChEMBL](https://projects.volkamerlab.org/teachopencadd/talktorials/T001_query_chembl.html) — Как скачать IC50 из ChEMBL, очистить и перевести в pIC50.

##### На русском

Русскоязычных материалов именно про QSAR и противовирусные соединения мало. Здесь лучшее из найденного: бустинг, графовые сети, метрики, программа курса.

- [Habr. Предсказание растворимости графовыми свёрточными сетями](https://habr.com/ru/post/565650) — Графовые сети против бустинга на дескрипторах на примере растворимости (ESOL).
- [Habr. Быстрый градиентный бустинг с CatBoost](https://habr.com/ru/post/527554) — Перевод обзора CatBoost: fit, cv, grid_search, параметры.
- [Habr. Борьба с дисбалансом классов. Стандартные методы](https://habr.com/ru/post/953158) — Метрики при дисбалансе, подбор порога, веса классов в бустинге.
- [Google ML Crash Course. ROC и AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc?hl=ru) — ROC и AUC на русском: как выбирать порог и когда нужны кривые точность-полнота.
- [Центральный университет. Программа курса «Хемоинформатика и машинное обучение»](https://static.centraluniversity.ru/documents/legal/education/programm/online-form/ii-v-biotehe/b1-o-dv-04-01-15-01-hemoinformatika-i-mashinnoe-obuchenie-3-sem.pdf) — Рабочая программа курса: темы, инструменты, базы данных.

##### Статьи и исследования

Теория: как валидировать, какие модели брать и что уже сделано для противовирусных соединений.

- [Bender et al. Evaluation guidelines for ML tools in the chemical sciences (2022)](https://research.chalmers.se/en/publication/530690) — Perspective в Nature Reviews Chemistry: как честно оценивать модели в химии.
- [Zheng et al. Beyond Scaffold Splits (arXiv 2607.10729)](https://arxiv.org/pdf/2607.10729) — Почему scaffold split недостаточен: проверка на редких и далёких молекулах.
- [Huang et al. Therapeutics Data Commons (arXiv 2102.09548)](https://arxiv.org/pdf/2102.09548) — Платформа TDC: наборы данных и типы разбиений.
- [Graff et al. Chemprop v2 (Cambridge Open Engage, 2025)](https://www.cambridge.org/engage/coe/article-details/68bb5b5b23be8e43d65c5cc5) — Chemprop v2: что нового и насколько быстрее.
- [Ross et al. MoLFormer (arXiv 2106.09553)](https://www.alphaxiv.org/abs/2106.09553v3) — MoLFormer: трансформер на SMILES, предобученный на 1,1 млрд молекул.
- [Xu et al. Selective antiviral agents from genomes and ML (Commun. Chem., 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12181400) — NCATS: модели избирательной противовирусной активности с учётом цитотоксичности.
- [Klimenko et al. DNA intercalating antiviral agents against VACV (2017)](https://dspace.kpfu.ru/xmlui/handle/net/129749) — Работа про вирус осповакцины: QSAR и фармакофор, отбор соединений с низкой цитотоксичностью.
- [Elton, Estep. Benchmarking open-source tools for antiviral discovery (arXiv 2605.04265)](https://arxiv.org/pdf/2605.04265) — Бенчмарк открытых инструментов для противовирусных препаратов, акцент на подготовке данных.
- [QSAR противовирусной активности на данных ChEMBL (PMC7162012)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7162012) — Модели процента ингибирования по вирусам из ChEMBL; осповакцина есть, но в малой доле.
- [Rolfe. Applicability domains in QSAR (Variational AI, 2024)](https://variational.ai/blog/applicability-domains-are-common-in-qsar-but-irrelevant-for-conventional-ml-tasks/) — Что такое область применимости и почему её порог условен.

*Порядок изучения*

#### Сначала минимум, потом всё остальное

**Минимум, около 2 часов**

Краткий пересказ каждого материала на вкладке «Выжимки».

1. [Google: ROC и AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc?hl=ru) — порог и метрики на русском
2. [RDKit: Getting Started](https://rdkit.readthedocs.io/en/latest/GettingStartedInPython.html) — SMILES, дескрипторы, отпечатки
3. [Pat Walters: ноутбуки](https://github.com/PatWalters/practical_cheminformatics_tutorials) — QSAR в 8 строк и сравнение моделей
4. [Chemprop: обучение из CLI](https://chemprop.readthedocs.io/en/latest/tutorial/cli/train.html) — первая команда и scaffold_balanced
5. [Xu et al. (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12181400) — как моделировать активность и токсичность отдельно

**Если останется время**

1. [TDC](https://arxiv.org/pdf/2102.09548) и [Beyond Scaffold Splits](https://arxiv.org/pdf/2607.10729) — почему scaffold split и что за ним
2. [Bender et al.](https://research.chalmers.se/en/publication/530690) — чек-лист честной оценки
3. [Habr: дисбаланс классов](https://habr.com/ru/post/953158) и [average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score) — порог и PR-AUC
4. [Optuna Examples](https://github.com/optuna/optuna-examples/blob/main/README.md) и [Habr: CatBoost](https://habr.com/ru/post/527554) — подбор параметров
5. [SHAP TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) и [область применимости](https://variational.ai/blog/applicability-domains-are-common-in-qsar-but-irrelevant-for-conventional-ml-tasks/) — объяснения и границы модели
6. [Chemprop: документация](https://chemprop.readthedocs.io/en/main/) и [Chemprop v2](https://www.cambridge.org/engage/coe/article-details/68bb5b5b23be8e43d65c5cc5) — multitask, дообучение, неопределённость
7. [Klimenko et al.](https://dspace.kpfu.ru/xmlui/handle/net/129749) и [Elton, Estep](https://arxiv.org/pdf/2605.04265) — VACV и подготовка данных
8. [TeachOpenCADD T001](https://projects.volkamerlab.org/teachopencadd/talktorials/T001_query_chembl.html) и [ChEMBL web services](https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services) — если разрешены внешние данные
9. [MoLFormer](https://www.alphaxiv.org/abs/2106.09553v3) и [Habr: графовые сети](https://habr.com/ru/post/565650) — эмбеддинги и графовые признаки

*Перед стартом*

#### Что уточнить у организаторов

##### Вопросы

- Какие колонки в данных и что считать целевой переменной: IC50, CC50, SI или готовый класс?
- Что значит «без поражения иных объектов»: токсичность для клеток или селективность по другим мишеням?
- Есть ли у тестовой выборки метки и как будут оценивать (метрика, формат сдачи)?
- Как записаны значения вида «>» и «<» и в каких единицах?
- Можно ли использовать внешние данные (ChEMBL и другие) и предобученные модели?
- Есть ли ограничения по вычислительным ресурсам и времени инференса?
- Нужна ли функция предсказания для новой молекулы или достаточно файла результатов?

##### Наши допущения

- Хакатон длится 24–48 часов.
- В команде четыре человека, все пишут на Python.
- GPU есть: своя или Colab и Kaggle.
- Метрика: ROC-AUC или PR-AUC для классификации, RMSE для регрессии.
- Селективным считаем SI ≥ 10, пока организаторы не скажут иначе.
- Данных немного (сотни или тысячи соединений), поэтому оценку надо делать повторами.

*Выжимки*

### Каждый материал за минуту

Для каждого источника три строки: суть, что взять в решение и на что обратить внимание. Это наш пересказ открытых страниц. Если в «Осторожно» отмечена неточность или ограничение, мы нашли их сами.

##### Код и библиотеки

Документация, руководства и репозитории с кодом. Здесь больше всего того, что можно запустить.

*Документация*

###### [RDKit. Getting Started with the RDKit in Python](https://rdkit.readthedocs.io/en/latest/GettingStartedInPython.html)

**Суть:** Официальное руководство RDKit. Разделы: чтение и запись молекул, работа с молекулами, поиск подструктур, отпечатки и молекулярное сходство, расчёт дескрипторов, реакции.

**Взять:** `Chem.MolFromSmiles` возвращает `None` для невалидной строки: на этом строится первая проверка данных. Дескрипторы: `Descriptors.MolLogP`, `Descriptors.TPSA`. Отпечаток Моргана: `AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)`, сходство через `DataStructs.FingerprintSimilarity` (по умолчанию Tanimoto).

**Осторожно:** По тексту страница относится к документации RDKit 2019.09.1 и показывает старый интерфейс `AllChem`, про `rdFingerprintGenerator` там нет. На странице useful_rdkit_utils для отключения предупреждений RDKit об устаревании показан `rdBase.BlockLogs()`. Сверяйтесь с документацией своей версии.

*Руководство*

###### [MolSSI. Introduction to RDKit](https://education.molssi.org/python-data-science-chemistry/rdkit_descriptors/rdkit.html)

**Суть:** Урок MolSSI (автор Jessica A. Nash, 2022): создание молекул из SMILES, подсчёт атомов, добавление водородов, дескрипторы `MolWt` и `NumAromaticRings`, стереохимия, поиск подструктур через `GetSubstructMatches`.

**Взять:** Самый мягкий вход для тех, кто не работал с RDKit. На нём удобно разобрать, что такое молекулярный граф: атомы и связи перебираются циклом.

**Осторожно:** Отпечатков, обучения моделей и работы с таблицами в уроке нет, это только основы RDKit.

*Документация*

###### [useful_rdkit_utils. Fingerprints and Descriptors](https://useful-rdkit-utils.readthedocs.io/en/latest/descriptors.html)

**Суть:** Справочник библиотеки useful_rdkit_utils 2.0.1: класс `Smi2Fp` (битовые, счётные и numpy-отпечатки), `RDKitDescriptors` (дескрипторы в DataFrame), `clean_and_scale_descriptors`, `Ro5Calculator`, `plot_properties`.

**Взять:** `compare_datasets` усредняет top-n сходств Tanimoto между обучающими и тестовыми отпечатками: быстрая оценка, насколько тест далёк от train. Подходит и для грубой проверки области применимости.

**Осторожно:** Это справочник без примеров, единственный фрагмент кода касается отключения предупреждений RDKit. Автор указан в копирайте страницы (Pat Walters, 2026).

*Документация*

###### [useful_rdkit_utils. Cross-validation and data splitting](https://useful-rdkit-utils.readthedocs.io/en/latest/split_utils.html)

**Суть:** Модуль `split_utils`: `GroupKFoldShuffle`, `get_scaffold`, `get_bemis_murcko_clusters`, `get_butina_clusters` (порог по умолчанию 0,65 по расстоянию 1 − Tanimoto), `get_kmeans_clusters`, `cross_validate` для вложенной кросс-валидации по нескольким моделям и способам группировки.

**Взять:** Получайте группы функцией кластеризации и подавайте их в `GroupKFoldShuffle`: каждая группа целиком попадает в одну часть. Так можно сравнить схемы (случайная, scaffold, Butina) на одной модели.

**Осторожно:** Примеров кода на странице нет, только сигнатуры. Порог Butina подберите под свои данные (рекомендация наша).

*Документация*

###### [scikit-fingerprints. scaffold_train_test_split](https://scikit-fingerprints.readthedocs.io/latest/modules/generated/skfp.model_selection.scaffold_train_test_split.html)

**Суть:** `scaffold_train_test_split`: детерминированное разбиение по каркасам Бемиса и Мерко, самые маленькие группы каркасов уходят в тест. Принимает SMILES или объекты RDKit, дополнительные массивы (например, метки) режутся вместе с ними.

**Взять:** Вызов вида `scaffold_train_test_split(smiles, train_size=0.8, test_size=0.2, return_indices=True)` даёт индексы, которые можно применить к таблице. Параметр `use_csk=True` обобщает каркас до циклического скелета (все атомы как углерод). В документации библиотеки есть также разделы про отпечатки, дескрипторы и область применимости.

**Осторожно:** Молекулы без колец не имеют каркаса и попадают в одну общую группу. Раз в тест уходят редкие каркасы, тест получается жёстче случайного (наш вывод из описания). Что за проект в целом, на странице прямо не описано.

*Репозиторий*

###### [Pat Walters. practical_cheminformatics_tutorials (GitHub)](https://github.com/PatWalters/practical_cheminformatics_tutorials)

**Суть:** Репозиторий Pat Walters (лицензия MIT), 41 ноутбук. В папке `ml_models`: «Build and Test a QSAR Model in 8 Lines of Python», построение и сравнение моделей классификации и регрессии, кросс-валидация. Отдельно: кластеризация (Butina, K-Means), анализ каркасов, два ноутбука по Chemprop.

**Взять:** Начните с ноутбука про QSAR в 8 строк, затем с сравнения моделей: это готовый шаблон, куда подставляются ваши данные. Ноутбуки по Chemprop пригодятся роли 3.

**Осторожно:** Содержание ноутбуков мы выводим по их названиям, сами ноутбуки не открывали. Отдельных руководств по очистке данных и разбиению в списке нет.

*Документация*

###### [MolVS: Molecule Validation and Standardization](https://molvs.readthedocs.io/en/latest/)

**Суть:** Инструмент на базе RDKit: валидация, нормализация функциональных групп, реионизация, нейтрализация зарядов, канонизация таутомеров, удаление солей и растворителей, получение «родительской» структуры.

**Взять:** Пример со страницы: `from molvs import standardize_smiles`, затем `standardize_smiles(...)` возвращает нормализованную строку. Страница задаёт хороший список того, что стоит проверить в данных.

**Осторожно:** Документация версии 0.1.1 с копирайтом 2016, о поддержке на странице ничего не сказано. Проверьте совместимость с вашей версией RDKit. В самом RDKit есть модуль `rdMolStandardize`, его документацию мы не открывали.

*Документация*

###### [CatBoost. Python quick start](https://catboost.ai/docs/en/concepts/python-quickstart)

**Суть:** Быстрый старт CatBoost: `CatBoostClassifier` (`fit`, `predict`, `predict_proba`) и `CatBoostRegressor` с `Pool`.

**Взять:** Каркас: `CatBoostClassifier(iterations=..., depth=..., learning_rate=..., loss_function='Logloss')`, затем `fit` и `predict_proba`. Для регрессии `loss_function='RMSE'`.

**Осторожно:** В примерах учебные значения (`iterations=2`, `depth=2`), для настоящей модели их надо менять. Установки, GPU и сохранения модели на странице нет.

*Репозиторий*

###### [Optuna Examples (README на GitHub)](https://github.com/optuna/optuna-examples/blob/main/README.md)

**Суть:** README репозитория optuna-examples со списком примеров. Градиентный бустинг: `lightgbm/lightgbm_simple.py`, `lightgbm/lightgbm_tuner_simple.py`, `catboost/catboost_simple.py`, `xgboost/xgboost_simple.py`. scikit-learn: `sklearn/sklearn_optuna_search_cv_simple.py` (`OptunaSearchCV`).

**Взять:** Откройте `catboost_simple.py` или `lightgbm_simple.py` и подставьте свои признаки. Метрику для оптимизации стоит брать PR-AUC на scaffold-фолдах (рекомендация наша). Есть варианты с ранней отсечкой неудачных попыток (pruning).

**Осторожно:** README это только список, сам код лежит в файлах. Рекомендаций по молекулярным данным в нём нет.

*Документация*

###### [SHAP. TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)

**Суть:** `shap.TreeExplainer` реализует Tree SHAP для ансамблей деревьев: XGBoost, LightGBM, CatBoost, PySpark и большинства деревьев scikit-learn.

**Взять:** Создать объясняющий объект по модели и вызвать `shap_values(X)` на нужных молекулах. Для режима interventional нужен фон примерно из 100–1000 объектов. Для отпечатков подпишите биты через `feature_names`, чтобы связать их с подструктурами (рекомендация наша).

**Осторожно:** Значения даны в пространстве выхода модели (для бинарного XGBoost это log-odds, не вероятность). Классификаторы scikit-learn возвращают значения по классам, XGBoost и LightGBM одно. Параметр `approximate` устарел и будет удалён в v0.49.0. Примеров кода на странице нет.

*Документация*

###### [scikit-learn. average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score)

**Суть:** Средняя точность (AP) как сводка кривой точность-полнота: сумма (Rn − Rn−1) · Pn по порогам.

**Взять:** `average_precision_score(y_true, y_score)` принимает оценки модели (вероятности), а не метки. Пример со страницы даёт 0,83.

**Осторожно:** AP не равна площади под PR-кривой, посчитанной по трапециям: линейная интерполяция даёт слишком оптимистичную оценку.

*Страница пакета*

###### [PyPI. chemprop](https://pypi.org/project/chemprop/)

**Суть:** Страница пакета chemprop на PyPI: версия 2.3.1 от 4 августа 2026, Python от 3.11 до 3.14, лицензия MIT.

**Взять:** `pip install chemprop`. Проверьте версию Python заранее, до начала хакатона.

**Осторожно:** Команд обучения на странице нет, они в документации (две следующие карточки).

*Документация*

###### [Chemprop. Документация](https://chemprop.readthedocs.io/en/main/)

**Суть:** Документация Chemprop 2.3.1, фреймворка на PyTorch для message-passing нейросетей. Руководства по CLI: обучение, предсказание, оптимизация гиперпараметров, отпечатки (fingerprint). Ноутбуки: многозадачные модели, дообучение с CheMeleon, оценка неопределённости, интерпретируемость, дополнительные признаки и дескрипторы, подбор гиперпараметров через RayTune или Optuna.

**Взять:** Для нашей задачи полезны ноутбуки про multitask (активность и токсичность вместе), extra features (подмешать дескрипторы RDKit) и uncertainty. Дообучение CheMeleon позволяет стартовать с предобученной модели, а не с нуля.

**Осторожно:** На главной странице нет команд и формата входных данных, лицензия не указана (MIT указан на PyPI). Что именно такое CheMeleon, по этой странице не ясно, читайте ноутбук.

*Руководство*

###### [Chemprop. CLI: обучение](https://chemprop.readthedocs.io/en/latest/tutorial/cli/train.html)

**Суть:** Руководство по обучению из командной строки. Базовая команда: `chemprop train --data-path ... --task-type ... --output-dir ...`. SMILES по умолчанию берутся из первого столбца, цели из остальных.

**Взять:** Классификация: `--task-type classification` (бинарные метки 0/1, потери BCE). Scaffold-разбиение: `--split-type scaffold_balanced`. Эпохи: `--epochs` (по умолчанию 50), ансамбль: `--ensemble-size`, повторы: `--num-replicates`. Свои разбиения: `--splits-column split` или `--splits-file splits.json`.

**Осторожно:** По умолчанию разбиение случайное 80/10/10, scaffold надо включить явно. Пример в документации использует тестовый файл из репозитория Chemprop.

*Руководство*

###### [Chemprop. Разбиение данных (Python)](https://chemprop.readthedocs.io/en/latest/tutorial/python/data/splitting.html)

**Суть:** Руководство по разбиению данных в Python-интерфейсе: типы `scaffold_balanced`, `random_with_repeated_smiles`, `random` (по умолчанию), `kennard_stone`, `kmeans` (реализация через библиотеку astartes). Функции `make_split_indices` и `split_data_by_indices`.

**Взять:** `random_with_repeated_smiles` держит повторяющиеся SMILES в одной части: полезно, если одно соединение измерено в нескольких опытах. `kmeans` даёт разбиение по структуре.

**Осторожно:** Примеры на странице построены на искусственных цепочках углерода и случайных значениях, не на реальных данных. Формат файла разбиений для CLI смотрите в руководстве по обучению.

##### Данные и базы

Как получить и подготовить данные об активности, если организаторы разрешат внешние источники.

*Документация*

###### [ChEMBL. Web services](https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services)

**Суть:** Документация веб-сервисов ChEMBL: ресурсы `activity`, `molecule`, `assay`, `target`, фильтры вида `[resource]?[field]__[filter_type]=[value]`, постраничная выдача, поиск по подструктуре и сходству. Клиент: `pip install chembl_webresource_client`.

**Взять:** Если организаторы разрешат внешние данные, можно собрать дополнительные измерения IC50 и CC50. Фильтр по типу активности строится по тому же шаблону (поле `standard_type`), но для IC50 и CC50 примера на странице нет, проверьте на живом API.

**Осторожно:** Кода в примерах нет, IC50 и CC50 на странице не фигурируют (пример с clearance). Упомянута версия ChEMBL_20, она устарела, актуальную показывает эндпоинт `status`. Выгрузок целиком на странице нет.

*Руководство*

###### [TeachOpenCADD T001. Данные из ChEMBL](https://projects.volkamerlab.org/teachopencadd/talktorials/T001_query_chembl.html)

**Суть:** Ноутбук TeachOpenCADD T001 (лаборатория Volkamer, Charité и FU Berlin): запрос к ChEMBL через `chembl_webresource_client` для мишени EGFR, отбор измерений IC50 с точным значением (relation «=») в нМ, удаление дубликатов, объединение с таблицей структур, сохранение в CSV на зафиксированной версии ChEMBL 27.

**Взять:** Формула `pIC50 = 9 − log10(IC50 в нМ)` и приём «оставить только relation «=» и единицы нМ» переносятся на наши данные. Для CC50 логика та же.

**Осторожно:** Пример для киназы EGFR, не для вирусов. Очистку структур ноутбук не делает, а отсылает к Standardiser и MolVS. Правило Липински вынесено в следующий ноутбук (T002).

##### На русском

Русскоязычных материалов именно про QSAR и противовирусные соединения мало. Здесь лучшее из найденного: бустинг, графовые сети, метрики, программа курса.

*Статья*

###### [Habr. Предсказание растворимости графовыми свёрточными сетями](https://habr.com/ru/post/565650)

**Суть:** Проект школьников СУНЦ МГУ с куратором из JetBrains Research (блог Питерской Вышки, 1 июля 2021): предсказание растворимости по SMILES. База ESOL из 1128 молекул, бейзлайн XGBoost на 2D-дескрипторах RDKit, затем графовая свёрточная сеть на PyTorch Geometric.

**Взять:** Готовая схема сравнения: сначала бустинг на дескрипторах, потом граф. Признаки атомов: атомный номер, число водородов, степень, заряд, хиральность, гибридизация, ароматичность, масса. Интерпретация по битам обученного отпечатка (гидроксильные и ароматические группы) пригодится как идея для защиты.

**Осторожно:** Это школьный проект про растворимость, не про противовирусную активность. Графовые сети оказались лишь немного лучше бустинга, а авторы сами пишут, что на малой выборке итог зависит от разбиения.

*Статья*

###### [Habr. Быстрый градиентный бустинг с CatBoost](https://habr.com/ru/post/527554)

**Суть:** Перевод статьи Derrick Mwiti с KDnuggets на Habr (блог OTUS, 11 ноября 2020). На примере `CatBoostRegressor`: обучение с графиками, кросс-валидация через `Pool` и `cv()`, `grid_search()`, `plot_tree()`, справочник параметров.

**Взять:** `grid_search()` и `cv()` как первый способ подобрать параметры до Optuna. Объяснение симметричных деревьев CatBoost пригодится для вопросов жюри.

**Осторожно:** Перевод 2020 года, про Optuna там нет. Пример для регрессии, для классификации придётся сменить функцию потерь.

*Статья*

###### [Habr. Борьба с дисбалансом классов. Стандартные методы](https://habr.com/ru/post/953158)

**Суть:** Статья Александра Приходько (блог KozhinDev, 3 октября 2025): accuracy, Precision, Recall, F1, ROC AUC и PR AUC при дисбалансе, подбор порога на валидации вместо 0,5, встроенные веса классов в LogisticRegression, XGBoost, LightGBM и CatBoost.

**Взять:** Выбирайте порог на валидационной части и сочетайте его с весами классов; для редких активных молекул смотрите PR AUC и F1 по миноритарному классу. Функция `optimize_threshold` перебирает пороги по квантилям вероятностей.

**Осторожно:** Все опыты на одном синтетическом наборе (100 тысяч объектов, 10 признаков), на реальных данных результат может отличаться. Подбор порога отнимает около 20% обучающих данных. Ресемплинг разобран в следующих частях серии, их мы не читали.

*Руководство*

###### [Google ML Crash Course. ROC и AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc?hl=ru)

**Суть:** Страница курса Google по машинному обучению: ROC-кривая показывает качество модели при всех порогах (TPR против FPR), AUC равна вероятности того, что случайный положительный пример получит оценку выше случайного отрицательного.

**Взять:** Порог выбирают по цене ошибки: если важнее не пропустить активное соединение, берут точку с большим TPR. При сильном дисбалансе страница советует кривые точность-полнота.

**Осторожно:** Общая страница без химии и кода. Раздел про PR-кривые помечен как необязательный.

*Программа курса*

###### [Центральный университет. Программа курса «Хемоинформатика и машинное обучение»](https://static.centraluniversity.ru/documents/legal/education/programm/online-form/ii-v-biotehe/b1-o-dv-04-01-15-01-hemoinformatika-i-mashinnoe-obuchenie-3-sem.pdf)

**Суть:** Рабочая программа курса «Хемоинформатика и машинное обучение» (Центральный университет, набор 2026): представление структур (SMILES, InChI, графы, 3D), дескрипторы, RDKit, построение и валидация QSAR и QSPR с принципами OECD, нейросети, кластеризация химического пространства, ADMET.

**Взять:** Список тем как чек-лист пробелов. Инструменты из программы: RDKit, Open Babel, Mordred, CDK, scikit-learn, DeepChem; базы: ChEMBL, PubChem, ZINC.

**Осторожно:** Это программа, а не учебник. Список литературы в ней общий по биотехнологии и молекулярной биологии, по хемоинформатике в нём ничего нет.

##### Статьи и исследования

Теория: как валидировать, какие модели брать и что уже сделано для противовирусных соединений.

*Статья*

###### [Bender et al. Evaluation guidelines for ML tools in the chemical sciences (2022)](https://research.chalmers.se/en/publication/530690)

**Суть:** Bender, Schneider, Segler, Walters, Engkvist, Rodrigues (Nature Reviews Chemistry, 2022, том 6, страницы 428–442). Авторы предлагают стандартизировать сравнение методов и дают чек-лист ретроспективных и проспективных проверок.

**Взять:** Основа для слайда «как мы проверяли»: чек-лист и принцип полной отчётности. Подходит для раздела об ограничениях.

**Осторожно:** Мы видели только аннотацию на странице Chalmers. Конкретные рекомендации по разбиениям, метрикам и статистическим тестам не проверены; полный текст по DOI 10.1038/s41570-022-00391-9.

*Препринт*

###### [Zheng et al. Beyond Scaffold Splits (arXiv 2607.10729)](https://arxiv.org/pdf/2607.10729)

**Суть:** Препринт «Beyond Scaffold Splits: Structural-Frontier Evaluation Reveals Hidden Failures in ADMET Models». Предлагают дополнительный тест на самых разреженных и физико-химически далёких группах каркасов. Медианная ошибка на таком тесте на 87% выше, чем на сопоставимом scaffold-тесте.

**Взять:** Идея для нашей валидации: помимо scaffold split добавить тест на наименее похожие молекулы и сообщать метрики по группам новизны. Сравнивать разбиения при одинаковой доле теста.

**Осторожно:** Данные из ADMET (BBB, HIA, CYP2D6, связывание с белками плазмы, растворимость, DILI), не противовирусные. Основная модель: MLP на отпечатках Моргана в 1024 бита. Дата публикации на странице не указана, рецензии нет.

*Статья*

###### [Huang et al. Therapeutics Data Commons (arXiv 2102.09548)](https://arxiv.org/pdf/2102.09548)

**Суть:** Huang, Fu, Gao и др.: платформа TDC с наборами данных и задачами для разработки лекарств (на момент статьи 66 наборов в 22 задачах), пять типов разбиений: случайное, по scaffold, cold-start, временное, комбинированное.

**Взять:** Обоснование scaffold split для питча: статья называет его более трудным, чем случайное, и более реалистичным. Python-пакет PyTDC даёт готовые разбиения.

**Осторожно:** Версия 2021 года, текущий API мог измениться, примеров кода в тексте нет. Противовирусных наборов под нашу задачу мы в тексте не проверяли.

*Препринт*

###### [Graff et al. Chemprop v2 (Cambridge Open Engage, 2025)](https://www.cambridge.org/engage/coe/article-details/68bb5b5b23be8e43d65c5cc5)

**Суть:** Graff et al., 15 сентября 2025: Chemprop переписан с нуля, модель D-MPNN, точность как у v1, примерно вдвое быстрее и втрое экономнее по памяти, масштабирование на несколько GPU, новые ноутбуки и документация.

**Взять:** Достаточно, чтобы объяснить жюри, что такое D-MPNN и почему выбрана именно эта реализация.

**Осторожно:** Рабочая версия без рецензирования. Установки и лицензии на странице нет (они на PyPI). Статью про первую версию мы открыть не смогли: страница потребовала проверку браузера.

*Препринт*

###### [Ross et al. MoLFormer (arXiv 2106.09553)](https://www.alphaxiv.org/abs/2106.09553v3)

**Суть:** Ross et al. (IBM Research): трансформер-энкодер с ротационными позиционными эмбеддингами и линейным вниманием, предобучение на примерно 1,1 млрд неразмеченных молекул из PubChem и ZINC, проверка на задачах MoleculeNet.

**Взять:** Кандидат для эмбеддингов: SMILES превращается в вектор, на нём обучают бустинг или линейную модель. На шести классификационных наборах MoLFormer-XL заметно лучше базовых в трёх и сопоставим в остальных.

**Осторожно:** Мы читали страницу-обзор alphaxiv, не саму статью. Выложены ли код и веса, на ней не указано, проверьте до начала использования.

*Статья*

###### [Xu et al. Selective antiviral agents from genomes and ML (Commun. Chem., 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12181400)

**Суть:** Xu, Xu, Zhang, Chen, Zheng, Huang (Communications Chemistry 8, 189, 2025). Модели предсказывают вирус-селективные и широкоспектральные противовирусные соединения: отпечатки ECFP4 на 1024 бита, Naive Bayes, SVM, нейросеть, случайный лес, XGBoost, консенсус RF и XGB. Отдельная модель цитотоксичности на контрольных соединениях Tox21.

**Взять:** Готовая идея под «действует на вирус без вреда прочему»: активность и токсичность моделируются отдельно, затем сводятся. Метрики: AUC-ROC, balanced accuracy, MCC; у вирус-селективных моделей RF даёт AUC-ROC 0,83, XGB 0,80. Входные данные и код на GitHub (TX-2017/antivirals_prediction и TX-2017/machine-learning).

**Осторожно:** Осповакцина (vaccinia) в статье не упоминается, натуральная оспа только в списке сокращений рисунка. Разбиения случайные 70/30 с 20 повторами; на scaffold-разбиении цифры, скорее всего, будут ниже (наше ожидание, не из статьи).

*Статья*

###### [Klimenko et al. DNA intercalating antiviral agents against VACV (2017)](https://dspace.kpfu.ru/xmlui/handle/net/129749)

**Суть:** Klimenko et al., 2017 (ISSN 0960-894X, запись репозитория КФУ): виртуальный скрининг DNA-интеркаляторов против VACV. Фармакофорную и QSAR-модель построили на данных про вирус везикулярного стоматита (VSV), отобрали из базы в 245 молекул 12 хитов, синтезировали и проверили против VACV; два соединения показали высокую активность при низкой цитотоксичности.

**Взять:** Прямой пример предметной области: отбор по активности против VACV вместе с оценкой цитотоксичности. Годится как контекст для вводной части питча.

**Осторожно:** Доступна только запись с аннотацией: структур, значений потенции и деталей модели там нет.

*Препринт*

###### [Elton, Estep. Benchmarking open-source tools for antiviral discovery (arXiv 2605.04265)](https://arxiv.org/pdf/2605.04265)

**Суть:** Elton, Estep (май 2026): сравнение docking-программ и ML-моделей по предсказанию аффинности. Собран набор из 43 005 измерений для пар вирусный белок и лиганд из BindingDB и других источников.

**Взять:** Главный вывод для нас: очистка данных важна не меньше выбора модели; дообучение на чистом противовирусном наборе подняло корреляцию с r≈0,5 до r≈0,7. В тексте упомянуты базы DrugVirus.info и AntiviralDB, где собраны значения CC50 и TC50.

**Осторожно:** Работа про связывание с белком, а не про IC50 и CC50 в клетках. Осповакцина в прочитанной части не упоминается; мы видели лишь фрагмент без приложений.

*Статья*

###### [QSAR противовирусной активности на данных ChEMBL (PMC7162012)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7162012)

**Суть:** Статья PMC7162012: процент ингибирования и SMILES из ChEMBL, около 18 000 дескрипторов и отпечатков (PaDEL), отбор признаков в Weka, SVM, 10-кратная кросс-валидация и независимая проверка, область применимости по графикам Вильямса. Есть общая модель по 26 вирусам.

**Взять:** Пример того, как собирают противовирусный QSAR-набор из ChEMBL. Для общей модели: Pearson r 0,71 в 10-fold CV, MCC 0,71 для классификации.

**Осторожно:** Название и авторов мы по тексту страницы подтвердить не смогли. IC50, CC50 и индекс селективности там не рассматриваются, цитотоксичность не оценивалась. Осповакцины в общем наборе мало: 12 и 22 соединения для записей vaccinia virus и vaccinia virus WR.

*Блог*

###### [Rolfe. Applicability domains in QSAR (Variational AI, 2024)](https://variational.ai/blog/applicability-domains-are-common-in-qsar-but-irrelevant-for-conventional-ml-tasks/)

**Суть:** Jason Rolfe, блог Variational AI, 14 сентября 2024. Область применимости это участок химического пространства около соединений с измеренной активностью; обычно её задают расстоянием Tanimoto на отпечатках Моргана до обучающей выборки с порогами 0,4 или 0,6.

**Взять:** Простая проверка: считать расстояние до ближайшего соседа из train и сравнивать с ошибкой модели. Автор показывает, что ошибка растёт с расстоянием для разных алгоритмов.

**Осторожно:** Блог компании, которая продаёт собственную модель, поэтому вывод о том, что границы условны, стоит взвешивать. Обещанные доказательства автор отнёс к будущему тексту.

Вернуться к плану

*По ролям*

### Что делать и читать каждому

Задачи каждой роли разбиты на подгруппы. Для каждой подгруппы есть теория (статьи и обзоры) и практика (документация, туториалы, репозитории с кодом). В первые два часа все читают общий минимум, потом каждый идёт в свой список.

*Роль 1*

#### Данные и химия

Превращают таблицу организаторов в чистые данные и понятный таргет.

**На выходе:** `data/clean.csv` (id, smiles_std, ic50, cc50, si, label) и журнал удалённых строк.

**Первые шаги**

1. Получить данные организаторов и открыть их в pandas.
2. Проверить SMILES через RDKit, посчитать долю невалидных.
3. Решить, что считать селективным соединением, и зафиксировать порог.
4. Передать `clean.csv` всем остальным как можно раньше, даже в сыром виде.

**Подгруппа 1.1**

##### Постановка и таргет

**Что сделать**

- Выяснить у организаторов формат данных, метрику и смысл фразы «без поражения иных объектов».
- Определить таргеты: pIC50, pCC50, SI = CC50 / IC50 и бинарную метку «селективен».
- Решить, как обращаться со значениями вида «>» и «<» и с повторными измерениями одного соединения.
- Зафиксировать порог для метки. Стартовое допущение: SI ≥ 10, подтвердить у организаторов.

Теория

- [Xu et al. Selective antiviral agents from genomes and ML (Commun. Chem., 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12181400) — Отдельная модель цитотоксичности рядом с моделью активности: подсказка, как строить таргет селективности.
- [Klimenko et al. DNA intercalating antiviral agents against VACV (2017)](https://dspace.kpfu.ru/xmlui/handle/net/129749) — Пример по VACV: активность и цитотоксичность оцениваются вместе.
- [Elton, Estep. Benchmarking open-source tools for antiviral discovery (arXiv 2605.04265)](https://arxiv.org/pdf/2605.04265) — Базы DrugVirus.info и AntiviralDB хранят CC50 и TC50; вывод про важность подготовки данных.

Код и библиотеки

- [TeachOpenCADD T001. Данные из ChEMBL](https://projects.volkamerlab.org/teachopencadd/talktorials/T001_query_chembl.html) — Формула pIC50 и фильтр «relation =», единицы нМ: переносятся на наши данные.

**Подгруппа 1.2**

##### Очистка и стандартизация структур

**Что сделать**

- Убрать невалидные SMILES, соли и растворители, нейтрализовать заряды.
- Найти дубликаты и противоречивые измерения одного соединения.
- Привести единицы измерения (нМ, мкМ) к одним.
- Вести журнал: сколько строк и почему удалено.

Теория

- [Elton, Estep. Benchmarking open-source tools for antiviral discovery (arXiv 2605.04265)](https://arxiv.org/pdf/2605.04265) — Очистка и разбор данных как главный источник выигрыша.
- [Центральный университет. Программа курса «Хемоинформатика и машинное обучение»](https://static.centraluniversity.ru/documents/legal/education/programm/online-form/ii-v-biotehe/b1-o-dv-04-01-15-01-hemoinformatika-i-mashinnoe-obuchenie-3-sem.pdf) — Чек-лист тем: представления структур, дескрипторы, принципы валидации.

Код и библиотеки

- [MolVS: Molecule Validation and Standardization](https://molvs.readthedocs.io/en/latest/) — Проверка и стандартизация молекул: соли, заряды, таутомеры.
- [RDKit. Getting Started with the RDKit in Python](https://rdkit.readthedocs.io/en/latest/GettingStartedInPython.html) — Проверка валидности через MolFromSmiles и работа с молекулой.
- [MolSSI. Introduction to RDKit](https://education.molssi.org/python-data-science-chemistry/rdkit_descriptors/rdkit.html) — Урок из курса Python for Data Science in Chemistry: молекулы из SMILES, дескрипторы, подструктуры.

**Подгруппа 1.3**

##### Анализ данных и внешние источники

**Что сделать**

- Распределения pIC50, pCC50, SI; баланс классов; выбросы.
- Размеры каркасов и кластеров: передать роли 4 для разбиений.
- Если организаторы разрешат, собрать дополнительные данные и убедиться, что они не пересекаются с тестом.
- Нарисовать пару молекул из каждого кластера для отчёта.

Теория

- [QSAR противовирусной активности на данных ChEMBL (PMC7162012)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7162012) — Как собирают противовирусный набор из ChEMBL и что делают с признаками.

Код и библиотеки

- [ChEMBL. Web services](https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services) — REST API ChEMBL и Python-клиент: как достать активности и структуры.
- [Pat Walters. practical_cheminformatics_tutorials (GitHub)](https://github.com/PatWalters/practical_cheminformatics_tutorials) — Ноутбуки про кластеризацию (Butina, K-Means) и анализ каркасов.
- [useful_rdkit_utils. Fingerprints and Descriptors](https://useful-rdkit-utils.readthedocs.io/en/latest/descriptors.html) — plot_properties и RDKitProperties для распределений.

*Роль 2*

#### Признаки и классические модели

Строят признаки и быстрый сильный бейзлайн, с которым сравнивают всё остальное.

**На выходе:** матрицы признаков в `features/` и предсказания каждой модели на val и test в `preds/`.

**Первые шаги**

1. Посчитать отпечаток Моргана (счётный, радиус 2) и дескрипторы RDKit для `clean.csv` .
2. Обучить CatBoost на scaffold-разбиении и записать метрики.
3. Сравнить три набора признаков одной и той же моделью.
4. Сохранить предсказания в общий формат для ансамбля.

**Подгруппа 2.1**

##### Признаки

**Что сделать**

- Отпечатки Моргана (биты и счётчики, радиус 2), MACCS.
- Дескрипторы RDKit с удалением констант и масштабированием.
- Сохранить признаки файлами, чтобы другие роли не пересчитывали.
- Сравнить наборы признаков одним и тем же бустингом.

Теория

- [Xu et al. Selective antiviral agents from genomes and ML (Commun. Chem., 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12181400) — Отпечатки ECFP4 на 1024 бита как рабочий стандарт для антивирусных моделей.
- [Habr. Предсказание растворимости графовыми свёрточными сетями](https://habr.com/ru/post/565650) — Бейзлайн на дескрипторах RDKit против графовой сети, по-русски.

Код и библиотеки

- [useful_rdkit_utils. Fingerprints and Descriptors](https://useful-rdkit-utils.readthedocs.io/en/latest/descriptors.html) — Готовые функции Pat Walters: отпечатки из SMILES, дескрипторы в pandas, сравнение train и test по Tanimoto.
- [RDKit. Getting Started with the RDKit in Python](https://rdkit.readthedocs.io/en/latest/GettingStartedInPython.html) — Официальное введение: чтение молекул, подструктуры, отпечатки и сходство, дескрипторы.
- [MolSSI. Introduction to RDKit](https://education.molssi.org/python-data-science-chemistry/rdkit_descriptors/rdkit.html) — Урок из курса Python for Data Science in Chemistry: молекулы из SMILES, дескрипторы, подструктуры.

**Подгруппа 2.2**

##### Бейзлайны

**Что сделать**

- Случайный лес, CatBoost, LightGBM, логистическая регрессия или SVM на одних признаках.
- Отдельные модели на IC50 и CC50 и модель на метку селективности: сравнить подходы.
- Фиксировать результаты на разбиениях роли 4.
- Записать таблицу: модель, признаки, метрики.

Теория

- [Xu et al. Selective antiviral agents from genomes and ML (Commun. Chem., 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12181400) — Сравнение RF, XGBoost, SVM, NN и Naive Bayes на антивирусных данных.

Код и библиотеки

- [CatBoost. Python quick start](https://catboost.ai/docs/en/concepts/python-quickstart) — Классификатор и регрессор CatBoost в нескольких строках, объект Pool.
- [Pat Walters. practical_cheminformatics_tutorials (GitHub)](https://github.com/PatWalters/practical_cheminformatics_tutorials) — Ноутбуки про построение и сравнение моделей классификации и регрессии.
- [Habr. Быстрый градиентный бустинг с CatBoost](https://habr.com/ru/post/527554) — Перевод обзора CatBoost: fit, cv, grid_search, параметры.

**Подгруппа 2.3**

##### Подбор гиперпараметров

**Что сделать**

- Optuna с целью PR-AUC на scaffold-фолдах.
- Ограничить время поиска и число попыток.
- Сохранять лучшие параметры в файл, чтобы повторить результат.

Код и библиотеки

- [Optuna Examples (README на GitHub)](https://github.com/optuna/optuna-examples/blob/main/README.md) — Индекс примеров Optuna для LightGBM, CatBoost, XGBoost и scikit-learn.
- [Habr. Быстрый градиентный бустинг с CatBoost](https://habr.com/ru/post/527554) — grid_search и cv как первый, более простой способ.

**Подгруппа 2.4**

##### Дисбаланс и порог

**Что сделать**

- Веса классов в бустинге.
- Порог по валидации, не по тесту.
- Сравнить PR-AUC, F1 и MCC до и после.

Теория

- [Google ML Crash Course. ROC и AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc?hl=ru) — ROC и AUC на русском: как выбирать порог и когда нужны кривые точность-полнота.
- [Habr. Борьба с дисбалансом классов. Стандартные методы](https://habr.com/ru/post/953158) — Метрики при дисбалансе, подбор порога, веса классов в бустинге.

Код и библиотеки

- [scikit-learn. average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score) — Средняя точность: метрика для редкого положительного класса.

*Роль 3*

#### Нейросети и представления

Графовая сеть Chemprop, предобученные модели и итоговый ансамбль.

**На выходе:** предсказания Chemprop на val и test в общем формате, при необходимости эмбеддинги молекул.

**Первые шаги**

1. Установить chemprop на машину с GPU или в Colab.
2. Обучить базовую модель на `clean.csv` , сначала со своим scaffold-разбиением.
3. Подключить разбиения роли 4 ( `--splits-column` ).
4. Сохранить предсказания в тот же формат, что у роли 2.

**Подгруппа 3.1**

##### Chemprop (D-MPNN)

**Что сделать**

- Установить пакет, проверить версию Python.
- Обучить классификацию и регрессию из командной строки.
- Подключить разбиения роли 4 вместо встроенных.
- Сравнить с лучшим бустингом роли 2 на тех же разбиениях.

Теория

- [Graff et al. Chemprop v2 (Cambridge Open Engage, 2025)](https://www.cambridge.org/engage/coe/article-details/68bb5b5b23be8e43d65c5cc5) — Chemprop v2: что нового и насколько быстрее.
- [Habr. Предсказание растворимости графовыми свёрточными сетями](https://habr.com/ru/post/565650) — Как устроены графовые признаки атомов, по-русски.

Код и библиотеки

- [PyPI. chemprop](https://pypi.org/project/chemprop/) — Установка Chemprop: версия, Python, лицензия.
- [Chemprop. Документация](https://chemprop.readthedocs.io/en/main/) — Руководства по CLI и Python, ноутбуки: multitask, дообучение, неопределённость, интерпретация.
- [Chemprop. CLI: обучение](https://chemprop.readthedocs.io/en/latest/tutorial/cli/train.html) — Команда chemprop train: типы задач, разбиения, эпохи, ансамбль.

**Подгруппа 3.2**

##### Multitask и предобучение

**Что сделать**

- Обучить модель на нескольких таргетах сразу (IC50 и CC50).
- Подмешать дескрипторы RDKit как дополнительные признаки.
- Попробовать дообучение предобученной модели (CheMeleon из документации).
- Эмбеддинги MoLFormer и бустинг на них, если веса доступны.

Теория

- [Ross et al. MoLFormer (arXiv 2106.09553)](https://www.alphaxiv.org/abs/2106.09553v3) — MoLFormer: трансформер на SMILES, предобученный на 1,1 млрд молекул.
- [Xu et al. Selective antiviral agents from genomes and ML (Commun. Chem., 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12181400) — Идея раздельных моделей активности и токсичности как альтернатива multitask.

Код и библиотеки

- [Chemprop. Документация](https://chemprop.readthedocs.io/en/main/) — Ноутбуки: multitask, extra features, transfer learning.
- [Pat Walters. practical_cheminformatics_tutorials (GitHub)](https://github.com/PatWalters/practical_cheminformatics_tutorials) — Два ноутбука по Chemprop, в том числе с дескрипторами RDKit.

**Подгруппа 3.3**

##### Ансамбль и неопределённость

**Что сделать**

- Ансамбль из нескольких моделей Chemprop.
- Усреднение или стекинг с моделями роли 2.
- Оценить неопределённость и сопоставить её с ошибкой.

Теория

- [Xu et al. Selective antiviral agents from genomes and ML (Commun. Chem., 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12181400) — Консенсус RF и XGB как образец сведения моделей.

Код и библиотеки

- [Chemprop. CLI: обучение](https://chemprop.readthedocs.io/en/latest/tutorial/cli/train.html) — Параметры --ensemble-size и --num-replicates.
- [Chemprop. Документация](https://chemprop.readthedocs.io/en/main/) — Ноутбук про неопределённость.

*Роль 4*

#### Валидация, интерпретация и питч

Честная схема проверки, метрики, объяснения и всё, что увидит жюри.

**На выходе:** `splits.csv` (в первые два часа), таблица метрик, рисунки SHAP, функция `predict(smiles)`, презентация.

**Первые шаги**

1. Получить `clean.csv` и сразу построить scaffold-разбиение и разбиение по кластерам.
2. Выдать всем `splits.csv` с колонкой split (train, val, test).
3. Написать функцию оценки: один вызов, одна таблица.
4. К концу первых суток подготовить скелет презентации.

**Подгруппа 4.1**

##### Схема валидации

**Что сделать**

- Scaffold-разбиение и разбиение по кластерам Butina; отдельный тест на самых далёких молекулах.
- Зафиксировать seed и выдать команде `splits.csv` .
- Проверить отсутствие общих молекул между частями (по канонической строке).
- Сравнить оценку на случайном и на scaffold-разбиении для слайда.

Теория

- [Huang et al. Therapeutics Data Commons (arXiv 2102.09548)](https://arxiv.org/pdf/2102.09548) — Платформа TDC: наборы данных и типы разбиений.
- [Zheng et al. Beyond Scaffold Splits (arXiv 2607.10729)](https://arxiv.org/pdf/2607.10729) — Почему scaffold split недостаточен: проверка на редких и далёких молекулах.
- [Bender et al. Evaluation guidelines for ML tools in the chemical sciences (2022)](https://research.chalmers.se/en/publication/530690) — Perspective в Nature Reviews Chemistry: как честно оценивать модели в химии.

Код и библиотеки

- [scikit-fingerprints. scaffold_train_test_split](https://scikit-fingerprints.readthedocs.io/latest/modules/generated/skfp.model_selection.scaffold_train_test_split.html) — Разбиение по scaffold Бемиса и Мерко одной функцией, с примером кода.
- [useful_rdkit_utils. Cross-validation and data splitting](https://useful-rdkit-utils.readthedocs.io/en/latest/split_utils.html) — Группы для разбиения: scaffold, кластеры Butina, KMeans, GroupKFold и вложенная кросс-валидация.
- [Chemprop. Разбиение данных (Python)](https://chemprop.readthedocs.io/en/latest/tutorial/python/data/splitting.html) — Типы разбиений: scaffold_balanced, kmeans, kennard_stone и защита от дубликатов.

**Подгруппа 4.2**

##### Метрики и калибровка

**Что сделать**

- ROC-AUC, PR-AUC, F1, MCC для классификации; RMSE, MAE и R² для регрессии.
- Повторы с разными seed и разброс метрик.
- Подбор порога на val и его перенос на test.
- Единая таблица результатов для всех моделей.

Теория

- [Google ML Crash Course. ROC и AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc?hl=ru) — ROC и AUC на русском: как выбирать порог и когда нужны кривые точность-полнота.
- [Habr. Борьба с дисбалансом классов. Стандартные методы](https://habr.com/ru/post/953158) — Метрики при дисбалансе, подбор порога, веса классов в бустинге.
- [Bender et al. Evaluation guidelines for ML tools in the chemical sciences (2022)](https://research.chalmers.se/en/publication/530690) — Perspective в Nature Reviews Chemistry: как честно оценивать модели в химии.

Код и библиотеки

- [scikit-learn. average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score) — Средняя точность: метрика для редкого положительного класса.

**Подгруппа 4.3**

##### Интерпретация и область применимости

**Что сделать**

- SHAP по лучшей модели на бустинге: какие биты и дескрипторы важны.
- Подсветка подструктур молекул для слайда.
- Расстояние до ближайшей обучающей молекулы против ошибки модели.
- Разбор нескольких крупных ошибок.

Теория

- [Rolfe. Applicability domains in QSAR (Variational AI, 2024)](https://variational.ai/blog/applicability-domains-are-common-in-qsar-but-irrelevant-for-conventional-ml-tasks/) — Что такое область применимости и почему её порог условен.
- [Habr. Предсказание растворимости графовыми свёрточными сетями](https://habr.com/ru/post/565650) — Пример связи битов отпечатка с подструктурами и химической интуицией.

Код и библиотеки

- [SHAP. TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) — TreeExplainer: объяснения для XGBoost, LightGBM, CatBoost и деревьев scikit-learn.
- [useful_rdkit_utils. Fingerprints and Descriptors](https://useful-rdkit-utils.readthedocs.io/en/latest/descriptors.html) — compare_datasets для расстояния между train и test.

**Подгруппа 4.4**

##### Сборка, predict(SMILES) и питч

**Что сделать**

- Один скрипт от CSV до файла предсказаний, фиксированные версии пакетов.
- Функция `predict(smiles)` : оценка активности и токсичности и пометка, если молекула далеко от обучающих.
- Слайды: постановка, схема, метрики, ошибки, ограничения.
- Репетиция демо на двух-трёх молекулах.

Теория

- [Klimenko et al. DNA intercalating antiviral agents against VACV (2017)](https://dspace.kpfu.ru/xmlui/handle/net/129749) — Контекст для вводной части: почему важны и активность, и безопасность.
- [Центральный университет. Программа курса «Хемоинформатика и машинное обучение»](https://static.centraluniversity.ru/documents/legal/education/programm/online-form/ii-v-biotehe/b1-o-dv-04-01-15-01-hemoinformatika-i-mashinnoe-obuchenie-3-sem.pdf) — Какие принципы валидации QSAR называют в курсах (OECD).

Код и библиотеки

- [Pat Walters. practical_cheminformatics_tutorials (GitHub)](https://github.com/PatWalters/practical_cheminformatics_tutorials) — Шаблоны ноутбуков для оформления результатов.
- [Chemprop. CLI: обучение](https://chemprop.readthedocs.io/en/latest/tutorial/cli/train.html) — Команда предсказания и сохранённые модели.

Время на изучение оценено нами. Краткий пересказ каждого материала на вкладке «Выжимки». Вернуться к плану

Страницу самого хакатона мы не нашли: план построен по описанию задачи со скриншота. Толкование «без поражения иных объектов» как селективности, порог SI ≥ 10 и распределение по ролям это наше предложение.
Фрагмент на Python и команды Chemprop не запускались: в рабочей среде не было доступа к менеджеру пакетов. Флаги Chemprop сверены с документацией версии 2.3.1, имена колонок в коде условные.
Выжимки написаны нами по открытым страницам. У Bender et al. и Klimenko et al. мы видели только аннотации, у Elton и Estep только фрагмент, у MoLFormer страницу-обзор. Ролики с видео мы не включали: проверенных материалов с известной длительностью не нашли.
Одну найденную страницу открыть не удалось, в список она не вошла. Время на изучение оценено нами.
