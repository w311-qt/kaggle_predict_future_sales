# Predict Future Sales

Домашние задания по курсу R&D в ML. Задача с Kaggle: [Predict Future Sales](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales). По дневным продажам сети 1С с января 2013 по октябрь 2015 нужно спрогнозировать, сколько штук каждого товара продаст каждый магазин в ноябре 2015. Метрика RMSE, таргет обрезается в [0, 20].

Каждое ДЗ сдаётся отдельным pull request и описано в своём README: что сделано, как запустить, результаты и выводы.

| ДЗ | что сделано | README | результат, RMSE на backtest 29-33 |
|---|---|---|---|
| 1 | постановка, EDA, baseline, протокол валидации | [hw1/README.md](hw1/README.md) | LightGBM на 15 лагах 0.778 ± 0.057 |

## Установка и данные

Нужен Python 3.13 и около 16 ГБ RAM. Команды запускаются из корня репозитория.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Данные берутся со [страницы соревнования](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/data): сначала принять правила, потом распаковать архив прямо в `data/`. Нужны `sales_train.csv`, `test.csv`, `items.csv`, `item_categories.csv`, `shops.csv`. Через Kaggle CLI:

```bash
kaggle competitions download -c competitive-data-science-predict-future-sales -p data
unzip data/competitive-data-science-predict-future-sales.zip -d data
```

Быстрая проверка, что всё работает (~5 секунд, 5% товаров):

```powershell
.\.venv\Scripts\python.exe smoke_test.py
```

Команды полного запуска для каждого ДЗ описаны в его README.

## Структура

```text
hw1/README.md           ДЗ1: постановка, baseline, протокол оценки
configs/baseline.yaml   всё настраиваемое: очистка, сетка, лаги, фолды, параметры LightGBM
src/data.py             загрузка, очистка, месячная сетка магазин × товар
src/features.py         лаговые признаки и срезы A/B/C
src/train.py            baseline: три модели на одних фолдах, сабмит
src/evaluate.py         метрики, срезы, кластерный bootstrap, графики
src/experiments.py      исследовательские вопросы ДЗ1
smoke_test.py           быстрый прогон пайплайна
docs/                   разбор постановки
notebooks/eda.ipynb     EDA
reports/                результаты и графики
```
