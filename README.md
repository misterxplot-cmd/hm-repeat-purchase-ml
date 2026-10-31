# hm-repeat-purchase-ml

Reproducible customer repeat-purchase prediction on H&M transaction history with PostgreSQL, temporal validation,
logistic regression and CatBoost.

Таблицы между маркерами `<!-- AUTO:… -->` генерирует `hmrp report` из сохранённых артефактов run; вручную
они не редактируются.

## Статус

<!-- AUTO:status:start -->
| Режим | Статус | run_id | Выполненные этапы |
|---|---|---|---|
| full | выполнен | `full-20261031T151306Z-b6b9e1` | ingest, audit, features, train, calibrate, evaluate, score |
| smoke | выполнен | `smoke-20261031T150445Z-f124ab` | ingest, audit, features, train, calibrate, evaluate, score |
<!-- AUTO:status:end -->

Режимы: **full** — все клиенты полного `transactions_train.csv`; **smoke** — детерминированный 1% реальных
клиентов (та же логика и окна); **fixture** — синтетические данные только в тестах. Незапущенные этапы — в разделе
[Ограничения](#ограничения-и-незапущенные-этапы).

## Задача

Бинарный прогноз: совершит ли клиент, покупавший в последние 90 дней, хотя бы одну наблюдаемую покупку в
следующие 30 дней. Задача Kaggle на этих данных — рекомендации товаров (MAP@12); здесь на той же истории решается
другая задача — ранжирование клиентов для CRM-сценария.

| Элемент | Определение |
|---|---|
| Единица наблюдения | клиентский срез `(customer_id, cutoff_date)` |
| История признаков | `[cutoff_date − 90 дней, cutoff_date)` |
| Целевое окно | `[cutoff_date, cutoff_date + 30 дней)`; покупка в `cutoff_date` относится к цели |
| Когорта | клиенты с ≥ 1 товарной записью в окне истории |
| target | 1 — есть ≥ 1 запись в целевом окне; 0 — нет наблюдаемой покупки в наборе (не «отток») |
| Разметка | только при `cutoff_date + 30 ≤ observation_end_exclusive` |
| Бюджет CRM | top 5% аудитории среза, `k = ceil(0.05·N)` (проектное допущение) |

Подробно: [docs/problem_statement.md](docs/problem_statement.md).

## Данные

- Источник: [H&M Personalized Fashion Recommendations (Kaggle)](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data).
  Условия использования данных — [Competition Rules](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules);
  раздел DATA ACCESS AND USE: «Non-Commercial Purposes & Academic Research». Данные в репозиторий не входят и
  под MIT не распространяются.
- Загрузка: публичная копия исходных CSV [microsoft/hnm-search-data](https://huggingface.co/datasets/microsoft/hnm-search-data)
  (`data/raw/`, ревизия `ac68fe6c`). Размеры и SHA-256 совпали с LFS-метаданными зеркала:

| Файл | Байт | SHA-256 |
|---|---|---|
| transactions_train.csv | 3 488 002 253 | `fff5e3b35c1b26016aa0bf2231be9525bc766eed32c94f1653420b465aef84fa` |
| customers.csv | 207 135 859 | `abf3450c37ee1c16124e0d3379371153f5371148742efdcfe44c97c0445df6b2` |
| articles.csv | 36 127 865 | `25f21882053517f7684ea43f524225a9481a7bb9aab9cf0abfc7fe8c4576a72b` |

Manifest загрузки: [reports/data/ingest_manifest_full.json](reports/data/ingest_manifest_full.json).
Фактический объём (из `source_audit.json` и `dataset_summary.json`):

<!-- AUTO:full:data:start -->
| Величина | Значение |
|---|---|
| Товарные записи transactions_train.csv | 31 788 324 |
| Уникальные customer_id в транзакциях | 1 362 281 |
| Уникальные article_id в транзакциях | 104 547 |
| Диапазон t_dat | 2018-09-20 … 2020-09-22 (734 дат) |
| Строки customers.csv / articles.csv | 1 371 980 / 105 542 |
| Клиенты в наборе `full_6114b4ad79` (доля 1.0) | 1 371 980 |
| Клиенты с ≥1 покупательским днём / с ≥2 | 1 362 281 / 914 308 |
| Строки customer_day | 9 080 179 |
| Клиентские срезы (все допустимые cutoff) | 10 719 039 |
| observation_end_exclusive | 2020-09-23 (max(t_dat) + 1 day: no missing dates and the last day is not truncated) |
<!-- AUTO:full:data:end -->

## Архитектура

```
CSV ──COPY FROM STDIN (Python-поток)──► staging (UNLOGGED text) ──типизация в SQL──► raw.* (партиция на load_id)
                                         отказы → raw.transactions_rejected
raw ──audit.sql, проверки many-to-one JOIN──► reports/data/source_audit.json
raw ──► derived.customer_key ──► derived.customer_day (customer_sk, t_dat) ──сверка строк/сумм с raw
     ──► derived.calendar ──► derived.snapshot (признаки) + derived.label (target)   [партиции на dataset_id]
     ──COPY TO STDOUT──► Parquet по cutoff ──► Python: walk-forward → selection_decision.json → refit
     ──► Platt-калибровка ──► final test + bootstrap ──► batch scoring (признаки тем же SQL) ──► hmrp report
```

- PostgreSQL 16.8 (Docker Compose, `127.0.0.1:55432`, named volume `hmrp_pgdata`, healthcheck): загрузка,
  типизация, аудит, агрегация, срезы, метки, проверки кардинальности, EXPLAIN ANALYZE.
- Python-пакет `hm_repeat_purchase` с CLI `hmrp`: оркестрация, обучение (scikit-learn, CatBoost CPU),
  калибровка, метрики, отчёты. Конфигурации: [configs/full.yaml](configs/full.yaml), [configs/smoke.yaml](configs/smoke.yaml).
- Идентичность набора `dataset_id` = хеш (SHA-256 источников, схема ingest, правило и доля выборки, версия
  признаков, окна); smoke и full — разные партиции. Повторный ingest того же файла не дублирует строки.
- Каждый run: `reports/runs/<run_id>/` (конфигурация, git commit, версии, метрики, решения, тайминги) и
  локально `artifacts/runs/<run_id>/` (модели, индивидуальные прогнозы).

Подробно: [docs/methodology.md](docs/methodology.md), [docs/data_contract.md](docs/data_contract.md),
[docs/decisions.md](docs/decisions.md).

## Признаки

Core (`core_v1`, ровно три признака; одинаковы для baseline, LR и CatBoost-core):

| Признак | Определение |
|---|---|
| `purchase_days_90d` | число различных дат с ≥ 1 записью клиента в окне истории (не число заказов) |
| `avg_scaled_spend_per_purchase_day_90d` | сумма `price` в окне / `purchase_days_90d`; предоставленная (масштабированная) шкала, не средний чек |
| `recency_days` | `cutoff_date − последняя дата покупки в окне`, 1…90 |

Extended (`ext_v1`, 16 признаков): core + покупательские дни и записи за 7/30 дней, суммы price за 7/30/90 дней,
средняя цена записи, давность первой покупки в окне, средний интервал между покупательскими днями, доля
записей канала 2, доля дней за последние 30 из 90. Определения: [docs/data_contract.md](docs/data_contract.md).
Поля `customers.csv` (age, FN, Active, club_member_status, fashion_news_frequency) в модели не используются.

Модели: recency baseline (`score = −recency_days`, не вероятность); LogisticRegression (Pipeline: median-импутация
→ log1p → StandardScaler, сетка C); CatBoostClassifier (CPU, ≤ 4 конфигурации, early stopping по validation
fold). Ранжирование всех моделей — по `raw_score`; основная вероятность — Platt-калиброванная.

## Временные границы (full)

<!-- AUTO:full:splits:start -->
| Этап | Train-срезы (cutoff) | N train | Доля target=1 train | Оценочный срез и окна | N оценки | Доля target=1 |
|---|---|---|---|---|---|---|
| dev fold 1 | 2019-01-02 … 2020-03-27 (16 срезов) | 8 160 990 | 32.99% | 2020-04-26 → [2020-04-26, 2020-05-26) | 482 273 | 29.01% |
| dev fold 2 | 2019-01-02 … 2020-04-26 (17 срезов) | 8 643 263 | 32.76% | 2020-05-26 → [2020-05-26, 2020-06-25) | 478 991 | 37.01% |
| dev fold 3 | 2019-01-02 … 2020-05-26 (18 срезов) | 9 122 254 | 32.99% | 2020-06-25 → [2020-06-25, 2020-07-25) | 520 860 | 34.93% |
| calibration | refit: 2019-01-02 … 2020-06-25 (19 срезов) | 9 643 114 | 33.09% | 2020-07-25 → [2020-07-25, 2020-08-24); история [2020-04-26, 2020-07-25) | 536 041 | 31.82% |
| final test | те же модели | — | — | 2020-08-24 → [2020-08-24, 2020-09-23); история [2020-05-26, 2020-08-24) | 539 884 | 31.34% |
<!-- AUTO:full:splits:end -->

## Установка и запуск

Проверено на Windows 11, Python 3.12.3, Docker Desktop (Engine 29.9), PostgreSQL 16.8.

```powershell
git clone https://github.com/misterxplot-cmd/hm-repeat-purchase-ml.git
cd hm-repeat-purchase-ml
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
Copy-Item .env.example .env          # задать HMRP_PG_PASSWORD (локальный пароль контейнера)
docker compose up -d                 # PostgreSQL на 127.0.0.1:55432
hmrp doctor --config configs/full.yaml
hmrp download --config configs/full.yaml   # 3 CSV (~3.7 ГБ) в data/raw + проверка SHA-256
hmrp ingest --config configs/full.yaml     # COPY 31.8 млн строк; повтор — reuse без дублей
hmrp audit --config configs/full.yaml
hmrp build-features --config configs/smoke.yaml
hmrp build-features --config configs/full.yaml
hmrp run --config configs/smoke.yaml --stages train,calibrate,evaluate,score,report
hmrp run --config configs/full.yaml --stages train,calibrate,evaluate,score,report
python -m pytest -q                  # unit + интеграционные тесты (нужен запущенный PostgreSQL)
```

Отдельные этапы: `hmrp train|calibrate|evaluate|report --config … [--run-id …]`,
`hmrp score --config configs/full.yaml --as-of 2020-09-23 [--run-id …]`. Повторная оценка test того же run
требует `--reevaluate-reason "…"` и записывается в `evaluation_log.json`. Данные можно хранить вне проекта:
`--data-dir D:\path` или переменная `HNM_DATA_DIR`. Остановка: `docker compose down` (volume сохраняется).

Bash/WSL: те же команды с `python3.12 -m venv .venv && source .venv/bin/activate`.

Тесты: 43 теста (`tests/`), из них интеграционные используют отдельную БД `hmrp_test` в том же контейнере и
пропускаются с явной причиной, если PostgreSQL недоступен. CI ([.github/workflows/ci.yml](.github/workflows/ci.yml))
запускает весь набор на синтетических данных с сервисом PostgreSQL 16.8; полный H&M в CI не скачивается.

## Результаты full

### Development walk-forward (validation AP по fold)

<!-- AUTO:full:dev:start -->
| Модель | Роль | Конфигурация | AP по fold 1/2/3 | mean AP | std AP | mean ROC-AUC | mean Precision@5% |
|---|---|---|---|---|---|---|---|
| recency_baseline | baseline | score=-recency_days | 0.3615 / 0.4674 / 0.4524 | 0.4271 | 0.0573 | 0.6107 | 50.45% |
| lr_core | кандидат | C=0.01 | 0.4966 / 0.5782 / 0.5663 | 0.5470 | 0.0441 | 0.6742 | 76.55% |
| catboost_core | кандидат | d6_lr0.05_l2_10 | 0.4987 / 0.5780 / 0.5672 | 0.5479 | 0.0430 | 0.6765 | 76.58% |
| catboost_ext | кандидат | d8_lr0.10_l2_10 | 0.5019 / 0.5845 / 0.5708 | 0.5524 | 0.0443 | 0.6859 | 76.55% |
| lr_ext | диагностика | C=1 | 0.4986 / 0.5832 / 0.5702 | 0.5507 | 0.0456 | 0.6818 | 76.47% |
<!-- AUTO:full:dev:end -->

### Выбор модели (до final test)

<!-- AUTO:full:selection:start -->
- Правило: mean_AP >= max_AP - 0.005 (boundary included); приоритет простоты: lr_core → catboost_core → catboost_ext.
- max mean AP среди кандидатов: 0.5524; near_best: catboost_core, catboost_ext.
- **Выбрана: `catboost_core`** (конфигурация `d6_lr0.05_l2_10`), решение зафиксировано 2026-10-31T16:59:12Z в [`selection_decision.json`](reports/runs/full-20261031T151306Z-b6b9e1/selection_decision.json).
<!-- AUTO:full:selection:end -->

### Калибровка

<!-- AUTO:full:calibration:start -->
Platt-калибровка на срезе 2020-07-25 (target [2020-07-25, 2020-08-24), N = 536 041, prevalence 31.82%); p = sigmoid(a·raw_score + b), базовые модели заморожены.

| Модель | Статус | a | b | Основная вероятность |
|---|---|---|---|---|
| lr_core | ok | 1.0086 | -0.0947 | calibrated_probability |
| catboost_core | ok | 1.0055 | -0.1025 | calibrated_probability |
| catboost_ext | ok | 1.0131 | -0.0815 | calibrated_probability |
<!-- AUTO:full:calibration:end -->

### Final test

<!-- AUTO:full:test:start -->
Final test: cutoff 2020-08-24, target [2020-08-24, 2020-09-23), N = 539 884 клиентов, positives = 169 180, бюджет top 5%: k = ceil(0.05·N) = 26 995. Run `full-20261031T151306Z-b6b9e1`.

| Модель | N | Prevalence | AP | ROC-AUC | P@1% | P@5% | P@10% | Lift@5% | Brier raw | Brier calibrated | LogLoss raw | LogLoss calibrated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| recency_baseline | 539 884 | 31.34% | 0.4053 | 0.6143 | 50.27% | 49.00% | 47.42% | 1.564 | — | — | — | — |
| lr_core | 539 884 | 31.34% | 0.5330 | 0.6854 | 88.74% | 75.01% | 66.48% | 2.394 | 0.1919 | 0.1913 | 0.5692 | 0.5677 |
| **catboost_core** | 539 884 | 31.34% | 0.5345 | 0.6876 | 88.89% | 75.09% | 66.65% | 2.396 | 0.1917 | 0.1910 | 0.5685 | 0.5670 |
| catboost_ext | 539 884 | 31.34% | 0.5394 | 0.6947 | 89.22% | 75.50% | 66.97% | 2.409 | 0.1906 | 0.1901 | 0.5658 | 0.5644 |
<!-- AUTO:full:test:end -->

<!-- AUTO:full:bootstrap:start -->
Парный bootstrap по клиентам: 500 из 500 ресэмплов (пропущено с одним классом: 0), seed 7, 95% percentile-интервалы. Интервал не учитывает неопределённость будущего временного сдвига.

| Разность | ΔAP [95% ДИ] | ΔROC-AUC [95% ДИ] | ΔPrecision@5% [95% ДИ] | Отн. изменение P@5% |
|---|---|---|---|---|
| catboost_core − recency_baseline | 0.1292 [0.1271; 0.1312] | 0.0733 [0.0718; 0.0746] | +26.09 п.п. [+25.39; +26.84] | +53.3% |
| lr_core − catboost_core | -0.0015 [-0.0018; -0.0012] | -0.0022 [-0.0026; -0.0018] | -0.08 п.п. [-0.29; +0.11] | -0.1% |
<!-- AUTO:full:bootstrap:end -->

### Batch scoring (CRM-выгрузка)

<!-- AUTO:full:scoring:start -->
| as_of_date | Модель | N | k (top 5%) | Основная вероятность | Окно target наблюдаемо | Выгрузка |
|---|---|---|---|---|---|---|
| 2020-08-24 | full-20261031T151306Z-b6b9e1/catboost_core | 539 884 | 26 995 | calibrated_probability | да | `artifacts/runs/full-20261031T151306Z-b6b9e1/scoring/scores_catboost_core_asof_2020-08-24.csv` (локально) |
| 2020-09-23 | full-20261031T151306Z-b6b9e1/catboost_core | 518 920 | 25 946 | calibrated_probability | нет (будущее неизвестно) | `artifacts/runs/full-20261031T151306Z-b6b9e1/scoring/scores_catboost_core_asof_2020-09-23.csv` (локально) |
<!-- AUTO:full:scoring:end -->

Выгрузки не содержат target; `rank` и `selected_for_top_segment` считаются по `raw_score`. Для 2020-09-23 будущее неизвестно, метрики не считаются.

### Выбор модели и выводы

<!-- AUTO:full:interpretation:start -->
- **Выбор.** Порог near_best = max mean AP − 0.005 = 0.5474. `lr_core`: mean AP 0.5470; `catboost_core`: mean AP 0.5479; `catboost_ext`: mean AP 0.5524. Вне near_best: `lr_core`. По приоритету простоты выбрана `catboost_core`; правило зафиксировано в конфигурации до запуска, решение — до чтения меток test.
- **Против baseline (final test).** `catboost_core`: AP 0.5345 против 0.4053 (Δ 0.1292, 95% ДИ [0.1271; 0.1312]); Precision@5% 75.09% против 49.00% (+26.09 п.п., относительно +53.3%). Интервалы по клиентам одного периода.
- **LR-core против CatBoost-core** (одинаковые три признака): на test ΔAP = -0.0015 [-0.0018; -0.0012], ΔPrecision@5% = -0.08 п.п. [-0.29; +0.11]. По AP интервал не включает 0; по Precision@5% — включает 0 (разница не различима).
- **Эффект признаков (абляция на dev, mean AP).** LR: core 0.5470 → ext 0.5507; CatBoost: core 0.5479 → ext 0.5524 (Δ 0.0037 и 0.0045). На test `catboost_ext` получил AP 0.5394; он не выбран правилом, результат показан для прозрачности и выбор задним числом не меняется.
- **Калибровка.** У `catboost_core` наклон Platt a = 1.0055: исходные вероятности уже близки к калиброванным; Brier на test 0.1917 → 0.1910, LogLoss 0.5685 → 0.5670.
- **Нестабильность early stopping.** best_iteration `catboost_core` по fold: 1635, 31, 2292; число деревьев final refit = 1636 (медиана, правило задано заранее); AP по fold — в таблице development.
- **Статус.** `catboost_core` ранжирует лучше recency baseline (нижняя граница ΔAP > 0); это прогноз наблюдаемой покупки на одном тестовом периоде. Эффект CRM-контакта и устойчивость на других сезонах не проверены, готовность к внедрению не заявляется.
<!-- AUTO:full:interpretation:end -->

## Результаты smoke (1% клиентов; не full)

<!-- AUTO:smoke:test:start -->
Final test: cutoff 2020-08-24, target [2020-08-24, 2020-09-23), N = 5 486 клиентов, positives = 1 673, бюджет top 5%: k = ceil(0.05·N) = 275. Run `smoke-20261031T150445Z-f124ab`.

| Модель | N | Prevalence | AP | ROC-AUC | P@1% | P@5% | P@10% | Lift@5% | Brier raw | Brier calibrated | LogLoss raw | LogLoss calibrated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| recency_baseline | 5 486 | 30.50% | 0.4068 | 0.6350 | 50.91% | 45.82% | 45.90% | 1.502 | — | — | — | — |
| **lr_core** | 5 486 | 30.50% | 0.5374 | 0.7039 | 87.27% | 74.55% | 66.30% | 2.444 | 0.1863 | 0.1857 | 0.5565 | 0.5548 |
| catboost_core | 5 486 | 30.50% | 0.5359 | 0.7041 | 85.45% | 74.55% | 66.67% | 2.444 | 0.1871 | 0.1858 | 0.5588 | 0.5550 |
| catboost_ext | 5 486 | 30.50% | 0.5373 | 0.7064 | 87.27% | 74.55% | 67.76% | 2.444 | 0.1856 | 0.1850 | 0.5546 | 0.5530 |
<!-- AUTO:smoke:test:end -->

## Ограничения и незапущенные этапы

- В источнике нет `order_id`, количества, времени, отмен и возвратов; одинаковые строки сохранены как единицы
  товара. `price` — недокументированная масштабированная шкала; суммы не являются выручкой.
- Один final test (2020-08-24) и три development-fold (апрель–июль 2020); bootstrap не учитывает временной сдвиг.
- Early stopping CatBoost — по validation текущего fold, поэтому dev-оценки CatBoost слегка оптимистичны.
- Прогноз наблюдаемой покупки, не эффекта контакта: uplift, ROI и причинный эффект не измерялись; модель не
  объявляется готовой к внедрению.
- not_run: GPU-обучение CatBoost, горизонты 60/90 дней, delivery-модель после теста, class_weight/oversampling,
  признаки категорий товаров.

Полный список: [docs/limitations.md](docs/limitations.md).

## Отчёты и артефакты

| Что | Где |
|---|---|
| Итоговый отчёт full / smoke | [reports/results.md](reports/results.md), [reports/results_smoke.md](reports/results_smoke.md) |
| Качество данных, сверки, JOIN | [reports/data_quality.md](reports/data_quality.md), [reports/data/source_audit.json](reports/data/source_audit.json) |
| Производительность (измерено) | [reports/performance.md](reports/performance.md) |
| Метрики, решения, манифесты run | `reports/runs/<run_id>/` (`selection_decision.json`, `test_metrics.json`, `bootstrap.json`, `calibration.json`, `run_manifest.json`, `stage_timings.json`, `dev_results.csv`) |
| EXPLAIN ANALYZE | `reports/data/<dataset_id>/explain_*.txt` |
| Демонстрация JOIN-ловушки (синтетика) | [reports/data/join_trap_demo.json](reports/data/join_trap_demo.json) |
| Ноутбуки (выполнены, читают артефакты) | [notebooks/01_data_audit.ipynb](notebooks/01_data_audit.ipynb), [notebooks/02_results_walkthrough.ipynb](notebooks/02_results_walkthrough.ipynb) |
| Документация | [problem_statement](docs/problem_statement.md), [data_contract](docs/data_contract.md), [methodology](docs/methodology.md), [decisions](docs/decisions.md), [limitations](docs/limitations.md) |
| Формат выгрузки scoring (пример, искусственные ID) | [docs/examples/scoring_output_example.csv](docs/examples/scoring_output_example.csv) |
| Обученные модели full | GitHub Release [`models-full-20261031T151306Z-b6b9e1`](https://github.com/misterxplot-cmd/hm-repeat-purchase-ml/releases/tag/models-full-20261031T151306Z-b6b9e1): `lr_core.joblib`, `catboost_core.cbm`, `catboost_ext.cbm`, метаданные, `SHA256SUMS.txt` (некоммерческое исследовательское использование по условиям данных H&M) |

Локально (не публикуются, воспроизводятся командами выше): исходные CSV (`data/raw/`), Parquet-матрицы признаков
и меток (`data/derived/<dataset_id>/`, ~1.1 ГБ для full), индивидуальные прогнозы test и выгрузки scoring
(`artifacts/runs/<run_id>/`), Docker volume `hmrp_pgdata`, `.env`.

## Лицензия

Код — MIT ([LICENSE](LICENSE)). Данные H&M в репозиторий не входят; их использование регулируется
[правилами соревнования](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules).
