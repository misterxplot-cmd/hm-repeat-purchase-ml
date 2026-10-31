# hm-repeat-purchase-ml

Reproducible customer repeat-purchase prediction on H&M transaction history with PostgreSQL, temporal validation,
logistic regression and CatBoost.

Таблицы между маркерами `<!-- AUTO:… -->` генерирует `hmrp report` из сохранённых артефактов run; вручную
они не редактируются.

## Статус

<!-- AUTO:status:start -->
| Режим | Статус | run_id | Выполненные этапы |
|---|---|---|---|
| full | не завершён (нет final test в опубликованных артефактах) | `full-20261031T151306Z-b6b9e1` | ingest, audit, features |
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
  данные в репозиторий не входят и под MIT не распространяются.
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
<!-- AUTO:full:dev:end -->

### Выбор модели (до final test)

<!-- AUTO:full:selection:start -->
<!-- AUTO:full:selection:end -->

### Калибровка

<!-- AUTO:full:calibration:start -->
<!-- AUTO:full:calibration:end -->

### Final test

<!-- AUTO:full:test:start -->
<!-- AUTO:full:test:end -->

<!-- AUTO:full:bootstrap:start -->
<!-- AUTO:full:bootstrap:end -->

<!-- INTERPRETATION -->
Full-эксперимент выполняется; таблицы и выводы будут добавлены из артефактов run.
<!-- /INTERPRETATION -->

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
| Обученные модели full | будет опубликовано в GitHub Release после завершения full run |

Локально (не публикуются, воспроизводятся командами выше): исходные CSV (`data/raw/`), Parquet-матрицы признаков
и меток (`data/derived/<dataset_id>/`, ~1.1 ГБ для full), индивидуальные прогнозы test и выгрузки scoring
(`artifacts/runs/<run_id>/`), Docker volume `hmrp_pgdata`, `.env`.

## Лицензия

Код — MIT ([LICENSE](LICENSE)). Данные H&M в репозиторий не входят; их использование регулируется
[правилами соревнования](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules).
