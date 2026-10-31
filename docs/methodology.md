# Методология

## Pipeline

```
CSV (data/raw) ──COPY FROM STDIN──► staging.transactions_text (UNLOGGED, text)
                                    │  проверка типов в SQL, отказы → raw.transactions_rejected
                                    ▼
                         raw.transactions  (партиция на load_id, PK (load_id, src_row))
                         raw.customers / raw.articles (csv-парсер Python → COPY)
                                    │  audit.sql, many-to-one JOIN checks
                                    ▼
             derived.customer_key → derived.customer_day (GROUP BY customer_sk, t_dat)
                                    │  сверка строк и сумм с raw
                                    ▼
          derived.calendar → derived.snapshot (признаки по окну) + derived.label (target)
                                    │  COPY TO STDOUT → Parquet по cutoff
                                    ▼
      Python: walk-forward (3 fold) → selection_decision.json → refit → Platt → final test → scoring
```

Роли: PostgreSQL выполняет загрузку, типизацию, аудит, агрегацию до `customer_day`, построение срезов и меток,
проверки кардинальности. Python (pandas/scikit-learn/CatBoost) — обучение, калибровка, метрики, отчёты.
Признаки scoring считаются тем же SQL-запросом, что и признаки обучения (`snapshot_features_select`).

## Временная схема

Сетка `cutoff_date`: от 2019-01-02 с шагом ровно 30 дней до 2020-08-24 (21 срез). Первый срез имеет полную
90-дневную историю (данные с 2018-09-20). Проверочные срезы на шаг до и после сетки записываются в календарь
как недопустимые с причиной (`insufficient_history`, `target_window_incomplete`).

| Роль | cutoff | Условие |
|---|---|---|
| Development | 2019-01-02 … 2020-06-25 | — |
| Fold 1 / 2 / 3 validation | 2020-04-26 / 2020-05-26 / 2020-06-25 | train: `cutoff + 30 ≤ validation_cutoff` |
| Final refit | 2019-01-02 … 2020-06-25 (19 срезов) | `cutoff + 30 ≤ 2020-07-25` |
| Calibration | 2020-07-25 | только Platt-преобразование |
| Final test | 2020-08-24, target [2020-08-24, 2020-09-23) | однократная оценка |

Условия зрелости проверяются в `splits.py` (`assert_matured_before`) и тестами.

## Обучение и выбор

- Preprocessing каждого fold обучается на train этого fold (Pipeline: SimpleImputer(median, indicator) → log1p →
  StandardScaler → LogisticRegression).
- LR: сетка `C` из конфигурации, lbfgs; сходимость записывается (`n_iter`, `converged`).
- CatBoost: CPU, `thread_count` из конфигурации, Logloss; early stopping по validation текущего fold;
  ≤ 4 конфигурации в full. В final refit число деревьев = медиана `best_iteration + 1` по трём fold.
- Правило выбора (зафиксировано в конфигурации до запуска): для каждого кандидата — конфигурация с лучшим mean AP
  по трём fold; `near_best = {mean_AP ≥ max_AP − 0.005}` (граница включена); выбор по приоритету простоты
  LR-core → CatBoost-core → CatBoost-extended. Baseline и LR-extended — диагностика.
- `selection_decision.json` пишется до чтения меток test; `evaluate` ссылается на его SHA-256.

## Калибровка

Platt: `p = sigmoid(a · raw_score + b)` на срезе 2020-07-25 для каждой замороженной модели. Базовая модель на
calibration не переобучается. Структурный отказ (класс < `min_class_count`, неконечные параметры, `a ≤ 0`)
переводит модель на `raw_probability` с отметкой `fallback_used` в `calibration.json` и метаданных scoring.

## Метрики и ties

- Ранжирование: `raw_score` по убыванию, ties — по `tie_key` по возрастанию (`sha256("hmrp-tie-v1:" +
  customer_id)`, первые 7 байт), не зависит от target.
- AP и ROC-AUC — scikit-learn по `raw_score` (равные score — один порог).
- `Precision@k = TP в top-k / k`, `Lift@k = Precision@k / prevalence`; при prevalence = 0 Lift не определён.
- Разница precision — в процентных пунктах и отдельно относительное изменение.

## Bootstrap

Парный bootstrap по клиентам на final test: один вектор мультиномиальных кратностей на ресэмпл для всех моделей;
top-k формируется заново в каждом ресэмпле с учётом кратностей и того же правила ties. Взвешенные AP/ROC-AUC/
Precision@k совпадают с явным ресэмплированием (`tests/test_metrics.py`). Число ресэмплов — конфигурация
(full: 500, smoke: 50). Интервалы — percentile 2.5–97.5%.

## Режимы

| Режим | Данные | Отличия |
|---|---|---|
| fixture | синтетические CSV из `tests/synthetic.py` | только проверка логики; результаты не публикуются как H&M |
| smoke | 1% реальных клиентов по `sha256("hmrp-sample-v1:" + customer_id)`, вся история | 2 конфигурации CatBoost, 50 bootstrap |
| full | все клиенты | 4 конфигурации CatBoost, 500 bootstrap |
