# Производительность

Все значения измерены во время запусков; отсутствующие помечены not_measured.

## Ingest (COPY FROM STDIN)

| Файл | Действие | Строк | Время загрузки | SHA-256 (время) |
|---|---|---|---|---|
| transactions_train.csv | loaded | 31 788 324 | 7.6 мин | 2.4 с |
| customers.csv | loaded | 1 371 980 | 3.0 с | 0.1 с |
| articles.csv | loaded | 105 542 | 0.8 с | 0.0 с |

## Аудит (SQL)

| Запрос | Время |
|---|---|
| tx_overview | 42.6 с |
| tx_price_stats | 3.9 с |
| tx_channels | 0.5 с |
| tx_monthly_channel | 15.5 с |
| tx_daily_by_channel | 9.0 с |
| tx_price_log_histogram | 0.6 с |
| tx_identical_rows | 13.9 с |
| tx_customer_coverage | 3.9 с |
| tx_article_coverage | 0.8 с |
| customers_overview | 5.8 с |
| customers_categoricals | 0.2 с |
| articles_overview | 0.1 с |
| rejected_by_reason | 0.0 с |

## Сборка набора `smoke_dfa33f484c`

| Этап | Время |
|---|---|
| customer_key | 21.6 с |
| customer_day | 3.9 с |
| coverage_calendar | 0.4 с |
| snapshots_labels | 0.6 с |
| export_parquet | 1.0 с |

| Таблица PostgreSQL | Размер |
|---|---|
| customer_day__smoke_dfa33f484c | 10.5 МБ |
| customer_key__smoke_dfa33f484c | 3.3 МБ |
| label__smoke_dfa33f484c | 9.4 МБ |
| snapshot__smoke_dfa33f484c | 23.5 МБ |

## Этапы run `smoke-20261031T150445Z-f124ab`

| Этап | Время | Пиковый RSS процесса, МБ |
|---|---|---|
| dev_fold_1_load | 0.1 с | 267.1 |
| dev_fold_1_lr_core_C=0.01 | 0.1 с | 267.1 |
| dev_fold_1_lr_core_C=0.1 | 0.1 с | 267.1 |
| dev_fold_1_lr_core_C=1 | 0.1 с | 267.1 |
| dev_fold_1_catboost_core_d4_lr0.10_l2_3 | 1.1 с | 275.9 |
| dev_fold_1_catboost_core_d6_lr0.10_l2_3 | 1.3 с | 284.5 |
| dev_fold_1_catboost_ext_d4_lr0.10_l2_3 | 1.2 с | 295.5 |
| dev_fold_1_catboost_ext_d6_lr0.10_l2_3 | 1.1 с | 296.1 |
| dev_fold_1_lr_ext_C=0.01 | 0.3 с | 328.1 |
| dev_fold_1_lr_ext_C=0.1 | 0.3 с | 328.1 |
| dev_fold_1_lr_ext_C=1 | 0.3 с | 328.1 |
| dev_fold_2_load | 0.1 с | 328.1 |
| dev_fold_2_lr_core_C=0.01 | 0.1 с | 328.1 |
| dev_fold_2_lr_core_C=0.1 | 0.1 с | 328.1 |
| dev_fold_2_lr_core_C=1 | 0.1 с | 328.1 |
| dev_fold_2_catboost_core_d4_lr0.10_l2_3 | 0.5 с | 328.1 |
| dev_fold_2_catboost_core_d6_lr0.10_l2_3 | 0.6 с | 328.1 |
| dev_fold_2_catboost_ext_d4_lr0.10_l2_3 | 0.6 с | 328.1 |
| dev_fold_2_catboost_ext_d6_lr0.10_l2_3 | 0.7 с | 328.1 |
| dev_fold_2_lr_ext_C=0.01 | 0.3 с | 347.9 |
| dev_fold_2_lr_ext_C=0.1 | 0.3 с | 347.9 |
| dev_fold_2_lr_ext_C=1 | 0.3 с | 347.9 |
| dev_fold_3_load | 0.1 с | 347.9 |
| dev_fold_3_lr_core_C=0.01 | 0.1 с | 347.9 |
| dev_fold_3_lr_core_C=0.1 | 0.1 с | 347.9 |
| dev_fold_3_lr_core_C=1 | 0.1 с | 347.9 |
| dev_fold_3_catboost_core_d4_lr0.10_l2_3 | 0.6 с | 347.9 |
| dev_fold_3_catboost_core_d6_lr0.10_l2_3 | 0.7 с | 347.9 |
| dev_fold_3_catboost_ext_d4_lr0.10_l2_3 | 1.1 с | 347.9 |
| dev_fold_3_catboost_ext_d6_lr0.10_l2_3 | 1.3 с | 347.9 |
| dev_fold_3_lr_ext_C=0.01 | 0.3 с | 356.4 |
| dev_fold_3_lr_ext_C=0.1 | 0.3 с | 356.4 |
| dev_fold_3_lr_ext_C=1 | 0.3 с | 356.4 |
| development_search | 14.6 с | 356.4 |
| final_refit_load | 0.1 с | 356.4 |
| final_refit_lr_core | 0.0 с | 356.4 |
| final_refit_catboost_core | 0.1 с | 356.4 |
| final_refit_catboost_ext | 0.4 с | 356.4 |
| final_refit | 0.6 с | 356.4 |
| calibration_load | 0.0 с | 356.4 |
| calibrate_lr_core | 0.0 с | 356.4 |
| calibrate_catboost_core | 0.0 с | 356.4 |
| calibrate_catboost_ext | 0.0 с | 356.4 |
| test_load | 0.0 с | 356.4 |
| test_predict_and_metrics | 0.1 с | 356.4 |
| test_bootstrap | 0.0 с | 356.4 |

## Окружение

Python 3.12.3, Windows-11-10.0.26200-SP0, 24 логических CPU, RAM 63.8 ГБ; CatBoost 1.2.8 на CPU (8 потоков). GPU-обучение не запускалось.

