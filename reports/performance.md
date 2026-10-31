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

## Сборка набора `full_6114b4ad79`

| Этап | Время |
|---|---|
| customer_key | 30.6 с |
| customer_day | 29.6 с |
| coverage_calendar | 0.4 с |
| snapshot_label_2019-01-02 | 1.7 с |
| snapshot_label_2019-02-01 | 1.6 с |
| snapshot_label_2019-03-03 | 1.5 с |
| snapshot_label_2019-04-02 | 1.6 с |
| snapshot_label_2019-05-02 | 1.6 с |
| snapshot_label_2019-06-01 | 1.7 с |
| snapshot_label_2019-07-01 | 1.7 с |
| snapshot_label_2019-07-31 | 1.8 с |
| snapshot_label_2019-08-30 | 1.8 с |
| snapshot_label_2019-09-29 | 1.7 с |
| snapshot_label_2019-10-29 | 1.7 с |
| snapshot_label_2019-11-28 | 1.8 с |
| snapshot_label_2019-12-28 | 1.8 с |
| snapshot_label_2020-01-27 | 1.8 с |
| snapshot_label_2020-02-26 | 1.8 с |
| snapshot_label_2020-03-27 | 1.7 с |
| snapshot_label_2020-04-26 | 1.8 с |
| snapshot_label_2020-05-26 | 1.9 с |
| snapshot_label_2020-06-25 | 2.0 с |
| snapshot_label_2020-07-25 | 2.0 с |
| snapshot_label_2020-08-24 | 2.4 с |
| snapshots_labels | 40.8 с |
| export_parquet | 79.3 с |

| Таблица PostgreSQL | Размер |
|---|---|
| customer_day__full_6114b4ad79 | 1052.3 МБ |
| customer_key__full_6114b4ad79 | 307.7 МБ |
| label__full_6114b4ad79 | 927.7 МБ |
| snapshot__full_6114b4ad79 | 2272.7 МБ |

## Этапы run `full-20261031T151306Z-b6b9e1`

| Этап | Время | Пиковый RSS процесса, МБ |
|---|---|---|
| dev_fold_1_load | 5.3 с | 4573.0 |
| dev_fold_1_lr_core_C=0.01 | 20.9 с | 4573.0 |
| dev_fold_1_lr_core_C=0.1 | 21.0 с | 4573.0 |
| dev_fold_1_lr_core_C=1 | 21.1 с | 4573.0 |
| dev_fold_1_catboost_core_d4_lr0.10_l2_3 | 3.5 мин | 4573.0 |
| dev_fold_1_catboost_core_d6_lr0.10_l2_3 | 3.0 мин | 4573.0 |
| dev_fold_1_catboost_core_d6_lr0.05_l2_10 | 5.0 мин | 4573.0 |
| dev_fold_1_catboost_core_d8_lr0.10_l2_10 | 2.5 мин | 4573.0 |
| dev_fold_1_catboost_ext_d4_lr0.10_l2_3 | 8.1 мин | 5447.1 |
| dev_fold_1_catboost_ext_d6_lr0.10_l2_3 | 4.6 мин | 5449.5 |
| dev_fold_1_catboost_ext_d6_lr0.05_l2_10 | 8.9 мин | 5453.4 |
| dev_fold_1_catboost_ext_d8_lr0.10_l2_10 | 4.3 мин | 5455.0 |
| dev_fold_1_lr_ext_C=0.01 | 45.6 с | 8507.7 |
| dev_fold_1_lr_ext_C=0.1 | 43.1 с | 8507.7 |
| dev_fold_1_lr_ext_C=1 | 43.1 с | 8507.7 |
| dev_fold_2_load | 5.9 с | 8507.7 |
| dev_fold_2_lr_core_C=0.01 | 22.7 с | 8507.7 |
| dev_fold_2_lr_core_C=0.1 | 22.4 с | 8507.7 |
| dev_fold_2_lr_core_C=1 | 23.0 с | 8507.7 |
| dev_fold_2_catboost_core_d4_lr0.10_l2_3 | 37.9 с | 8507.7 |
| dev_fold_2_catboost_core_d6_lr0.10_l2_3 | 36.4 с | 8507.7 |
| dev_fold_2_catboost_core_d6_lr0.05_l2_10 | 40.7 с | 8507.7 |
| dev_fold_2_catboost_core_d8_lr0.10_l2_10 | 38.6 с | 8507.7 |
| dev_fold_2_catboost_ext_d4_lr0.10_l2_3 | 36.7 с | 8507.7 |
| dev_fold_2_catboost_ext_d6_lr0.10_l2_3 | 40.8 с | 8507.7 |
| dev_fold_2_catboost_ext_d6_lr0.05_l2_10 | 44.8 с | 8507.7 |
| dev_fold_2_catboost_ext_d8_lr0.10_l2_10 | 45.7 с | 8507.7 |
| dev_fold_2_lr_ext_C=0.01 | 44.0 с | 9214.0 |
| dev_fold_2_lr_ext_C=0.1 | 44.6 с | 9214.0 |
| dev_fold_2_lr_ext_C=1 | 44.1 с | 9214.0 |
| dev_fold_3_load | 6.1 с | 9214.0 |
| dev_fold_3_lr_core_C=0.01 | 23.6 с | 9214.0 |
| dev_fold_3_lr_core_C=0.1 | 24.1 с | 9214.0 |
| dev_fold_3_lr_core_C=1 | 23.5 с | 9214.0 |
| dev_fold_3_catboost_core_d4_lr0.10_l2_3 | 5.2 мин | 9214.0 |
| dev_fold_3_catboost_core_d6_lr0.10_l2_3 | 5.9 мин | 9214.0 |
| dev_fold_3_catboost_core_d6_lr0.05_l2_10 | 7.6 мин | 9214.0 |
| dev_fold_3_catboost_core_d8_lr0.10_l2_10 | 3.7 мин | 9214.0 |
| dev_fold_3_catboost_ext_d4_lr0.10_l2_3 | 8.1 мин | 9214.0 |
| dev_fold_3_catboost_ext_d6_lr0.10_l2_3 | 4.9 мин | 9214.0 |
| dev_fold_3_catboost_ext_d6_lr0.05_l2_10 | 10.0 мин | 9214.0 |
| dev_fold_3_catboost_ext_d8_lr0.10_l2_10 | 4.3 мин | 9214.0 |
| dev_fold_3_lr_ext_C=0.01 | 46.9 с | 9781.7 |
| dev_fold_3_lr_ext_C=0.1 | 47.4 с | 9781.7 |
| dev_fold_3_lr_ext_C=1 | 47.5 с | 9781.7 |
| development_search | 106.1 мин | 9781.7 |
| final_refit_load | 6.3 с | 9781.7 |
| final_refit_lr_core | 4.4 с | 9781.7 |
| final_refit_catboost_core | 5.6 мин | 9781.7 |
| final_refit_catboost_ext | 3.7 мин | 9781.7 |
| final_refit | 9.5 мин | 9781.7 |
| calibration_load | 0.2 с | 9781.7 |
| calibrate_lr_core | 0.7 с | 9781.7 |
| calibrate_catboost_core | 0.7 с | 9781.7 |
| calibrate_catboost_ext | 0.9 с | 9781.7 |
| test_load | 0.2 с | 9781.7 |
| test_predict_and_metrics | 14.1 с | 9781.7 |
| test_bootstrap | 36.5 с | 9781.7 |

## Окружение

Python 3.12.3, Windows-11-10.0.26200-SP0, 24 логических CPU, RAM 63.8 ГБ; CatBoost 1.2.8 на CPU (16 потоков). GPU-обучение не запускалось.

