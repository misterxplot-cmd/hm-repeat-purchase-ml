# Контракт данных

## Источник

- Оригинал: [H&M Personalized Fashion Recommendations, Kaggle](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data);
  условия использования — [Competition Rules](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules).
- Использованная копия: [microsoft/hnm-search-data](https://huggingface.co/datasets/microsoft/hnm-search-data),
  каталог `data/raw/`, ревизия `ac68fe6c5372c7f0bbfe0b7a6510f9e17addc81f`. Карточка набора ссылается на исходные
  условия H&M/Kaggle: «DATA ACCESS AND USE: Non-Commercial Purposes & Academic Research».
- Контрольные суммы и размеры сверены с LFS-метаданными зеркала (Hugging Face tree API) и записаны в
  [`configs/full.yaml`](../configs/full.yaml) и `reports/data/ingest_manifest_full.json`.

| Файл | Байт | SHA-256 |
|---|---|---|
| transactions_train.csv | 3 488 002 253 | `fff5e3b35c1b26016aa0bf2231be9525bc766eed32c94f1653420b465aef84fa` |
| customers.csv | 207 135 859 | `abf3450c37ee1c16124e0d3379371153f5371148742efdcfe44c97c0445df6b2` |
| articles.csv | 36 127 865 | `25f21882053517f7684ea43f524225a9481a7bb9aab9cf0abfc7fe8c4576a72b` |

Изображения товаров не загружаются.

## transactions_train.csv → `raw.transactions`

| Поле | Тип в PostgreSQL | Правило проверки при загрузке |
|---|---|---|
| `t_dat` | `date` | `YYYY-MM-DD` и существующая календарная дата |
| `customer_id` | `text` | 64 символа `[0-9a-f]`; хранится строкой, в модель не подаётся |
| `article_id` | `text` | 10 цифр; ведущие нули сохраняются |
| `price` | `double precision` | десятичное/экспоненциальное число; значения не меняются |
| `sales_channel_id` | `smallint` | целое; смысл кодов источником не раскрыт |
| `load_id`, `src_row` | `integer`, `bigint` | технический ключ загрузки и номер исходной строки (PK) |

Строка без ровно пяти неэкранированных полей → `raw.transactions_rejected` (`reason = structure`); строка с
нарушением типа → та же таблица с перечнем причин. Значения не заменяются нулями и не импутируются. UNIQUE по
содержательным полям нет: одинаковые строки допустимы (нет `order_id` и количества).

## customers.csv, articles.csv

Хранятся как исходный текст (`raw.customers`, `raw.articles`), пустое поле CSV → NULL. Используются для аудита
уникальности ключей, покрытия и проверочных JOIN. Поля `FN`, `Active`, `club_member_status`,
`fashion_news_frequency`, `age`, `postal_code` — снимок без даты актуальности; в модели не используются.

## Производные таблицы (на `dataset_id`)

| Таблица | Grain | Назначение |
|---|---|---|
| `derived.customer_key` | `customer_sk` (PK), `customer_id` (UNIQUE) | обратимое отображение; `customer_sk` = ранг `customer_id` в байтовом порядке |
| `derived.customer_day` | `(customer_sk, t_dat)` (PK) | `n_rows`, `spend`, `n_distinct_articles`, `n_rows_channel_2` |
| `derived.calendar` | `(dataset_id, cutoff_date)` | границы окон, роль, допустимость и причина исключения |
| `derived.snapshot` | `(cutoff_date, customer_sk)` (PK) | признаки + аудитные поля, без target |
| `derived.label` | `(cutoff_date, customer_sk)` (PK) | `target`, `target_start`, `target_end_exclusive` |

`dataset_id = <mode>_<sha256(identity)[:10]>`; identity включает SHA-256 исходных файлов, версию схемы ingest,
правило и долю выборки клиентов, версию признаков и параметры окон. Smoke и full — разные `dataset_id` и разные
партиции; производные таблицы партиционированы по `dataset_id`.

## Экспорт для Python (локально, не публикуется)

`data/derived/<dataset_id>/features/cutoff=YYYY-MM-DD.parquet`, `labels/cutoff=…parquet`,
`customer_keys.parquet` (`customer_sk`, `customer_id`, `tie_key`). Метки test-среза читаются только с
`purpose="final_test"` (`TestAccessError` иначе).

## Extended-признаки (`ext_v1`, 16 шт.)

Все из окна `[cutoff − 90, cutoff)`; подокна 7/30 дней — `[cutoff − n, cutoff)`.

| Признак | Определение |
|---|---|
| core (3) | см. [problem_statement.md](problem_statement.md) |
| `purchase_days_7d`, `purchase_days_30d` | различные покупательские дни в подокне |
| `rows_7d`, `rows_30d`, `rows_90d` | товарные записи |
| `spend_7d`, `spend_30d`, `spend_90d` | сумма `price`, предоставленная шкала |
| `avg_price_per_row_90d` | `spend_90d / rows_90d` |
| `first_purchase_age_days_90d` | `cutoff − min(t_dat в окне)` |
| `mean_gap_days_90d` | `(max − min) / (purchase_days_90d − 1)`; NULL при одном дне (в LR — медиана train + индикатор) |
| `share_rows_channel_2_90d` | доля записей с `sales_channel_id = 2` |
| `recent_share_purchase_days_30d` | `purchase_days_30d / purchase_days_90d` (знаменатель ≥ 1 по определению когорты) |

Метаданные товаров в признаки не входят.

## Выгрузка scoring (CRM-демонстрация)

`customer_id, as_of_date, horizon_days, raw_score, raw_probability, calibrated_probability, rank,
selected_for_top_segment, model_version, feature_set_version, run_id`. Поля target нет. Формат — см.
[`docs/examples/scoring_output_example.csv`](examples/scoring_output_example.csv) (искусственные идентификаторы).
