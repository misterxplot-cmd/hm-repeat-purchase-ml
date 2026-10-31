-- Ingest: staging text -> typed raw partition. {part} is the raw partition for this load.

-- name: create_raw_partition
CREATE TABLE {part} PARTITION OF {parent} FOR VALUES IN ({load_id});

-- name: staging_distinct_dates
SELECT DISTINCT t_dat FROM staging.transactions_text WHERE load_id = %(load_id)s;

-- name: insert_typed_transactions
-- Type rules: date via validated date_map, 64-char lowercase hex customer_id, 10-digit article_id,
-- decimal/scientific price, integer sales_channel_id. Values are not altered or imputed.
INSERT INTO {part} (load_id, src_row, t_dat, customer_id, article_id, price, sales_channel_id)
SELECT s.load_id, s.src_row, dm.d, s.customer_id, s.article_id, s.price::double precision, s.sales_channel_id::smallint
FROM staging.transactions_text s
JOIN staging.date_map dm ON dm.load_id = s.load_id AND dm.txt = s.t_dat AND dm.d IS NOT NULL
WHERE s.load_id = %(load_id)s
  AND s.customer_id ~ '^[0-9a-f]{{64}}$'
  AND s.article_id ~ '^[0-9]{{10}}$'
  AND s.price ~ '^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$'
  AND s.sales_channel_id ~ '^[0-9]{{1,4}}$';

-- name: insert_rejected_transactions
INSERT INTO raw.transactions_rejected (load_id, src_row, reason, raw_line)
SELECT s.load_id, s.src_row,
       concat_ws(';',
           CASE WHEN dm.d IS NULL THEN 'invalid_t_dat' END,
           CASE WHEN s.customer_id IS NULL OR s.customer_id !~ '^[0-9a-f]{{64}}$' THEN 'invalid_customer_id' END,
           CASE WHEN s.article_id IS NULL OR s.article_id !~ '^[0-9]{{10}}$' THEN 'invalid_article_id' END,
           CASE WHEN s.price IS NULL OR s.price !~ '^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$' THEN 'invalid_price' END,
           CASE WHEN s.sales_channel_id IS NULL OR s.sales_channel_id !~ '^[0-9]{{1,4}}$' THEN 'invalid_sales_channel_id' END),
       concat_ws(',', s.t_dat, s.customer_id, s.article_id, s.price, s.sales_channel_id)
FROM staging.transactions_text s
LEFT JOIN staging.date_map dm ON dm.load_id = s.load_id AND dm.txt = s.t_dat
WHERE s.load_id = %(load_id)s
  AND NOT (dm.d IS NOT NULL
           AND coalesce(s.customer_id ~ '^[0-9a-f]{{64}}$', false)
           AND coalesce(s.article_id ~ '^[0-9]{{10}}$', false)
           AND coalesce(s.price ~ '^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$', false)
           AND coalesce(s.sales_channel_id ~ '^[0-9]{{1,4}}$', false));

-- name: raw_partition_totals
SELECT count(*) AS n_rows, coalesce(sum(price), 0) AS sum_price FROM {part};

-- name: staging_totals
SELECT count(*) AS n_rows FROM staging.transactions_text WHERE load_id = %(load_id)s;
