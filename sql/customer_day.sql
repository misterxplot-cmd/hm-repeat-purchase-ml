-- Dataset-scoped key map and customer-day aggregate.
-- Placeholders: {key_part}, {day_part} are partitions of derived.customer_key / derived.customer_day.

-- name: create_dataset_partition
CREATE TABLE {part} PARTITION OF {parent} FOR VALUES IN ({dataset_id});

-- name: customer_key
-- All customer_id values seen in customers.csv or transactions, restricted to the sampled hash buckets.
-- customer_sk = rank of customer_id in byte order: deterministic, one-to-one, reversible via this table.
INSERT INTO {key_part} (dataset_id, customer_sk, customer_id, in_customers_csv, in_transactions)
WITH ids AS (
    SELECT customer_id, bool_or(src = 'c') AS in_c, bool_or(src = 't') AS in_t
    FROM (
        SELECT customer_id, 'c' AS src FROM raw.customers
        WHERE load_id = %(customers_load_id)s AND customer_id IS NOT NULL
        UNION ALL
        SELECT DISTINCT customer_id, 't' AS src FROM raw.transactions
        WHERE load_id = %(transactions_load_id)s
    ) u
    GROUP BY customer_id
), sampled AS (
    SELECT * FROM ids
    WHERE (('x' || substr(encode(sha256(convert_to(%(salt)s || ':' || customer_id, 'UTF8')), 'hex'), 1, 8))::bit(32)::bigint
           %% %(buckets)s) < %(threshold)s
)
SELECT %(dataset_id)s, (row_number() OVER (ORDER BY customer_id COLLATE "C"))::integer, customer_id, in_c, in_t
FROM sampled;

-- name: customer_key_constraints
ALTER TABLE {key_part} ADD PRIMARY KEY (customer_sk);
CREATE UNIQUE INDEX ON {key_part} (customer_id);

-- name: customer_day
-- Grain (customer_sk, t_dat). Identical item rows are kept: they add to n_rows and spend,
-- while the day still counts once in purchase_days.
INSERT INTO {day_part} (dataset_id, customer_sk, t_dat, n_rows, spend, n_distinct_articles, n_rows_channel_2)
SELECT %(dataset_id)s, k.customer_sk, t.t_dat,
       count(*)::integer,
       sum(t.price),
       count(DISTINCT t.article_id)::integer,
       (count(*) FILTER (WHERE t.sales_channel_id = 2))::integer
FROM raw.transactions t
JOIN {key_part} k ON k.customer_id = t.customer_id
WHERE t.load_id = %(transactions_load_id)s
GROUP BY k.customer_sk, t.t_dat;

-- name: customer_day_constraints
-- The primary key doubles as the grain check: it fails if (customer_sk, t_dat) is not unique.
ALTER TABLE {day_part} ADD PRIMARY KEY (customer_sk, t_dat);
CREATE INDEX ON {day_part} (t_dat, customer_sk);
ANALYZE {day_part};
