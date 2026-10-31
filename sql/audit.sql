-- Source-level audit queries over raw partitions. {tx}, {cust}, {art} are raw partitions of one load each.
-- Nothing here modifies data.

-- name: tx_overview
SELECT count(*)                    AS n_rows,
       count(DISTINCT customer_id) AS n_customers,
       count(DISTINCT article_id)  AS n_articles,
       min(t_dat)                  AS min_date,
       max(t_dat)                  AS max_date,
       count(DISTINCT t_dat)       AS n_distinct_dates,
       sum(price)                  AS sum_price
FROM {tx};

-- name: tx_daily_by_channel
SELECT t_dat, sales_channel_id, count(*) AS n_rows, sum(price) AS sum_price,
       count(DISTINCT customer_id) AS n_customers
FROM {tx}
GROUP BY t_dat, sales_channel_id
ORDER BY t_dat, sales_channel_id;

-- name: tx_price_stats
SELECT min(price) AS min, max(price) AS max, avg(price) AS mean, stddev_samp(price) AS std,
       percentile_disc(ARRAY[0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]) WITHIN GROUP (ORDER BY price) AS quantiles,
       count(*) FILTER (WHERE price = 0)  AS n_zero,
       count(*) FILTER (WHERE price < 0)  AS n_negative,
       count(*) FILTER (WHERE price = 'NaN'::double precision OR price IN ('Infinity', '-Infinity')) AS n_non_finite
FROM {tx};

-- name: tx_channels
SELECT sales_channel_id, count(*) AS n_rows, sum(price) AS sum_price
FROM {tx} GROUP BY sales_channel_id ORDER BY sales_channel_id;

-- name: tx_monthly_channel
SELECT date_trunc('month', t_dat)::date AS month, sales_channel_id, count(*) AS n_rows,
       count(DISTINCT customer_id) AS n_customers
FROM {tx} GROUP BY 1, 2 ORDER BY 1, 2;

-- name: tx_identical_rows
-- Multiplicity of fully identical rows (all five source fields). Reported, never removed:
-- without order_id or quantity, identical rows can be several units of one item.
SELECT multiplicity, count(*) AS n_groups, sum(multiplicity) AS n_rows
FROM (
    SELECT count(*) AS multiplicity
    FROM {tx}
    GROUP BY t_dat, customer_id, article_id, price, sales_channel_id
) g
GROUP BY multiplicity ORDER BY multiplicity;

-- name: tx_customer_coverage
SELECT count(*) AS n_tx_customers,
       count(*) FILTER (WHERE c.customer_id IS NULL) AS n_not_in_customers_csv
FROM (SELECT DISTINCT customer_id FROM {tx}) t
LEFT JOIN (SELECT DISTINCT customer_id FROM {cust}) c USING (customer_id);

-- name: tx_article_coverage
SELECT count(*) AS n_tx_articles,
       count(*) FILTER (WHERE a.article_id IS NULL) AS n_not_in_articles_csv
FROM (SELECT DISTINCT article_id FROM {tx}) t
LEFT JOIN (SELECT DISTINCT article_id FROM {art}) a USING (article_id);

-- name: customers_overview
SELECT count(*) AS n_rows,
       count(DISTINCT customer_id) AS n_distinct_customer_id,
       count(*) FILTER (WHERE customer_id IS NULL OR customer_id = '') AS n_null_customer_id,
       count(*) FILTER (WHERE customer_id !~ '^[0-9a-f]{{64}}$') AS n_bad_format_customer_id,
       count(*) FILTER (WHERE fn IS NULL OR fn = '') AS n_null_fn,
       count(*) FILTER (WHERE active IS NULL OR active = '') AS n_null_active,
       count(*) FILTER (WHERE club_member_status IS NULL OR club_member_status = '') AS n_null_club_member_status,
       count(*) FILTER (WHERE fashion_news_frequency IS NULL OR fashion_news_frequency = '') AS n_null_fashion_news_frequency,
       count(*) FILTER (WHERE age IS NULL OR age = '') AS n_null_age,
       count(*) FILTER (WHERE age !~ '^[0-9]+(\.0+)?$' AND age <> '') AS n_non_integer_age,
       count(*) FILTER (WHERE postal_code IS NULL OR postal_code = '') AS n_null_postal_code
FROM {cust};

-- name: customers_categoricals
SELECT 'fn' AS field, coalesce(nullif(fn, ''), '<empty>') AS value, count(*) AS n FROM {cust} GROUP BY 2
UNION ALL
SELECT 'active', coalesce(nullif(active, ''), '<empty>'), count(*) FROM {cust} GROUP BY 2
UNION ALL
SELECT 'club_member_status', coalesce(nullif(club_member_status, ''), '<empty>'), count(*) FROM {cust} GROUP BY 2
UNION ALL
SELECT 'fashion_news_frequency', coalesce(nullif(fashion_news_frequency, ''), '<empty>'), count(*) FROM {cust} GROUP BY 2
ORDER BY 1, 3 DESC;

-- name: articles_overview
SELECT count(*) AS n_rows,
       count(DISTINCT article_id) AS n_distinct_article_id,
       count(*) FILTER (WHERE article_id IS NULL OR article_id = '') AS n_null_article_id,
       count(*) FILTER (WHERE article_id !~ '^[0-9]{{10}}$') AS n_bad_format_article_id,
       count(*) FILTER (WHERE detail_desc IS NULL OR detail_desc = '') AS n_null_detail_desc,
       count(DISTINCT product_group_name) AS n_product_groups
FROM {art};

-- name: rejected_by_reason
SELECT reason, count(*) AS n FROM raw.transactions_rejected WHERE load_id = %(load_id)s GROUP BY reason ORDER BY 2 DESC;

-- name: tx_price_log_histogram
-- 0.1-wide buckets of log10(price) for positive prices.
SELECT floor(log(price) * 10) / 10.0 AS bucket_low, count(*) AS n
FROM {tx} WHERE price > 0 GROUP BY 1 ORDER BY 1;
