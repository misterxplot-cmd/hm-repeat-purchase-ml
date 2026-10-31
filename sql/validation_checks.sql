-- Cardinality and reconciliation checks. Each query returns numbers that Python compares
-- against explicit expectations; a failed expectation stops the stage.

-- name: key_uniqueness
-- Generic: {table} with key column {key}.
SELECT count(*) AS n_rows, count(DISTINCT {key}) AS n_distinct, count(*) FILTER (WHERE {key} IS NULL) AS n_null
FROM {table};

-- name: many_to_one_join
-- Fact rows joined to a dimension on {key}. If the dimension key is unique, row count and sum are
-- unchanged by the LEFT JOIN; n_unmatched counts fact rows without a dimension row.
SELECT
    (SELECT count(*) FROM {fact})                                AS fact_rows,
    (SELECT coalesce(sum({measure}), 0) FROM {fact})             AS fact_sum,
    count(*)                                                     AS joined_rows,
    coalesce(sum(f.{measure}), 0)                                AS joined_sum,
    count(*) FILTER (WHERE d.{key} IS NULL)                      AS n_unmatched
FROM {fact} f
LEFT JOIN {dim} d ON d.{key} = f.{key};

-- name: customer_day_reconciliation
-- Raw rows of the sampled customers vs. the customer_day aggregate.
SELECT
    (SELECT count(*) FROM raw.transactions t JOIN {key_part} k ON k.customer_id = t.customer_id
      WHERE t.load_id = %(transactions_load_id)s)                         AS raw_rows,
    (SELECT coalesce(sum(t.price), 0) FROM raw.transactions t JOIN {key_part} k ON k.customer_id = t.customer_id
      WHERE t.load_id = %(transactions_load_id)s)                         AS raw_sum_price,
    (SELECT coalesce(sum(n_rows), 0) FROM {day_part})                     AS day_rows,
    (SELECT coalesce(sum(spend), 0) FROM {day_part})                      AS day_sum_price,
    (SELECT count(*) FROM {day_part})                                     AS n_customer_days,
    (SELECT count(DISTINCT customer_sk) FROM {day_part})                  AS n_customers_with_purchases;

-- name: customer_day_distribution
-- Purchase days per customer over the whole observed period.
SELECT purchase_days, count(*) AS n_customers
FROM (SELECT customer_sk, count(*) AS purchase_days FROM {day_part} GROUP BY customer_sk) x
GROUP BY purchase_days ORDER BY purchase_days;

-- name: snapshot_label_alignment
-- Grain check before the feature/label join: both sides must be unique on (cutoff_date, customer_sk)
-- and match one-to-one.
SELECT s.cutoff_date,
       count(*)                               AS n_snapshot,
       count(l.customer_sk)                   AS n_labelled,
       sum(l.target)                          AS n_positive,
       min(l.target_end_exclusive)            AS target_end_exclusive
FROM {snap_part} s
LEFT JOIN {label_part} l ON l.cutoff_date = s.cutoff_date AND l.customer_sk = s.customer_sk
GROUP BY s.cutoff_date ORDER BY s.cutoff_date;

-- name: label_maturity
-- No label may have a target window ending after observation_end_exclusive.
SELECT count(*) AS n_immature FROM {label_part} WHERE target_end_exclusive > %(observation_end_exclusive)s::date;

-- name: history_bounds
-- Snapshot features must never use dates outside [history_start, cutoff_date).
SELECT count(*) AS n_violations
FROM {snap_part}
WHERE history_end_exclusive <> cutoff_date
   OR history_start <> cutoff_date - %(history_days)s
   OR recency_days < 1 OR recency_days > %(history_days)s
   OR first_purchase_age_days_90d < recency_days;
