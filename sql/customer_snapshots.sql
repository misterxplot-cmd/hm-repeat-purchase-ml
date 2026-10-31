-- Customer snapshot features for one cutoff_date.
-- History window: [history_start, cutoff_date), history_start = cutoff_date - history_days.
-- Only customer-days strictly before cutoff_date are read, so a purchase on cutoff_date
-- belongs to the target window and never to the features.
-- Cohort: customers with at least one purchase day inside the history window.

-- name: snapshot_features_select
SELECT
    cd.customer_sk,
    count(*)::integer                                                            AS purchase_days_90d,
    sum(cd.spend) / count(*)                                                     AS avg_scaled_spend_per_purchase_day_90d,
    ({cutoff} - max(cd.t_dat))::integer                                  AS recency_days,
    (count(*) FILTER (WHERE cd.t_dat >= {cutoff} - 7))::integer          AS purchase_days_7d,
    (count(*) FILTER (WHERE cd.t_dat >= {cutoff} - 30))::integer         AS purchase_days_30d,
    coalesce(sum(cd.n_rows) FILTER (WHERE cd.t_dat >= {cutoff} - 7), 0)::integer  AS rows_7d,
    coalesce(sum(cd.n_rows) FILTER (WHERE cd.t_dat >= {cutoff} - 30), 0)::integer AS rows_30d,
    sum(cd.n_rows)::integer                                                      AS rows_90d,
    coalesce(sum(cd.spend) FILTER (WHERE cd.t_dat >= {cutoff} - 7), 0)   AS spend_7d,
    coalesce(sum(cd.spend) FILTER (WHERE cd.t_dat >= {cutoff} - 30), 0)  AS spend_30d,
    sum(cd.spend)                                                                AS spend_90d,
    sum(cd.spend) / sum(cd.n_rows)                                               AS avg_price_per_row_90d,
    ({cutoff} - min(cd.t_dat))::integer                                  AS first_purchase_age_days_90d,
    CASE WHEN count(*) > 1
         THEN (max(cd.t_dat) - min(cd.t_dat))::double precision / (count(*) - 1)
    END                                                                          AS mean_gap_days_90d,
    sum(cd.n_rows_channel_2)::double precision / sum(cd.n_rows)                  AS share_rows_channel_2_90d,
    (count(*) FILTER (WHERE cd.t_dat >= {cutoff} - 30))::double precision / count(*)
                                                                                 AS recent_share_purchase_days_30d
FROM {day_part} cd
WHERE cd.t_dat >= {history_start}
  AND cd.t_dat <  {cutoff}
GROUP BY cd.customer_sk

-- name: insert_snapshot
INSERT INTO {snap_part} (
    dataset_id, cutoff_date, customer_sk, history_start, history_end_exclusive,
    feature_build_version, eligibility_reason,
    purchase_days_90d, avg_scaled_spend_per_purchase_day_90d, recency_days,
    purchase_days_7d, purchase_days_30d, rows_7d, rows_30d, rows_90d,
    spend_7d, spend_30d, spend_90d, avg_price_per_row_90d, first_purchase_age_days_90d,
    mean_gap_days_90d, share_rows_channel_2_90d, recent_share_purchase_days_30d)
SELECT %(dataset_id)s, %(cutoff)s::date, f.customer_sk, %(history_start)s::date, %(cutoff)s::date,
       %(feature_build_version)s, 'purchase_in_history_window',
       f.purchase_days_90d, f.avg_scaled_spend_per_purchase_day_90d, f.recency_days,
       f.purchase_days_7d, f.purchase_days_30d, f.rows_7d, f.rows_30d, f.rows_90d,
       f.spend_7d, f.spend_30d, f.spend_90d, f.avg_price_per_row_90d, f.first_purchase_age_days_90d,
       f.mean_gap_days_90d, f.share_rows_channel_2_90d, f.recent_share_purchase_days_30d
FROM ({features}) f;

-- name: snapshot_constraints
ALTER TABLE {snap_part} ADD PRIMARY KEY (cutoff_date, customer_sk);
ANALYZE {snap_part};

-- name: export_snapshot
COPY (
    SELECT s.cutoff_date, s.customer_sk, k.customer_id, s.history_start, s.history_end_exclusive,
           s.feature_build_version, s.eligibility_reason,
           s.purchase_days_90d, s.avg_scaled_spend_per_purchase_day_90d, s.recency_days,
           s.purchase_days_7d, s.purchase_days_30d, s.rows_7d, s.rows_30d, s.rows_90d,
           s.spend_7d, s.spend_30d, s.spend_90d, s.avg_price_per_row_90d, s.first_purchase_age_days_90d,
           s.mean_gap_days_90d, s.share_rows_channel_2_90d, s.recent_share_purchase_days_30d
    FROM {snap_part} s
    JOIN {key_part} k ON k.customer_sk = s.customer_sk
    WHERE s.cutoff_date = {cutoff}
    ORDER BY s.customer_sk
) TO STDOUT WITH (FORMAT csv, HEADER true)

-- name: export_scoring_features
COPY (
    SELECT f.*, k.customer_id
    FROM ({features}) f
    JOIN {key_part} k ON k.customer_sk = f.customer_sk
    ORDER BY f.customer_sk
) TO STDOUT WITH (FORMAT csv, HEADER true)
