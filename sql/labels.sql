-- Labels for one cutoff. Built only for snapshot customers and only when the full target window
-- [cutoff_date, cutoff_date + horizon_days) lies before observation_end_exclusive (checked in Python
-- and again by validation_checks.sql). target = 1 if the customer has at least one purchase row in
-- the window; several items or days in the window still give a single positive label.

-- name: insert_labels
INSERT INTO {label_part} (dataset_id, cutoff_date, customer_sk, target_start, target_end_exclusive, target)
SELECT %(dataset_id)s, s.cutoff_date, s.customer_sk, %(cutoff)s::date, %(target_end)s::date,
       CASE WHEN f.customer_sk IS NULL THEN 0 ELSE 1 END
FROM {snap_part} s
LEFT JOIN (
    SELECT DISTINCT customer_sk
    FROM {day_part}
    WHERE t_dat >= %(cutoff)s::date AND t_dat < %(target_end)s::date
) f ON f.customer_sk = s.customer_sk
WHERE s.cutoff_date = %(cutoff)s::date;

-- name: label_constraints
ALTER TABLE {label_part} ADD PRIMARY KEY (cutoff_date, customer_sk);
ANALYZE {label_part};

-- name: export_labels
COPY (
    SELECT l.cutoff_date, l.customer_sk, l.target_start, l.target_end_exclusive, l.target
    FROM {label_part} l
    WHERE l.cutoff_date = {cutoff}
    ORDER BY l.customer_sk
) TO STDOUT WITH (FORMAT csv, HEADER true)
