-- Schema layers:
--   meta     : source file manifest, loads, datasets
--   staging  : untyped text copies used only during ingest (UNLOGGED, truncated after load)
--   raw      : typed, immutable source rows, partitioned by load_id
--   derived  : dataset-scoped aggregates, partitioned by dataset_id
-- Partitions are created by the Python code (one per load_id / dataset_id), so a rebuild
-- drops exactly one partition and never touches other loads or datasets.

-- name: create_schema
CREATE SCHEMA IF NOT EXISTS meta;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS derived;

CREATE TABLE IF NOT EXISTS meta.schema_version (
    version     text PRIMARY KEY,
    applied_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS meta.source_file (
    sha256              text PRIMARY KEY,
    file_role           text NOT NULL,
    file_name           text NOT NULL,
    size_bytes          bigint NOT NULL,
    first_registered_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS meta.load (
    load_id                 serial PRIMARY KEY,
    file_role               text NOT NULL CHECK (file_role IN ('transactions', 'customers', 'articles')),
    sha256                  text NOT NULL REFERENCES meta.source_file (sha256),
    schema_version          text NOT NULL,
    status                  text NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'superseded')),
    rows_read               bigint,
    rows_loaded             bigint,
    rows_rejected_structure bigint,
    rows_rejected_type      bigint,
    started_at              timestamptz NOT NULL DEFAULT now(),
    finished_at             timestamptz,
    error                   text
);
CREATE UNIQUE INDEX IF NOT EXISTS load_one_completed
    ON meta.load (file_role, sha256, schema_version) WHERE status = 'completed';

CREATE UNLOGGED TABLE IF NOT EXISTS staging.transactions_text (
    load_id          integer NOT NULL,
    src_row          bigint  NOT NULL,
    t_dat            text,
    customer_id      text,
    article_id       text,
    price            text,
    sales_channel_id text
);

CREATE UNLOGGED TABLE IF NOT EXISTS staging.date_map (
    load_id integer NOT NULL,
    txt     text    NOT NULL,
    d       date,
    PRIMARY KEY (load_id, txt)
);

CREATE TABLE IF NOT EXISTS raw.transactions (
    load_id          integer          NOT NULL,
    src_row          bigint           NOT NULL,
    t_dat            date             NOT NULL,
    customer_id      text             NOT NULL,
    article_id       text             NOT NULL,
    price            double precision NOT NULL,
    sales_channel_id smallint         NOT NULL
) PARTITION BY LIST (load_id);

CREATE TABLE IF NOT EXISTS raw.transactions_rejected (
    load_id  integer NOT NULL,
    src_row  bigint  NOT NULL,
    reason   text    NOT NULL,
    raw_line text,
    PRIMARY KEY (load_id, src_row)
);

-- Reference tables are kept as source text: they are used for audit and descriptive JOIN checks only.
CREATE TABLE IF NOT EXISTS raw.customers (
    load_id                integer NOT NULL,
    src_row                bigint  NOT NULL,
    customer_id            text,
    fn                     text,
    active                 text,
    club_member_status     text,
    fashion_news_frequency text,
    age                    text,
    postal_code            text
) PARTITION BY LIST (load_id);

CREATE TABLE IF NOT EXISTS raw.articles (
    load_id                      integer NOT NULL,
    src_row                      bigint  NOT NULL,
    article_id                   text,
    product_code                 text,
    prod_name                    text,
    product_type_no              text,
    product_type_name            text,
    product_group_name           text,
    graphical_appearance_no      text,
    graphical_appearance_name    text,
    colour_group_code            text,
    colour_group_name            text,
    perceived_colour_value_id    text,
    perceived_colour_value_name  text,
    perceived_colour_master_id   text,
    perceived_colour_master_name text,
    department_no                text,
    department_name              text,
    index_code                   text,
    index_name                   text,
    index_group_no               text,
    index_group_name             text,
    section_no                   text,
    section_name                 text,
    garment_group_no             text,
    garment_group_name           text,
    detail_desc                  text
) PARTITION BY LIST (load_id);

CREATE TABLE IF NOT EXISTS meta.dataset (
    dataset_id           text PRIMARY KEY,
    mode                 text NOT NULL,
    sample_fraction      double precision NOT NULL,
    hash_rule            text NOT NULL,
    transactions_load_id integer NOT NULL REFERENCES meta.load (load_id),
    customers_load_id    integer NOT NULL REFERENCES meta.load (load_id),
    articles_load_id     integer NOT NULL REFERENCES meta.load (load_id),
    identity             jsonb NOT NULL,
    status               text NOT NULL CHECK (status IN ('building', 'completed', 'failed')),
    created_at           timestamptz NOT NULL DEFAULT now(),
    completed_at         timestamptz
);

-- Reversible mapping customer_id <-> customer_sk, one per dataset.
CREATE TABLE IF NOT EXISTS derived.customer_key (
    dataset_id       text    NOT NULL,
    customer_sk      integer NOT NULL,
    customer_id      text    NOT NULL,
    in_customers_csv boolean NOT NULL,
    in_transactions  boolean NOT NULL
) PARTITION BY LIST (dataset_id);

-- Grain: (dataset_id, customer_sk, t_dat).
CREATE TABLE IF NOT EXISTS derived.customer_day (
    dataset_id          text             NOT NULL,
    customer_sk         integer          NOT NULL,
    t_dat               date             NOT NULL,
    n_rows              integer          NOT NULL,
    spend               double precision NOT NULL,
    n_distinct_articles integer          NOT NULL,
    n_rows_channel_2    integer          NOT NULL
) PARTITION BY LIST (dataset_id);

CREATE TABLE IF NOT EXISTS derived.calendar (
    dataset_id            text    NOT NULL,
    cutoff_date           date    NOT NULL,
    history_start         date    NOT NULL,
    history_end_exclusive date    NOT NULL,
    target_start          date    NOT NULL,
    target_end_exclusive  date    NOT NULL,
    role                  text    NOT NULL,
    is_eligible           boolean NOT NULL,
    eligibility_reason    text    NOT NULL,
    PRIMARY KEY (dataset_id, cutoff_date)
);

-- Grain: (dataset_id, cutoff_date, customer_sk). Features only; labels are stored separately.
CREATE TABLE IF NOT EXISTS derived.snapshot (
    dataset_id                            text             NOT NULL,
    cutoff_date                           date             NOT NULL,
    customer_sk                           integer          NOT NULL,
    history_start                         date             NOT NULL,
    history_end_exclusive                 date             NOT NULL,
    feature_build_version                 text             NOT NULL,
    eligibility_reason                    text             NOT NULL,
    purchase_days_90d                     integer          NOT NULL,
    avg_scaled_spend_per_purchase_day_90d double precision NOT NULL,
    recency_days                          integer          NOT NULL,
    purchase_days_7d                      integer          NOT NULL,
    purchase_days_30d                     integer          NOT NULL,
    rows_7d                               integer          NOT NULL,
    rows_30d                              integer          NOT NULL,
    rows_90d                              integer          NOT NULL,
    spend_7d                              double precision NOT NULL,
    spend_30d                             double precision NOT NULL,
    spend_90d                             double precision NOT NULL,
    avg_price_per_row_90d                 double precision NOT NULL,
    first_purchase_age_days_90d           integer          NOT NULL,
    mean_gap_days_90d                     double precision,
    share_rows_channel_2_90d              double precision NOT NULL,
    recent_share_purchase_days_30d        double precision NOT NULL
) PARTITION BY LIST (dataset_id);

CREATE TABLE IF NOT EXISTS derived.label (
    dataset_id           text     NOT NULL,
    cutoff_date          date     NOT NULL,
    customer_sk          integer  NOT NULL,
    target_start         date     NOT NULL,
    target_end_exclusive date     NOT NULL,
    target               smallint NOT NULL CHECK (target IN (0, 1))
) PARTITION BY LIST (dataset_id);

INSERT INTO meta.schema_version (version) VALUES ('hmrp_schema_v1') ON CONFLICT DO NOTHING;
