"""Synthetic H&M-shaped CSV files for tests. Artificial identifiers; never H&M results."""
from __future__ import annotations

import csv
import hashlib
from datetime import date, timedelta
from pathlib import Path

import numpy as np

TX_HEADER = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
CUST_HEADER = ["customer_id", "FN", "Active", "club_member_status", "fashion_news_frequency", "age", "postal_code"]
ART_HEADER = [
    "article_id", "product_code", "prod_name", "product_type_no", "product_type_name", "product_group_name",
    "graphical_appearance_no", "graphical_appearance_name", "colour_group_code", "colour_group_name",
    "perceived_colour_value_id", "perceived_colour_value_name", "perceived_colour_master_id",
    "perceived_colour_master_name", "department_no", "department_name", "index_code", "index_name",
    "index_group_no", "index_group_name", "section_no", "section_name", "garment_group_no",
    "garment_group_name", "detail_desc",
]


def cid(i: int) -> str:
    return hashlib.sha256(f"synthetic-customer-{i}".encode()).hexdigest()


def aid(i: int) -> str:
    return f"{i:010d}"


def write_csv(path: Path, header: list[str], rows: list[list]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return path


def articles_rows(n: int = 40) -> list[list]:
    rows = []
    for i in range(1, n + 1):
        desc = 'Soft top, "relaxed" fit, with ribbing' if i % 3 == 0 else "Jersey top"
        rows.append([aid(i), f"{i:07d}", f"Item {i}", "253", "Vest top", "Garment Upper body", "1010016", "Solid",
                     "09", "Black", "4", "Dark", "5", "Black", "1676", "Jersey Basic", "A", "Ladieswear", "1",
                     "Ladieswear", "16", "Womens Everyday Basics", "1002", "Jersey Basic", desc])
    return rows


def generate(out_dir: Path, n_customers: int = 240, seed: int = 11,
             start: date = date(2018, 9, 20), end: date = date(2020, 9, 22)) -> dict[str, Path]:
    """Customers with heterogeneous purchase rates; several items per day; some identical rows."""
    rng = np.random.default_rng(seed)
    n_days = (end - start).days + 1
    tx = []
    for i in range(n_customers):
        rate = rng.choice([0.004, 0.01, 0.03, 0.08])
        active_from = int(rng.integers(0, n_days // 2))
        for d in range(active_from, n_days):
            if rng.random() < rate:
                day = start + timedelta(days=d)
                channel = int(rng.choice([1, 2]))
                for _ in range(int(rng.integers(1, 4))):
                    a = int(rng.integers(1, 41))
                    price = round(float(rng.uniform(0.005, 0.08)), 6)
                    copies = 2 if rng.random() < 0.1 else 1  # identical rows: several units of one item
                    tx.extend([[day.isoformat(), cid(i), aid(a), repr(price), channel]] * copies)
    # Every calendar day has at least one row (as in the source); one customer buys every day.
    for d in range(n_days):
        tx.append([(start + timedelta(days=d)).isoformat(), cid(n_customers), aid(1), "0.01", 2])
    tx.sort(key=lambda r: r[0])
    customers = [[cid(i), "1.0" if i % 2 else "", "1.0" if i % 3 else "", "ACTIVE", "NONE", str(20 + i % 50),
                  hashlib.sha256(f"pc{i % 7}".encode()).hexdigest()] for i in range(n_customers + 1)]
    return {
        "transactions": write_csv(out_dir / "transactions_train.csv", TX_HEADER, tx),
        "customers": write_csv(out_dir / "customers.csv", CUST_HEADER, customers),
        "articles": write_csv(out_dir / "articles.csv", ART_HEADER, articles_rows()),
    }
