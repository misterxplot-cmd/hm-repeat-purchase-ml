"""Ranking and probability metrics, deterministic tie handling and paired cluster bootstrap.

Ordering contract: rank by raw_score descending; ties broken by tie_key ascending (a hash of
customer_id, independent of the target). AP and ROC-AUC are threshold metrics on raw_score
(tied scores form one threshold, as in scikit-learn); Precision@k / Lift@k use the full order.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def ranking_order(score: np.ndarray, tie_key: np.ndarray) -> np.ndarray:
    """Indices from best to worst: score desc, then tie_key asc."""
    return np.lexsort((np.asarray(tie_key), -np.asarray(score, dtype=float)))


def budget_k(n: int, share: float) -> int:
    return int(math.ceil(share * n))


def precision_lift_at(y: np.ndarray, score: np.ndarray, tie_key: np.ndarray, share: float) -> dict[str, Any]:
    y = np.asarray(y)
    n = len(y)
    if n == 0:
        return {"k": 0, "n": 0, "status": "empty"}
    k = budget_k(n, share)
    order = ranking_order(score, tie_key)
    hits = int(y[order[:k]].sum())
    precision = hits / k
    prevalence = float(y.mean())
    out = {"share": share, "k": k, "n": n, "hits": hits, "precision": precision, "prevalence": prevalence}
    if prevalence == 0:
        out.update(lift=None, lift_status="undefined_prevalence_zero")
    else:
        out.update(lift=precision / prevalence, lift_status="ok")
    return out


def _both_classes(y: np.ndarray) -> bool:
    return 0 < int(np.sum(y)) < len(y)


def evaluate_scores(y: np.ndarray, raw_score: np.ndarray, tie_key: np.ndarray, budgets: Sequence[float],
                    probability: np.ndarray | None = None, calibrated: np.ndarray | None = None) -> dict[str, Any]:
    y = np.asarray(y).astype(int)
    n = len(y)
    out: dict[str, Any] = {"n": n, "n_positive": int(y.sum()), "prevalence": float(y.mean()) if n else None}
    if _both_classes(y):
        out["roc_auc"] = float(roc_auc_score(y, raw_score))
        out["average_precision"] = float(average_precision_score(y, raw_score))
        out["ranking_status"] = "ok"
    else:
        out["roc_auc"] = out["average_precision"] = None
        out["ranking_status"] = "not_available_single_class"
    out["at_budget"] = {f"top_{round(b * 100)}pct": precision_lift_at(y, raw_score, tie_key, b) for b in budgets}
    for name, p in (("raw_probability", probability), ("calibrated_probability", calibrated)):
        if p is None:
            out[name] = {"status": "not_applicable"}
            continue
        p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
        res = {"brier": float(brier_score_loss(y, p)), "mean_predicted": float(p.mean()), "status": "ok"}
        res["log_loss"] = float(log_loss(y, p, labels=[0, 1])) if n else None
        out[name] = res
    return out


def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    """Equal-frequency bins of predicted probability."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    rows = []
    for idx in np.array_split(order, n_bins):
        if len(idx):
            rows.append({"mean_pred": float(p[idx].mean()), "frac_pos": float(y[idx].mean()), "n": int(len(idx))})
    return rows


# --------------------------------------------------------------------------- weighted (bootstrap) metrics
# A bootstrap resample is represented by integer multiplicities w (multinomial). The functions below give
# exactly the statistics of the explicitly repeated sample (tests compare them with scikit-learn on np.repeat).

class SortedScores:
    """Pre-sorted view of one model's scores for repeated weighted evaluation."""

    def __init__(self, y: np.ndarray, score: np.ndarray, tie_key: np.ndarray):
        self.order = ranking_order(score, tie_key)
        s = np.asarray(score, dtype=float)[self.order]
        self.y = np.asarray(y, dtype=float)[self.order]
        # Group boundaries of equal scores (for AP / AUC, which treat ties as one threshold).
        new_group = np.empty(len(s), dtype=bool)
        new_group[0] = True
        new_group[1:] = s[1:] != s[:-1]
        self.group_id = np.cumsum(new_group) - 1
        self.n_groups = int(self.group_id[-1]) + 1


def weighted_ap_auc(ss: SortedScores, w: np.ndarray) -> tuple[float | None, float | None]:
    w = np.asarray(w, dtype=float)[ss.order]
    tp_g = np.bincount(ss.group_id, weights=w * ss.y, minlength=ss.n_groups)
    fp_g = np.bincount(ss.group_id, weights=w * (1 - ss.y), minlength=ss.n_groups)
    P, N = tp_g.sum(), fp_g.sum()
    if P == 0 or N == 0:
        return None, None
    tp = np.cumsum(tp_g)
    fp = np.cumsum(fp_g)
    keep = (tp_g + fp_g) > 0
    tp_k, fp_k = tp[keep], fp[keep]
    precision = tp_k / (tp_k + fp_k)
    recall = tp_k / P
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    ap = float(np.sum((recall - recall_prev) * precision))
    # AUC: P(score_pos > score_neg) + 0.5 P(tie); negatives strictly below a group = N - cumulative fp.
    neg_below = N - fp
    auc = float(np.sum(tp_g * (neg_below + 0.5 * fp_g)) / (P * N))
    return ap, auc


def weighted_precision_at(ss: SortedScores, w: np.ndarray, share: float) -> float:
    w = np.asarray(w, dtype=float)[ss.order]
    n = w.sum()
    k = budget_k(int(round(n)), share)
    cum = np.cumsum(w)
    # Items fully inside the top-k, plus the copies of the boundary item that still fit.
    inside = cum <= k
    hits = float(np.sum(w[inside] * ss.y[inside]))
    j = int(np.searchsorted(cum, k, side="left"))
    if j < len(w) and not inside[j]:
        before = cum[j - 1] if j > 0 else 0.0
        hits += (k - before) * ss.y[j]
    return hits / k


def paired_bootstrap(y: np.ndarray, scores: dict[str, np.ndarray], tie_key: np.ndarray, share: float,
                     n_resamples: int, seed: int, pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Customer-level paired bootstrap: one multinomial weight vector per resample, shared by all models.
    One row per customer in the final test slice, so the customer is the resampling cluster."""
    rng = np.random.default_rng(seed)
    n = len(y)
    sorted_scores = {m: SortedScores(y, s, tie_key) for m, s in scores.items()}
    stats = {m: {"average_precision": [], "roc_auc": [], "precision_at": []} for m in scores}
    skipped = 0
    for _ in range(n_resamples):
        w = rng.multinomial(n, np.full(n, 1.0 / n))
        if np.sum(w * y) in (0, n):
            skipped += 1
            continue
        for m, ss in sorted_scores.items():
            ap, auc = weighted_ap_auc(ss, w)
            stats[m]["average_precision"].append(ap)
            stats[m]["roc_auc"].append(auc)
            stats[m]["precision_at"].append(weighted_precision_at(ss, w, share))

    def ci(values: list[float]) -> dict[str, float | None]:
        a = np.asarray([v for v in values if v is not None], dtype=float)
        if len(a) == 0:
            return {"low": None, "high": None, "mean": None, "n": 0}
        return {"low": float(np.quantile(a, 0.025)), "high": float(np.quantile(a, 0.975)),
                "mean": float(a.mean()), "n": int(len(a))}

    out: dict[str, Any] = {"n_resamples_requested": n_resamples, "n_resamples_used": n_resamples - skipped,
                           "n_skipped_single_class": skipped, "seed": seed, "share": share,
                           "cluster": "customer (one row per customer in the final test slice)",
                           "interval": "percentile 2.5-97.5", "models": {}, "differences": {}}
    for m, st in stats.items():
        out["models"][m] = {k: ci(v) for k, v in st.items()}
    for a, b in pairs:
        out["differences"][f"{a}_minus_{b}"] = {
            k: ci(list(np.asarray(stats[a][k], dtype=float) - np.asarray(stats[b][k], dtype=float)))
            for k in ("average_precision", "roc_auc", "precision_at")
        }
    return out
