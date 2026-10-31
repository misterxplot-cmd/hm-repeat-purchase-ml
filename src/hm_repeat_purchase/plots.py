"""Static report figures (matplotlib). Fixed categorical order; recessive grid; one y-axis per chart."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Colour follows the entity: each model keeps its slot in every figure.
MODEL_COLORS = {
    "recency_baseline": MUTED,
    "lr_core": SERIES[0],
    "catboost_core": SERIES[1],
    "catboost_ext": SERIES[2],
    "lr_ext": SERIES[3],
}
MODEL_LABELS = {
    "recency_baseline": "Recency baseline",
    "lr_core": "LR core",
    "catboost_core": "CatBoost core",
    "catboost_ext": "CatBoost extended",
    "lr_ext": "LR extended (diagnostic)",
}


def _style(ax: plt.Axes, title: str, ylabel: str | None = None, xlabel: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path


def daily_rows_by_channel(daily: pd.DataFrame, path: Path) -> Path:
    """daily: t_dat, sales_channel_id, n_rows."""
    piv = daily.pivot_table(index="t_dat", columns="sales_channel_id", values="n_rows", aggfunc="sum").fillna(0)
    piv.index = pd.to_datetime(piv.index)
    weekly = piv.resample("W").sum() / 1e3
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, ch in enumerate(weekly.columns):
        ax.plot(weekly.index, weekly[ch], color=SERIES[i], linewidth=2, label=f"sales_channel_id = {ch}")
    _style(ax, "Item rows per week by sales channel", ylabel="item rows per week, thousands")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return _save(fig, path)


def monthly_channel_share(monthly: pd.DataFrame, path: Path) -> Path:
    piv = monthly.pivot_table(index="month", columns="sales_channel_id", values="n_rows", aggfunc="sum").fillna(0)
    share = piv.div(piv.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    x = pd.to_datetime(share.index)
    col = share.columns[-1]
    ax.plot(x, share[col] * 100, color=SERIES[0], linewidth=2, marker="o", markersize=4)
    _style(ax, f"Share of item rows with sales_channel_id = {col}, by month", ylabel="% of item rows")
    ax.set_ylim(0, 100)
    return _save(fig, path)


def price_histogram(hist: pd.DataFrame, path: Path) -> Path:
    """hist: bucket_low (log10 price), n."""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(hist["bucket_low"], hist["n"] / 1e6, width=0.1 * 0.92, align="edge", color=SERIES[0])
    _style(ax, "Distribution of provided (scaled) price per item row", ylabel="item rows, millions",
           xlabel="log10(price), provided scale")
    return _save(fig, path)


def cohort_and_prevalence(cohort: pd.DataFrame, path: Path) -> Path:
    """Two separate panels (no dual axis): snapshot size and prevalence per cutoff."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
    x = pd.to_datetime(cohort["cutoff_date"])
    axes[0].bar(x, cohort["n_rows"] / 1e3, width=18, color=SERIES[0])
    _style(axes[0], "Customer snapshots per cutoff (purchase in previous 90 days)", ylabel="snapshots, thousands")
    axes[1].plot(x, cohort["prevalence"] * 100, color=SERIES[0], linewidth=2, marker="o", markersize=4)
    _style(axes[1], "Share with a purchase in the next 30 days (target = 1)", ylabel="% of snapshots")
    for role, marker in (("calibration", "s"), ("test", "D")):
        sel = cohort["role"] == role
        if sel.any():
            axes[1].scatter(x[sel], cohort.loc[sel, "prevalence"] * 100, s=60, color=SERIES[1], marker=marker,
                            zorder=3, edgecolor=SURFACE, linewidth=2)
            axes[1].annotate(role, (x[sel].iloc[0], cohort.loc[sel, "prevalence"].iloc[0] * 100),
                             textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8, color=INK_2)
    return _save(fig, path)


def feature_distributions(df: pd.DataFrame, features: Sequence[str], path: Path, title_suffix: str) -> Path:
    fig, axes = plt.subplots(1, len(features), figsize=(3.4 * len(features), 3.0))
    for ax, f in zip(np.atleast_1d(axes), features):
        v = df[f].dropna()
        if f == "avg_scaled_spend_per_purchase_day_90d":
            ax.hist(np.log10(v[v > 0]), bins=50, color=SERIES[0])
            _style(ax, f, xlabel="log10, provided price scale")
        else:
            upper = int(np.nanpercentile(v, 99)) + 1
            ax.hist(v.clip(upper=upper), bins=min(50, upper), color=SERIES[0])
            _style(ax, f, xlabel="days (capped at p99)")
    fig.suptitle(f"Core feature distributions, {title_suffix}", x=0.01, ha="left", fontsize=10, color=INK_2)
    return _save(fig, path)


def walk_forward(dev: pd.DataFrame, metric: str, path: Path, ylabel: str) -> Path:
    """dev: model, validation_cutoff, <metric> for the selected configuration of each model."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    models = [m for m in MODEL_COLORS if m in set(dev["model"])]
    for m in models:
        d = dev[dev["model"] == m].sort_values("validation_cutoff")
        ax.plot(pd.to_datetime(d["validation_cutoff"]), d[metric], color=MODEL_COLORS[m], linewidth=2, marker="o",
                markersize=5, label=MODEL_LABELS[m], linestyle="--" if m in ("recency_baseline", "lr_ext") else "-")
    _style(ax, f"Development walk-forward: {ylabel} per validation cutoff", ylabel=ylabel)
    ax.legend(frameon=False, fontsize=8, loc="best")
    return _save(fig, path)


def precision_lift_by_budget(curves: pd.DataFrame, prevalence: float, path: Path) -> Path:
    """curves: model, budget, precision, lift. Two panels instead of a dual axis."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for m in [m for m in MODEL_COLORS if m in set(curves["model"])]:
        d = curves[curves["model"] == m].sort_values("budget")
        ls = "--" if m == "recency_baseline" else "-"
        axes[0].plot(d["budget"] * 100, d["precision"] * 100, color=MODEL_COLORS[m], linewidth=2, linestyle=ls,
                     label=MODEL_LABELS[m])
        axes[1].plot(d["budget"] * 100, d["lift"], color=MODEL_COLORS[m], linewidth=2, linestyle=ls,
                     label=MODEL_LABELS[m])
    axes[0].axhline(prevalence * 100, color=BASELINE, linewidth=1)
    axes[0].annotate("prevalence", (axes[0].get_xlim()[1], prevalence * 100), ha="right", va="bottom", fontsize=8,
                     color=MUTED)
    axes[1].axhline(1.0, color=BASELINE, linewidth=1)
    _style(axes[0], "Final test: Precision@top-k", ylabel="precision, %", xlabel="top share of audience, %")
    _style(axes[1], "Final test: Lift@top-k", ylabel="lift (precision / prevalence)", xlabel="top share of audience, %")
    axes[0].legend(frameon=False, fontsize=8)
    return _save(fig, path)


def reliability(bins: pd.DataFrame, path: Path) -> Path:
    """bins: model, kind (raw/calibrated), mean_pred, frac_pos, n."""
    kinds = list(dict.fromkeys(bins["kind"]))
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.6 * len(kinds), 4.0), sharey=True)
    for ax, kind in zip(np.atleast_1d(axes), kinds):
        ax.plot([0, 1], [0, 1], color=BASELINE, linewidth=1)
        for m in [m for m in MODEL_COLORS if m in set(bins["model"])]:
            d = bins[(bins["model"] == m) & (bins["kind"] == kind)]
            ax.plot(d["mean_pred"], d["frac_pos"], color=MODEL_COLORS[m], linewidth=2, marker="o", markersize=4,
                    label=MODEL_LABELS[m])
        _style(ax, f"Final test reliability: {kind} probability", ylabel="observed share target = 1",
               xlabel="mean predicted probability (decile bins)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, loc="upper left")
    return _save(fig, path)


def stage_durations(stages: pd.DataFrame, path: Path) -> Path:
    d = stages.sort_values("wall_seconds")
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(d) + 1.2))
    ax.barh(d["stage"], d["wall_seconds"] / 60, color=SERIES[0], height=0.7)
    for y, v in enumerate(d["wall_seconds"] / 60):
        ax.annotate(f"{v:.1f}", (v, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8,
                    color=INK_2)
    _style(ax, "Measured wall time per stage", xlabel="minutes")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    return _save(fig, path)
