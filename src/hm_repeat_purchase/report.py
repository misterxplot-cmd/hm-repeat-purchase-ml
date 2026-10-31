"""Markdown reports, figures and README blocks generated only from saved run artifacts."""
from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from . import features as feat
from . import plots
from .config import Config, project_root
from .tracking import read_json, relpath, resolve_run_id, run_paths, utc_now, write_json

log = logging.getLogger(__name__)

MODEL_ORDER = ["recency_baseline", "lr_core", "catboost_core", "catboost_ext", "lr_ext"]


# --------------------------------------------------------------------------- formatting

def f4(x: Any) -> str:
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.4f}"


def pct(x: Any, nd: int = 2) -> str:
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{100 * x:.{nd}f}%"


def num(x: Any) -> str:
    return "—" if x is None else f"{int(x):,}".replace(",", " ")


def secs(x: Any) -> str:
    if x is None:
        return "not_measured"
    return f"{x:.1f} с" if x < 120 else f"{x / 60:.1f} мин"


def table(header: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def link(path: Path, base: Path) -> str:
    try:
        return Path(path).resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return relpath(path)


# --------------------------------------------------------------------------- loading

def load_run(config: Config, run_id: str) -> dict[str, Any]:
    paths = run_paths(config, run_id)
    pub = paths["published"]
    run: dict[str, Any] = {"run_id": run_id, "paths": paths}
    for name in ("run_manifest", "selection_decision", "calibration", "test_metrics", "bootstrap", "evaluation_log"):
        p = pub / f"{name}.json"
        run[name] = read_json(p) if p.exists() else None
    run["stage_timings"] = read_json(pub / "stage_timings.json") if (pub / "stage_timings.json").exists() else []
    run["dev"] = pd.read_csv(pub / "dev_results.csv") if (pub / "dev_results.csv").exists() else None
    for name in ("test_budget_curves", "test_reliability_bins"):
        p = pub / f"{name}.csv"
        run[name] = pd.read_csv(p) if p.exists() else None
    run["scoring"] = [read_json(p) for p in sorted((pub / "scoring").glob("*.json"))] if (pub / "scoring").exists() else []
    ds = (run["selection_decision"] or {}).get("dataset_id") or (run["run_manifest"] or {}).get("dataset_id")
    run["dataset_id"] = ds
    sp = config.reports_dir / "data" / ds / "dataset_summary.json" if ds else None
    run["dataset"] = read_json(sp) if sp and sp.exists() else None
    return run


# --------------------------------------------------------------------------- blocks

def block_splits(run: dict) -> str:
    ds = run["dataset"]
    cohort = {c["cutoff_date"]: c for c in ds["cohort_by_cutoff"]}
    dec = run["selection_decision"]
    rows = []
    for f in dec["folds"]:
        tr = [str(c) for c in f["train_cutoffs"]]
        n_tr = sum(cohort[c]["n_rows"] for c in tr)
        pos_tr = sum(cohort[c]["n_positive"] for c in tr)
        v = cohort[str(f["validation_cutoff"])]
        rows.append([f"dev fold {f['fold']}", f"{tr[0]} … {tr[-1]} ({len(tr)} срезов)", num(n_tr), pct(pos_tr / n_tr),
                     f"{f['validation_cutoff']} → [{f['validation_window']['target_start']}, "
                     f"{f['validation_window']['target_end_exclusive']})", num(v["n_rows"]), pct(v["prevalence"])])
    ref = dec["final_refit"]
    tr = [str(c) for c in ref["train_cutoffs"]]
    for role in ("calibration", "test"):
        c = next(x for x in ds["calendar"] if x["role"] == role)
        v = cohort[c["cutoff_date"]]
        label = "calibration" if role == "calibration" else "final test"
        rows.append([label, f"refit: {tr[0]} … {tr[-1]} ({len(tr)} срезов)" if role == "calibration" else "те же модели",
                     num(ref["n_train"]) if role == "calibration" else "—",
                     pct(ref["prevalence"]) if role == "calibration" else "—",
                     f"{c['cutoff_date']} → [{c['target_start']}, {c['target_end_exclusive']}); история "
                     f"[{c['history_start']}, {c['history_end_exclusive']})",
                     num(v["n_rows"]), pct(v["prevalence"])])
    return table(["Этап", "Train-срезы (cutoff)", "N train", "Доля target=1 train", "Оценочный срез и окна",
                  "N оценки", "Доля target=1"], rows)


def block_dev(run: dict) -> str:
    dec = run["selection_decision"]
    val = run["dev"][run["dev"]["split"] == "validation"]
    rows = []
    best = {m: d["config"] for m, d in dec["development_scores"].items()}
    best["recency_baseline"] = "score=-recency_days"
    for m in [m for m in MODEL_ORDER if m in best]:
        d = val[(val["model"] == m) & (val["config"] == best[m])].sort_values("fold")
        per_fold = " / ".join(f4(x) for x in d["average_precision"])
        role = ("кандидат" if m in dec["candidates"] else "диагностика" if m in dec["diagnostics"] else "baseline")
        rows.append([m, role, best[m], per_fold, f4(d["average_precision"].mean()), f4(d["average_precision"].std()),
                     f4(d["roc_auc"].mean()), pct(d["precision_top_5pct"].mean())])
    return table(["Модель", "Роль", "Конфигурация", "AP по fold 1/2/3", "mean AP", "std AP", "mean ROC-AUC",
                  "mean Precision@5%"], rows)


def block_selection(run: dict) -> str:
    dec = run["selection_decision"]
    lines = [
        f"- Правило: {dec['rule']['near_best']}; приоритет простоты: {' → '.join(dec['rule']['priority'])}.",
        f"- max mean AP среди кандидатов: {f4(dec['max_ap'])}; near_best: {', '.join(dec['near_best'])}.",
        f"- **Выбрана: `{dec['selected_model']}`** (конфигурация "
        f"`{dec['development_scores'][dec['selected_model']]['config']}`), решение зафиксировано "
        f"{dec['frozen_at']} в [`selection_decision.json`]({{decision_link}}).",
    ]
    return "\n".join(lines)


def block_test(run: dict) -> str:
    t = run["test_metrics"]
    rows = []
    for m in [m for m in MODEL_ORDER if m in t["models"]]:
        r = t["models"][m]
        b = r["at_budget"]
        rows.append([("**" + m + "**") if r["selected"] else m, num(r["n"]), pct(r["prevalence"]),
                     f4(r["average_precision"]), f4(r["roc_auc"]),
                     pct(b["top_1pct"]["precision"]), pct(b["top_5pct"]["precision"]), pct(b["top_10pct"]["precision"]),
                     f"{b['top_5pct']['lift']:.3f}" if b["top_5pct"]["lift"] is not None else "—",
                     f4(r["raw_probability"].get("brier")), f4(r["calibrated_probability"].get("brier")),
                     f4(r["raw_probability"].get("log_loss")), f4(r["calibrated_probability"].get("log_loss"))])
    k = t["models"]["recency_baseline"]["at_budget"]["top_5pct"]["k"]
    head = (f"Final test: cutoff {t['test_window']['cutoff_date']}, target [{t['test_window']['target_start']}, "
            f"{t['test_window']['target_end_exclusive']}), N = {num(t['n'])} клиентов, positives = "
            f"{num(t['n_positive'])}, бюджет top 5%: k = ceil(0.05·N) = {num(k)}. Run `{run['run_id']}`.\n\n")
    return head + table(["Модель", "N", "Prevalence", "AP", "ROC-AUC", "P@1%", "P@5%", "P@10%", "Lift@5%",
                         "Brier raw", "Brier calibrated", "LogLoss raw", "LogLoss calibrated"], rows)


def block_bootstrap(run: dict) -> str:
    b = run["bootstrap"]
    t = run["test_metrics"]
    rows = []
    for key, d in b["differences"].items():
        a, c = key.split("_minus_")
        pa = t["models"][a]["at_budget"]["top_5pct"]["precision"]
        pc = t["models"][c]["at_budget"]["top_5pct"]["precision"]
        rows.append([f"{a} − {c}",
                     f"{f4(t['models'][a]['average_precision'] - t['models'][c]['average_precision'])} "
                     f"[{f4(d['average_precision']['low'])}; {f4(d['average_precision']['high'])}]",
                     f"{f4(t['models'][a]['roc_auc'] - t['models'][c]['roc_auc'])} "
                     f"[{f4(d['roc_auc']['low'])}; {f4(d['roc_auc']['high'])}]",
                     f"{100 * (pa - pc):+.2f} п.п. [{100 * d['precision_at']['low']:+.2f}; "
                     f"{100 * d['precision_at']['high']:+.2f}]",
                     f"{100 * (pa / pc - 1):+.1f}%"])
    head = (f"Парный bootstrap по клиентам: {b['n_resamples_used']} из {b['n_resamples_requested']} ресэмплов "
            f"(пропущено с одним классом: {b['n_skipped_single_class']}), seed {b['seed']}, 95% percentile-интервалы. "
            "Интервал не учитывает неопределённость будущего временного сдвига.\n\n")
    return head + table(["Разность", "ΔAP [95% ДИ]", "ΔROC-AUC [95% ДИ]", "ΔPrecision@5% [95% ДИ]",
                         "Отн. изменение P@5%"], rows)


def block_calibration(run: dict) -> str:
    c = run["calibration"]
    rows = []
    for m, r in c["models"].items():
        p = r["params"]
        rows.append([m, p["status"], f"{p.get('a', float('nan')):.4f}", f"{p.get('b', float('nan')):.4f}",
                     r["primary_probability"]])
    w = c["calibration_window"]
    head = (f"Platt-калибровка на срезе {w['cutoff_date']} (target [{w['target_start']}, {w['target_end_exclusive']}), "
            f"N = {num(c['n'])}, prevalence {pct(c['prevalence'])}); p = sigmoid(a·raw_score + b), базовые модели "
            "заморожены.\n\n")
    return head + table(["Модель", "Статус", "a", "b", "Основная вероятность"], rows)


def block_scoring(run: dict) -> str:
    rows = []
    for s in run["scoring"]:
        rows.append([s["as_of_date"], s["model_version"], num(s["n_scored"]), num(s["k"]),
                     s["primary_probability_field"], "да" if s["label_window_observed"] else "нет (будущее неизвестно)",
                     f"`{s['output_file']}` (локально)"])
    return table(["as_of_date", "Модель", "N", "k (top 5%)", "Основная вероятность", "Окно target наблюдаемо",
                  "Выгрузка"], rows)


def block_data(audit: dict | None, run: dict) -> str:
    rows = []
    if audit:
        tx = audit["transactions"]
        rows += [["Товарные записи transactions_train.csv", num(tx["n_rows"])],
                 ["Уникальные customer_id в транзакциях", num(tx["n_customers"])],
                 ["Уникальные article_id в транзакциях", num(tx["n_articles"])],
                 ["Диапазон t_dat", f"{tx['min_date']} … {tx['max_date']} ({tx['n_distinct_dates']} дат)"],
                 ["Строки customers.csv / articles.csv", f"{num(audit['customers_csv']['n_rows'])} / "
                                                          f"{num(audit['articles_csv']['n_rows'])}"]]
    ds = run["dataset"]
    if ds:
        pdc = ds["purchase_days_per_customer"]
        rows += [[f"Клиенты в наборе `{ds['dataset_id']}` (доля {ds['identity']['sampling']['fraction']})",
                  num(ds["customer_keys"]["n_keys"])],
                 ["Клиенты с ≥1 покупательским днём / с ≥2", f"{num(pdc['n_customers_with_purchases'])} / "
                                                             f"{num(pdc['n_customers_with_2plus_purchase_days'])}"],
                 ["Строки customer_day", num(ds["checks"]["customer_day_reconciliation"]["n_customer_days"])],
                 ["Клиентские срезы (все допустимые cutoff)", num(ds["n_snapshot_rows_total"])],
                 ["observation_end_exclusive", f"{ds['coverage']['observation_end_exclusive']} "
                                               f"({ds['coverage']['observation_end_decision']})"]]
    return table(["Величина", "Значение"], rows)


# --------------------------------------------------------------------------- figures

def run_figures(config: Config, run: dict) -> dict[str, Path]:
    fig_dir = run["paths"]["figures"]
    figs: dict[str, Path] = {}
    ds = run["dataset"]
    cohort = pd.DataFrame(ds["cohort_by_cutoff"])
    figs["cohort"] = plots.cohort_and_prevalence(cohort, fig_dir / "cohort_and_prevalence.png")
    dev_cut = date.fromisoformat(str(max(f["validation_cutoff"] for f in run["selection_decision"]["folds"])))
    df = feat.load_features(config, run["dataset_id"], [dev_cut], feat.CORE_FEATURES)
    figs["features"] = plots.feature_distributions(df, feat.CORE_FEATURES, fig_dir / "core_feature_distributions.png",
                                                   f"cutoff {dev_cut}")
    dec = run["selection_decision"]
    val = run["dev"][run["dev"]["split"] == "validation"]
    best = {m: d["config"] for m, d in dec["development_scores"].items()}
    best["recency_baseline"] = "score=-recency_days"
    sel = pd.concat([val[(val["model"] == m) & (val["config"] == c)] for m, c in best.items()])
    figs["walk_ap"] = plots.walk_forward(sel, "average_precision", fig_dir / "walk_forward_ap.png", "AP")
    sel = sel.assign(p5=sel["precision_top_5pct"] * 100)
    figs["walk_p5"] = plots.walk_forward(sel, "p5", fig_dir / "walk_forward_precision_top5.png", "Precision@5%, %")
    if run["test_budget_curves"] is not None:
        figs["budget"] = plots.precision_lift_by_budget(run["test_budget_curves"], run["test_metrics"]["prevalence"],
                                                        fig_dir / "test_precision_lift_by_budget.png")
        figs["reliability"] = plots.reliability(run["test_reliability_bins"], fig_dir / "test_reliability.png")
    if run["stage_timings"]:
        st = pd.DataFrame(run["stage_timings"])
        top = st[~st["stage"].str.contains("_C=|_d[0-9]", regex=True)]
        figs["stages"] = plots.stage_durations(top, fig_dir / "stage_durations.png")
    return figs


# --------------------------------------------------------------------------- documents

def results_md(config: Config, run: dict, figs: dict[str, Path], out: Path) -> str:
    base = out.parent
    dec_link = link(run["paths"]["published"] / "selection_decision.json", base)
    parts = [
        f"# Результаты: режим `{config.mode}`, run `{run['run_id']}`",
        "",
        f"Сгенерировано командой `hmrp report` из артефактов run ({utc_now()}). Набор данных `{run['dataset_id']}`. "
        f"Машинно-читаемые файлы: [`{link(run['paths']['published'], base)}/`]({link(run['paths']['published'], base)}/).",
        "",
        "## Данные и когорта", "", block_data(_audit(config), run), "",
        "## Временные границы", "", block_splits(run), "",
        "## Development walk-forward (validation)", "",
        "Для каждой модели показана конфигурация с лучшим mean AP; все конфигурации — в "
        f"[`dev_results.csv`]({link(run['paths']['published'] / 'dev_results.csv', base)}).", "",
        block_dev(run), "",
        "## Выбор модели до final test", "", block_selection(run).replace("{decision_link}", dec_link), "",
    ]
    if run["calibration"]:
        parts += ["## Калибровка", "", block_calibration(run), ""]
    if run["test_metrics"]:
        parts += ["## Final test", "", block_test(run), "",
                  "Ранжирующие метрики (AP, ROC-AUC, Precision/Lift) считаются по `raw_score`; Brier/LogLoss — по "
                  "вероятностям; у baseline вероятности нет (not_applicable).", "",
                  "## Bootstrap-интервалы", "", block_bootstrap(run), ""]
    if run["scoring"]:
        parts += ["## Batch scoring", "", block_scoring(run), ""]
    parts += ["## Графики", ""]
    parts += [f"![{k}]({link(p, base)})" for k, p in figs.items()]
    return "\n".join(parts) + "\n"


def _audit(config: Config) -> dict | None:
    p = config.reports_dir / "data" / "source_audit.json"
    return read_json(p) if p.exists() else None


def data_quality_md(config: Config, run: dict, out: Path) -> str:
    a = _audit(config)
    ds = run["dataset"]
    base = out.parent
    lines = ["# Качество данных", "",
             f"Сгенерировано `hmrp report` из `reports/data/source_audit.json` и `dataset_summary.json` "
             f"набора `{ds['dataset_id']}`. Исправлений значений не выполнялось; правила ниже.", ""]
    if a:
        tx, pr = a["transactions"], a["price"]
        lines += ["## Источник и загрузка", "",
                  table(["Файл", "load_id", "Прочитано строк", "Загружено", "Отклонено (структура / тип)", "Байт",
                         "SHA-256"],
                        [[r["file_name"], r["load_id"], num(r["rows_read"]), num(r["rows_loaded"]),
                          f"{num(r['rows_rejected_structure'])} / {num(r['rows_rejected_type'])}", num(r["size_bytes"]),
                          f"`{r['sha256'][:16]}…`"] for r in a["loads"]]), "",
                  f"Причины отклонения: {a['rejected_by_reason'] or 'нет отклонённых строк'}.", "",
                  "## Транзакции", "",
                  table(["Показатель", "Значение"], [
                      ["Строк", num(tx["n_rows"])], ["Клиентов", num(tx["n_customers"])],
                      ["Товаров", num(tx["n_articles"])], ["Дат с записями", num(tx["n_distinct_dates"])],
                      ["Диапазон", f"{tx['min_date']} … {tx['max_date']}"],
                      ["Пропущенные календарные даты", ", ".join(a["calendar"]["missing_dates"]) or "нет"],
                      ["Сумма price (предоставленная шкала)", f"{tx['sum_price']:.4f}"],
                  ]), "",
                  "### Цена (предоставленная/масштабированная шкала)", "",
                  "Шкала price в источнике не документирована как валюта; суммы не интерпретируются как выручка.", "",
                  table(["min", "p1", "p50", "p99", "max", "mean", "нулевых", "отрицательных", "нечисловых/inf"],
                        [[f"{pr['min']:.6f}", f"{pr['quantiles'][1]:.6f}", f"{pr['quantiles'][4]:.6f}",
                          f"{pr['quantiles'][7]:.6f}", f"{pr['max']:.6f}", f"{pr['mean']:.6f}", num(pr["n_zero"]),
                          num(pr["n_negative"]), num(pr["n_non_finite"])]]), "",
                  "### Полностью совпадающие строки", "",
                  "Строки с одинаковыми (t_dat, customer_id, article_id, price, sales_channel_id) сохранены: в "
                  "источнике нет order_id и количества, повтор может означать несколько единиц товара.", "",
                  table(["Кратность", "Групп", "Строк"],
                        [[r["multiplicity"], num(r["n_groups"]), num(r["n_rows"])] for r in a["identical_rows"][:8]]),
                  "",
                  f"Строк в группах с кратностью > 1: {num(a['identical_rows_summary']['rows_in_groups_with_multiplicity_gt_1'])}; "
                  f"сверх первой строки группы: {num(a['identical_rows_summary']['rows_beyond_first_in_group'])}.", "",
                  "### Каналы продаж", "",
                  table(["sales_channel_id", "Строк", "Доля"],
                        [[r["sales_channel_id"], num(r["n_rows"]), pct(r["n_rows"] / tx["n_rows"])] for r in a["channels"]]),
                  "", "Значение кодов каналов источником не раскрыто; в признаках используется только доля "
                      "`sales_channel_id = 2`.", "",
                  "## Справочники и покрытие ключей", "",
                  table(["Проверка", "Результат"], [
                      ["customers.csv: строк / уникальных customer_id / пустых",
                       f"{num(a['customers_csv']['n_rows'])} / {num(a['customers_csv']['n_distinct_customer_id'])} / "
                       f"{num(a['customers_csv']['n_null_customer_id'])}"],
                      ["articles.csv: строк / уникальных article_id / пустых",
                       f"{num(a['articles_csv']['n_rows'])} / {num(a['articles_csv']['n_distinct_article_id'])} / "
                       f"{num(a['articles_csv']['n_null_article_id'])}"],
                      ["Клиенты транзакций без строки в customers.csv",
                       num(a["coverage"]["customers"]["n_not_in_customers_csv"])],
                      ["Товары транзакций без строки в articles.csv",
                       num(a["coverage"]["articles"]["n_not_in_articles_csv"])],
                      ["Пропуски customers.csv: FN / Active / age / club_member_status / fashion_news_frequency",
                       " / ".join(num(a["customers_csv"][k]) for k in ("n_null_fn", "n_null_active", "n_null_age",
                                                                      "n_null_club_member_status",
                                                                      "n_null_fashion_news_frequency"))],
                  ]), "",
                  "Поля customers.csv — снимок текущего состояния без истории; в модель не подаются.", "",
                  "## Проверки JOIN (фактические, на H&M)", "",
                  table(["JOIN", "Ключ справочника уникален", "Строк до", "Строк после", "Сумма price до", "после",
                         "Без пары", "Статус"],
                        [[k, "да" if v.get("dim_key", {}).get("n_rows") == v.get("dim_key", {}).get("n_distinct") else "нет",
                          num(v.get("fact_rows")), num(v.get("joined_rows")), f"{v.get('fact_sum', 0):.4f}",
                          f"{v.get('joined_sum', 0):.4f}", num(v.get("n_unmatched")), v["status"]]
                         for k, v in a["join_checks"].items()]), "",
                  "Демонстрация JOIN-ловушки на искусственных данных (не находка в H&M): "
                  "[`data/join_trap_demo.json`](data/join_trap_demo.json).", "",
                  "![daily](data/figures/daily_rows_by_channel.png)", "",
                  "![channel share](data/figures/monthly_channel_share.png)", "",
                  "![price](data/figures/price_log10_histogram.png)", ""]
    rec = ds["checks"]["customer_day_reconciliation"]
    cov = ds["coverage"]
    lines += [f"## Набор `{ds['dataset_id']}`: сверка слоёв", "",
              table(["Слой", "Строк (товарных записей)", "Сумма price"],
                    [["raw.transactions (клиенты набора)", num(rec["raw_rows"]), f"{rec['raw_sum_price']:.6f}"],
                     ["derived.customer_day (sum n_rows / sum spend)", num(rec["day_rows"]), f"{rec['day_sum_price']:.6f}"]]),
              "",
              f"Строки customer_day: {num(rec['n_customer_days'])}; клиентов с покупками: "
              f"{num(rec['n_customers_with_purchases'])}. Совпадение строк: {rec['rows_match']}, сумм (rel 1e-9): "
              f"{rec['sum_match_rel_1e-9']}. Метки с незрелым окном: {ds['checks']['n_immature_labels']}; нарушения "
              f"границ истории: {ds['checks']['n_history_bound_violations']}.", "",
              f"Покрытие календаря: {cov['n_dates_with_rows']} дат из {cov['n_calendar_days']}; дни с объёмом < 0.5 "
              f"скользящей медианы за 28 дней: {len(cov['low_volume_days_ratio_below_0_5'])} "
              f"({', '.join(list(cov['low_volume_days_ratio_below_0_5'])[:10]) or 'нет'}). Последний день: "
              f"{num(cov['last_day_rows'])} строк, отношение к медиане {cov['last_day_ratio_to_trailing_28d_median']:.2f}. "
              f"Решение: {cov['observation_end_decision']}.", "",
              "Покупательские дни на клиента за весь период: квантили "
              + ", ".join(f"q{k}={v:g}" for k, v in ds["purchase_days_per_customer"]["quantiles"].items())
              + f"; доля клиентов с ≥2 днями: {pct(ds['purchase_days_per_customer']['share_2plus'])}.", "",
              "EXPLAIN ANALYZE тяжёлых запросов (выполнены в рамках сборки): "
              + ", ".join(f"[`{p.name}`]({link(p, base)})" for p in
                          sorted((config.reports_dir / "data" / ds["dataset_id"]).glob("explain_*.txt"))), ""]
    return "\n".join(lines) + "\n"


def performance_md(config: Config, run: dict, out: Path) -> str:
    ds = run["dataset"]
    lines = ["# Производительность", "", "Все значения измерены во время запусков; отсутствующие помечены not_measured.", ""]
    ing = config.reports_dir / "data" / f"ingest_manifest_{config.mode}.json"
    full_ing = config.reports_dir / "data" / "ingest_manifest_full.json"
    src = ing if ing.exists() else full_ing
    if src.exists():
        m = read_json(src)
        lines += ["## Ingest (COPY FROM STDIN)", "",
                  table(["Файл", "Действие", "Строк", "Время загрузки", "SHA-256 (время)"],
                        [[f["file_name"], f["action"], num(f.get("rows_loaded")), secs(f.get("load_seconds")),
                          secs(f.get("sha256_seconds"))] for f in m["files"].values()]), ""]
    a = _audit(config)
    if a:
        lines += ["## Аудит (SQL)", "", table(["Запрос", "Время"], [[k, secs(v)] for k, v in a["query_seconds"].items()]),
                  ""]
    lines += [f"## Сборка набора `{ds['dataset_id']}`", "",
              table(["Этап", "Время"], [[k, secs(v)] for k, v in ds["timings_seconds"].items()]), "",
              table(["Таблица PostgreSQL", "Размер"], [[k, f"{v / 2**20:.1f} МБ"] for k, v in
                                                        ds["table_sizes_bytes"].items()]), ""]
    if run["stage_timings"]:
        st = pd.DataFrame(run["stage_timings"])
        lines += [f"## Этапы run `{run['run_id']}`", "",
                  table(["Этап", "Время", "Пиковый RSS процесса, МБ"],
                        [[r["stage"], secs(r["wall_seconds"]), r.get("process_peak_rss_mb") or "not_measured"]
                         for _, r in st.iterrows()]), ""]
    env = (run["run_manifest"] or {}).get("environment", {})
    if env:
        lines += ["## Окружение", "",
                  f"Python {env.get('python')}, {env.get('platform')}, {env.get('cpu_logical')} логических CPU, "
                  f"RAM {env.get('ram_total_gb')} ГБ; CatBoost {env['packages'].get('catboost')} на "
                  f"{env['compute'].get('catboost_task_type')} ({env['compute'].get('catboost_threads')} потоков). "
                  "GPU-обучение не запускалось.", ""]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- README blocks

def _replace_block(text: str, name: str, body: str) -> str:
    pattern = re.compile(rf"(<!-- AUTO:{name}:start -->\n)(?:.*?\n)?(<!-- AUTO:{name}:end -->)", re.DOTALL)
    if not pattern.search(text):
        return text
    return pattern.sub(lambda m: m.group(1) + body.rstrip("\n") + "\n" + m.group(2), text)


def update_readme(config: Config) -> list[str]:
    readme = project_root() / "README.md"
    if not readme.exists():
        return []
    index_path = config.reports_dir / "runs" / "index.json"
    index = read_json(index_path) if index_path.exists() else {}
    text = readme.read_text(encoding="utf-8")
    changed = []
    status_rows = []
    for mode in ("full", "smoke"):
        rid = index.get(mode)
        if not rid:
            partial = sorted(p.name for p in (config.reports_dir / "runs").glob(f"{mode}-*")
                             if (p / "run_manifest.json").exists())
            if partial:
                status_rows.append([mode, "не завершён (нет final test в опубликованных артефактах)",
                                    f"`{partial[-1]}`", "ingest, audit, features"])
            else:
                status_rows.append([mode, "not_run", "—", "—"])
            continue
        r = load_run(config, rid)
        stages = [s for s, ok in (("train", r["selection_decision"]), ("calibrate", r["calibration"]),
                                  ("evaluate", r["test_metrics"]), ("score", r["scoring"])) if ok]
        status_rows.append([mode, "выполнен", f"`{rid}`", ", ".join(["ingest", "audit", "features", *stages])])
    text = _replace_block(text, "status", table(["Режим", "Статус", "run_id", "Выполненные этапы"], status_rows))
    changed.append("status")
    for mode in ("full", "smoke"):
        rid = index.get(mode)
        if not rid:
            continue
        r = load_run(config, rid)
        base = project_root()
        if r["dataset"]:
            text = _replace_block(text, f"{mode}:data", block_data(_audit(config), r))
            text = _replace_block(text, f"{mode}:splits", block_splits(r))
        if r["dev"] is not None:
            text = _replace_block(text, f"{mode}:dev", block_dev(r))
            text = _replace_block(text, f"{mode}:selection", block_selection(r).replace(
                "{decision_link}", link(r["paths"]["published"] / "selection_decision.json", base)))
        if r["test_metrics"]:
            text = _replace_block(text, f"{mode}:test", block_test(r))
            text = _replace_block(text, f"{mode}:bootstrap", block_bootstrap(r))
        if r["calibration"]:
            text = _replace_block(text, f"{mode}:calibration", block_calibration(r))
        changed.append(mode)
    readme.write_text(text, encoding="utf-8")
    return changed


# --------------------------------------------------------------------------- entry point

def build_reports(config: Config, run_id: str | None = None, execute_notebooks: bool = False) -> dict[str, Any]:
    run_id = resolve_run_id(config, run_id)
    run = load_run(config, run_id)
    if run["selection_decision"] is None or run["dataset"] is None:
        raise SystemExit(f"run {run_id} has no frozen selection decision or dataset summary")
    figs = run_figures(config, run)
    rep = config.reports_dir
    name = "results.md" if config.mode == "full" else f"results_{config.mode}.md"
    out = rep / name
    out.write_text(results_md(config, run, figs, out), encoding="utf-8")
    written = [out]
    dq = rep / ("data_quality.md" if config.mode == "full" else f"data_quality_{config.mode}.md")
    dq.write_text(data_quality_md(config, run, dq), encoding="utf-8")
    perf = rep / ("performance.md" if config.mode == "full" else f"performance_{config.mode}.md")
    perf.write_text(performance_md(config, run, perf), encoding="utf-8")
    written += [dq, perf]
    index_path = rep / "runs" / "index.json"
    index = read_json(index_path) if index_path.exists() else {}
    if run["test_metrics"] is not None:
        index[config.mode] = run_id
    write_json(index_path, index)
    readme_blocks = update_readme(config) if config.mode in ("full", "smoke") else []
    if execute_notebooks:
        execute_all_notebooks()
    return {"run_id": run_id, "written": [relpath(p) for p in written], "figures": [relpath(p) for p in figs.values()],
            "readme_blocks": readme_blocks}


def execute_all_notebooks() -> list[str]:
    import nbformat
    from nbclient import NotebookClient

    done = []
    for nb_path in sorted((project_root() / "notebooks").glob("*.ipynb")):
        nb = nbformat.read(nb_path, as_version=4)
        NotebookClient(nb, timeout=600, kernel_name="python3",
                       resources={"metadata": {"path": str(nb_path.parent)}}).execute()
        nbformat.write(nb, nb_path)
        done.append(relpath(nb_path))
    return done
