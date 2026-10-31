"""Cutoff calendar, half-open windows, development folds and maturity checks."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

from .config import Config


@dataclass(frozen=True)
class Window:
    cutoff_date: date
    history_start: date            # inclusive
    history_end_exclusive: date    # == cutoff_date
    target_start: date             # == cutoff_date, inclusive
    target_end_exclusive: date

    def as_dict(self) -> dict:
        return {k: getattr(self, k).isoformat() for k in self.__dataclass_fields__}


def make_window(cutoff: date, history_days: int, horizon_days: int) -> Window:
    return Window(
        cutoff_date=cutoff,
        history_start=cutoff - timedelta(days=history_days),
        history_end_exclusive=cutoff,
        target_start=cutoff,
        target_end_exclusive=cutoff + timedelta(days=horizon_days),
    )


def config_window(config: Config, cutoff: date) -> Window:
    return make_window(cutoff, int(config.get("time.history_days")), int(config.get("time.horizon_days")))


def candidate_grid(config: Config) -> list[date]:
    start, end = config.get("time.grid_start"), config.get("time.grid_end")
    step = timedelta(days=int(config.get("time.step_days")))
    out, c = [], start
    while c <= end:
        out.append(c)
        c += step
    return out


def cutoff_role(config: Config, cutoff: date) -> str:
    if cutoff == config.get("time.test_cutoff"):
        return "test"
    if cutoff == config.get("time.calibration_cutoff"):
        return "calibration"
    if cutoff <= max(config.get("time.dev_validation_cutoffs")):
        return "development"
    return "unused"


def build_calendar(config: Config, data_start: date, observation_end_exclusive: date) -> list[dict]:
    """Candidate grid plus one probe step on each side, so exclusion rules are visible in the output."""
    grid = candidate_grid(config)
    step = timedelta(days=int(config.get("time.step_days")))
    rows = []
    for cutoff, is_probe in [(grid[0] - step, True), *((c, False) for c in grid), (grid[-1] + step, True)]:
        w = config_window(config, cutoff)
        if w.history_start < data_start:
            reason, ok = f"insufficient_history: history_start {w.history_start} < data_start {data_start}", False
        elif w.target_end_exclusive > observation_end_exclusive:
            reason, ok = (f"target_window_incomplete: target_end_exclusive {w.target_end_exclusive} > "
                          f"observation_end_exclusive {observation_end_exclusive}"), False
        else:
            reason, ok = "eligible", True
        rows.append({**w.as_dict(), "role": "probe" if is_probe else cutoff_role(config, cutoff),
                     "is_eligible": ok and not is_probe, "eligibility_reason": reason})
    for key in ("calibration_cutoff", "test_cutoff"):
        c = config.get(f"time.{key}")
        match = [r for r in rows if r["cutoff_date"] == c.isoformat()]
        if not match or not match[0]["is_eligible"]:
            raise ValueError(f"{key} {c} is not an eligible grid cutoff: {match}")
    for c in config.get("time.dev_validation_cutoffs"):
        if c not in grid:
            raise ValueError(f"dev validation cutoff {c} is not on the 30-day grid")
    return rows


def eligible_cutoffs(calendar: Iterable[dict], role: str | None = None) -> list[date]:
    return [date.fromisoformat(r["cutoff_date"]) for r in calendar
            if r["is_eligible"] and (role is None or r["role"] == role)]


def assert_matured_before(train_cutoffs: Iterable[date], eval_cutoff: date, horizon_days: int) -> None:
    """Every train label window must end no later than the evaluation cutoff."""
    bad = [c for c in train_cutoffs if c + timedelta(days=horizon_days) > eval_cutoff]
    if bad:
        raise ValueError(f"train cutoffs {bad} have labels ending after {eval_cutoff}")


def development_folds(config: Config, calendar: list[dict]) -> list[dict]:
    horizon = int(config.get("time.horizon_days"))
    dev = eligible_cutoffs(calendar, "development")
    folds = []
    for i, val in enumerate(config.get("time.dev_validation_cutoffs"), start=1):
        if val not in dev:
            raise ValueError(f"validation cutoff {val} is not an eligible development cutoff")
        train = [c for c in dev if c + timedelta(days=horizon) <= val]
        if not train:
            raise ValueError(f"fold {i}: no matured train cutoffs before {val}")
        assert_matured_before(train, val, horizon)
        folds.append({
            "fold": i,
            "train_cutoffs": train,
            "validation_cutoff": val,
            "train_label_end_exclusive_max": max(train) + timedelta(days=horizon),
            "validation_window": config_window(config, val).as_dict(),
        })
    return folds


def final_refit_cutoffs(config: Config, calendar: list[dict]) -> list[date]:
    horizon = int(config.get("time.horizon_days"))
    cal = config.get("time.calibration_cutoff")
    train = [c for c in eligible_cutoffs(calendar, "development") if c + timedelta(days=horizon) <= cal]
    assert_matured_before(train, cal, horizon)
    return train
