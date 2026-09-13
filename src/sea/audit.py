"""Percentage-based stabilization / diminishing-return audit (proposal Eqs. 2–3)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


SIGN = {"macro_auroc": 1.0, "brier": -1.0, "ece": -1.0}


def delta_m(value: float, baseline: float, metric: str) -> float:
    return SIGN[metric] * (value - baseline)


def pmin_delta(percentages: Iterable[float], values: Iterable[float], tau: float) -> float | None:
    ps = list(percentages)
    vs = list(values)
    for i in range(len(ps) - 1):
        if abs(vs[i + 1] - vs[i]) < tau:
            return float(ps[i])
    return None


def summarize_adaptation(rows: list[dict], taus: dict[str, float]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    summaries = []
    group_cols = ["source", "target", "method", "metric"]
    for keys, part in df.groupby(group_cols, dropna=False):
        source, target, method, metric = keys
        baseline = part.loc[part["P"] == 0, "value"].mean() if (part["P"] == 0).any() else np.nan
        ordered = part[part["P"] > 0].groupby("P")["value"].mean().sort_index()
        pmin = pmin_delta(ordered.index.tolist(), ordered.tolist(), taus[metric]) if len(ordered) >= 2 else None
        last = ordered.iloc[-1] if len(ordered) else np.nan
        summaries.append(
            {
                "source": source,
                "target": target,
                "method": method,
                "metric": metric,
                "M0": baseline,
                "M_Pmax": last,
                "delta_M_Pmax": delta_m(last, baseline, metric) if not np.isnan(baseline) else np.nan,
                "Pmin_delta": pmin,
            }
        )
    return pd.DataFrame(summaries)
