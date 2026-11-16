"""Shared business rules: which model to run, and what a score means for the shop.

Used by the dashboard and the REST API so both give the same answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

ALPHA = 0.05


def significant_vs(pairs: pl.DataFrame | None, k: int, a: str, b: str) -> tuple[float, bool] | None:
    """(PR-AUC of a minus b, significant after Holm) from the paired tests, or None if not tested."""
    if pairs is None:
        return None
    p = pairs.filter(pl.col("k") == k)
    for x, y, sign in ((a, b, 1), (b, a, -1)):
        r = p.filter((pl.col("model_a") == x) & (pl.col("model_b") == y))
        if r.height:
            return sign * float(r["pr_auc_diff"][0]), float(r["p_bootstrap_holm"][0]) < ALPHA
    return None


def recommended(rq1: pl.DataFrame, pairs: pl.DataFrame | None, eff: pl.DataFrame | None,
                k: int) -> tuple[str, list[str]]:
    """The model to ship at K: the fastest of those statistically tied with the most accurate one.

    Returns (model, tied models best-first). Without benchmarks, the most accurate model.
    """
    part = rq1.filter((pl.col("k") == k) & (pl.col("model") != "rule_based")).sort("pr_auc_mean", descending=True)
    top = part["model"][0]
    tied = [top] + [m for m in part["model"][1:] if (r := significant_vs(pairs, k, top, m)) is not None and not r[1]]
    if eff is None:
        return top, tied
    e = eff.filter(pl.col("k") == k)
    lat = dict(zip(e["model"], e["latency_b1_p95_ms"]))
    return min(tied, key=lambda m: lat.get(m, float("inf"))), tied


@dataclass(frozen=True)
class Intent:
    level: str        # "high", "warm" or "browsing"
    label: str
    action: str


INTENTS = {
    "high": Intent("high", "High intent", "Keep the path clear: express checkout, no discount needed."),
    "warm": Intent("warm", "Warm", "Give a nudge: free shipping, a low-stock note or a reminder of the cart."),
    "browsing": Intent("browsing", "Just browsing",
                       "Don't spend a coupon here; show inspiration and similar products instead."),
}


def intent(score: float, threshold: float) -> Intent:
    """High intent at or above the tuned threshold, warm from half of it, otherwise browsing."""
    if score >= threshold:
        return INTENTS["high"]
    if score >= threshold / 2:
        return INTENTS["warm"]
    return INTENTS["browsing"]
