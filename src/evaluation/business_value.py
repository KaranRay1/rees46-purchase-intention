"""Business value of a purchase-intent model: profit of a discount campaign, with honest error bars.

The Campaign planner (``src.serving.campaign``) values one model under settings the user picks.
This module compares **all models in money**, so the report can ask whether the PR-AUC ranking
(and the ML ~ DL ties) still hold when the yardstick is profit.

Campaign (same assumptions as the planner)
------------------------------------------
Everyone is offered a discount except visits whose score is at or above a cutoff
("suppression": sure buyers don't need it). For the offered visits with ``b`` buyers-anyway
and ``n`` non-buyers::

    net = n * uplift * aov * (1 - discount) - b * aov * discount - (b + n) * contact_cost

Honest evaluation
-----------------
* The cutoff is **chosen on one half** of the test visits (random, fixed seed) and the profit is
  **measured on the other half**, so no model gets credit for a threshold tuned on the visits it
  is scored on.
* Uncertainty: bootstrap (resampling visits with replacement) on the evaluation half, with the **same resamples for every
  model**, so model differences are paired (as for PR-AUC).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.evaluation.statistics import _resample_weights
from src.serving.campaign import Offer, best_share, gain_table, suppression_value

SHARES = np.round(np.arange(0.0, 0.7001, 0.01), 2)


@dataclass
class HalfSplit:
    choose: np.ndarray   # indices used to pick the cutoff
    measure: np.ndarray  # indices the profit is measured on


def split_halves(n: int, seed: int = 0) -> HalfSplit:
    idx = np.random.default_rng(seed).permutation(n)
    return HalfSplit(choose=np.sort(idx[: n // 2]), measure=np.sort(idx[n // 2:]))


def choose_cutoff(y: np.ndarray, score: np.ndarray, offer: Offer) -> tuple[float, float]:
    """Best suppression rule on (y, score) as ``(cutoff, tie_share)``.

    Visits scoring above ``cutoff`` are skipped, and so is a share ``tie_share`` of the visits
    scoring exactly ``cutoff`` (coarse scores, e.g. the rule-based model, have large ties).
    ``(inf, 0)`` = offer to everyone.
    """
    best = best_share(suppression_value(gain_table(y, score, SHARES), offer))
    if best["share"] == 0:
        return float("inf"), 0.0
    cut = float(best["score_cutoff"])
    above, tied = int((score > cut).sum()), int((score == cut).sum())
    return cut, (int(best["skipped"]) - above) / max(tied, 1)


def _parts(y: np.ndarray, score: np.ndarray, rule: tuple[float, float], seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """0/1 vectors: offered buyer, offered non-buyer, under a ``choose_cutoff`` rule."""
    cut, tie_share = rule
    skipped = score > cut
    tied = np.flatnonzero(score == cut)
    if tied.size and tie_share > 0:
        skipped[tied[np.random.default_rng(seed).random(tied.size) < tie_share]] = True
    offered = ~skipped
    return (offered & (y == 1)).astype(np.float64), (offered & (y == 0)).astype(np.float64)


def net_value(buyers: float | np.ndarray, non: float | np.ndarray, offer: Offer) -> float | np.ndarray:
    return (non * offer.uplift * offer.aov * (1 - offer.discount) - buyers * offer.aov * offer.discount
            - (buyers + non) * offer.contact_cost)


def compare_models(y: np.ndarray, scores: Mapping[str, np.ndarray], offer: Offer, n_resamples: int = 1000,
                   seed: int = 0, confidence: float = 0.95) -> dict:
    """Profit per model (cutoff chosen on one half, measured on the other) with paired bootstrap draws.

    Returns ``{"split", "rows", "draws", "blanket", "blanket_draws"}``: ``rows`` has, per model, the
    cutoff, the share of measured visits skipped, the net value and its CI; ``draws[model]`` holds
    the bootstrap net values (same resamples for every model) for paired tests.
    """
    y = np.asarray(y, dtype=np.int64)
    sp = split_halves(len(y), seed)
    ym = y[sp.measure]
    weights = np.stack([w.astype(np.float32) for w in _resample_weights(len(ym), n_resamples, seed)])  # (R, n)
    alpha = (1 - confidence) / 2
    rows, draws = [], {}
    for name, s in scores.items():
        s = np.asarray(s, dtype=np.float64)
        rule = choose_cutoff(y[sp.choose], s[sp.choose], offer)
        b, n = _parts(ym, s[sp.measure], rule, seed)
        net = float(net_value(b.sum(), n.sum(), offer))
        d = net_value(weights @ b, weights @ n, offer)
        draws[name] = d
        rows.append({"model": name, "cutoff": rule[0] if math.isfinite(rule[0]) else None, "tie_share": rule[1],
                     "share_skipped": float(1 - (b.sum() + n.sum()) / len(ym)), "net": net,
                     "net_low": float(np.quantile(d, alpha)), "net_high": float(np.quantile(d, 1 - alpha))})
    bl = net_value(float(ym.sum()), float(len(ym) - ym.sum()), offer)
    bl_draws = net_value(weights @ ym.astype(np.float64), weights @ (1 - ym).astype(np.float64), offer)
    return {"split": sp, "rows": rows, "draws": draws, "blanket": float(bl), "blanket_draws": bl_draws}


def paired_diff(da: np.ndarray, db: np.ndarray, a: float, b: float, confidence: float = 0.95) -> dict:
    """Difference ``a − b`` with its bootstrap CI and two-sided sign-flip p-value (as in statistics.py)."""
    d = np.asarray(da) - np.asarray(db)
    alpha = (1 - confidence) / 2
    return {"diff": a - b, "lower": float(np.quantile(d, alpha)), "upper": float(np.quantile(d, 1 - alpha)),
            "p_value": float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))}
