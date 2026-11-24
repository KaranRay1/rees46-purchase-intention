"""Segment (fairness) analysis: does a model work equally well, and treat visits alike, across groups?

REES46 has no protected attributes (no age, gender or location), so "fairness" here means
**consistency across business segments**: product category, price level, whether the visitor
has carted, time of day, weekday vs weekend, and new vs returning users.

For each segment and model, on the test weeks:

* **Ranking quality**: PR-AUC, and *PR-AUC lift* = PR-AUC / buyer rate. Segments have different
  buyer rates and PR-AUC rises with the base rate, so lift is the fair comparison across segments.
* **Calibration**: mean calibrated probability vs the actual buyer rate (over/under-prediction).
* **Treatment at the decision point**: the engine flags the top ``flag_share`` of all visits as
  "likely buyer" (one global cutoff, as deployed). Per segment we report the flag rate, the
  true-positive rate (share of buyers flagged: *equal opportunity*) and the false-positive rate.
  In the discount campaign a flagged visit gets **no** discount, so a gap in flag rates between
  groups is a gap in who gets offers.

Segments are formed from prefix features only (known at prediction time).
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

TIME_OF_DAY = [(0, 6, "night (0-6 h)"), (6, 12, "morning (6-12 h)"), (12, 18, "afternoon (12-18 h)"),
               (18, 24, "evening (18-24 h)")]
DIMENSIONS = ["category", "price", "cart", "time_of_day", "day_type", "user_type"]
SHORT_LABELS = {"category": "Category", "price": "Price", "cart": "Cart", "time_of_day": "Time of day",
                "day_type": "Weekday/end", "user_type": "New/returning"}
DIMENSION_LABELS = {"category": "Product category", "price": "Price level (prefix average)",
                    "cart": "Cart in the first clicks", "time_of_day": "Time of day (UTC)",
                    "day_type": "Weekday or weekend", "user_type": "New or returning user"}


def price_edges(train_prices: pl.Series) -> list[float]:
    """Quartile edges of the prefix average price, fitted on the training split."""
    return [float(train_prices.quantile(q)) for q in (0.25, 0.5, 0.75)]


def assign_segments(feats: pl.DataFrame, edges: list[float], top_categories: int = 8,
                    returning: pl.DataFrame | None = None) -> pl.DataFrame:
    """One column per dimension (``seg_<dimension>``) for each visit in ``feats``.

    Args:
        feats: feature rows (``user_session``, ``main_category``, ``price_mean``, ``has_cart``,
            ``hour_of_day``, ``day_of_week``).
        edges: price quartile edges from :func:`price_edges`.
        top_categories: categories outside the most frequent ones (in ``feats``) become "other".
        returning: optional ``user_session``, ``uh_is_returning`` (G6, step 17).
    """
    top = (feats["main_category"].value_counts().sort("count", descending=True)
           .filter(pl.col("main_category") != "unknown")["main_category"].head(top_categories).to_list())
    q1, q2, q3 = edges
    price = (pl.when(pl.col("price_mean") < q1).then(pl.lit("1 low"))
             .when(pl.col("price_mean") < q2).then(pl.lit("2 mid-low"))
             .when(pl.col("price_mean") < q3).then(pl.lit("3 mid-high"))
             .otherwise(pl.lit("4 high")))
    tod = pl.lit(None, dtype=pl.String)
    for lo, hi, name in reversed(TIME_OF_DAY):
        tod = pl.when((pl.col("hour_of_day") >= lo) & (pl.col("hour_of_day") < hi)).then(pl.lit(name)).otherwise(tod)
    out = feats.select(
        "user_session",
        seg_category=pl.when(pl.col("main_category").is_in(top)).then(pl.col("main_category"))
        .when(pl.col("main_category") == "unknown").then(pl.lit("unknown"))
        .otherwise(pl.lit("other")),
        seg_price=price,
        seg_cart=pl.when(pl.col("has_cart") == 1).then(pl.lit("carted")).otherwise(pl.lit("no cart")),
        seg_time_of_day=tod,
        seg_day_type=pl.when(pl.col("day_of_week") >= 6).then(pl.lit("weekend")).otherwise(pl.lit("weekday")),
    )
    if returning is not None:
        out = out.join(returning.select("user_session", seg_user_type=pl.when(pl.col("uh_is_returning") == 1)
                                        .then(pl.lit("returning")).otherwise(pl.lit("new"))),
                       on="user_session", how="left")
    return out


def _safe(fn, y, s) -> float:
    return float(fn(y, s)) if 0 < y.sum() < len(y) else math.nan


def segment_table(y: np.ndarray, scores: Mapping[str, np.ndarray], segs: pl.DataFrame,
                  probs: Mapping[str, np.ndarray] | None = None, flag_share: float = 0.13,
                  min_size: int = 200) -> pl.DataFrame:
    """Metrics per (model, dimension, segment); ``segs`` rows are aligned with ``y``.

    The flag cutoff is the global top ``flag_share`` of each model's scores (one cutoff for all
    segments, as deployed). Segments smaller than ``min_size`` visits are left out.
    """
    y = np.asarray(y, dtype=np.int64)
    rows = []
    for model, s in scores.items():
        s = np.asarray(s, dtype=np.float64)
        flag = s >= np.quantile(s, 1 - flag_share)
        p = None if probs is None or model not in probs else np.asarray(probs[model])
        overall = {"pr_auc": _safe(average_precision_score, y, s), "base_rate": float(y.mean()),
                   "flag_rate": float(flag.mean())}
        for dim in [c.removeprefix("seg_") for c in segs.columns if c.startswith("seg_")]:
            lab = segs[f"seg_{dim}"].to_numpy()
            for seg in sorted({x for x in lab if x is not None}):
                m = lab == seg
                n = int(m.sum())
                if n < min_size:
                    continue
                ys, ss, fs = y[m], s[m], flag[m]
                pos = int(ys.sum())
                pr = _safe(average_precision_score, ys, ss)
                rows.append({
                    "model": model, "dimension": dim, "segment": seg, "n": n, "share": n / len(y),
                    "positives": pos, "base_rate": pos / n, "pr_auc": pr,
                    "pr_auc_lift": pr / (pos / n) if pos else math.nan,
                    "roc_auc": _safe(roc_auc_score, ys, ss),
                    "mean_prob": float(p[m].mean()) if p is not None else math.nan,
                    "calibration_gap": float(p[m].mean() - pos / n) if p is not None else math.nan,
                    "flag_rate": float(fs.mean()),
                    "tpr": float(fs[ys == 1].mean()) if pos else math.nan,
                    "fpr": float(fs[ys == 0].mean()) if pos < n else math.nan,
                    "precision_flagged": float(ys[fs].mean()) if fs.any() else math.nan,
                    "overall_pr_auc": overall["pr_auc"], "overall_base_rate": overall["base_rate"],
                })
    return pl.DataFrame(rows)


def disparity(table: pl.DataFrame) -> pl.DataFrame:
    """Per (model, dimension): the spread between the best and worst segment.

    ``lift_ratio`` = worst / best PR-AUC lift (1 = equal ranking quality);
    ``tpr_gap`` = largest − smallest true-positive rate (0 = equal opportunity);
    ``max_abs_calibration_gap`` = the worst over/under-prediction of the buyer rate.
    """
    return table.group_by("model", "dimension").agg(
        segments=pl.len(),
        worst_segment=pl.col("segment").sort_by("pr_auc_lift").first(),
        best_segment=pl.col("segment").sort_by("pr_auc_lift").last(),
        lift_ratio=pl.col("pr_auc_lift").min() / pl.col("pr_auc_lift").max(),
        tpr_gap=pl.col("tpr").max() - pl.col("tpr").min(),
        flag_rate_gap=pl.col("flag_rate").max() - pl.col("flag_rate").min(),
        max_abs_calibration_gap=pl.col("calibration_gap").abs().max(),
    ).sort("model", "dimension")
