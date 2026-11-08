"""Uncertainty and significance for the model comparisons (H1–H3).

What "significantly higher" means in this project
-------------------------------------------------
* **Confidence intervals** — percentile bootstrap over test sessions
  (``bootstrap_ci``): resample sessions with replacement, recompute the
  metric, report the 2.5th–97.5th percentiles (95% by default).
* **PR-AUC difference between two models** — *paired* bootstrap
  (``paired_bootstrap``): both models are scored on the same resampled
  sessions, so the CI and p-value are for ``metric(A) − metric(B)``.
* **Label-level disagreement** — McNemar's test (``mcnemar``) on the
  sessions where exactly one of the two models is correct (exact binomial
  when there are fewer than 25 such sessions, otherwise χ² with continuity
  correction).
* **Many comparisons** — Holm–Bonferroni adjustment (``holm``) across the
  family of pairwise tests (7 models × 3 values of K).
* **Seeds** — ``seed_summary`` gives mean ± std over the 3 training seeds,
  which captures training randomness (the bootstrap captures test-sample
  randomness).

Speed
-----
REES46 test splits can hold hundreds of thousands of sessions. Instead of
re-sorting for every resample, scores are sorted **once** and each bootstrap
resample is expressed as integer weights (how often each session was drawn).
PR-AUC, ROC-AUC and thresholded metrics are then weighted cumulative sums —
one O(n) pass per resample. The results equal scikit-learn's
``average_precision_score`` / ``roc_auc_score`` with ``sample_weight``.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import stats

FAST_METRICS = ("pr_auc", "roc_auc", "precision", "recall", "f1")


# -----------------------------------------------------------------------------
# Weighted metrics on pre-sorted scores
# -----------------------------------------------------------------------------
class _Sorted:
    """Scores sorted descending once, with the end index of each tie group."""

    def __init__(self, y_true: np.ndarray, y_score: np.ndarray) -> None:
        y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
        y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
        if y_true.shape != y_score.shape:
            raise ValueError("y_true and y_score lengths differ")
        if not np.isfinite(y_score).all():
            raise ValueError("y_score contains NaN or infinite values")
        self.order = np.argsort(-y_score, kind="mergesort")
        self.y = y_true[self.order].astype(np.float64)
        s = y_score[self.order]
        # last position of every run of equal scores = one PR/ROC point per distinct threshold
        self.ends = np.flatnonzero(np.r_[s[1:] != s[:-1], True])
        self.n = len(y_true)

    def curve(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Cumulative weighted TP / FP at each distinct threshold, plus totals P, N."""
        ws = w[self.order]
        tp = np.cumsum(ws * self.y)[self.ends]
        fp = np.cumsum(ws * (1.0 - self.y))[self.ends]
        return tp, fp, tp[-1] if len(tp) else 0.0, fp[-1] if len(fp) else 0.0


def _weighted_pr_auc(sv: _Sorted, w: np.ndarray) -> float:
    tp, fp, P, _ = sv.curve(w)
    if P <= 0:
        return math.nan
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = tp / P
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _weighted_roc_auc(sv: _Sorted, w: np.ndarray) -> float:
    tp, fp, P, N = sv.curve(w)
    if P <= 0 or N <= 0:
        return math.nan
    tpr = np.r_[0.0, tp / P]
    fpr = np.r_[0.0, fp / N]
    return float(np.trapezoid(tpr, fpr))


def _weighted_threshold_metric(y: np.ndarray, pred: np.ndarray, w: np.ndarray, name: str) -> float:
    tp = float(np.sum(w * (y == 1) * (pred == 1)))
    fp = float(np.sum(w * (y == 0) * (pred == 1)))
    fn = float(np.sum(w * (y == 1) * (pred == 0)))
    if name == "precision":
        return tp / (tp + fp) if tp + fp else 0.0
    if name == "recall":
        return tp / (tp + fn) if tp + fn else math.nan
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0   # f1


def weighted_metric(
    name: str, y_true: np.ndarray, y_score: np.ndarray, weights: np.ndarray | None = None, threshold: float = 0.5
) -> float:
    """One metric with optional per-session weights (weights = bootstrap counts)."""
    if name not in FAST_METRICS:
        raise ValueError(f"metric must be one of {FAST_METRICS}")
    w = np.ones(len(y_true)) if weights is None else np.asarray(weights, dtype=np.float64)
    if name in ("pr_auc", "roc_auc"):
        sv = _Sorted(y_true, y_score)
        return _weighted_pr_auc(sv, w) if name == "pr_auc" else _weighted_roc_auc(sv, w)
    y = np.asarray(y_true).astype(np.int64)
    pred = (np.asarray(y_score) >= threshold).astype(np.int64)
    return _weighted_threshold_metric(y, pred, w, name)


# -----------------------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------------------
def _resample_weights(n: int, n_resamples: int, seed: int):
    """Yield bootstrap weights: multinomial counts = sampling n sessions with replacement."""
    rng = np.random.default_rng(seed)
    p = np.full(n, 1.0 / n)
    for _ in range(n_resamples):
        yield rng.multinomial(n, p).astype(np.float64)


def _metric_fn(name: str | Callable, y_true, y_score, threshold: float):
    """Return f(weights) -> value, reusing the sort for curve metrics."""
    if callable(name):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        def generic(w):  # slow path: materialise the resample
            idx = np.repeat(np.arange(len(w)), w.astype(np.int64))
            return float(name(y_true[idx], y_score[idx]))
        return generic
    if name == "pr_auc":
        sv = _Sorted(y_true, y_score)
        return lambda w: _weighted_pr_auc(sv, w)
    if name == "roc_auc":
        sv = _Sorted(y_true, y_score)
        return lambda w: _weighted_roc_auc(sv, w)
    if name in ("precision", "recall", "f1"):
        y = np.asarray(y_true).astype(np.int64)
        pred = (np.asarray(y_score) >= threshold).astype(np.int64)
        return lambda w: _weighted_threshold_metric(y, pred, w, name)
    raise ValueError(f"metric must be one of {FAST_METRICS} or a callable")


def _ci(values: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = 1.0 - confidence
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str | Callable = "pr_auc",
    n_resamples: int = 1000,
    confidence: float = 0.95,
    threshold: float = 0.5,
    seed: int = 0,
) -> dict[str, Any]:
    """Percentile bootstrap CI for one model's metric on one split.

    Returns:
        ``{"metric", "estimate", "lower", "upper", "std", "n_resamples", "n_invalid"}``;
        resamples where the metric is undefined (e.g. no positives) are skipped
        and counted in ``n_invalid``.
    """
    f = _metric_fn(metric, y_true, y_score, threshold)
    n = len(np.asarray(y_true))
    estimate = f(np.ones(n))
    values = np.array([f(w) for w in _resample_weights(n, n_resamples, seed)])
    valid = values[np.isfinite(values)]
    lower, upper = _ci(valid, confidence) if len(valid) else (math.nan, math.nan)
    return {
        "metric": metric if isinstance(metric, str) else getattr(metric, "__name__", "custom"),
        "estimate": float(estimate),
        "lower": lower,
        "upper": upper,
        "std": float(valid.std(ddof=1)) if len(valid) > 1 else math.nan,
        "n_resamples": int(n_resamples),
        "n_invalid": int(len(values) - len(valid)),
    }


def paired_bootstrap(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    metric: str | Callable = "pr_auc",
    n_resamples: int = 1000,
    confidence: float = 0.95,
    threshold_a: float = 0.5,
    threshold_b: float = 0.5,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired bootstrap for ``metric(A) − metric(B)`` on the same sessions.

    The p-value is two-sided: twice the smaller of the shares of resampled
    differences ≤ 0 and ≥ 0 (capped at 1), i.e. how often the sign of the
    difference flips under resampling.
    """
    y = np.asarray(y_true)
    if not (len(y) == len(score_a) == len(score_b)):
        raise ValueError("y_true, score_a and score_b must have the same length")
    fa = _metric_fn(metric, y, score_a, threshold_a)
    fb = _metric_fn(metric, y, score_b, threshold_b)
    n = len(y)
    ones = np.ones(n)
    a0, b0 = fa(ones), fb(ones)
    diffs = []
    for w in _resample_weights(n, n_resamples, seed):
        d = fa(w) - fb(w)
        if math.isfinite(d):
            diffs.append(d)
    diffs = np.asarray(diffs)
    if len(diffs) == 0:
        lower = upper = p = math.nan
    else:
        lower, upper = _ci(diffs, confidence)
        p = float(min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))))
    return {
        "metric": metric if isinstance(metric, str) else getattr(metric, "__name__", "custom"),
        "a": float(a0),
        "b": float(b0),
        "diff": float(a0 - b0),
        "lower": lower,
        "upper": upper,
        "p_value": p,
        "share_a_better": float(np.mean(diffs > 0)) if len(diffs) else math.nan,
        "n_resamples": int(n_resamples),
        "n_invalid": int(n_resamples - len(diffs)),
    }


# -----------------------------------------------------------------------------
# McNemar
# -----------------------------------------------------------------------------
def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, exact_below: int = 25) -> dict[str, Any]:
    """McNemar's test on paired 0/1 predictions.

    ``b`` = sessions A gets right and B gets wrong; ``c`` = the reverse.
    Uses the exact binomial test when ``b + c < exact_below``, otherwise the
    χ² statistic with continuity correction ``(|b − c| − 1)² / (b + c)``.
    """
    y = np.asarray(y_true).astype(np.int64)
    a_ok = np.asarray(pred_a).astype(np.int64) == y
    b_ok = np.asarray(pred_b).astype(np.int64) == y
    if not (len(y) == len(a_ok) == len(b_ok)):
        raise ValueError("y_true, pred_a and pred_b must have the same length")
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    n_disc = b + c
    if n_disc == 0:
        return {"b": 0, "c": 0, "statistic": 0.0, "p_value": 1.0, "method": "none (no disagreements)"}
    if n_disc < exact_below:
        p = float(stats.binomtest(b, n_disc, 0.5).pvalue)
        return {"b": b, "c": c, "statistic": float(b), "p_value": min(p, 1.0), "method": "exact binomial"}
    chi2 = (abs(b - c) - 1) ** 2 / n_disc
    return {"b": b, "c": c, "statistic": float(chi2), "p_value": float(stats.chi2.sf(chi2, df=1)),
            "method": "chi-squared, continuity corrected"}


# -----------------------------------------------------------------------------
# Multiple comparisons and seeds
# -----------------------------------------------------------------------------
def holm(p_values: Sequence[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values (same order as the input; NaN kept as NaN)."""
    p = np.asarray(p_values, dtype=np.float64)
    out = np.full_like(p, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    m = len(idx)
    if m == 0:
        return out.tolist()
    order = idx[np.argsort(p[idx], kind="mergesort")]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out.tolist()


def seed_summary(values: Sequence[float]) -> dict[str, float]:
    """Mean, sample std, min and max of a metric across training seeds."""
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=np.float64)
    if len(v) == 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan, "n": 0}
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
        "n": int(len(v)),
    }


def bootstrap_settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """``n_resamples`` and ``confidence`` from ``config.yaml → evaluation.bootstrap``."""
    b = cfg["evaluation"]["bootstrap"]
    return {"n_resamples": int(b["n_resamples"]), "confidence": float(b["confidence"])}
