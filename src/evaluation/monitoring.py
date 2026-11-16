"""Production-style monitoring: accuracy over time and input drift.

* :func:`weekly_performance` scores a model's saved predictions week by week (by
  session start) with a bootstrap 95 % CI on PR-AUC, so a dip can be told apart from
  noise.
* :func:`psi` / :func:`drift_table` compare each month's feature distribution with
  the training period using the Population Stability Index, the usual credit-risk and
  marketing-model drift check. Rule of thumb: < 0.10 stable, 0.10-0.25 moderate shift,
  > 0.25 major shift.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import polars as pl

from src.evaluation.metrics import pr_auc
from src.evaluation.statistics import bootstrap_ci

PSI_MODERATE, PSI_MAJOR = 0.10, 0.25
_EPS = 1e-4


def drift_status(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return "major" if value > PSI_MAJOR else "moderate" if value > PSI_MODERATE else "stable"


def weekly_performance(preds: pl.DataFrame, threshold: float, n_resamples: int = 200,
                       min_positives: int = 20, seed: int = 0) -> pl.DataFrame:
    """PR-AUC (with 95 % CI), lift, precision, recall and calibration per ISO week.

    Args:
        preds: ``session_start``, ``y_true``, ``score`` (and optionally ``split``).
        threshold: the model's validation-tuned decision threshold.
        min_positives: weeks with fewer buyers get no CI (too few to estimate).
    """
    by = ["week"] + (["split"] if "split" in preds.columns else [])
    df = preds.with_columns(week=pl.col("session_start").dt.truncate("1w").dt.date())
    rows = []
    for key, g in df.group_by(by, maintain_order=False):
        y, s = g["y_true"].to_numpy(), g["score"].to_numpy()
        pos, n = int(y.sum()), len(y)
        if pos == 0 or pos == n:
            continue
        pred = s >= threshold
        tp = int((pred & (y == 1)).sum())
        ci = (bootstrap_ci(y, s, "pr_auc", n_resamples=n_resamples, seed=seed)
              if pos >= min_positives else {"lower": np.nan, "upper": np.nan})
        auc = pr_auc(y, s)
        rows.append({
            **dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True)),
            "sessions": n, "buyers": pos, "base_rate": pos / n,
            "pr_auc": auc, "pr_auc_low": ci["lower"], "pr_auc_high": ci["upper"], "lift": auc / (pos / n),
            "precision": tp / max(int(pred.sum()), 1), "recall": tp / pos, "mean_score": float(s.mean()),
        })
    return pl.DataFrame(rows).sort("week") if rows else pl.DataFrame()


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index of ``actual`` against ``expected`` (numeric, quantile bins)."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected, actual = expected[np.isfinite(expected)], actual[np.isfinite(actual)]
    if not len(expected) or not len(actual):
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:                         # constant in training: compare "same value" vs "other"
        e = np.array([1.0, 0.0])
        a = np.array([np.mean(actual == expected[0]), np.mean(actual != expected[0])])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected)
        a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, _EPS, None), np.clip(a, _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorical(expected: Sequence, actual: Sequence) -> float:
    """PSI over the categories seen in either sample."""
    e = pl.Series(list(expected)).value_counts(normalize=True, name="p")
    a = pl.Series(list(actual)).value_counts(normalize=True, name="q")
    col = e.columns[0]
    j = e.join(a.rename({a.columns[0]: col}), on=col, how="full", coalesce=True).fill_null(0.0)
    p, q = np.clip(j["p"].to_numpy(), _EPS, None), np.clip(j["q"].to_numpy(), _EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def drift_table(features: pl.DataFrame, columns: Sequence[str], reference: pl.DataFrame,
                by: str = "month") -> pl.DataFrame:
    """PSI of every column for every period in ``features`` against ``reference`` (the training rows).

    ``features`` needs the ``by`` column (a month or week label); string columns use categorical PSI.
    """
    rows = []
    for (month,), g in features.group_by([by], maintain_order=False):
        for c in columns:
            if features.schema[c] in (pl.String, pl.Categorical) or isinstance(features.schema[c], pl.Enum):
                v = psi_categorical(reference[c].cast(pl.String).to_list(), g[c].cast(pl.String).to_list())
            else:
                v = psi(reference[c].cast(pl.Float64).to_numpy(), g[c].cast(pl.Float64).to_numpy())
            rows.append({by: month, "feature": c, "psi": v, "status": drift_status(v), "sessions": g.height})
    return pl.DataFrame(rows).sort(by, "feature")
