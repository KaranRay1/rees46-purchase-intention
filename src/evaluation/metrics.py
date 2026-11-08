"""Classification metrics and validation-set threshold tuning (RQ1–RQ3).

Metrics (config ``evaluation``)
-------------------------------
Primary   precision, recall, f1 (positive class = purchase), pr_auc
Secondary macro_f1, roc_auc
Context   base_rate, predicted_positive_rate, tp / fp / fn / tn, threshold

``pr_auc`` is **average precision** (``sklearn.metrics.average_precision_score``),
the step-wise area under the precision–recall curve. It is preferred over the
trapezoidal ``auc(recall, precision)`` mentioned in the original blueprint,
because straight-line interpolation between PR points is not valid in PR
space and can misstate the area (Davis & Goadrich, 2006). The trapezoid
version is still available as ``pr_auc_trapezoid`` for comparison. A random
classifier scores the base rate.

Threshold policy (config ``imbalance.threshold``)
-------------------------------------------------
Scores are turned into 0/1 predictions with a threshold chosen on the
**validation** split to maximise positive-class F1, then applied unchanged to
the test split. Threshold-free metrics (PR-AUC, ROC-AUC) do not depend on it.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

# -----------------------------------------------------------------------------
# Single metrics
# -----------------------------------------------------------------------------
def _check(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y_true.shape != y_score.shape:
        raise ValueError(f"y_true and y_score lengths differ: {y_true.shape} vs {y_score.shape}")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("y_true must be binary 0/1.")
    if not np.isfinite(y_score).all():
        raise ValueError("y_score contains NaN or infinite values.")
    return y_true, y_score


def _both_classes(y_true: np.ndarray) -> bool:
    return 0 < y_true.sum() < len(y_true)


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision (step-wise PR-AUC). NaN if only one class is present."""
    y_true, y_score = _check(y_true, y_score)
    return float(average_precision_score(y_true, y_score)) if _both_classes(y_true) else math.nan


def pr_auc_trapezoid(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Trapezoidal area under the PR curve (interpolates linearly; for comparison only)."""
    y_true, y_score = _check(y_true, y_score)
    if not _both_classes(y_true):
        return math.nan
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(auc(recall, precision))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC. NaN if only one class is present."""
    y_true, y_score = _check(y_true, y_score)
    return float(roc_auc_score(y_true, y_score)) if _both_classes(y_true) else math.nan


# -----------------------------------------------------------------------------
# Threshold tuning
# -----------------------------------------------------------------------------
def tune_threshold(y_true: np.ndarray, y_score: np.ndarray, objective: str = "f1", beta: float = 1.0) -> float:
    """Threshold that maximises positive-class F-beta on (validation) data.

    Predictions are ``score >= threshold``. Candidates are the distinct
    scores (exact search via the precision–recall curve). Ties go to the
    highest threshold (fewest positives predicted). If there are no positives
    or the scores are constant, 0.5 is returned.

    Args:
        objective: ``"f1"`` or ``"fbeta"`` (uses ``beta``).
    """
    if objective not in ("f1", "fbeta"):
        raise ValueError("objective must be 'f1' or 'fbeta'")
    b = 1.0 if objective == "f1" else float(beta)
    y_true, y_score = _check(y_true, y_score)
    if y_true.sum() == 0 or np.unique(y_score).size < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    p, r = precision[:-1], recall[:-1]          # last point has no threshold
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (1 + b**2) * p * r / (b**2 * p + r)
    f = np.nan_to_num(f, nan=-1.0)
    best = np.flatnonzero(f == f.max())
    return float(thresholds[best[-1]])          # thresholds are increasing -> highest of the ties


# -----------------------------------------------------------------------------
# Metric bundle
# -----------------------------------------------------------------------------
def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """All primary, secondary and context metrics at one threshold.

    Returns:
        Dict with ``precision``, ``recall``, ``f1``, ``pr_auc``, ``macro_f1``,
        ``roc_auc``, ``pr_auc_trapezoid``, ``base_rate``,
        ``predicted_positive_rate``, ``tp``, ``fp``, ``fn``, ``tn``,
        ``threshold``, ``n``, ``n_positive``.
    """
    y_true, y_score = _check(y_true, y_score)
    y_pred = (y_score >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    n = len(y_true)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "pr_auc": pr_auc(y_true, y_score),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)),
        "roc_auc": roc_auc(y_true, y_score),
        "pr_auc_trapezoid": pr_auc_trapezoid(y_true, y_score),
        "base_rate": float(y_true.mean()) if n else math.nan,
        "predicted_positive_rate": float(y_pred.mean()) if n else math.nan,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "threshold": float(threshold),
        "n": int(n),
        "n_positive": int(y_true.sum()),
    }


# -----------------------------------------------------------------------------
# Model-level helpers
# -----------------------------------------------------------------------------
def _xy(split: Any) -> tuple[Any, np.ndarray]:
    """Model input and labels from SplitData (tabular) or SequenceData (sequence)."""
    if hasattr(split, "X"):
        return split.X, split.y
    if hasattr(split, "cat"):
        return split, split.y
    raise TypeError("Expected SplitData (with .X) or SequenceData (with .cat).")


def evaluate_model(model: Any, split: Any, threshold: float | None = None) -> dict[str, Any]:
    """Metrics for a fitted model on one split (threshold defaults to the model's)."""
    X, y = _xy(split)
    t = model.threshold if threshold is None else threshold
    return classification_metrics(y, model.predict_proba(X), t)


def tune_and_evaluate(
    model: Any,
    val: Any,
    test: Any,
    cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Tune the threshold on ``val`` (if the model allows), then score val and test.

    Sets ``model.threshold``. Returns ``{"threshold", "val": metrics, "test": metrics,
    "val_scores", "test_scores"}`` — scores are kept for statistics and error analysis.
    """
    objective = "f1"
    if cfg is not None:
        objective = cfg["imbalance"]["threshold"].get("objective", "f1")
        model.threshold = float(cfg["imbalance"]["threshold"].get("default", 0.5))

    X_val, y_val = _xy(val)
    X_test, y_test = _xy(test)
    val_scores = model.predict_proba(X_val)
    if getattr(model, "tunable_threshold", True):
        model.set_threshold(tune_threshold(y_val, val_scores, objective))
    test_scores = model.predict_proba(X_test)
    return {
        "threshold": model.threshold,
        "val": classification_metrics(y_val, val_scores, model.threshold),
        "test": classification_metrics(y_test, test_scores, model.threshold),
        "val_scores": val_scores,
        "test_scores": test_scores,
    }


def metrics_by_group(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: Sequence[Any],
    threshold: float,
    metrics: Sequence[str] = ("n", "base_rate", "precision", "recall", "f1", "pr_auc"),
) -> pl.DataFrame:
    """Metrics per cohort (e.g. session-length bin, month) for robustness / error analysis."""
    y_true, y_score = _check(y_true, y_score)
    groups = np.asarray(groups)
    if len(groups) != len(y_true):
        raise ValueError("groups must have one entry per session")
    rows = []
    for g in sorted(set(groups.tolist()), key=str):
        mask = groups == g
        m = classification_metrics(y_true[mask], y_score[mask], threshold)
        rows.append({"group": str(g), **{k: m[k] for k in metrics}})
    return pl.DataFrame(rows)
