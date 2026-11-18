"""Stacked ensembles of the trained models (stacked generalisation, Wolpert 1992).

The base models were trained on the training weeks; their scores on the **validation**
weeks are out-of-sample, so a small meta-model can be fitted on them without leakage
and judged on the untouched **test** weeks.

Meta-model: L2-regularised logistic regression on the logits of the base scores
(``C`` chosen by time-blocked cross-validation inside the validation weeks). Compared with
two parameter-free combiners: the mean of the logits and the mean rank.

Ensembles: machine learning only (Merve's LR, RF, LightGBM), deep learning only
(Karan's MLP, TabNet, GRU), and both teams together. The question is whether the two
teams' models make *different* mistakes: if they do, combining them beats either team.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

ML = ["logistic_regression", "random_forest", "lightgbm"]
DL = ["mlp", "tabnet", "gru"]
ENSEMBLES = {"stack_ml": ML, "stack_dl": DL, "stack_all": ML + DL}


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_meta(X_val: np.ndarray, y_val: np.ndarray, blocks: int = 5, Cs: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
             seed: int = 0) -> tuple[LogisticRegression, float, dict]:
    """Choose ``C`` by blocked CV over time-ordered validation rows, then refit on all of them.

    ``X_val`` rows must be in time order (blocks are contiguous slices, not random folds).
    """
    n = len(y_val)
    edges = np.linspace(0, n, blocks + 1).astype(int)
    cv: dict[float, float] = {}
    for C in Cs:
        scores = []
        for i in range(blocks):
            te = np.arange(edges[i], edges[i + 1])
            tr = np.setdiff1d(np.arange(n), te)
            m = LogisticRegression(C=C, max_iter=1000, random_state=seed).fit(X_val[tr], y_val[tr])
            scores.append(average_precision_score(y_val[te], m.predict_proba(X_val[te])[:, 1]))
        cv[C] = float(np.mean(scores))
    best = max(cv, key=cv.get)
    return LogisticRegression(C=best, max_iter=1000, random_state=seed).fit(X_val, y_val), best, cv


def mean_logit(X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-X.mean(axis=1)))


def mean_rank(S: np.ndarray) -> np.ndarray:
    return np.mean([rankdata(S[:, j]) / len(S) for j in range(S.shape[1])], axis=0)


def error_overlap(y: np.ndarray, preds: dict[str, np.ndarray]) -> list[dict]:
    """For each pair of models: share of buyers missed by both, and Cohen's kappa of their decisions."""
    names = list(preds)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = preds[a].astype(bool), preds[b].astype(bool)
            miss_a, miss_b = (y == 1) & ~pa, (y == 1) & ~pb
            po = float((pa == pb).mean())
            pe = float(pa.mean() * pb.mean() + (1 - pa.mean()) * (1 - pb.mean()))
            out.append({"model_a": a, "model_b": b, "agreement": po, "kappa": (po - pe) / (1 - pe) if pe < 1 else 1.0,
                        "missed_by_both": float((miss_a & miss_b).sum() / max((miss_a | miss_b).sum(), 1))})
    return out
