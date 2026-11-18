"""Conformal prediction: label sets with a distribution-free coverage guarantee.

A point score says "buyer" or "not buyer" even when the model is unsure. Split conformal
prediction (Vovk et al., 2005; Angelopoulos & Bates, 2023) instead returns a *set* of
labels per visit -- ``{buyer}``, ``{not buyer}``, ``{both}`` (unsure) or, rarely, ``{}`` --
calibrated on held-out data so that the true label is in the set for at least
``1 - alpha`` of new visits, whatever the model.

* **Non-conformity score**: ``1 - s`` for a buyer and ``s`` for a non-buyer, where ``s`` is
  the model's score (a probability is not needed).
* **Class-conditional (Mondrian)**: one threshold per class, so coverage holds *within*
  buyers and within non-buyers. With about 10 % buyers, a single threshold would reach
  90 % coverage by covering the non-buyers and failing the buyers.
* **Calibration set**: the validation weeks (the model never trained on them); coverage is
  then measured on the later test weeks. The guarantee assumes exchangeability, which time
  can break, so coverage is also reported week by week.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


def _quantile(scores: np.ndarray, alpha: float) -> float:
    """Conformal quantile: the ceil((n + 1)(1 - alpha))-th smallest score."""
    n = len(scores)
    if n == 0:
        return np.inf
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


@dataclass(frozen=True)
class MondrianConformal:
    """Class-conditional split conformal classifier for a binary score."""

    q_buyer: float        # a visit can be "buyer" when 1 - s <= q_buyer
    q_non: float          # a visit can be "not buyer" when s <= q_non
    alpha: float

    @classmethod
    def fit(cls, y_cal: np.ndarray, s_cal: np.ndarray, alpha: float = 0.1) -> "MondrianConformal":
        y, s = np.asarray(y_cal), np.asarray(s_cal, dtype=float)
        return cls(_quantile(1 - s[y == 1], alpha), _quantile(s[y == 0], alpha), alpha)

    def sets(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Boolean arrays (buyer in set, not-buyer in set)."""
        s = np.asarray(s, dtype=float)
        return (1 - s) <= self.q_buyer, s <= self.q_non

    @staticmethod
    def kind(has_buyer: np.ndarray, has_non: np.ndarray) -> np.ndarray:
        return np.where(has_buyer & has_non, "unsure", np.where(has_buyer, "buyer",
                                                                 np.where(has_non, "not_buyer", "empty")))

    def evaluate(self, y: np.ndarray, s: np.ndarray) -> dict[str, float]:
        y = np.asarray(y)
        b, nb = self.sets(s)
        covered = np.where(y == 1, b, nb)
        single_b, single_nb = b & ~nb, nb & ~b
        return {
            "alpha": self.alpha, "target_coverage": 1 - self.alpha, "n": int(len(y)),
            "coverage": float(covered.mean()),
            "coverage_buyers": float(covered[y == 1].mean()) if (y == 1).any() else np.nan,
            "coverage_non_buyers": float(covered[y == 0].mean()) if (y == 0).any() else np.nan,
            "avg_set_size": float((b.astype(int) + nb.astype(int)).mean()),
            "share_confident_buyer": float(single_b.mean()),
            "share_confident_non_buyer": float(single_nb.mean()),
            "share_unsure": float((b & nb).mean()),
            "share_empty": float((~b & ~nb).mean()),
            "precision_confident_buyer": float(y[single_b].mean()) if single_b.any() else np.nan,
            "npv_confident_non_buyer": float(1 - y[single_nb].mean()) if single_nb.any() else np.nan,
            "q_buyer": self.q_buyer, "q_non": self.q_non,
        }


def weekly_coverage(cp: MondrianConformal, y: np.ndarray, s: np.ndarray, week: np.ndarray) -> pl.DataFrame:
    """Coverage per week (by session start): does the guarantee survive the passage of time?"""
    b, nb = cp.sets(s)
    covered = np.where(np.asarray(y) == 1, b, nb)
    df = pl.DataFrame({"week": week, "y": np.asarray(y), "covered": covered, "unsure": b & nb})
    return df.group_by("week").agg(
        n=pl.len(), coverage=pl.col("covered").mean(),
        coverage_buyers=pl.col("covered").filter(pl.col("y") == 1).mean(),
        coverage_non_buyers=pl.col("covered").filter(pl.col("y") == 0).mean(),
        share_unsure=pl.col("unsure").mean(),
    ).sort("week")
