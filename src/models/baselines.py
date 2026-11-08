"""Baselines for RQ1: a cart-presence rule and logistic regression.

RuleBasedBaseline
    The proposals' minimum-performance reference: predict a purchase if the
    prefix contains at least one cart event (``cart_count > 0``). No training,
    no threshold tuning. Its score is 0/1, so PR-AUC reduces to a single
    operating point.

LogisticRegressionBaseline
    The linear baseline both hypotheses (H1a, H1b) are tested against. Runs on
    the ``onehot_scaled`` preprocessing profile (log1p + standardised numbers,
    one-hot categories), with balanced class weights from ``config.yaml``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.models.base import BaseModel

RULES = {"cart_count_gt_0": "cart_count"}


class RuleBasedBaseline(BaseModel):
    """``1`` if ``cart_count > 0`` in the prefix, else ``0``.

    Use with the ``raw`` preprocessing profile so ``cart_count`` is an
    unscaled count (a standardised value would make "> 0" meaningless).

    Args:
        columns: preprocessor output columns; must contain ``cart_count``.
        rule: rule name from config (``cart_count_gt_0``).
    """

    name = "rule_based"
    tunable_threshold = False

    def __init__(self, columns: Sequence[str] | None, rule: str = "cart_count_gt_0", seed: int = 42) -> None:
        super().__init__(seed)
        if rule not in RULES:
            raise ValueError(f"Unknown rule {rule!r}; choose from {sorted(RULES)}")
        if not columns or RULES[rule] not in columns:
            raise ValueError(f"RuleBasedBaseline needs a {RULES[rule]!r} column in the input.")
        self.rule = rule
        self.col_idx = list(columns).index(RULES[rule])

    def _fit(self, X_train, y_train, X_val, y_val) -> None:
        pass  # nothing to learn

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X)[:, self.col_idx] > 0).astype(np.float64)

    def get_params(self) -> dict[str, Any]:
        return {"seed": self.seed, "rule": self.rule}


class LogisticRegressionBaseline(BaseModel):
    """L2-regularised logistic regression (scikit-learn).

    Args:
        params: ``models.logistic_regression`` block from config
            (``C``, ``penalty``, ``solver``, ``max_iter``).
        class_weight: ``"balanced"`` or ``None``.
        columns: preprocessor output columns (for named coefficients).
    """

    name = "logistic_regression"

    def __init__(
        self,
        params: Mapping[str, Any],
        class_weight: str | None = "balanced",
        seed: int = 42,
        columns: Sequence[str] | None = None,
    ) -> None:
        super().__init__(seed)
        self.params = dict(params)
        self.class_weight = class_weight
        self.columns = list(columns) if columns is not None else None
        self.model = LogisticRegression(
            C=self._effective_C(),
            l1_ratio=self._l1_ratio(),
            solver=self.params.get("solver", "lbfgs"),
            max_iter=int(self.params.get("max_iter", 1000)),
            class_weight=class_weight,
            random_state=seed,
        )

    # scikit-learn >= 1.8 deprecates `penalty`; translate it to l1_ratio / C.
    def _l1_ratio(self) -> float:
        penalty = self.params.get("penalty", "l2")
        mapping = {"l2": 0.0, "l1": 1.0, "none": 0.0}
        if penalty == "elasticnet":
            return float(self.params.get("l1_ratio", 0.5))
        if penalty not in mapping:
            raise ValueError(f"Unsupported penalty {penalty!r}")
        return mapping[penalty]

    def _effective_C(self) -> float:
        return float("inf") if self.params.get("penalty") == "none" else float(self.params.get("C", 1.0))

    def _fit(self, X_train, y_train, X_val, y_val) -> None:
        self.model.fit(X_train, y_train)
        self.n_iter_ = int(np.max(self.model.n_iter_))
        self.converged_ = self.n_iter_ < self.model.max_iter

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def coefficients(self) -> dict[str, float]:
        """Coefficient per input column (log-odds per standard deviation for numbers)."""
        if not self.fitted:
            raise RuntimeError("Model is not fitted.")
        names = self.columns or [f"x{i}" for i in range(self.model.coef_.shape[1])]
        return dict(zip(names, self.model.coef_[0].tolist()))

    def get_params(self) -> dict[str, Any]:
        return {"seed": self.seed, "class_weight": self.class_weight, **self.params}
