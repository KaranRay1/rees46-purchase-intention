"""Tree ensembles for RQ1 (Merve's half): Random Forest and LightGBM.

Both use the ``raw`` preprocessing profile: numbers untouched (trees do not
need scaling), categories as integer codes (0 = "other").

RandomForestModel
    scikit-learn ``RandomForestClassifier`` with balanced class weights.
    Integer category codes are treated as ordered numbers — a known RF
    limitation, noted in the report; LightGBM handles them natively.
    The same seed gives the same trees; with ``n_jobs > 1`` the per-tree
    probabilities are summed in thread order, so predictions can differ in
    the last floating-point digits (~1e-16) between runs.

LightGBMModel
    ``LGBMClassifier`` with balanced class weights, native categorical splits
    on the code columns, and **early stopping on validation PR-AUC**
    (``average_precision``): it trains up to ``n_estimators`` trees and keeps
    the iteration with the best validation PR-AUC. Training is made
    deterministic (``deterministic=True``, row-wise histograms) so a seed
    always gives the same model.

Both expose ``feature_importances()`` (named) for the report; SHAP values
come later from ``src.explainability``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.base import BaseModel

try:  # LightGBM is optional at import time so RF still works without it
    import lightgbm as lgb
except (ImportError, OSError) as _exc:  # OSError: missing libomp on macOS
    lgb = None
    _LGB_IMPORT_ERROR = _exc


def _named(values: np.ndarray, columns: Sequence[str] | None) -> dict[str, float]:
    names = list(columns) if columns is not None else [f"x{i}" for i in range(len(values))]
    return dict(zip(names, np.asarray(values, dtype=float).tolist()))


# -----------------------------------------------------------------------------
# Random Forest
# -----------------------------------------------------------------------------
class RandomForestModel(BaseModel):
    """Random Forest (scikit-learn).

    Args:
        params: ``models.random_forest`` block (``n_estimators``, ``max_depth``,
            ``min_samples_split``, ``min_samples_leaf``, ``max_features``).
        class_weight: ``"balanced"`` or ``None``.
        n_jobs: CPU cores for training and prediction.
        columns: input column names (for importances).
    """

    name = "random_forest"

    def __init__(
        self,
        params: Mapping[str, Any],
        class_weight: str | None = "balanced",
        seed: int = 42,
        n_jobs: int = -1,
        columns: Sequence[str] | None = None,
    ) -> None:
        super().__init__(seed)
        self.params = dict(params)
        self.class_weight = class_weight
        self.columns = list(columns) if columns is not None else None
        self.model = RandomForestClassifier(
            n_estimators=int(self.params.get("n_estimators", 300)),
            max_depth=self.params.get("max_depth"),
            min_samples_split=int(self.params.get("min_samples_split", 2)),
            min_samples_leaf=int(self.params.get("min_samples_leaf", 1)),
            max_features=self.params.get("max_features", "sqrt"),
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=seed,
        )

    def _fit(self, X_train, y_train, X_val, y_val) -> None:
        self.model.fit(X_train, y_train)

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def set_n_jobs(self, n_jobs: int) -> None:
        """Change prediction threads (RQ4 benchmarks use 1)."""
        self.model.set_params(n_jobs=n_jobs)

    def feature_importances(self) -> dict[str, float]:
        """Mean decrease in impurity per column (sums to 1)."""
        if not self.fitted:
            raise RuntimeError("Model is not fitted.")
        return _named(self.model.feature_importances_, self.columns)

    def get_params(self) -> dict[str, Any]:
        return {"seed": self.seed, "class_weight": self.class_weight, **self.params}


# -----------------------------------------------------------------------------
# LightGBM
# -----------------------------------------------------------------------------
class LightGBMModel(BaseModel):
    """Gradient-boosted trees (LightGBM) with early stopping on validation PR-AUC.

    Args:
        params: ``models.lightgbm`` block (``n_estimators``, ``learning_rate``,
            ``num_leaves``, ``max_depth``, ``min_child_samples``, ``subsample``,
            ``subsample_freq``, ``colsample_bytree``, ``early_stopping_rounds``,
            ``eval_metric``).
        class_weight: ``"balanced"`` or ``None``.
        n_jobs: CPU threads.
        columns: input column names.
        categorical_features: names of integer-coded categorical columns
            (must be in ``columns``); LightGBM splits on them natively.
    """

    name = "lightgbm"

    def __init__(
        self,
        params: Mapping[str, Any],
        class_weight: str | None = "balanced",
        seed: int = 42,
        n_jobs: int = -1,
        columns: Sequence[str] | None = None,
        categorical_features: Sequence[str] | None = None,
    ) -> None:
        if lgb is None:
            raise ImportError(
                "LightGBM could not be imported. On macOS without Homebrew run "
                "`bash scripts/fix_macos_lightgbm.sh`."
            ) from _LGB_IMPORT_ERROR
        super().__init__(seed)
        self.params = dict(params)
        self.class_weight = class_weight
        self.columns = list(columns) if columns is not None else None
        cats = list(categorical_features or [])
        if cats and self.columns is None:
            raise ValueError("columns are required when categorical_features are given.")
        missing = [c for c in cats if c not in (self.columns or [])]
        if missing:
            raise ValueError(f"categorical_features not in columns: {missing}")
        self.categorical_features = cats
        self.categorical_indices = [self.columns.index(c) for c in cats] if cats else []
        self.eval_metric = self.params.get("eval_metric", "average_precision")
        self.early_stopping_rounds = int(self.params.get("early_stopping_rounds", 50))

        self.model = lgb.LGBMClassifier(
            n_estimators=int(self.params.get("n_estimators", 1000)),
            learning_rate=float(self.params.get("learning_rate", 0.05)),
            num_leaves=int(self.params.get("num_leaves", 31)),
            max_depth=int(self.params.get("max_depth", -1)),
            min_child_samples=int(self.params.get("min_child_samples", 20)),
            subsample=float(self.params.get("subsample", 1.0)),
            subsample_freq=int(self.params.get("subsample_freq", 0)),
            colsample_bytree=float(self.params.get("colsample_bytree", 1.0)),
            reg_lambda=float(self.params.get("reg_lambda", 0.0)),   # LightGBM's default; tuned in step 15
            class_weight=class_weight,
            metric=self.eval_metric,        # replaces the default logloss
            random_state=seed,
            n_jobs=n_jobs,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
        )

    def _fit(self, X_train, y_train, X_val, y_val) -> None:
        if X_val is None or y_val is None:
            raise ValueError("LightGBMModel needs a validation split for early stopping.")
        self.model.fit(
            X_train,
            y_train,
            eval_X=(X_val,),
            eval_y=(y_val,),
            eval_names=["val"],
            categorical_feature=self.categorical_indices or "auto",
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, first_metric_only=True, verbose=False)],
        )
        self.best_iteration_ = int(self.model.best_iteration_ or self.model.n_estimators)
        self.best_val_score_ = float(self.model.best_score_["val"][self.eval_metric])

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        # LightGBM uses best_iteration_ automatically after early stopping.
        return self.model.predict_proba(X)[:, 1]

    def set_n_jobs(self, n_jobs: int) -> None:
        """Change prediction threads (RQ4 benchmarks use 1)."""
        self.model.set_params(n_jobs=n_jobs)

    def feature_importances(self, kind: str = "gain") -> dict[str, float]:
        """Total split gain (``"gain"``) or split count (``"split"``) per column."""
        if not self.fitted:
            raise RuntimeError("Model is not fitted.")
        values = self.model.booster_.feature_importance(importance_type=kind)
        return _named(values, self.columns)

    def get_params(self) -> dict[str, Any]:
        out = {"seed": self.seed, "class_weight": self.class_weight, **self.params,
               "categorical_features": self.categorical_features}
        if self.fitted:
            out["best_iteration"] = self.best_iteration_
        return out
