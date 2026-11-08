"""Shared interface for all seven models, plus a config-driven factory.

Every model — rule, logistic regression, random forest, LightGBM, MLP,
TabNet, GRU — exposes the same methods, so experiment scripts, evaluation,
efficiency benchmarks and SHAP code never need to know which model they hold:

    model = build_model("lightgbm", cfg, seed=42, columns=data["train"].columns)
    model.fit(X_train, y_train, X_val, y_val)
    p = model.predict_proba(X_test)        # P(purchase), shape (n,)
    yhat = model.predict(X_test)           # uses model.threshold

``input_kind`` tells the caller which data to pass:
``"tabular"`` → ``SplitData.X`` from ``preprocessing.prepare_splits``;
``"sequence"`` → ``(cat, num)`` from ``sequences.prepare_sequence_splits``.

The decision threshold defaults to 0.5. Experiment code tunes it on the
validation split (``evaluation.metrics``) and stores it with ``set_threshold``.
"""

from __future__ import annotations

import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.config import model_params


class BaseModel(ABC):
    """Common behaviour: timing, thresholding, persistence."""

    name: str = "base"
    input_kind: str = "tabular"          # "tabular" or "sequence"
    tunable_threshold: bool = True       # False for fixed rules

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.threshold: float = 0.5
        self.train_time_sec: float | None = None
        self.fitted: bool = False

    # ------------------------------------------------------------------ API
    def fit(self, X_train: Any, y_train: np.ndarray, X_val: Any = None, y_val: np.ndarray | None = None) -> "BaseModel":
        """Train the model; records wall-clock ``train_time_sec``."""
        start = time.perf_counter()
        self._fit(X_train, np.asarray(y_train), X_val, None if y_val is None else np.asarray(y_val))
        self.train_time_sec = time.perf_counter() - start
        self.fitted = True
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Probability-like score for the positive class, shape ``(n,)``, in [0, 1]."""
        if not self.fitted:
            raise RuntimeError(f"{self.name} is not fitted.")
        return np.asarray(self._predict_proba(X), dtype=np.float64).reshape(-1)

    def predict(self, X: Any, threshold: float | None = None) -> np.ndarray:
        """0/1 predictions at ``threshold`` (default: ``self.threshold``)."""
        t = self.threshold if threshold is None else threshold
        return (self.predict_proba(X) >= t).astype(np.int8)

    def set_threshold(self, threshold: float) -> None:
        if not self.tunable_threshold:
            return
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = float(threshold)

    def get_params(self) -> dict[str, Any]:
        """Hyperparameters to log with a run."""
        return {"seed": self.seed}

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        """Pickle the whole model (sklearn / LightGBM). Torch models override this."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @staticmethod
    def load(path: str | Path) -> "BaseModel":
        with Path(path).open("rb") as fh:
            return pickle.load(fh)

    # ------------------------------------------------------------------ to implement
    @abstractmethod
    def _fit(self, X_train: Any, y_train: np.ndarray, X_val: Any, y_val: np.ndarray | None) -> None: ...

    @abstractmethod
    def _predict_proba(self, X: Any) -> np.ndarray: ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(seed={self.seed}, threshold={self.threshold:.3f})"


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------
def build_model(
    name: str,
    cfg: Mapping[str, Any],
    seed: int,
    columns: Sequence[str] | None = None,
    **extra: Any,
) -> BaseModel:
    """Create an unfitted model from ``config.yaml``.

    Args:
        name: key in ``models.enabled`` (e.g. ``"lightgbm"``).
        seed: random seed for this run.
        columns: output columns of the preprocessor (needed by the rule baseline
            and for feature-named importances / SHAP).
        extra: model-specific arguments (e.g. TabNet categorical indices).
    """
    params = model_params(cfg, name)
    class_weight = cfg["imbalance"]["class_weight"]

    if name == "rule_based":
        from src.models.baselines import RuleBasedBaseline
        return RuleBasedBaseline(columns=columns, rule=params["rule"], seed=seed)
    if name == "logistic_regression":
        from src.models.baselines import LogisticRegressionBaseline
        return LogisticRegressionBaseline(params, class_weight=class_weight, seed=seed, columns=columns)

    n_jobs = int(cfg["project"].get("n_jobs", -1))
    if name == "random_forest":
        from src.models.tree_models import RandomForestModel
        return RandomForestModel(params, class_weight=class_weight, seed=seed, n_jobs=n_jobs, columns=columns)
    if name == "lightgbm":
        from src.models.tree_models import LightGBMModel
        cats = [c for c in cfg["features"]["categorical"] if columns is not None and c in columns]
        return LightGBMModel(
            params, class_weight=class_weight, seed=seed, n_jobs=n_jobs,
            columns=columns, categorical_features=cats,
        )

    device = str(cfg["project"].get("device", "cpu"))
    if name == "mlp":
        from src.models.tabular_nn import MLPModel
        return MLPModel(params, class_weight=class_weight, seed=seed, device=device, columns=columns)
    if name == "tabnet":
        from src.models.tabular_nn import TabNetModel
        if "cat_dims" not in extra:
            raise ValueError("tabnet needs cat_dims=preprocessor.categorical_dims()")
        return TabNetModel(
            params, class_weight=class_weight, seed=seed, device=device,
            columns=columns, cat_dims=extra["cat_dims"],
        )
    if name == "gru":
        from src.models.sequence_models import GRUModel
        if "embedding_spec" not in extra:
            raise ValueError("gru needs embedding_spec=encoder.embedding_spec()")
        return GRUModel(
            params,
            embedding_spec=extra["embedding_spec"],
            num_features=extra.get("num_features", list(cfg["sequences"]["numerical"])),
            class_weight=class_weight, seed=seed, device=device,
        )
    raise ValueError(f"Unknown model {name!r}; expected one of models.enabled.")
