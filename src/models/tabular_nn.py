"""Deep tabular models for RQ1 (Karan's half): MLP and TabNet.

MLPModel  (PyTorch, ``onehot_scaled`` profile)
    Dense blocks ``Linear → BatchNorm → ReLU → Dropout`` (sizes from
    ``models.mlp.hidden_layers``), then one output logit. Trained with the
    shared loop in ``torch_base.py``: weighted BCE, Adam, early stopping on
    validation PR-AUC, best weights restored.

TabNetModel  (pytorch-tabnet, ``ordinal_scaled`` profile)
    Attentive tabular network (Arik & Pfister, 2021). Categorical columns are
    integer codes that TabNet embeds (``cat_emb_dim``). Uses the same recipe
    as the other models where the library allows it:
      * balanced class weights in the cross-entropy loss,
      * early stopping on validation PR-AUC (custom metric below),
      * best weights restored at the end (pytorch-tabnet default).
    ``explain(X)`` returns TabNet's own per-feature attention masks (RQ5).
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score

from src.models.base import BaseModel
from src.models.torch_base import TorchBinaryClassifier, _require_torch

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


# -----------------------------------------------------------------------------
# MLP
# -----------------------------------------------------------------------------
def _mlp_network(n_inputs: int, hidden: Sequence[int], dropout: float, batch_norm: bool, activation: str) -> "nn.Module":
    acts = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
    if activation not in acts:
        raise ValueError(f"Unsupported activation {activation!r}; choose from {sorted(acts)}")
    layers: list[nn.Module] = []
    width = n_inputs
    for h in hidden:
        layers.append(nn.Linear(width, h))
        if batch_norm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(acts[activation]())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        width = h
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


class MLPModel(TorchBinaryClassifier):
    """Multi-layer perceptron on the scaled, one-hot tabular features.

    Args:
        params: ``models.mlp`` block (``hidden_layers``, ``activation``,
            ``batch_norm``, ``dropout`` + the shared training settings).
        columns: input column names (kept for SHAP / reporting).
    """

    name = "mlp"

    def __init__(
        self,
        params: Mapping[str, Any],
        class_weight: str | None = "balanced",
        seed: int = 42,
        device: str = "cpu",
        columns: Sequence[str] | None = None,
    ) -> None:
        super().__init__(params, class_weight, seed, device)
        self.columns = list(columns) if columns is not None else None

    def _build_module(self, example):
        return _mlp_network(
            n_inputs=example[0].shape[1],
            hidden=[int(h) for h in self.params.get("hidden_layers", [128, 64])],
            dropout=float(self.params.get("dropout", 0.0)),
            batch_norm=bool(self.params.get("batch_norm", True)),
            activation=str(self.params.get("activation", "relu")),
        )

    def _to_tensors(self, X):
        return (torch.as_tensor(np.asarray(X, dtype=np.float32)),)


# -----------------------------------------------------------------------------
# TabNet
# -----------------------------------------------------------------------------
try:
    with warnings.catch_warnings():  # pytorch-tabnet imports a deprecated scipy path
        warnings.simplefilter("ignore", DeprecationWarning)
        from pytorch_tabnet.metrics import Metric as _TabNetMetric
except ImportError:  # pragma: no cover
    _TabNetMetric = object


class TabNetPRAUC(_TabNetMetric):
    """Validation PR-AUC for pytorch-tabnet early stopping (module level so models pickle)."""

    def __init__(self):
        self._name = "pr_auc"
        self._maximize = True

    def __call__(self, y_true, y_score):
        return float(average_precision_score(y_true, y_score[:, 1]))


class TabNetModel(BaseModel):
    """TabNet classifier (pytorch-tabnet).

    Args:
        params: ``models.tabnet`` block (``n_d``, ``n_a``, ``n_steps``, ``gamma``,
            ``lambda_sparse``, ``learning_rate``, ``batch_size``,
            ``virtual_batch_size``, ``max_epochs``, ``early_stopping_patience``,
            ``cat_emb_dim``).
        columns: input column names (from the ``ordinal_scaled`` preprocessor).
        cat_dims: ``{categorical column: number of codes}`` from
            ``TabularPreprocessor.categorical_dims()``.
    """

    name = "tabnet"

    def __init__(
        self,
        params: Mapping[str, Any],
        class_weight: str | None = "balanced",
        seed: int = 42,
        device: str = "cpu",
        columns: Sequence[str] | None = None,
        cat_dims: Mapping[str, int] | None = None,
    ) -> None:
        _require_torch()
        super().__init__(seed)
        self.params = dict(params)
        self.class_weight = class_weight
        self.device = device
        self.columns = list(columns) if columns is not None else None
        cat_dims = dict(cat_dims or {})
        if cat_dims and self.columns is None:
            raise ValueError("columns are required when cat_dims are given.")
        missing = [c for c in cat_dims if c not in (self.columns or [])]
        if missing:
            raise ValueError(f"cat_dims columns not in columns: {missing}")
        self.cat_names = list(cat_dims)
        self.cat_idxs = [self.columns.index(c) for c in self.cat_names]
        self.cat_dims = [int(cat_dims[c]) for c in self.cat_names]
        self.model = None
        self.best_epoch_: int | None = None
        self.best_val_score_: float | None = None

    def _make(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from pytorch_tabnet.tab_model import TabNetClassifier

        return TabNetClassifier(
            n_d=int(self.params.get("n_d", 16)),
            n_a=int(self.params.get("n_a", 16)),
            n_steps=int(self.params.get("n_steps", 4)),
            gamma=float(self.params.get("gamma", 1.5)),
            lambda_sparse=float(self.params.get("lambda_sparse", 1e-4)),
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            cat_emb_dim=int(self.params.get("cat_emb_dim", 4)),
            optimizer_fn=torch.optim.Adam,
            optimizer_params={"lr": float(self.params.get("learning_rate", 2e-2))},
            seed=self.seed,
            device_name=self.device,
            verbose=0,
        )

    def _fit(self, X_train, y_train, X_val, y_val) -> None:
        if X_val is None or y_val is None:
            raise ValueError("TabNetModel needs a validation split for early stopping.")
        torch.manual_seed(self.seed)
        X_train = np.asarray(X_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_train = np.asarray(y_train).astype(np.int64)
        y_val = np.asarray(y_val).astype(np.int64)

        weight = None
        if self.class_weight == "balanced":
            counts = np.bincount(y_train, minlength=2).astype(np.float64)
            if (counts > 0).all():
                weight = torch.tensor(len(y_train) / (2.0 * counts), dtype=torch.float32)
        self.model = self._make()
        with warnings.catch_warnings():
            # Informational only: the library restores the best epoch's weights.
            warnings.filterwarnings("ignore", message="Best weights from best epoch")
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_name=["val"],
                eval_metric=[TabNetPRAUC],
                loss_fn=torch.nn.CrossEntropyLoss(weight=weight),
                max_epochs=int(self.params.get("max_epochs", 50)),
                patience=int(self.params.get("early_stopping_patience", 5)),
                batch_size=int(self.params.get("batch_size", 1024)),
                virtual_batch_size=int(self.params.get("virtual_batch_size", 128)),
                drop_last=False,
            )
        self.best_epoch_ = int(self.model.best_epoch) + 1   # library counts from 0
        self.best_val_score_ = float(self.model.best_cost)

    def _predict_proba(self, X):
        return self.model.predict_proba(np.asarray(X, dtype=np.float32))[:, 1]

    def feature_importances(self) -> dict[str, float]:
        """TabNet's global importances (aggregated attention masks, sums to 1)."""
        if not self.fitted:
            raise RuntimeError("Model is not fitted.")
        names = self.columns or [f"x{i}" for i in range(len(self.model.feature_importances_))]
        return dict(zip(names, np.asarray(self.model.feature_importances_, dtype=float).tolist()))

    def explain(self, X) -> np.ndarray:
        """Per-session, per-feature attention attribution, shape (n, n_columns)."""
        if not self.fitted:
            raise RuntimeError("Model is not fitted.")
        masks, _ = self.model.explain(np.asarray(X, dtype=np.float32))
        return masks

    def set_n_jobs(self, n_jobs: int) -> None:
        if n_jobs and n_jobs > 0:
            torch.set_num_threads(n_jobs)

    def n_parameters(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.network.parameters() if p.requires_grad)

    def get_params(self) -> dict[str, Any]:
        out = {"seed": self.seed, "class_weight": self.class_weight, "device": self.device,
               **self.params, "categorical_columns": self.cat_names}
        if self.fitted:
            out["best_epoch"] = self.best_epoch_
        return out
