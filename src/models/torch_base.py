"""Shared PyTorch training loop for the MLP (tabular) and GRU (sequence) models.

Both neural models are trained the same way, so differences in RQ1 come from
the architecture, not the training recipe:

* binary cross-entropy on logits, with ``pos_weight = n_negative / n_positive``
  when ``class_weight = "balanced"`` (same idea as the tree/linear models);
* Adam with weight decay;
* mini-batches shuffled by a seeded generator;
* **early stopping on validation PR-AUC**: after every epoch the validation
  PR-AUC is computed, the best weights are kept, and training stops after
  ``early_stopping_patience`` epochs without improvement; the best weights are
  restored at the end;
* CPU by default (``project.device``) so latency comparisons in RQ4 are fair.

Subclasses implement ``_build_module(example_inputs)`` and ``_to_tensors(X)``.
"""

from __future__ import annotations

import copy
from abc import abstractmethod
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import average_precision_score

from src.models.base import BaseModel

try:
    import torch
    from torch import nn
except ImportError as _exc:  # pragma: no cover - torch is in requirements.txt
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = _exc


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required for the neural models; install requirements.txt.") from _TORCH_IMPORT_ERROR


class TorchBinaryClassifier(BaseModel):
    """Base class: training loop, early stopping, batched inference.

    Args:
        params: model block from config; uses ``learning_rate``, ``weight_decay``,
            ``batch_size``, ``max_epochs``, ``early_stopping_patience``.
        class_weight: ``"balanced"`` (positive-class weighting) or ``None``.
        device: ``"cpu"`` (default), ``"mps"`` or ``"cuda"``.
    """

    def __init__(
        self,
        params: Mapping[str, Any],
        class_weight: str | None = "balanced",
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        _require_torch()
        super().__init__(seed)
        self.params = dict(params)
        self.class_weight = class_weight
        self.device = device
        self.module: "nn.Module | None" = None
        self.history_: list[dict[str, float]] = []
        self.best_epoch_: int | None = None
        self.best_val_score_: float | None = None

    # ------------------------------------------------------------------ to implement
    @abstractmethod
    def _build_module(self, example: tuple["torch.Tensor", ...]) -> "nn.Module":
        """Create the network; ``example`` is the tensor tuple of the training inputs."""

    @abstractmethod
    def _to_tensors(self, X: Any) -> tuple["torch.Tensor", ...]:
        """Convert model input (array or tuple of arrays) to a tuple of CPU tensors."""

    # ------------------------------------------------------------------ training
    def _fit(self, X_train, y_train, X_val, y_val) -> None:
        if X_val is None or y_val is None:
            raise ValueError(f"{self.name} needs a validation split for early stopping.")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        xs = self._to_tensors(X_train)
        y = torch.as_tensor(np.asarray(y_train), dtype=torch.float32)
        n = len(y)
        self.module = self._build_module(xs).to(self.device)

        n_pos = float(y.sum())
        pos_weight = None
        if self.class_weight == "balanced" and 0 < n_pos < n:
            pos_weight = torch.tensor([(n - n_pos) / n_pos], device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(
            self.module.parameters(),
            lr=float(self.params.get("learning_rate", 1e-3)),
            weight_decay=float(self.params.get("weight_decay", 0.0)),
        )

        batch = int(self.params.get("batch_size", 1024))
        max_epochs = int(self.params.get("max_epochs", 50))
        patience = int(self.params.get("early_stopping_patience", 5))
        gen = torch.Generator().manual_seed(self.seed)

        best_state, best_score, best_epoch, bad = None, -np.inf, 0, 0
        self.history_ = []
        for epoch in range(1, max_epochs + 1):
            self.module.train()
            order = torch.randperm(n, generator=gen)
            total, seen = 0.0, 0
            for start in range(0, n, batch):
                idx = order[start : start + batch]
                if len(idx) < 2:  # BatchNorm cannot train on a single row
                    continue
                inputs = tuple(t[idx].to(self.device) for t in xs)
                target = y[idx].to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.module(*inputs).reshape(-1), target)
                loss.backward()
                opt.step()
                total += loss.detach().item() * len(idx)
                seen += len(idx)

            val_score = float(average_precision_score(y_val, self._predict_proba(X_val)))
            self.history_.append({"epoch": epoch, "train_loss": total / max(seen, 1), "val_pr_auc": val_score})
            if val_score > best_score:
                best_score, best_epoch, bad = val_score, epoch, 0
                best_state = copy.deepcopy(self.module.state_dict())
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            self.module.load_state_dict(best_state)
        self.best_epoch_, self.best_val_score_ = best_epoch, best_score

    # ------------------------------------------------------------------ inference
    def _predict_proba(self, X: Any) -> np.ndarray:
        if self.module is None:
            raise RuntimeError(f"{self.name} has no network yet.")
        xs = self._to_tensors(X)
        n = len(xs[0])
        batch = max(int(self.params.get("batch_size", 1024)), 1)
        self.module.eval()
        out = []
        with torch.no_grad():
            for start in range(0, n, batch):
                inputs = tuple(t[start : start + batch].to(self.device) for t in xs)
                out.append(torch.sigmoid(self.module(*inputs).reshape(-1)).cpu())
        return torch.cat(out).numpy() if out else np.zeros(0)

    def set_n_jobs(self, n_jobs: int) -> None:
        """Set PyTorch CPU threads (process-wide; RQ4 benchmarks use 1)."""
        if n_jobs and n_jobs > 0:
            torch.set_num_threads(n_jobs)

    def n_parameters(self) -> int:
        """Trainable parameter count (reported with model size in RQ4)."""
        if self.module is None:
            return 0
        return sum(p.numel() for p in self.module.parameters() if p.requires_grad)

    def get_params(self) -> dict[str, Any]:
        out = {"seed": self.seed, "class_weight": self.class_weight, "device": self.device, **self.params}
        if self.fitted:
            out["best_epoch"] = self.best_epoch_
        return out
