"""GRU over the K prefix events — the sequence-model extension in RQ1 (H1c).

Unlike every other model, the GRU does not see aggregated features: it reads
the K events of the prefix one step at a time (from ``sequences.py``) and
learns its own summary of the session. Comparing it with the tabular models
on the *same sessions* asks whether event order adds information beyond
counts and ratios.

Architecture (sizes from ``models.gru`` and ``sequences`` in config)::

    per step:  [emb(event_type) | emb(main_category) | log_price | log_gap_sec]
            -> GRU(hidden_size, num_layers, bidirectional?)
            -> final hidden state (both directions if bidirectional)
            -> Dropout -> Linear -> purchase logit

Every prefix has exactly K events, so no padding or masking is needed and
the final hidden state is the state after event K (the prediction time).
Training uses the shared recipe in ``torch_base.py`` (weighted BCE, Adam,
early stopping on validation PR-AUC, best weights restored).

Input ``X`` is either a ``(cat, num)`` tuple of arrays or a
``SequenceData`` object from ``prepare_sequence_splits``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.models.torch_base import TorchBinaryClassifier

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


class _GRUNet(nn.Module if nn is not None else object):
    """Embeddings + GRU + linear head."""

    def __init__(
        self,
        embedding_spec: Sequence[tuple[int, int]],
        n_numeric: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(n, d) for n, d in embedding_spec)
        input_size = sum(d for _, d in embedding_spec) + n_numeric
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,  # GRU dropout acts between layers only
            bidirectional=bidirectional,
        )
        self.bidirectional = bidirectional
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * (2 if bidirectional else 1), 1),
        )

    def forward(self, cat: "torch.Tensor", num: "torch.Tensor") -> "torch.Tensor":
        parts = [emb(cat[..., i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([*parts, num], dim=-1)            # (batch, K, input_size)
        _, h_n = self.gru(x)                            # (layers * directions, batch, hidden)
        last = torch.cat([h_n[-2], h_n[-1]], dim=-1) if self.bidirectional else h_n[-1]
        return self.head(last).squeeze(-1)


class GRUModel(TorchBinaryClassifier):
    """GRU classifier on per-event sequences.

    Args:
        params: ``models.gru`` block (``hidden_size``, ``num_layers``, ``dropout``,
            ``bidirectional`` + shared training settings).
        embedding_spec: ``{categorical feature: (n_codes, emb_dim)}`` from
            ``SequenceEncoder.embedding_spec()``; order = last axis of ``cat``.
        num_features: names of the numeric step features (order = last axis of ``num``).
    """

    name = "gru"
    input_kind = "sequence"

    def __init__(
        self,
        params: Mapping[str, Any],
        embedding_spec: Mapping[str, tuple[int, int]],
        num_features: Sequence[str],
        class_weight: str | None = "balanced",
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        super().__init__(params, class_weight, seed, device)
        if not embedding_spec:
            raise ValueError("GRUModel needs at least one categorical step feature (embedding_spec).")
        self.cat_features = list(embedding_spec)
        self.embedding_spec = [tuple(int(v) for v in embedding_spec[f]) for f in self.cat_features]
        self.num_features = list(num_features)

    def _build_module(self, example):
        return _GRUNet(
            embedding_spec=self.embedding_spec,
            n_numeric=len(self.num_features),
            hidden_size=int(self.params.get("hidden_size", 64)),
            num_layers=int(self.params.get("num_layers", 1)),
            dropout=float(self.params.get("dropout", 0.0)),
            bidirectional=bool(self.params.get("bidirectional", False)),
        )

    def _to_tensors(self, X):
        cat, num = (X.cat, X.num) if hasattr(X, "cat") else X
        cat = np.asarray(cat)
        num = np.asarray(num, dtype=np.float32)
        if cat.ndim != 3 or num.ndim != 3 or cat.shape[:2] != num.shape[:2]:
            raise ValueError(f"Expected cat/num of shape (n, K, ·); got {cat.shape} and {num.shape}.")
        if cat.shape[2] != len(self.cat_features) or num.shape[2] != len(self.num_features):
            raise ValueError(
                f"Expected {len(self.cat_features)} categorical and {len(self.num_features)} numeric "
                f"step features; got {cat.shape[2]} and {num.shape[2]}."
            )
        limits = np.array([n for n, _ in self.embedding_spec])
        if cat.size and ((cat < 0).any() or (cat >= limits).any()):
            raise ValueError("Categorical codes outside the embedding range; refit the SequenceEncoder.")
        return torch.as_tensor(cat.astype(np.int64)), torch.as_tensor(num)

    def get_params(self) -> dict[str, Any]:
        out = super().get_params()
        out["embedding_spec"] = dict(zip(self.cat_features, self.embedding_spec))
        out["num_features"] = self.num_features
        return out
