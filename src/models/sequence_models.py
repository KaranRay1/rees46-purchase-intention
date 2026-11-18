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


# =============================================================================
# Transformer encoder over the K prefix events (advanced extension)
# =============================================================================
class _AttentionBlock(nn.Module if nn is not None else object):
    """Pre-norm self-attention block that also returns its attention weights."""

    def __init__(self, d_model: int, n_heads: int, ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        a, w = self.attn(h, h, h, need_weights=True, average_attn_weights=True)
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.norm2(x))), w


class _TransformerNet(nn.Module if nn is not None else object):
    """Per-step embeddings -> linear projection + learned positions + [CLS] -> attention blocks -> head.

    Unlike the GRU, every click can attend to every other click directly, and the [CLS]
    token's attention shows which clicks the prediction relied on.
    """

    def __init__(self, embedding_spec, n_numeric: int, d_model: int, n_heads: int, n_layers: int,
                 ff: int, dropout: float, max_len: int = 64) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(n, d) for n, d in embedding_spec)
        self.proj = nn.Linear(sum(d for _, d in embedding_spec) + n_numeric, d_model)
        self.pos = nn.Embedding(max_len + 1, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks = nn.ModuleList(_AttentionBlock(d_model, n_heads, ff, dropout) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))
        nn.init.normal_(self.cls, std=0.02)

    def encode(self, cat, num):
        parts = [emb(cat[..., i]) for i, emb in enumerate(self.embeddings)]
        x = self.proj(torch.cat([*parts, num], dim=-1))                     # (batch, K, d)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], dim=1)          # (batch, 1 + K, d)
        x = x + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        weights = []
        for blk in self.blocks:
            x, w = blk(x)
            weights.append(w)
        return self.norm(x[:, 0]), weights

    def forward(self, cat, num):
        h, _ = self.encode(cat, num)
        return self.head(h).squeeze(-1)


class TransformerModel(GRUModel):
    """Transformer-encoder classifier on the same per-event sequences as the GRU.

    Params (``models.transformer``): ``d_model``, ``n_heads``, ``n_layers``, ``ff_dim``,
    ``dropout`` + the shared training settings (``torch_base.py``).
    """

    name = "transformer"

    def _build_module(self, example):
        return _TransformerNet(
            embedding_spec=self.embedding_spec, n_numeric=len(self.num_features),
            d_model=int(self.params.get("d_model", 64)), n_heads=int(self.params.get("n_heads", 4)),
            n_layers=int(self.params.get("n_layers", 2)), ff=int(self.params.get("ff_dim", 128)),
            dropout=float(self.params.get("dropout", 0.1)),
        )

    def attention(self, X, max_rows: int = 5000) -> np.ndarray:
        """Mean attention of [CLS] on each click position in the last layer, shape (K,), summing to ~1."""
        cat, num = self._to_tensors(X)
        cat, num = cat[:max_rows], num[:max_rows]
        self.module.eval()
        with torch.no_grad():
            _, weights = self.module.encode(cat.to(self.device), num.to(self.device))
        w = weights[-1][:, 0, 1:]                                           # CLS row, skip CLS column
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return w.mean(dim=0).cpu().numpy()
