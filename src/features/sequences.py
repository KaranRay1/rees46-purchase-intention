"""Per-event sequences of the K prefix events, for the GRU.

The tabular models see one aggregated row per session; the GRU sees the same
K events **in order**, one step per event. Both are built from the same
``prefix_events`` and the same split / target labels, and sessions are
returned in the same order (sorted by ``user_session``) as
``preprocessing.prepare_splits``, so predictions can be compared pair-wise
(McNemar, paired bootstrap).

Per step (config ``sequences``)
-------------------------------
categorical  event_type     -> integer code (embedding in the GRU)
             main_category  -> top level of this event's category_code, top-N
                               from train, rare/unseen -> 0 ("other")
numerical    log_price      -> log1p(price), standardised on train
             log_gap_sec    -> log1p(seconds since previous event; 0 at step 1),
                               standardised on train

Arrays per split::

    cat : int64   (n_sessions, K, n_categorical)   codes, 0 = other/unseen
    num : float32 (n_sessions, K, n_numerical)
    y   : int8    (n_sessions,)

Typical use::

    data, enc = prepare_sequence_splits(cfg, k=5)
    enc.embedding_spec()   # {"event_type": (n_codes, 4), "main_category": (n_codes, 8)}
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl

from src.data.sessionize import TARGET, load_prefixes
from src.data.split import load_split_sessions

OTHER = "other"
SPLITS = ("train", "val", "test")


@dataclass
class SequenceData:
    """Model-ready sequence arrays for one split."""

    cat: np.ndarray
    num: np.ndarray
    y: np.ndarray
    session_ids: list[str]
    cat_features: list[str]
    num_features: list[str]


# -----------------------------------------------------------------------------
# Per-event table
# -----------------------------------------------------------------------------
def event_steps(prefix_events: pl.LazyFrame, labels: pl.LazyFrame, unknown_category: str) -> pl.DataFrame:
    """One row per (session, step) with raw step features, split and target.

    Args:
        prefix_events: first K events per session with ``pos``.
        labels: ``user_session``, ``split``, target (purged sessions already removed).
        unknown_category: token used by the loader for a missing category_code.
    """
    code = pl.col("category_code")
    return (
        prefix_events.join(labels, on="user_session", how="inner")
        .sort(["user_session", "pos"])
        .with_columns(
            event_type=pl.col("event_type").cast(pl.String),
            main_category=pl.when(code == unknown_category)
            .then(pl.lit(unknown_category))
            .otherwise(code.str.split(".").list.first())
            .fill_null(unknown_category),
            log_price=pl.col("price").clip(lower_bound=0.0).log1p(),
            log_gap_sec=pl.col("event_time")
            .diff()
            .over("user_session")
            .dt.total_seconds()
            .cast(pl.Float64)
            .fill_null(0.0)
            .clip(lower_bound=0.0)
            .log1p(),
        )
        .select(
            "user_session", "pos", "split", TARGET,
            "event_type", "main_category", "log_price", "log_gap_sec",
        )
        .collect()
    )


# -----------------------------------------------------------------------------
# Encoder (fitted on train steps only)
# -----------------------------------------------------------------------------
@dataclass
class SequenceEncoder:
    """Vocabularies and scaling for sequence steps, fitted on the training split.

    Args:
        cat_features: categorical step features (order = last axis of ``cat``).
        num_features: numerical step features (order = last axis of ``num``).
        embedding_dims: embedding size per categorical feature (for the GRU).
        top_n: max categories kept for features in ``top_n_features``.
    """

    cat_features: list[str]
    num_features: list[str]
    embedding_dims: dict[str, int] = field(default_factory=dict)
    top_n: int = 20
    top_n_features: list[str] = field(default_factory=lambda: ["main_category"])

    vocab: dict[str, list[str]] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    fitted: bool = False

    def fit(self, train_steps: pl.DataFrame) -> "SequenceEncoder":
        if train_steps.height == 0:
            raise ValueError("Cannot fit on an empty training split.")
        self.vocab = {}
        for f in self.cat_features:
            counts = (
                train_steps.group_by(f).len()
                .filter(pl.col(f).is_not_null() & (pl.col(f) != OTHER))
                .sort(["len", f], descending=[True, False])
            )
            values = counts[f].to_list()
            self.vocab[f] = values[: self.top_n] if f in self.top_n_features else values
        self.means, self.stds = {}, {}
        for f in self.num_features:
            col = train_steps[f]
            self.means[f] = float(col.mean())
            s = col.std(ddof=0)
            self.stds[f] = float(s) if s is not None and s > 0 and math.isfinite(s) else 1.0
        self.fitted = True
        return self

    def embedding_spec(self) -> dict[str, tuple[int, int]]:
        """``{feature: (number_of_codes, embedding_dim)}``; code 0 is "other"."""
        self._check_fitted()
        return {f: (len(self.vocab[f]) + 1, int(self.embedding_dims.get(f, 8))) for f in self.cat_features}

    def transform(self, steps: pl.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Encode steps (sorted by session, pos) into ``(cat, num)`` arrays of shape (n, K, ·)."""
        self._check_fitted()
        if steps.height % k:
            raise ValueError(f"{steps.height} steps is not a multiple of K={k}.")
        n = steps.height // k

        cats = [
            steps[f].replace_strict(
                {v: i + 1 for i, v in enumerate(self.vocab[f])}, default=0, return_dtype=pl.Int64
            ).to_numpy()
            for f in self.cat_features
        ]
        nums = [
            ((steps[f] - self.means[f]) / self.stds[f]).cast(pl.Float32).to_numpy()
            for f in self.num_features
        ]
        cat = np.stack(cats, axis=-1).reshape(n, k, len(self.cat_features)) if cats else np.zeros((n, k, 0), np.int64)
        num = np.stack(nums, axis=-1).reshape(n, k, len(self.num_features)) if nums else np.zeros((n, k, 0), np.float32)
        return cat.astype(np.int64, copy=False), num.astype(np.float32, copy=False)

    # persistence
    def to_dict(self) -> dict[str, Any]:
        self._check_fitted()
        return {
            "cat_features": self.cat_features, "num_features": self.num_features,
            "embedding_dims": self.embedding_dims, "top_n": self.top_n,
            "top_n_features": self.top_n_features, "vocab": self.vocab,
            "means": self.means, "stds": self.stds,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SequenceEncoder":
        state = {k: d[k] for k in ("vocab", "means", "stds")}
        obj = cls(**{k: v for k, v in d.items() if k not in state})
        obj.vocab, obj.means, obj.stds = state["vocab"], state["means"], state["stds"]
        obj.fitted = True
        return obj

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SequenceEncoder":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("SequenceEncoder is not fitted; call fit() on the training split.")


def encoder_from_config(cfg: Mapping[str, Any]) -> SequenceEncoder:
    """Unfitted encoder with features and embedding sizes from ``cfg["sequences"]``."""
    seq = cfg["sequences"]
    if seq.get("padding", "none") != "none":
        raise ValueError("Only sequences.padding = none is supported (every prefix has exactly K events).")
    return SequenceEncoder(
        cat_features=list(seq["categorical"]),
        num_features=list(seq["numerical"]),
        embedding_dims={f: spec["embedding_dim"] for f, spec in seq["categorical"].items()},
        top_n=cfg["features"]["top_n_categories"],
    )


# -----------------------------------------------------------------------------
# Config-driven entry point
# -----------------------------------------------------------------------------
def prepare_sequence_splits(
    cfg: Mapping[str, Any], k: int, steps: pl.DataFrame | None = None
) -> tuple[dict[str, SequenceData], SequenceEncoder]:
    """Build, fit (train only) and encode sequences for every split.

    Args:
        steps: optional pre-built output of ``event_steps`` (mainly for tests).

    Returns:
        ({"train": SequenceData, "val": ..., "test": ...}, fitted encoder)
    """
    if steps is None:
        labels = load_split_sessions(cfg, k).select("user_session", "split", TARGET)
        steps = event_steps(
            load_prefixes(cfg, k).prefix_events, labels, cfg["data"]["cleaning"]["missing_category_token"]
        )
    enc = encoder_from_config(cfg)
    parts = {s: steps.filter(pl.col("split") == s) for s in SPLITS}
    enc.fit(parts["train"])  # <- the only place vocabularies and statistics are learned

    out: dict[str, SequenceData] = {}
    for s, part in parts.items():
        cat, num = enc.transform(part, k)
        first = part.filter(pl.col("pos") == 1)
        out[s] = SequenceData(
            cat=cat,
            num=num,
            y=first[TARGET].to_numpy().astype(np.int8),
            session_ids=first["user_session"].to_list(),
            cat_features=enc.cat_features,
            num_features=enc.num_features,
        )
    return out, enc
