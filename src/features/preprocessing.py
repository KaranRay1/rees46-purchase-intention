"""Train-only preprocessing for the tabular models.

``TabularPreprocessor`` is fitted on the **training split only** and then
applied unchanged to validation and test, so no statistic (mean, standard
deviation, category vocabulary) is ever learned from held-out data.

Steps, in order
---------------
1. **Rare categories.** ``main_category`` keeps the ``top_n_categories`` most
   frequent training values; everything else becomes ``"other"``. For every
   categorical feature, values never seen in training also map to ``"other"``.
2. **log1p** on heavy-tailed non-negative features (config ``log1p_features``),
   which tames the ratio blow-up when a denominator is 0.
3. **Standardise** numeric features with training mean / std (std 0 → 1).
4. **Encode** categories: one-hot columns ``feature=value`` (LR, MLP) or
   integer codes 0..n where 0 = ``"other"`` (TabNet embeddings, trees).

Which steps run is set per model by ``preprocessing.model_profiles`` in
``config.yaml``. The fitted state is plain JSON, so it can be saved next to
a run and inspected by hand.

Typical use::

    data, pre = prepare_splits(cfg, k=5, model_name="mlp")
    X_train, y_train = data["train"].X, data["train"].y
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import polars as pl

from src.config import feature_list
from src.data.sessionize import TARGET
from src.features.build_features import load_feature_table

OTHER = "other"


@dataclass
class SplitData:
    """Model-ready arrays for one split."""

    X: np.ndarray                  # float32, shape (n_sessions, n_columns)
    y: np.ndarray                  # int8, shape (n_sessions,)
    session_ids: list[str]
    columns: list[str]


@dataclass
class TabularPreprocessor:
    """Fit-on-train, apply-everywhere feature transformer.

    Args:
        features: input feature names (e.g. one rung of the ablation ladder).
        categorical: which of ``features`` are categorical.
        log1p_features: numeric features to ``log1p`` before scaling.
        log1p: apply the log step at all.
        scale: standardise numeric features.
        categorical_encoding: ``"onehot"`` or ``"ordinal"``.
        top_n: categories kept for ``main_category``; others → ``"other"``.
        top_n_features: categorical features the top-N limit applies to.
    """

    features: list[str]
    categorical: list[str]
    log1p_features: list[str] = field(default_factory=list)
    log1p: bool = True
    scale: bool = True
    categorical_encoding: str = "onehot"
    top_n: int = 20
    top_n_features: list[str] = field(default_factory=lambda: ["main_category"])

    # fitted state
    vocab: dict[str, list[str]] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    fitted: bool = False

    def __post_init__(self) -> None:
        if self.categorical_encoding not in ("onehot", "ordinal"):
            raise ValueError("categorical_encoding must be 'onehot' or 'ordinal'")
        self.categorical = [f for f in self.features if f in set(self.categorical)]
        self.log1p_features = [f for f in self.log1p_features if f in set(self.numeric)]

    # ------------------------------------------------------------------ props
    @property
    def numeric(self) -> list[str]:
        return [f for f in self.features if f not in set(self.categorical)]

    @property
    def output_columns(self) -> list[str]:
        """Column names of the transformed matrix, in order."""
        self._check_fitted()
        cols = list(self.numeric)
        for f in self.categorical:
            if self.categorical_encoding == "onehot":
                cols += [f"{f}={v}" for v in [OTHER, *self.vocab[f]]]
            else:
                cols.append(f)
        return cols

    def categorical_dims(self) -> dict[str, int]:
        """Number of integer codes per categorical column (ordinal mode; for TabNet)."""
        self._check_fitted()
        return {f: len(self.vocab[f]) + 1 for f in self.categorical}

    def categorical_indices(self) -> list[int]:
        """Positions of categorical columns in the output matrix (ordinal mode)."""
        if self.categorical_encoding != "ordinal":
            raise ValueError("categorical_indices only applies to ordinal encoding")
        cols = self.output_columns
        return [cols.index(f) for f in self.categorical]

    # ------------------------------------------------------------------ fit
    def fit(self, train: pl.DataFrame) -> "TabularPreprocessor":
        """Learn vocabularies and scaling statistics from the training split."""
        missing = [f for f in self.features if f not in train.columns]
        if missing:
            raise KeyError(f"Training frame is missing features: {missing}")
        if train.height == 0:
            raise ValueError("Cannot fit on an empty training split.")

        self.vocab = {}
        for f in self.categorical:
            counts = (
                train.select(pl.col(f).cast(pl.String))
                .group_by(f)
                .len()
                .filter(pl.col(f).is_not_null() & (pl.col(f) != OTHER))
                .sort(["len", f], descending=[True, False])
            )
            values = counts[f].to_list()
            if f in self.top_n_features:
                values = values[: self.top_n]
            self.vocab[f] = values

        num = self._numeric_frame(train)
        self.means = {c: float(num[c].mean()) for c in self.numeric}
        self.stds = {}
        for c in self.numeric:
            s = num[c].std(ddof=0)
            self.stds[c] = float(s) if s is not None and s > 0 and math.isfinite(s) else 1.0
        self.fitted = True
        return self

    # ------------------------------------------------------------------ transform
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply the fitted transformation; returns a frame with ``output_columns``."""
        self._check_fitted()
        num = self._numeric_frame(df)
        if self.scale:
            num = num.with_columns(
                ((pl.col(c) - self.means[c]) / self.stds[c]).alias(c) for c in self.numeric
            )
        parts = [num.select(self.numeric).cast(pl.Float32)]

        for f in self.categorical:
            values = df[f].cast(pl.String)
            known = self.vocab[f]
            mapped = values.replace_strict(
                {v: i + 1 for i, v in enumerate(known)}, default=0, return_dtype=pl.Int32
            )
            if self.categorical_encoding == "ordinal":
                parts.append(pl.DataFrame({f: mapped.cast(pl.Float32)}))
            else:
                n = len(known) + 1
                onehot = np.zeros((df.height, n), dtype=np.float32)
                onehot[np.arange(df.height), mapped.to_numpy()] = 1.0
                names = [f"{f}={v}" for v in [OTHER, *known]]
                parts.append(pl.DataFrame(onehot, schema=names, orient="row"))

        out = pl.concat(parts, how="horizontal")
        return out.select(self.output_columns)

    def fit_transform(self, train: pl.DataFrame) -> pl.DataFrame:
        return self.fit(train).transform(train)

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        self._check_fitted()
        return {
            "features": self.features, "categorical": self.categorical,
            "log1p_features": self.log1p_features, "log1p": self.log1p, "scale": self.scale,
            "categorical_encoding": self.categorical_encoding, "top_n": self.top_n,
            "top_n_features": self.top_n_features, "vocab": self.vocab,
            "means": self.means, "stds": self.stds,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TabularPreprocessor":
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
    def load(cls, path: str | Path) -> "TabularPreprocessor":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ------------------------------------------------------------------ helpers
    def _numeric_frame(self, df: pl.DataFrame) -> pl.DataFrame:
        exprs = []
        for c in self.numeric:
            e = pl.col(c).cast(pl.Float64).fill_null(0.0)
            if self.log1p and c in self.log1p_features:
                e = e.clip(lower_bound=0.0).log1p()
            exprs.append(e.alias(c))
        return df.select(exprs) if exprs else pl.DataFrame()

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("TabularPreprocessor is not fitted; call fit() on the training split.")


# -----------------------------------------------------------------------------
# Config-driven helpers
# -----------------------------------------------------------------------------
def preprocessor_for(
    cfg: Mapping[str, Any], model_name: str, groups: Iterable[str] | str | None = None
) -> TabularPreprocessor:
    """Unfitted preprocessor configured for ``model_name`` and a feature-group set."""
    pre = cfg["preprocessing"]
    try:
        profile = pre["profiles"][pre["model_profiles"][model_name]]
    except KeyError as exc:
        raise KeyError(f"No preprocessing profile for model {model_name!r}") from exc
    return TabularPreprocessor(
        features=feature_list(cfg, groups),
        categorical=cfg["features"]["categorical"],
        log1p_features=pre["log1p_features"],
        log1p=profile["log1p"],
        scale=profile["scale"],
        categorical_encoding=profile["categorical"],
        top_n=cfg["features"]["top_n_categories"],
    )


def prepare_splits(
    cfg: Mapping[str, Any],
    k: int,
    model_name: str,
    groups: Iterable[str] | str | None = None,
    table: pl.DataFrame | None = None,
) -> tuple[dict[str, SplitData], TabularPreprocessor]:
    """Load features for K, fit on train, transform train/val/test.

    Args:
        table: optional pre-loaded feature table (defaults to ``features.parquet`` for K).

    Returns:
        ({"train": SplitData, "val": ..., "test": ...}, fitted preprocessor)
    """
    table = load_feature_table(cfg, k) if table is None else table
    pre = preprocessor_for(cfg, model_name, groups)
    splits = {s: table.filter(pl.col("split") == s) for s in ("train", "val", "test")}
    pre.fit(splits["train"])  # <- the only place statistics are learned

    out: dict[str, SplitData] = {}
    for name, part in splits.items():
        X = pre.transform(part)
        out[name] = SplitData(
            X=X.to_numpy().astype(np.float32, copy=False),
            y=part[TARGET].to_numpy().astype(np.int8),
            session_ids=part["user_session"].to_list(),
            columns=X.columns,
        )
    return out, pre
