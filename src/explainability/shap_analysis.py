"""Explainability for RQ5: which behaviours drive purchase predictions, do the
model families rely on the same signals, and what do their errors look like?

Explainer per model
-------------------
=====================  =====================  ===============================
Model                  Method                 Output scale of the values
=====================  =====================  ===============================
random_forest          TreeSHAP (exact)       probability of purchase
lightgbm               TreeSHAP (exact)       log-odds
logistic_regression    LinearSHAP (exact)     log-odds
mlp                    DeepSHAP (DeepLIFT)    logit; GradientSHAP as fallback
tabnet                 TabNet attention masks share of attention (not SHAP)
=====================  =====================  ===============================

Values are on different scales across models, so cross-model comparisons use
the **share of mean |attribution|** per feature (each model's importances sum
to 1) and **rank correlation** — both scale-free.

One-hot columns (``main_category=electronics`` …) are summed back to their
source feature (``main_category``) so every model is compared on the same 20
features. (Summing is exact for additive SHAP values.)

Error analysis
--------------
Test sessions are grouped into TP / FP / FN / TN at the model's tuned
threshold; ``shap_by_error_group`` gives the mean signed attribution per
feature in each group (e.g. "false positives have high cart_count SHAP but
also high remove_to_cart_ratio") and ``pick_examples`` selects the most
confident session of each group for waterfall plots.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl

import matplotlib

matplotlib.use("Agg")  # files only, no display needed
import matplotlib.pyplot as plt  # noqa: E402

ERROR_GROUPS = ("true_positive", "false_positive", "false_negative", "true_negative")


@dataclass
class Attribution:
    """Per-session, per-feature attributions for one model."""

    model: str
    method: str
    scale: str                      # "probability", "log-odds", "logit", "attention share"
    values: np.ndarray              # (n_sessions, n_features)
    base_value: float               # expected model output (SHAP); NaN for attention
    data: np.ndarray                # feature values shown in plots, (n_sessions, n_features)
    features: list[str]

    def to_explanation(self):
        import shap

        return shap.Explanation(
            values=self.values,
            base_values=np.full(len(self.values), self.base_value),
            data=self.data,
            feature_names=self.features,
        )


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------
def sample_rows(n_rows: int, n: int, seed: int = 0) -> np.ndarray:
    """Sorted random row indices (all rows if ``n >= n_rows``)."""
    if n >= n_rows:
        return np.arange(n_rows)
    return np.sort(np.random.default_rng(seed).choice(n_rows, size=n, replace=False))


# -----------------------------------------------------------------------------
# Explainers
# -----------------------------------------------------------------------------
def _positive_class(values: Any, expected: Any) -> tuple[np.ndarray, float]:
    """Normalise SHAP outputs to (n, f) values and a scalar base for the purchase class."""
    v = values[1] if isinstance(values, list) else np.asarray(values)
    if v.ndim == 3:
        v = v[..., 1] if v.shape[-1] == 2 else v[..., 0]
    e = np.asarray(expected, dtype=np.float64).reshape(-1)
    base = float(e[1] if e.size == 2 else e[0])
    return np.asarray(v, dtype=np.float64), base


def explain_model(
    model: Any,
    X_explain: np.ndarray,
    columns: Sequence[str],
    X_background: np.ndarray | None = None,
) -> Attribution:
    """Attributions for a fitted tabular model on ``X_explain``.

    Args:
        model: fitted BaseModel (rule and GRU are not supported).
        X_explain: preprocessed test rows to explain.
        columns: preprocessor output columns (same order as ``X_explain``).
        X_background: training rows for LinearSHAP / DeepSHAP (required for LR and MLP).
    """
    import shap

    name = model.name
    X_explain = np.asarray(X_explain, dtype=np.float32)
    cols = list(columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # shap emits format-change notices for LightGBM / torch
        if name in ("random_forest", "lightgbm"):
            ex = shap.TreeExplainer(model.model)
            values, base = _positive_class(ex.shap_values(X_explain), ex.expected_value)
            scale = "probability" if name == "random_forest" else "log-odds"
            return Attribution(name, "TreeSHAP", scale, values, base, X_explain, cols)

        if name == "logistic_regression":
            if X_background is None:
                raise ValueError("logistic_regression needs X_background (training rows).")
            bg = np.asarray(X_background, dtype=np.float32)
            masker = shap.maskers.Independent(bg, max_samples=len(bg))
            ex = shap.LinearExplainer(model.model, masker)
            values, base = _positive_class(ex.shap_values(X_explain), ex.expected_value)
            return Attribution(name, "LinearSHAP", "log-odds", values, base, X_explain, cols)

        if name == "mlp":
            if X_background is None:
                raise ValueError("mlp needs X_background (training rows).")
            import torch

            net = model.module.eval()
            bg = torch.as_tensor(np.asarray(X_background, dtype=np.float32))
            xt = torch.as_tensor(X_explain)
            try:
                ex = shap.DeepExplainer(net, bg)
                method = "DeepSHAP"
            except Exception:  # noqa: BLE001 - unsupported layer: fall back
                ex = shap.GradientExplainer(net, bg)
                method = "GradientSHAP"
            values = np.asarray(ex.shap_values(xt), dtype=np.float64).reshape(len(X_explain), -1)
            with torch.no_grad():
                base = float(net(bg).mean())
            return Attribution(name, method, "logit", values, base, X_explain, cols)

        if name == "tabnet":
            masks = np.asarray(model.explain(X_explain), dtype=np.float64)
            totals = masks.sum(axis=1, keepdims=True)
            share = np.divide(masks, totals, out=np.zeros_like(masks), where=totals > 0)
            return Attribution(name, "TabNet attention", "attention share", share, float("nan"), X_explain, cols)

    raise ValueError(f"No explainer for model {name!r} (supported: RF, LightGBM, LR, MLP, TabNet).")


# -----------------------------------------------------------------------------
# One-hot aggregation and global importance
# -----------------------------------------------------------------------------
def aggregate_onehot(attr: Attribution, features: Sequence[str]) -> Attribution:
    """Sum ``feature=value`` columns back to ``feature``; output columns follow ``features``.

    The plotted data for an aggregated one-hot feature is the index of the
    active category (for colour only).
    """
    cols = attr.features
    vals = np.zeros((len(attr.values), len(features)))
    data = np.zeros((len(attr.values), len(features)))
    for j, f in enumerate(features):
        if f in cols:
            i = cols.index(f)
            vals[:, j] = attr.values[:, i]
            data[:, j] = attr.data[:, i]
            continue
        idx = [i for i, c in enumerate(cols) if c.startswith(f"{f}=")]
        if not idx:
            raise KeyError(f"Feature {f!r} not found in attribution columns.")
        vals[:, j] = attr.values[:, idx].sum(axis=1)
        data[:, j] = attr.data[:, idx].argmax(axis=1)
    return Attribution(attr.model, attr.method, attr.scale, vals, attr.base_value, data, list(features))


def global_importance(attr: Attribution) -> pl.DataFrame:
    """Mean |attribution| per feature, its share of the total, and its rank (1 = most important)."""
    mean_abs = np.abs(attr.values).mean(axis=0)
    total = mean_abs.sum()
    share = mean_abs / total if total > 0 else np.zeros_like(mean_abs)
    return (
        pl.DataFrame({"feature": attr.features, "mean_abs": mean_abs, "share": share})
        .sort("mean_abs", descending=True)
        .with_columns(rank=pl.int_range(1, pl.len() + 1))
    )


def compare_importance(attrs: Mapping[str, Attribution]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Cross-model comparison (H5).

    Returns:
        (wide table: feature × model importance share, sorted by the mean share;
         pairwise Spearman rank correlation of the importances between models)
    """
    from scipy.stats import spearmanr

    tables = {m: global_importance(a).select("feature", pl.col("share").alias(m)) for m, a in attrs.items()}
    names = list(tables)
    wide = tables[names[0]]
    for m in names[1:]:
        wide = wide.join(tables[m], on="feature", how="full", coalesce=True)
    wide = wide.fill_null(0.0).with_columns(mean_share=pl.mean_horizontal(names)).sort("mean_share", descending=True)

    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            rho = spearmanr(wide[a].to_numpy(), wide[b].to_numpy()).statistic
            rows.append({"model_a": a, "model_b": b, "spearman_rho": float(rho)})
    return wide, pl.DataFrame(rows)


# -----------------------------------------------------------------------------
# Error groups
# -----------------------------------------------------------------------------
def error_groups(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Label each session TP / FP / FN / TN (strings from ``ERROR_GROUPS``)."""
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(y_pred).astype(bool)
    out = np.empty(len(y), dtype=object)
    out[y & p] = "true_positive"
    out[~y & p] = "false_positive"
    out[y & ~p] = "false_negative"
    out[~y & ~p] = "true_negative"
    return out


def shap_by_error_group(attr: Attribution, groups: Sequence[str]) -> pl.DataFrame:
    """Mean signed attribution per feature in each error group (long format)."""
    groups = np.asarray(groups)
    if len(groups) != len(attr.values):
        raise ValueError("groups must have one entry per explained session")
    rows = []
    for g in ERROR_GROUPS:
        mask = groups == g
        if not mask.any():
            continue
        means = attr.values[mask].mean(axis=0)
        for f, v in zip(attr.features, means):
            rows.append({"group": g, "n": int(mask.sum()), "feature": f, "mean_attribution": float(v)})
    return pl.DataFrame(rows)


def pick_examples(scores: np.ndarray, groups: Sequence[str], n_per_group: int = 3) -> dict[str, list[int]]:
    """Row indices of the most confident sessions per group.

    TP / FP: highest scores (most confident "will buy"); FN / TN: lowest scores.
    """
    scores = np.asarray(scores)
    groups = np.asarray(groups)
    out: dict[str, list[int]] = {}
    for g in ERROR_GROUPS:
        idx = np.flatnonzero(groups == g)
        if len(idx) == 0:
            continue
        order = np.argsort(-scores[idx] if g in ("true_positive", "false_positive") else scores[idx], kind="mergesort")
        out[g] = idx[order[:n_per_group]].tolist()
    return out


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def _save(fig, path: str | Path, dpi: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_beeswarm(attr: Attribution, path: str | Path, max_display: int = 20, dpi: int = 150) -> Path:
    """Global beeswarm: each dot is a session, colour = feature value."""
    import shap

    plt.figure()
    shap.plots.beeswarm(attr.to_explanation(), max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle(f"{attr.model} — {attr.method} ({attr.scale})", fontsize=10)
    return _save(fig, path, dpi)


def plot_importance_bar(attr: Attribution, path: str | Path, max_display: int = 20, dpi: int = 150) -> Path:
    """Mean |attribution| bar chart."""
    imp = global_importance(attr).head(max_display).reverse()
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(imp) + 1))
    ax.barh(imp["feature"].to_list(), imp["mean_abs"].to_list(), color="#3b6ea5")
    ax.set_xlabel(f"mean |attribution| ({attr.scale})")
    ax.set_title(f"{attr.model} — {attr.method}", fontsize=10)
    return _save(fig, path, dpi)


def plot_waterfall(attr: Attribution, row: int, path: str | Path, title: str = "", max_display: int = 12, dpi: int = 150) -> Path:
    """Local explanation of one session (SHAP methods only)."""
    import shap

    if not np.isfinite(attr.base_value):
        raise ValueError(f"Waterfall needs a SHAP base value; {attr.method} has none.")
    plt.figure()
    shap.plots.waterfall(attr.to_explanation()[row], max_display=max_display, show=False)
    fig = plt.gcf()
    if title:
        fig.suptitle(title, fontsize=10)
    return _save(fig, path, dpi)


def plot_importance_comparison(wide: pl.DataFrame, path: str | Path, top: int = 12, dpi: int = 150) -> Path:
    """Grouped bars: importance share of the top features for every model (H5)."""
    models = [c for c in wide.columns if c not in ("feature", "mean_share")]
    sub = wide.head(top)
    y = np.arange(len(sub))
    h = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(sub) + 1.5))
    for i, m in enumerate(models):
        ax.barh(y + i * h, sub[m].to_list(), height=h, label=m)
    ax.set_yticks(y + h * (len(models) - 1) / 2, sub["feature"].to_list())
    ax.invert_yaxis()
    ax.set_xlabel("share of mean |attribution|")
    ax.legend(fontsize=8)
    return _save(fig, path, dpi)
