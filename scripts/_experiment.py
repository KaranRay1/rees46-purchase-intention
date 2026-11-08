"""Shared training / logging / lookup helpers for the experiment scripts (03–06).

Keeps every experiment on exactly the same recipe:
fit (time + memory) → tune threshold on validation → score val and test →
log params, metrics, per-session predictions, model and preprocessor.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import polars as pl

from src.config import resolve_path, set_seed
from src.evaluation.efficiency import fit_with_memory
from src.evaluation.metrics import tune_and_evaluate
from src.features.preprocessing import prepare_splits
from src.features.sequences import prepare_sequence_splits
from src.models.base import build_model
from src.utils.results_logger import load_predictions, load_runs, start_run


class InputCache:
    """Prepared inputs per (K, preprocessing profile, feature groups) and GRU sequences per K."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self._tab: dict[tuple, Any] = {}
        self._seq: dict[int, Any] = {}

    def get(self, k: int, model: str, groups: Sequence[str] | None = None):
        """Returns ``(data, preprocessor_or_encoder, model_extra_kwargs, columns)``."""
        if model == "gru":
            if groups is not None:
                raise ValueError("Feature groups do not apply to the GRU (it reads raw event sequences).")
            if k not in self._seq:
                self._seq[k] = prepare_sequence_splits(self.cfg, k)
            data, enc = self._seq[k]
            return data, enc, {"embedding_spec": enc.embedding_spec(), "num_features": enc.num_features}, None
        profile = self.cfg["preprocessing"]["model_profiles"][model]
        key = (k, profile, tuple(groups) if groups else None)
        if key not in self._tab:
            self._tab[key] = prepare_splits(self.cfg, k, model, groups=groups)
        data, pre = self._tab[key]
        extra = {"cat_dims": pre.categorical_dims()} if model == "tabnet" else {}
        return data, pre, extra, pre.output_columns


def xy(split_data: Any, model_name: str):
    """Model input and labels for one split."""
    return (split_data, split_data.y) if model_name == "gru" else (split_data.X, split_data.y)


def train_and_log(
    cfg: Mapping[str, Any],
    cache: InputCache,
    experiment: str,
    model_name: str,
    k: int,
    seed: int,
    groups: Sequence[str] | None = None,
    feature_set: str = "all",
) -> dict[str, Any]:
    """Train, evaluate and log one run. Returns the run's flat row from ``all_runs.csv``."""
    data, prep, extra, columns = cache.get(k, model_name, groups)
    set_seed(seed)
    model = build_model(model_name, cfg, seed=seed, columns=columns, **extra)
    X_tr, y_tr = xy(data["train"], model_name)
    X_va, y_va = xy(data["val"], model_name)

    with start_run(cfg, experiment, model_name, k=k, seed=seed, feature_set=feature_set) as run:
        train_mem = fit_with_memory(model, X_tr, y_tr, X_va, y_va)
        res = tune_and_evaluate(model, data["val"], data["test"], cfg)
        training = {"train_time_sec": model.train_time_sec, "train_peak_memory_mb": train_mem,
                    "n_train": int(len(y_tr)), "n_inputs": len(columns) if columns else None}
        for attr in ("best_iteration_", "best_epoch_", "best_val_score_"):
            if getattr(model, attr, None) is not None:
                training[attr.rstrip("_")] = getattr(model, attr)
        if hasattr(model, "n_parameters"):
            training["n_parameters"] = model.n_parameters()

        run.log_params({**model.get_params(), "feature_groups": list(groups) if groups else "all"})
        run.log_metrics({"threshold": res["threshold"], "val": res["val"], "test": res["test"], "training": training})
        run.log_predictions("val", data["val"].session_ids, y_va, res["val_scores"], model.threshold)
        run.log_predictions("test", data["test"].session_ids, data["test"].y, res["test_scores"], model.threshold)
        run.log_model(model)
        run.log_artifact("preprocessor" if model_name != "gru" else "sequence_encoder", prep)
        if getattr(model, "history_", None):
            run.log_artifact("training_history", model.history_)

    return load_runs(cfg, experiment).filter(pl.col("run_id") == run.run_id).row(0, named=True)


def latest_runs(cfg: Mapping[str, Any], experiment: str, keys: Sequence[str] = ("k", "model", "seed")) -> pl.DataFrame:
    """Most recent completed run per ``keys`` for the current data mode."""
    runs = load_runs(cfg, experiment)
    if runs.height == 0:
        raise RuntimeError(f"No completed {experiment} runs found; train first.")
    runs = runs.filter(pl.col("data_mode") == cfg["data"]["mode"])
    if runs.height == 0:
        raise RuntimeError(f"No completed {experiment} runs for data.mode={cfg['data']['mode']}.")
    return runs.sort("started_at").group_by(list(keys), maintain_order=True).last()


def primary_run(cfg: Mapping[str, Any], part: pl.DataFrame) -> pl.DataFrame:
    """The primary-seed row of a group of runs (first project seed; falls back to the first row)."""
    prim = part.filter(pl.col("seed") == cfg["project"]["seeds"][0])
    return prim if prim.height else part.head(1)


def primary_predictions(cfg: Mapping[str, Any], part: pl.DataFrame, split: str = "test") -> pl.DataFrame:
    """Per-session predictions of the primary-seed run in ``part``, sorted by session."""
    run_id = primary_run(cfg, part)["run_id"][0]
    return load_predictions(resolve_path(cfg, "runs_dir") / run_id, split).sort("user_session")
