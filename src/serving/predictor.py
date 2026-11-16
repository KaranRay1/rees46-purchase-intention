"""Score a live session prefix with the models saved by step 03 (RQ1).

For each model, the **primary-seed run** (``project.seeds[0]``) of experiment
``rq1_models`` for the current ``data.mode`` and K is loaded with its fitted
preprocessor (or, for the GRU, its sequence encoder). Raw events go through the
same feature code as training (``build_features`` / ``event_steps``), so a
prediction here is computed exactly like a test-set prediction.

Usage::

    from src.config import load_config
    from src.serving.predictor import SessionPredictor

    cfg = load_config(overrides=["data.mode=dev"])
    predictor = SessionPredictor(cfg, k=5)
    scores = predictor.predict(events)      # events: polars DataFrame, >= K rows

``events`` needs the columns in ``EVENT_COLUMNS``; only the first K rows (in
time order) are used, as at prediction time nothing later is known.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl

from src.config import resolve_path
from src.data.sessionize import TARGET
from src.features.build_features import build_features
from src.features.preprocessing import TabularPreprocessor
from src.features.sequences import SequenceData, SequenceEncoder, event_steps
from src.models.base import BaseModel
from src.utils.results_logger import load_runs

EXPERIMENT = "rq1_models"
EVENT_TYPES = ("view", "cart", "purchase")
EVENT_COLUMNS: dict[str, Any] = {
    "event_time": pl.Datetime("us", "UTC"),
    "event_type": pl.String,
    "product_id": pl.Int64,
    "category_id": pl.Int64,
    "category_code": pl.String,
    "brand": pl.String,
    "price": pl.Float64,
}


@dataclass(frozen=True)
class ModelScore:
    """One model's verdict on one session."""

    model: str
    score: float          # probability-like score in [0, 1]
    threshold: float      # decision threshold tuned on the validation split
    decision: bool        # score >= threshold
    latency_ms: float     # preprocessing + inference for this session (wall clock)
    run_id: str


@dataclass
class LoadedModel:
    name: str
    run_id: str
    model: BaseModel
    transformer: TabularPreprocessor | SequenceEncoder


def available_runs(cfg: Mapping[str, Any]) -> pl.DataFrame:
    """Primary-seed ``rq1_models`` runs for the current mode: one row per (k, model), newest first."""
    try:
        runs = load_runs(cfg, EXPERIMENT)
    except FileNotFoundError:
        return pl.DataFrame()
    if runs.height == 0 or "data_mode" not in runs.columns:
        return pl.DataFrame()
    runs = runs.filter(pl.col("data_mode") == cfg["data"]["mode"])
    seed = cfg["project"]["seeds"][0]
    prim = runs.filter(pl.col("seed") == seed)
    runs = prim if prim.height else runs
    return runs.sort("started_at").group_by(["k", "model"], maintain_order=True).last()


def available_ks(cfg: Mapping[str, Any]) -> list[int]:
    runs = available_runs(cfg)
    return sorted(set(runs["k"].to_list())) if runs.height else []


def prepare_events(events: pl.DataFrame, k: int) -> pl.DataFrame:
    """Validate raw events and keep the first K (by time) as a one-session prefix with ``pos``."""
    missing = [c for c in EVENT_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"events are missing columns: {missing}")
    if events.height < k:
        raise ValueError(f"need at least K={k} events, got {events.height}")
    df = events.select(
        pl.col(c).cast(t, strict=False) if c != "event_time" else pl.col(c) for c, t in EVENT_COLUMNS.items()
    )
    if df["event_time"].dtype != EVENT_COLUMNS["event_time"]:
        df = df.with_columns(pl.col("event_time").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"))
    bad = set(df["event_type"].to_list()) - set(EVENT_TYPES)
    if bad:
        raise ValueError(f"unknown event types {sorted(bad)}; expected {EVENT_TYPES}")
    if df["price"].null_count() or (df["price"] <= 0).any():
        raise ValueError("every event needs a positive price")
    df = df.sort("event_time", maintain_order=True).head(k)
    if "purchase" in df["event_type"].to_list():
        raise ValueError("the prefix already contains a purchase; the task predicts purchases after event K")
    return df.with_columns(
        user_session=pl.lit("live"),
        pos=pl.int_range(1, k + 1, dtype=pl.Int32),
        category_code=pl.col("category_code").fill_null(pl.lit("unknown")),
        brand=pl.col("brand").fill_null(pl.lit("unknown")),
    )


class SessionPredictor:
    """All saved models for one (mode, K), ready to score single sessions.

    Args:
        cfg: loaded config; ``data.mode`` selects which trained runs to use.
        k: prefix length the models were trained for.
        models: subset to load (default: every model with a saved run).
    """

    def __init__(self, cfg: Mapping[str, Any], k: int, models: Sequence[str] | None = None) -> None:
        self.cfg, self.k = cfg, int(k)
        runs = available_runs(cfg)
        if runs.height == 0:
            raise FileNotFoundError(
                f"No trained {EXPERIMENT} runs for data.mode={cfg['data']['mode']!r}; "
                f"run `python scripts/03_run_experiments.py --mode {cfg['data']['mode']}` first."
            )
        runs = runs.filter(pl.col("k") == self.k)
        order = [m for m in cfg["models"]["enabled"] if m in set(runs["model"].to_list())]
        if models is not None:
            order = [m for m in order if m in set(models)]
        if not order:
            raise FileNotFoundError(f"No trained models for K={self.k} in mode {cfg['data']['mode']!r}.")
        runs_dir = resolve_path(cfg, "runs_dir")
        self.loaded: dict[str, LoadedModel] = {}
        for name in order:
            run_id = runs.filter(pl.col("model") == name)["run_id"][0]
            self.loaded[name] = _load(runs_dir / run_id, name, run_id)

    @property
    def models(self) -> list[str]:
        return list(self.loaded)

    # ------------------------------------------------------------------ features
    def features(self, events: pl.DataFrame) -> pl.DataFrame:
        """The 18 tabular features for this prefix (one row), exactly as used in training."""
        prefix = prepare_events(events, self.k)
        return build_features(prefix.lazy(), self.k, self.cfg).collect().drop("user_session")

    # ------------------------------------------------------------------ predict
    def predict(self, events: pl.DataFrame, models: Sequence[str] | None = None) -> list[ModelScore]:
        """Score the session with every loaded model (or only ``models``), in ``models.enabled`` order."""
        prefix = prepare_events(events, self.k)
        feats = build_features(prefix.lazy(), self.k, self.cfg).collect()
        out: list[ModelScore] = []
        for lm in self.loaded.values():
            if models is not None and lm.name not in models:
                continue
            start = time.perf_counter()
            X = self._inputs(lm, prefix, feats)
            score = float(lm.model.predict_proba(X)[0])
            ms = (time.perf_counter() - start) * 1000
            out.append(ModelScore(lm.name, score, float(lm.model.threshold), score >= lm.model.threshold, ms, lm.run_id))
        return out

    def contributions(self, events: pl.DataFrame) -> pl.DataFrame | None:
        """Per-feature TreeSHAP contributions (log-odds) from LightGBM, or ``None`` if not loaded.

        Columns: ``feature``, ``value`` (raw feature value as text), ``contribution``;
        sorted by absolute contribution. The bias term is returned as feature ``"(base)"``.
        """
        lm = self.loaded.get("lightgbm")
        if lm is None:
            return None
        prefix = prepare_events(events, self.k)
        feats = build_features(prefix.lazy(), self.k, self.cfg).collect()
        X = self._inputs(lm, prefix, feats)
        contrib = np.asarray(lm.model.model.predict(X, pred_contrib=True))[0]
        cols = lm.transformer.output_columns
        values = [_fmt(feats[c][0]) if c in feats.columns else "" for c in cols]
        df = pl.DataFrame({"feature": cols, "value": values, "contribution": contrib[: len(cols)].astype(float)})
        df = df.sort(pl.col("contribution").abs(), descending=True)
        return pl.concat([df, pl.DataFrame({"feature": ["(base)"], "value": [""], "contribution": [float(contrib[-1])]})])

    # ------------------------------------------------------------------ helpers
    def _inputs(self, lm: LoadedModel, prefix: pl.DataFrame, feats: pl.DataFrame) -> Any:
        if isinstance(lm.transformer, SequenceEncoder):
            labels = pl.DataFrame({"user_session": ["live"], "split": ["live"], TARGET: [0]}).lazy()
            steps = event_steps(prefix.lazy(), labels, self.cfg["data"]["cleaning"]["missing_category_token"])
            cat, num = lm.transformer.transform(steps, self.k)
            return SequenceData(cat, num, np.zeros(1, np.int8), ["live"], lm.transformer.cat_features,
                                lm.transformer.num_features)
        return lm.transformer.transform(feats).to_numpy().astype(np.float32)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _load(run_dir: Path, name: str, run_id: str) -> LoadedModel:
    model_path = run_dir / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} missing; train with logging.save_models=true.")
    model = BaseModel.load(model_path)
    art = run_dir / "artifacts"
    if model.input_kind == "sequence":
        transformer: Any = SequenceEncoder.load(art / "sequence_encoder.json")
    else:
        transformer = TabularPreprocessor.load(art / "preprocessor.json")
    return LoadedModel(name, run_id, model, transformer)
