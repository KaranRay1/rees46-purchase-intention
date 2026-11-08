"""Lightweight run logger — the coursework replacement for MLflow.

Every experiment run gets its own folder and one row in a shared table::

    results/runs/<run_id>/
        run.json               who / what / when: experiment, model, K, seed,
                               feature set, git commit, status, duration
        config_snapshot.yaml   the exact config used
        params.json            model hyperparameters (model.get_params())
        metrics.json           all metrics, nested as logged
        predictions.parquet    session_id, split, y_true, score, y_pred
        model.pkl              fitted model (if logging.save_models)
        artifacts/             anything else: JSON, tables, figures, preprocessor

    results/tables/all_runs.csv   one flat row per run (params + metrics);
                                  new columns are added as needed

Usage::

    with start_run(cfg, experiment="rq1_models", model="lightgbm", k=5, seed=42) as run:
        model.fit(...)
        res = tune_and_evaluate(model, data["val"], data["test"], cfg)
        run.log_params(model.get_params())
        run.log_metrics({"val": res["val"], "test": res["test"]})
        run.log_predictions("test", ids, y, res["test_scores"], model.threshold)
        run.log_model(model)

If the block raises, the run is still recorded with ``status = "failed"`` and
the error message, then the exception is re-raised.

Nothing here talks to a server, so it can be swapped for MLflow later
(product phase) without changing the experiment scripts' structure.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl

from src.config import PROJECT_ROOT, resolve_path, save_config_snapshot

ID_COLUMNS = ["run_id", "experiment", "model", "k", "seed", "feature_set", "tag", "status",
              "started_at", "duration_sec", "git_commit", "git_dirty", "data_mode"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def git_state(cwd: Path = PROJECT_ROOT) -> tuple[str | None, bool | None]:
    """(commit hash, has uncommitted changes) — ``(None, None)`` outside a git repo."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd, capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def _jsonable(obj: Any) -> Any:
    """Convert numpy / Path / non-finite values to plain JSON types."""
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def flatten(d: Mapping[str, Any], prefix: str = "", sep: str = "_") -> dict[str, Any]:
    """Flatten nested dicts to ``a_b_c`` keys, keeping only scalars (arrays/lists are dropped)."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            out.update(flatten(v, key, sep))
        elif isinstance(v, (np.generic,)):
            out[key] = v.item()
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[key] = v
    return out


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
class Run:
    """One experiment run. Create with ``start_run``; use as a context manager."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        experiment: str,
        model: str,
        k: int | None = None,
        seed: int | None = None,
        feature_set: str | None = None,
        tag: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.started = time.perf_counter()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        parts = [stamp, experiment, model] + ([f"k{k}"] if k is not None else []) \
            + ([f"s{seed}"] if seed is not None else []) + ([tag] if tag else []) + [uuid.uuid4().hex[:6]]
        self.run_id = "_".join(parts)
        self.dir = resolve_path(cfg, "runs_dir") / self.run_id
        self.dir.mkdir(parents=True, exist_ok=False)
        commit, dirty = git_state()
        self.info: dict[str, Any] = {
            "run_id": self.run_id, "experiment": experiment, "model": model, "k": k, "seed": seed,
            "feature_set": feature_set, "tag": tag, "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "duration_sec": None, "git_commit": commit, "git_dirty": dirty,
            "data_mode": cfg["data"]["mode"],
        }
        self.params: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}
        self._pred_parts: list[pl.DataFrame] = []
        save_config_snapshot(cfg, self.dir)
        self._write_info()

    # ------------------------------------------------------------------ context manager
    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None:
            self.finish("completed")
        else:
            self.info["error"] = "".join(traceback.format_exception_only(exc_type, exc)).strip()
            self.finish("failed")
        return False  # never swallow the exception

    # ------------------------------------------------------------------ logging
    def log_params(self, params: Mapping[str, Any]) -> None:
        self.params.update(_jsonable(dict(params)))
        _atomic_write(self.dir / "params.json", json.dumps(self.params, indent=2))

    def log_metrics(self, metrics: Mapping[str, Any]) -> None:
        """Log (possibly nested) metrics; arrays such as raw scores are ignored."""
        clean = {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
        self.metrics.update(_jsonable(clean))
        _atomic_write(self.dir / "metrics.json", json.dumps(self.metrics, indent=2))

    def log_predictions(
        self,
        split: str,
        session_ids: Sequence[str],
        y_true: Sequence[int],
        scores: Sequence[float],
        threshold: float,
    ) -> Path | None:
        """Store per-session scores for one split (needed for paired tests and error analysis)."""
        if not self.cfg["logging"].get("save_predictions", True):
            return None
        scores = np.asarray(scores, dtype=np.float64)
        part = pl.DataFrame({
            "user_session": list(session_ids),
            "split": [split] * len(scores),
            "y_true": np.asarray(y_true, dtype=np.int8),
            "score": scores,
            "y_pred": (scores >= threshold).astype(np.int8),
        })
        if not (part.height == len(y_true)):
            raise ValueError("session_ids, y_true and scores must have the same length")
        self._pred_parts = [p for p in self._pred_parts if p["split"][0] != split] + [part]
        path = self.dir / "predictions.parquet"
        pl.concat(self._pred_parts).write_parquet(path)
        return path

    def log_model(self, model: Any) -> Path | None:
        if not self.cfg["logging"].get("save_models", True):
            return None
        return model.save(self.dir / "model.pkl")

    def log_artifact(self, name: str, obj: Any) -> Path:
        """Save an extra result under ``artifacts/``.

        ``obj`` may be a dict / list (→ JSON), a polars DataFrame (→ parquet),
        a matplotlib Figure (→ image in ``logging.figure_format``), an object
        with ``save(path)`` (e.g. a preprocessor; → JSON), or an existing file
        path (copied).
        """
        art = self.dir / "artifacts"
        art.mkdir(exist_ok=True)
        if isinstance(obj, (dict, list)):
            path = art / f"{name}.json"
            _atomic_write(path, json.dumps(_jsonable(obj), indent=2))
        elif isinstance(obj, pl.DataFrame):
            path = art / f"{name}.parquet"
            obj.write_parquet(path)
        elif hasattr(obj, "savefig"):
            fmt = self.cfg["logging"].get("figure_format", "png")
            path = art / f"{name}.{fmt}"
            obj.savefig(path, dpi=self.cfg["logging"].get("figure_dpi", 150), bbox_inches="tight")
        elif hasattr(obj, "save"):
            path = obj.save(art / f"{name}.json")
        elif isinstance(obj, (str, Path)) and Path(obj).is_file():
            src = Path(obj)
            path = art / (name + src.suffix if not name.endswith(src.suffix) else name)
            path.write_bytes(src.read_bytes())
        else:
            raise TypeError(f"Cannot log artifact of type {type(obj).__name__}")
        return Path(path)

    # ------------------------------------------------------------------ finish
    def finish(self, status: str = "completed") -> None:
        """Write final run info and append/refresh this run's row in ``all_runs.csv``."""
        self.info["status"] = status
        self.info["duration_sec"] = round(time.perf_counter() - self.started, 3)
        self._write_info()
        row = {
            **{c: self.info.get(c) for c in ID_COLUMNS},
            **{f"param_{k}": v for k, v in flatten(self.params).items()},
            **flatten(self.metrics),
        }
        if "error" in self.info:
            row["error"] = self.info["error"]
        append_row(self.cfg, row)

    def _write_info(self) -> None:
        _atomic_write(self.dir / "run.json", json.dumps(_jsonable(self.info), indent=2))


def start_run(
    cfg: Mapping[str, Any],
    experiment: str,
    model: str,
    k: int | None = None,
    seed: int | None = None,
    feature_set: str | None = None,
    tag: str | None = None,
) -> Run:
    """Create the run folder, save the config snapshot and return a ``Run``."""
    return Run(cfg, experiment, model, k, seed, feature_set, tag)


# -----------------------------------------------------------------------------
# The shared table
# -----------------------------------------------------------------------------
def _cell(v: Any) -> Any:
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(_jsonable(v))
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def append_row(cfg: Mapping[str, Any], row: Mapping[str, Any]) -> Path:
    """Append one run to ``all_runs.csv`` (replacing an earlier row with the same run_id).

    Columns are the union of all runs' columns; missing values stay empty.
    Every value is stored as text in the CSV and typed again by ``load_runs``.
    """
    path = resolve_path(cfg, "all_runs_csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pl.DataFrame([{k: (None if _cell(v) is None else str(_cell(v))) for k, v in row.items()}],
                       schema={k: pl.String for k in row})
    if path.exists():
        old = pl.read_csv(path, infer_schema=False)
        old = old.filter(pl.col("run_id") != row["run_id"])
        new = pl.concat([old, new], how="diagonal")
    ordered = [c for c in ID_COLUMNS if c in new.columns] + [c for c in new.columns if c not in ID_COLUMNS]
    tmp = path.with_suffix(".csv.tmp")
    new.select(ordered).write_csv(tmp)
    tmp.replace(path)
    return path


def load_runs(cfg: Mapping[str, Any], experiment: str | None = None, status: str | None = "completed") -> pl.DataFrame:
    """Read ``all_runs.csv`` with numeric columns typed; optionally filter experiment / status."""
    path = resolve_path(cfg, "all_runs_csv")
    if not path.exists():
        return pl.DataFrame()
    df = pl.read_csv(path, infer_schema=False)
    casts = []
    for c in df.columns:
        if c in ("run_id", "experiment", "model", "feature_set", "tag", "status", "started_at",
                 "git_commit", "data_mode", "error"):
            continue
        as_num = df[c].cast(pl.Float64, strict=False)
        if as_num.null_count() == df[c].null_count():   # every non-empty value is numeric
            casts.append(as_num.alias(c))
    if casts:
        df = df.with_columns(casts)
    for c in ("k", "seed"):
        if c in df.columns and df[c].dtype == pl.Float64:
            df = df.with_columns(pl.col(c).cast(pl.Int64))
    if experiment is not None:
        df = df.filter(pl.col("experiment") == experiment)
    if status is not None and "status" in df.columns:
        df = df.filter(pl.col("status") == status)
    return df


def load_predictions(run_dir: str | Path, split: str | None = None) -> pl.DataFrame:
    """Per-session predictions saved by a run."""
    df = pl.read_parquet(Path(run_dir) / "predictions.parquet")
    return df if split is None else df.filter(pl.col("split") == split)
