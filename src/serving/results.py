"""Read the tables written by scripts 01–06 and notebooks 03–04, for one data mode.

Every file is ``results/tables/<stem>_<mode>.csv|json``; a missing file returns
``None`` so a caller (dashboard page, report generator, API) can skip a section
instead of failing when only some steps have been run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from src.config import load_config, resolve_path

MODE_ORDER = ("full", "dev", "synthetic")


def modes_with_results(tables_dir: Path) -> list[str]:
    """Data modes that have at least a data summary or RQ1 table, best first."""
    found = set()
    for p in tables_dir.glob("*_*.*"):
        m = re.match(r"(?:data_summary|rq1_model_comparison)_(\w+)\.(?:csv|json)$", p.name)
        if m:
            found.add(m.group(1))
    return [m for m in MODE_ORDER if m in found]


class ResultStore:
    """Cached access to the result tables of one mode."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.mode: str = cfg["data"]["mode"]
        self.tables_dir = resolve_path(cfg, "tables_dir")
        self.figures_dir = resolve_path(cfg, "figures_dir")
        self._cache: dict[str, Any] = {}

    @classmethod
    def for_mode(cls, mode: str, overrides: list[str] | None = None) -> "ResultStore":
        return cls(load_config(overrides=[*(overrides or []), f"data.mode={mode}"]))

    def path(self, stem: str, ext: str) -> Path:
        return self.tables_dir / f"{stem}_{self.mode}.{ext}"

    def csv(self, stem: str) -> pl.DataFrame | None:
        key = f"{stem}.csv"
        if key not in self._cache:
            p = self.path(stem, "csv")
            self._cache[key] = pl.read_csv(p) if p.exists() else None
        return self._cache[key]

    def json(self, stem: str) -> dict[str, Any] | None:
        key = f"{stem}.json"
        if key not in self._cache:
            p = self.path(stem, "json")
            self._cache[key] = json.loads(p.read_text()) if p.exists() else None
        return self._cache[key]

    def figures(self, prefix: str) -> list[Path]:
        """Saved PNG figures ``<prefix>*_<mode>*.png``, sorted by name."""
        return sorted(p for p in self.figures_dir.glob(f"{prefix}*.png") if f"_{self.mode}" in p.name)


def monthly_profile(cfg: Mapping[str, Any]) -> pl.DataFrame | None:
    """Events, sessions and buying sessions per calendar month of the cleaned events.

    Cached as ``monthly_profile_<mode>.csv`` next to the other tables; computed from
    ``data/interim/events_<mode>.parquet`` on first use. ``None`` if neither exists.
    """
    from src.data.loader import cleaned_events_path

    store = ResultStore(cfg)
    cached = store.csv("monthly_profile")
    if cached is not None:
        return cached
    src = cleaned_events_path(cfg)
    if not src.exists():
        return None
    out = (
        pl.scan_parquet(src)
        .with_columns(month=pl.col("event_time").dt.strftime("%Y-%m"))
        .group_by("month")
        .agg(
            events=pl.len(),
            sessions=pl.col("user_session").n_unique(),
            buying_sessions=pl.col("user_session").filter(pl.col("event_type") == "purchase").n_unique(),
        )
        .sort("month")
        .collect(engine="streaming")
        .with_columns(conversion=pl.col("buying_sessions") / pl.col("sessions"))
    )
    store.tables_dir.mkdir(parents=True, exist_ok=True)
    out.write_csv(store.path("monthly_profile", "csv"))
    return out


def demo_dir(cfg: Mapping[str, Any]) -> Path:
    """Folder of the lightweight demo bundle for the current mode (``results/demo/<mode>``)."""
    return resolve_path(cfg, "tables_dir").parent / "demo" / cfg["data"]["mode"]


def demo_overrides(cfg: Mapping[str, Any]) -> list[str]:
    """Config overrides that point the runs at the demo bundle, or ``[]`` to use the real runs.

    The bundle (``scripts/09_build_demo.py``) is used when ``PIE_DEMO=1``, or when it exists
    and none of the primary-seed run folders listed in ``all_runs.csv`` are on this machine
    (a fresh clone or a hosted demo: ``results/runs`` is not in git).
    """
    import os

    from src.serving.predictor import available_runs

    d = demo_dir(cfg)
    if not (d / "runs.csv").exists() or os.environ.get("PIE_DEMO") == "0":
        return []
    if os.environ.get("PIE_DEMO") != "1":
        runs = available_runs(cfg)
        runs_dir = resolve_path(cfg, "runs_dir")
        if runs.height and any((runs_dir / r).exists() for r in runs["run_id"]):
            return []
    return [f"paths.runs_dir={d / 'runs'}", f"paths.all_runs_csv={d / 'runs.csv'}"]


def test_predictions(cfg: Mapping[str, Any], k: int, model: str) -> pl.DataFrame | None:
    """Primary-seed test-split ``y_true`` and ``score`` (and ``user_session`` when known) for one model at K.

    Read from the newest logged run; without runs on this machine, from the demo bundle
    (``scripts/08_build_demo.py``). ``None`` if neither has it.
    """
    from src.serving.predictor import available_runs

    runs = available_runs(cfg)
    if runs.height:
        r = runs.filter((pl.col("k") == k) & (pl.col("model") == model))
        if r.height:
            p = resolve_path(cfg, "runs_dir") / r["run_id"][0] / "predictions.parquet"
            if p.exists():
                return (pl.scan_parquet(p).filter(pl.col("split") == "test")
                        .select("user_session", "y_true", "score").collect())
    p = demo_dir(cfg) / f"test_predictions_k{k}.parquet"
    if p.exists():
        return pl.scan_parquet(p).filter(pl.col("model") == model).select("y_true", "score").collect()
    return None
