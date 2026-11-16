"""Monitoring: PSI drift and weekly performance (src/evaluation/monitoring.py, scripts/07_monitor.py)."""

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.config import PROJECT_ROOT
from src.evaluation.monitoring import drift_status, drift_table, psi, psi_categorical, weekly_performance
from tests.test_serving import trained  # noqa: F401  (module-scoped fixture: synthetic data + models)


def test_psi_is_zero_for_the_same_distribution_and_grows_with_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=20_000)
    assert psi(ref, rng.normal(size=20_000)) < 0.01
    small, big = psi(ref, rng.normal(0.3, 1, 20_000)), psi(ref, rng.normal(1.5, 1, 20_000))
    assert 0.01 < small < big and big > 0.25
    assert drift_status(big) == "major" and drift_status(0.15) == "moderate" and drift_status(0.01) == "stable"


def test_psi_handles_constant_and_categorical_features():
    assert psi(np.zeros(100), np.zeros(50)) == pytest.approx(0, abs=1e-9)
    assert psi(np.zeros(100), np.ones(50)) > 0.25
    assert psi_categorical(["a"] * 50 + ["b"] * 50, ["a"] * 50 + ["b"] * 50) == pytest.approx(0, abs=1e-9)
    assert psi_categorical(["a"] * 90 + ["b"] * 10, ["a"] * 10 + ["b"] * 80 + ["c"] * 10) > 0.25


def test_drift_table_has_one_row_per_month_and_feature():
    df = pl.DataFrame({"month": ["2020-01"] * 50 + ["2020-02"] * 50, "x": list(range(100)), "c": ["a", "b"] * 50})
    d = drift_table(df, ["x", "c"], df.filter(pl.col("month") == "2020-01"))
    assert d.height == 4 and set(d["status"]) <= {"stable", "moderate", "major"}
    assert d.filter((pl.col("month") == "2020-02") & (pl.col("feature") == "x"))["status"][0] == "major"


def test_weekly_performance():
    rng = np.random.default_rng(1)
    n = 3000
    t0 = datetime(2020, 3, 2, tzinfo=timezone.utc)
    y = (rng.random(n) < 0.2).astype(np.int8)
    df = pl.DataFrame({"session_start": [t0 + timedelta(hours=int(h)) for h in rng.integers(0, 24 * 21, n)],
                       "y_true": y, "score": np.clip(y * 0.4 + rng.random(n) * 0.6, 0, 1)})
    w = weekly_performance(df, threshold=0.5, n_resamples=50)
    assert w.height == 3 and w["sessions"].sum() == n
    assert (w["pr_auc_low"] <= w["pr_auc"]).all() and (w["pr_auc"] <= w["pr_auc_high"]).all()
    assert (w["lift"] > 1).all()


def test_monitor_script_on_synthetic_runs(trained):  # noqa: F811
    spec = importlib.util.spec_from_file_location("monitor", PROJECT_ROOT / "scripts" / "07_monitor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = ["--mode", "synthetic", "--n-resamples", "20", "--ks", "5"]
    for p in ("interim_dir", "processed_dir", "tables_dir", "runs_dir", "figures_dir", "all_runs_csv"):
        args += ["--set", f"paths.{p}={trained['paths'][p]}"]
    mod.main(args)
    tables = trained["paths"]["tables_dir"]
    drift = pl.read_csv(Path(tables) / "monitoring_drift_synthetic.csv")
    assert {"period", "granularity", "feature", "psi", "status", "k", "split"} <= set(drift.columns)
    assert (Path(tables) / "monitoring_summary_synthetic.json").exists()


def test_monitor_refuses_processed_data_from_another_mode(trained):  # noqa: F811
    spec = importlib.util.spec_from_file_location("monitor", PROJECT_ROOT / "scripts" / "07_monitor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = ["--mode", "dev", "--ks", "5"]
    for p in ("interim_dir", "processed_dir", "tables_dir", "runs_dir", "figures_dir", "all_runs_csv"):
        args += ["--set", f"paths.{p}={trained['paths'][p]}"]
    with pytest.raises(SystemExit, match="holds K=5: synthetic, not 'dev'"):
        mod.main(args)
