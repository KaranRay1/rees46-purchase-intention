"""Tests for src/evaluation/efficiency.py."""

import time

import numpy as np
import pytest

from src.config import load_config
from src.data import sessionize, split
from src.data.loader import ingest
from src.evaluation.efficiency import (
    benchmark_model,
    fit_with_memory,
    hardware_info,
    limit_threads,
    measure_latency,
    measure_peak_memory,
    model_size_mb,
)
from src.features import build_features
from src.features.preprocessing import prepare_splits
from src.models.base import build_model


class _SlowModel:
    """Fake model: 2 ms per call, plus 0.01 ms per row."""

    name = "slow"
    fitted = True
    train_time_sec = 1.5

    def predict_proba(self, X):
        n = len(X[0]) if isinstance(X, tuple) else len(X)
        time.sleep(0.002 + 0.00001 * n)
        return np.zeros(n)

    def save(self, path):
        path.write_bytes(b"x" * 2_000_000)
        return path


def test_latency_percentiles_and_throughput():
    X = np.zeros((1000, 4), dtype=np.float32)
    r = measure_latency(_SlowModel(), X, batch_size=100, n_warmup=2, n_repeats=20)
    assert r["batch_size"] == 100
    assert 2.0 <= r["p50_ms"] <= r["p95_ms"] <= r["p99_ms"]
    assert r["per_session_ms"] == pytest.approx(r["p50_ms"] / 100)
    assert 100 / (r["mean_ms"] / 1e3) == pytest.approx(r["throughput_per_sec"])


def test_batch_larger_than_data_is_clipped():
    r = measure_latency(_SlowModel(), np.zeros((10, 2)), batch_size=4096, n_warmup=1, n_repeats=3)
    assert r["batch_size"] == 10


def test_latency_accepts_sequence_tuple():
    X = (np.zeros((50, 5, 2), dtype=np.int64), np.zeros((50, 5, 2), dtype=np.float32))
    r = measure_latency(_SlowModel(), X, batch_size=1, n_warmup=1, n_repeats=5)
    assert r["batch_size"] == 1 and r["p50_ms"] > 0


def test_peak_memory_sees_large_allocation():
    def allocate():
        a = np.ones(25_000_000)          # 200 MB
        time.sleep(0.05)
        return float(a[0])

    result, mb = measure_peak_memory(allocate)
    assert result == 1.0 and mb > 150


def test_model_size_from_save():
    assert model_size_mb(_SlowModel()) == pytest.approx(2.0)


def test_limit_threads_restores_torch():
    torch = pytest.importorskip("torch")
    before = torch.get_num_threads()
    with limit_threads(1):
        assert torch.get_num_threads() == 1
    assert torch.get_num_threads() == before


def test_hardware_info_keys():
    info = hardware_info()
    assert {"platform", "cpu_logical", "ram_gb", "python", "numpy_version"} <= set(info)
    assert info["cpu_logical"] >= 1


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("eff")
    cfg = load_config(overrides={
        "data.mode": "synthetic",
        "data.synthetic.n_sessions": 4000,
        "paths.interim_dir": str(tmp / "interim"),
        "paths.processed_dir": str(tmp / "processed"),
        "efficiency.n_warmup": 2,
        "efficiency.n_repeats": 10,
        "efficiency.batch_sizes": [1, 64],
    })
    ingest(cfg)
    sessionize.run(cfg)
    split.run(cfg)
    build_features.run(cfg)
    return cfg


def test_benchmark_logistic_regression(built):
    data, pre = prepare_splits(built, 5, "logistic_regression")
    lr = build_model("logistic_regression", built, seed=0, columns=pre.output_columns)
    train_mem = fit_with_memory(lr, data["train"].X, data["train"].y)
    res = benchmark_model(lr, data["test"], built)
    assert res["model"] == "logistic_regression" and res["n_threads"] == 1
    assert res["train_time_sec"] > 0 and res["model_size_mb"] > 0 and train_mem >= 0
    for b in (1, 64):
        assert res[f"latency_b{b}_p50_ms"] <= res[f"latency_b{b}_p99_ms"]
        assert res[f"throughput_b{b}_per_sec"] > 0
    assert res["throughput_b64_per_sec"] > res["throughput_b1_per_sec"]   # batching pays off
    assert res["infer_peak_memory_mb"] >= 0


def test_benchmark_restores_model_threads(built):
    pytest.importorskip("lightgbm", exc_type=ImportError)
    data, pre = prepare_splits(built, 5, "lightgbm")
    cfg = load_config(overrides={"models.lightgbm.n_estimators": 50})
    m = build_model("lightgbm", cfg, seed=0, columns=pre.output_columns)
    m.fit(data["train"].X, data["train"].y, data["val"].X, data["val"].y)
    before = m.model.get_params()["n_jobs"]
    benchmark_model(m, data["test"], built)
    assert m.model.get_params()["n_jobs"] == before


def test_benchmark_rejects_unfitted(built):
    lr = build_model("logistic_regression", built, seed=0)
    with pytest.raises(RuntimeError, match="fitted"):
        benchmark_model(lr, np.zeros((5, 3)), built)
