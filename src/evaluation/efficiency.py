"""Efficiency benchmarks for RQ4: is real-time purchase prediction feasible?

Measured offline, in-process, on fixed hardware — no API server or container
(the MLOps serving study is parked for the product phase).

Measurements (config ``efficiency``)
------------------------------------
train_time_sec        wall-clock ``fit`` time (recorded by ``BaseModel.fit``)
train_peak_memory_mb  peak extra resident memory during ``fit``
latency_ms            per-call latency of ``predict_proba`` for batch sizes
                      1 (one session = the real-time case), 256 and 4096;
                      reported as p50 / p95 / p99 over ``n_repeats`` calls
                      after ``n_warmup`` untimed calls
latency_per_session   batch latency ÷ batch size (ms per session)
throughput_per_sec    sessions scored per second at each batch size
infer_peak_memory_mb  peak extra resident memory while scoring the test split
model_size_mb         size of the saved model file (pickle)
n_parameters          trainable parameters (neural models) where available

Fairness rules
--------------
* Inference is timed with **one CPU thread** for every model
  (``n_threads: 1``): native thread pools (BLAS/OpenMP via threadpoolctl),
  PyTorch and the model's own ``n_jobs`` are all limited, then restored.
* Every model is timed on the same test inputs, with its own preprocessing
  already applied (preprocessing cost is excluded and identical in kind).
* Memory is sampled from the process RSS every millisecond in a background
  thread, so it includes native (LightGBM / PyTorch) allocations that Python's
  ``tracemalloc`` cannot see. It is approximate (±a few MB) and reported as the
  increase over the starting RSS.
"""

from __future__ import annotations

import contextlib
import platform
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import psutil

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - shipped with scikit-learn
    threadpool_limits = None


# -----------------------------------------------------------------------------
# Thread limiting
# -----------------------------------------------------------------------------
@contextlib.contextmanager
def limit_threads(n_threads: int, model: Any = None) -> Iterator[None]:
    """Limit BLAS/OpenMP pools, PyTorch and the model's ``n_jobs`` to ``n_threads``; restore after."""
    torch_mod = None
    torch_before = None
    try:
        import torch as torch_mod  # noqa: F401
        torch_before = torch_mod.get_num_threads()
    except ImportError:
        torch_mod = None

    # sklearn / LightGBM wrappers keep n_jobs on the inner estimator; remember it.
    inner = getattr(model, "model", None)
    jobs_before = inner.get_params().get("n_jobs") if hasattr(inner, "get_params") else None

    pools = threadpool_limits(limits=n_threads) if threadpool_limits else contextlib.nullcontext()
    with pools:
        if torch_mod is not None:
            torch_mod.set_num_threads(n_threads)
        if model is not None and hasattr(model, "set_n_jobs"):
            model.set_n_jobs(n_threads)
        try:
            yield
        finally:
            if model is not None and hasattr(model, "set_n_jobs") and jobs_before is not None:
                model.set_n_jobs(jobs_before)
            if torch_mod is not None and torch_before is not None:
                torch_mod.set_num_threads(torch_before)  # also undoes torch models' set_n_jobs


# -----------------------------------------------------------------------------
# Input handling (tabular arrays or GRU sequences)
# -----------------------------------------------------------------------------
def _n_rows(X: Any) -> int:
    if hasattr(X, "cat"):
        return len(X.cat)
    if isinstance(X, tuple):
        return len(X[0])
    return len(X)


def _take(X: Any, start: int, size: int) -> Any:
    """Contiguous slice of ``size`` sessions starting at ``start``."""
    sl = slice(start, start + size)
    if hasattr(X, "cat"):
        return (X.cat[sl], X.num[sl])
    if isinstance(X, tuple):
        return tuple(a[sl] for a in X)
    return X[sl]


def _model_input(split: Any) -> Any:
    """Model input from SplitData (``.X``), SequenceData (``.cat``) or a raw array/tuple."""
    if hasattr(split, "X"):
        return split.X
    return split


# -----------------------------------------------------------------------------
# Memory
# -----------------------------------------------------------------------------
def measure_peak_memory(fn: Callable[[], Any], interval_sec: float = 0.001) -> tuple[Any, float]:
    """Run ``fn`` and return ``(result, peak RSS increase in MB)``."""
    proc = psutil.Process()
    start = proc.memory_info().rss
    peak = start
    stop = threading.Event()

    def sampler() -> None:
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, proc.memory_info().rss)
            stop.wait(interval_sec)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    try:
        result = fn()
    finally:
        stop.set()
        t.join()
        peak = max(peak, proc.memory_info().rss)
    return result, max(0.0, (peak - start) / 1e6)


# -----------------------------------------------------------------------------
# Latency / throughput
# -----------------------------------------------------------------------------
def measure_latency(
    model: Any,
    X: Any,
    batch_size: int,
    n_warmup: int = 20,
    n_repeats: int = 200,
    percentiles: Sequence[int] = (50, 95, 99),
    seed: int = 0,
) -> dict[str, float]:
    """Time ``model.predict_proba`` on random contiguous batches.

    Returns per-call latency percentiles (ms), per-session latency at the
    median (ms), and throughput (sessions / second, from mean call time).
    ``batch_size`` larger than the data is clipped to the data size.
    """
    X = _model_input(X)
    n = _n_rows(X)
    if n == 0:
        raise ValueError("No sessions to benchmark.")
    size = min(int(batch_size), n)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - size + 1, size=n_warmup + n_repeats)

    for s in starts[:n_warmup]:
        model.predict_proba(_take(X, int(s), size))
    times_ms = np.empty(n_repeats)
    for i, s in enumerate(starts[n_warmup:]):
        batch = _take(X, int(s), size)
        t0 = time.perf_counter_ns()
        model.predict_proba(batch)
        times_ms[i] = (time.perf_counter_ns() - t0) / 1e6

    out = {"batch_size": float(size)}
    for p in percentiles:
        out[f"p{p}_ms"] = float(np.percentile(times_ms, p))
    out["mean_ms"] = float(times_ms.mean())
    out["per_session_ms"] = float(np.percentile(times_ms, 50) / size)
    out["throughput_per_sec"] = float(size / (times_ms.mean() / 1e3)) if times_ms.mean() > 0 else float("inf")
    return out


# -----------------------------------------------------------------------------
# Size
# -----------------------------------------------------------------------------
def model_size_mb(model: Any) -> float:
    """Size of the model as saved by ``model.save`` (MB)."""
    with tempfile.TemporaryDirectory() as d:
        path = model.save(Path(d) / "model.pkl")
        return Path(path).stat().st_size / 1e6


# -----------------------------------------------------------------------------
# Training + full benchmark
# -----------------------------------------------------------------------------
def fit_with_memory(model: Any, X_train: Any, y_train: Any, X_val: Any = None, y_val: Any = None) -> float:
    """Fit ``model`` (all threads allowed) and return peak extra memory in MB."""
    _, mem = measure_peak_memory(lambda: model.fit(X_train, y_train, X_val, y_val))
    return mem


def benchmark_model(
    model: Any,
    X_test: Any,
    cfg: Mapping[str, Any] | None = None,
    batch_sizes: Sequence[int] | None = None,
    n_warmup: int | None = None,
    n_repeats: int | None = None,
    percentiles: Sequence[int] | None = None,
    n_threads: int | None = None,
) -> dict[str, Any]:
    """All inference-side RQ4 numbers for one fitted model, as a flat dict.

    Keys: ``model``, ``train_time_sec``, ``model_size_mb``, ``n_parameters``,
    ``infer_peak_memory_mb``, ``n_threads``, and per batch size ``b``:
    ``latency_b{b}_p{p}_ms``, ``latency_b{b}_per_session_ms``,
    ``throughput_b{b}_per_sec``. Arguments override ``cfg["efficiency"]``.
    """
    eff = dict(cfg["efficiency"]) if cfg is not None else {}
    batch_sizes = list(batch_sizes or eff.get("batch_sizes", [1, 256, 4096]))
    n_warmup = int(n_warmup if n_warmup is not None else eff.get("n_warmup", 20))
    n_repeats = int(n_repeats if n_repeats is not None else eff.get("n_repeats", 200))
    percentiles = list(percentiles or eff.get("latency_percentiles", [50, 95, 99]))
    n_threads = int(n_threads or eff.get("n_threads", 1))
    if not getattr(model, "fitted", True):
        raise RuntimeError("Benchmark needs a fitted model.")

    X = _model_input(X_test)
    out: dict[str, Any] = {
        "model": getattr(model, "name", type(model).__name__),
        "train_time_sec": getattr(model, "train_time_sec", None),
        "model_size_mb": model_size_mb(model),
        "n_parameters": model.n_parameters() if hasattr(model, "n_parameters") else None,
        "n_threads": n_threads,
    }
    with limit_threads(n_threads, model):
        _, out["infer_peak_memory_mb"] = measure_peak_memory(lambda: model.predict_proba(X))
        for b in batch_sizes:
            lat = measure_latency(model, X, b, n_warmup, n_repeats, percentiles)
            tag = f"b{int(b)}"
            for p in percentiles:
                out[f"latency_{tag}_p{p}_ms"] = lat[f"p{p}_ms"]
            out[f"latency_{tag}_per_session_ms"] = lat["per_session_ms"]
            out[f"throughput_{tag}_per_sec"] = lat["throughput_per_sec"]
    return out


def hardware_info() -> dict[str, Any]:
    """Machine and library versions, saved with every efficiency result."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "cpu_physical": psutil.cpu_count(logical=False),
        "cpu_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "python": platform.python_version(),
    }
    for lib in ("numpy", "polars", "sklearn", "lightgbm", "torch", "pytorch_tabnet"):
        try:
            mod = __import__(lib)
            info[f"{lib}_version"] = getattr(mod, "__version__", "unknown")
        except Exception:  # noqa: BLE001 - optional libraries
            info[f"{lib}_version"] = None
    return info
