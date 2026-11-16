"""Tests for src/evaluation/statistics.py."""

import math

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.config import load_config
from src.evaluation.statistics import (
    bootstrap_ci,
    bootstrap_settings,
    holm,
    mcnemar,
    paired_bootstrap,
    seed_summary,
    weighted_metric,
)


def _data(n=3000, base=0.1, signal=0.4, seed=0, round_to=None):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < base).astype(int)
    s = np.clip(signal * y + rng.random(n) * (1 - signal), 0, 1)
    if round_to is not None:
        s = np.round(s, round_to)          # create many ties
    return y, s


# ---------------------------------------------------------------- weighted metrics = sklearn
@pytest.mark.parametrize("round_to", [None, 1, 2])
def test_weighted_curve_metrics_match_sklearn(round_to):
    y, s = _data(round_to=round_to)
    w = np.random.default_rng(5).integers(0, 4, len(y)).astype(float)
    assert weighted_metric("pr_auc", y, s) == pytest.approx(average_precision_score(y, s))
    assert weighted_metric("pr_auc", y, s, w) == pytest.approx(average_precision_score(y, s, sample_weight=w))
    assert weighted_metric("roc_auc", y, s) == pytest.approx(roc_auc_score(y, s))
    assert weighted_metric("roc_auc", y, s, w) == pytest.approx(roc_auc_score(y, s, sample_weight=w))


def test_weighted_threshold_metrics_match_sklearn():
    y, s = _data()
    w = np.random.default_rng(6).integers(0, 3, len(y)).astype(float)
    pred = s >= 0.6
    assert weighted_metric("f1", y, s, w, threshold=0.6) == pytest.approx(f1_score(y, pred, sample_weight=w))
    assert weighted_metric("recall", y, s, threshold=0.6) == pytest.approx(pred[y == 1].mean())


# ---------------------------------------------------------------- bootstrap CI
def test_ci_brackets_estimate_and_is_reproducible():
    y, s = _data()
    a = bootstrap_ci(y, s, "pr_auc", n_resamples=300, seed=1)
    b = bootstrap_ci(y, s, "pr_auc", n_resamples=300, seed=1)
    assert a == b
    assert a["lower"] < a["estimate"] < a["upper"]
    assert a["estimate"] == pytest.approx(average_precision_score(y, s))
    assert a["n_invalid"] == 0


def test_ci_narrows_with_more_sessions():
    small = bootstrap_ci(*_data(n=1000, seed=2), n_resamples=300)
    large = bootstrap_ci(*_data(n=16000, seed=2), n_resamples=300)
    assert (large["upper"] - large["lower"]) < 0.6 * (small["upper"] - small["lower"])


def test_ci_with_callable_metric_slow_path():
    y, s = _data(n=600)
    res = bootstrap_ci(y, s, average_precision_score, n_resamples=50, seed=3)
    fast = bootstrap_ci(y, s, "pr_auc", n_resamples=50, seed=3)
    assert res["estimate"] == pytest.approx(fast["estimate"])
    assert res["lower"] == pytest.approx(fast["lower"]) and res["upper"] == pytest.approx(fast["upper"])


def test_rare_positives_counted_as_invalid():
    y = np.zeros(200, dtype=int)
    y[0] = 1
    res = bootstrap_ci(y, np.linspace(0, 1, 200), "pr_auc", n_resamples=200, seed=0)
    assert res["n_invalid"] > 0 and math.isfinite(res["lower"])


# ---------------------------------------------------------------- paired bootstrap
def test_identical_models_not_different():
    y, s = _data()
    r = paired_bootstrap(y, s, s, n_resamples=200)
    assert r["diff"] == 0 and r["lower"] == 0 and r["upper"] == 0 and r["p_value"] == 1.0


def test_clearly_better_model_is_significant():
    y, strong = _data(signal=0.6, seed=4)
    _, weak = _data(signal=0.6, seed=4)
    weak = 0.5 * weak + 0.5 * np.random.default_rng(9).random(len(y))      # blur the signal
    r = paired_bootstrap(y, strong, weak, n_resamples=500, seed=0)
    assert r["diff"] > 0 and r["lower"] > 0 and r["p_value"] < 0.01 and r["share_a_better"] > 0.99
    rev = paired_bootstrap(y, weak, strong, n_resamples=500, seed=0)
    assert rev["diff"] == pytest.approx(-r["diff"]) and rev["upper"] < 0


def test_paired_threshold_metric():
    y, s = _data()
    r = paired_bootstrap(y, s, s, metric="f1", threshold_a=0.5, threshold_b=0.9, n_resamples=100)
    assert r["a"] == pytest.approx(f1_score(y, s >= 0.5)) and r["b"] == pytest.approx(f1_score(y, s >= 0.9))


def test_paired_length_check():
    with pytest.raises(ValueError):
        paired_bootstrap([0, 1], [0.1, 0.9], [0.2])


# ---------------------------------------------------------------- McNemar
def test_mcnemar_exact_small_counts():
    y = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    a = np.array([1, 1, 1, 0, 0, 0, 0, 1])     # wrong on last two
    b = np.array([0, 0, 1, 0, 1, 0, 1, 0])     # wrong on 0, 1, 4
    r = mcnemar(y, a, b)
    assert (r["b"], r["c"]) == (3, 2) and r["method"] == "exact binomial"
    assert r["p_value"] == pytest.approx(1.0)


def test_mcnemar_chi2_large_counts():
    y = np.ones(200, dtype=int)
    a = np.ones(200, dtype=int)
    a[:10] = 0                                       # A wrong on 10
    b = np.ones(200, dtype=int)
    b[100:160] = 0                                   # B wrong on 60
    r = mcnemar(y, a, b)
    assert (r["b"], r["c"]) == (60, 10)
    assert r["statistic"] == pytest.approx((50 - 1) ** 2 / 70)
    assert r["p_value"] < 1e-6 and "chi" in r["method"]


def test_mcnemar_no_disagreement():
    y = np.array([0, 1, 1])
    assert mcnemar(y, y, y)["p_value"] == 1.0


# ---------------------------------------------------------------- Holm and seeds
def test_holm_known_example():
    adj = holm([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_keeps_nan_and_monotone():
    adj = holm([0.2, float("nan"), 0.01])
    assert math.isnan(adj[1]) and adj[2] == pytest.approx(0.02) and adj[0] == pytest.approx(0.2)


def test_seed_summary():
    s = seed_summary([0.30, 0.32, 0.31, float("nan")])
    assert s["n"] == 3 and s["mean"] == pytest.approx(0.31) and s["std"] == pytest.approx(0.01)
    assert seed_summary([])["n"] == 0


def test_settings_from_config():
    assert bootstrap_settings(load_config()) == {"n_resamples": 1000, "confidence": 0.95}
