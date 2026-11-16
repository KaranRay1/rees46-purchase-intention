"""Score calibration (src/serving/calibration.py, scripts/10_calibrate.py)."""

import numpy as np
import polars as pl
import pytest

from src.serving.calibration import Calibrator, brier, ece, fit_platt, reliability
from tests.test_product_scripts import _run
from tests.test_serving import trained  # noqa: F401  (module-scoped fixture: synthetic data + models)


def _biased(n=20000, seed=0):
    """Scores from a known Platt map: true p = sigmoid(0.8 * logit(s) - 1.5)."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.01, 0.99, n)
    p = 1 / (1 + np.exp(-(0.8 * np.log(s / (1 - s)) - 1.5)))
    return (rng.random(n) < p).astype(int), s


def test_fit_recovers_known_map():
    y, s = _biased()
    cal = fit_platt(y, s)
    assert cal.a == pytest.approx(0.8, abs=0.08) and cal.b == pytest.approx(-1.5, abs=0.1)


def test_calibration_is_monotone_and_improves_error():
    y, s = _biased()
    cal = fit_platt(y, s)
    p = cal(np.sort(s))
    assert np.all(np.diff(p) >= 0)                      # ranking (and so PR-AUC) unchanged
    assert ece(y, cal(s)) < ece(y, s) and brier(y, cal(s)) < brier(y, s)
    assert abs(cal(s).mean() - y.mean()) < 0.01


def test_scalar_save_load_and_reliability(tmp_path):
    cal = Calibrator(1.2, -0.3)
    assert isinstance(cal(0.5), float)
    cal.save(tmp_path, fitted_on="val")
    assert Calibrator.load(tmp_path) == cal and Calibrator.load(tmp_path / "missing") is None
    rel = reliability(np.array([0, 1, 1, 0]), np.array([0.05, 0.95, 0.9, 0.15]), bins=10)
    assert sum(r["n"] for r in rel) == 4 and all(0 <= r["observed"] <= 1 for r in rel)


def test_script_calibrates_runs_and_predictor_reports_probability(trained):  # noqa: F811
    from src.serving.predictor import SessionPredictor
    from src.serving.results import ResultStore
    from tests.test_api import _events

    before = SessionPredictor(trained, 5, models=["lightgbm"])
    assert before.predict(pl.DataFrame(_api_events(_events(5))))[0].probability is None
    _run("10_calibrate.py", trained)
    table = ResultStore(trained).csv("calibration")
    assert table is not None and {"k", "model", "a", "b", "ece", "ece_raw", "mean_prob"} <= set(table.columns)
    assert "rule_based" in table["model"].to_list()
    s = SessionPredictor(trained, 5, models=["lightgbm"]).predict(pl.DataFrame(_api_events(_events(5))))[0]
    assert s.probability is not None and 0 <= s.probability <= 1
    assert (s.probability >= s.threshold_probability) == s.decision


def _api_events(events):
    from datetime import datetime

    return [{**e, "event_time": datetime.fromisoformat(e["event_time"].replace("Z", "+00:00")),
             "category_id": 1} for e in events]


def test_two_valued_scores_map_to_observed_rates():
    rng = np.random.default_rng(1)
    s = rng.integers(0, 2, 20000).astype(float)          # a rule: flagged (1) or not (0)
    y = np.where(s == 1, rng.random(20000) < 0.4, rng.random(20000) < 0.05).astype(int)
    cal = fit_platt(y, s)
    assert cal(1.0) == pytest.approx(y[s == 1].mean(), abs=0.01)
    assert cal(0.0) == pytest.approx(y[s == 0].mean(), abs=0.01)
