"""Tests for src/models/tree_models.py (run in .venv: LightGBM needs it)."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from src.config import load_config
from src.data import sessionize, split
from src.data.loader import ingest
from src.features import build_features
from src.features.preprocessing import prepare_splits
from src.models.base import BaseModel, build_model

lgb = pytest.importorskip("lightgbm", exc_type=ImportError)

CFG = load_config()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("trees")
    cfg = load_config(overrides={
        "data.mode": "synthetic",
        "data.synthetic.n_sessions": 8000,
        "paths.interim_dir": str(tmp / "interim"),
        "paths.processed_dir": str(tmp / "processed"),
        # small and fast for tests
        "models.random_forest.n_estimators": 60,
        "models.lightgbm.n_estimators": 400,
        "models.lightgbm.early_stopping_rounds": 20,
    })
    ingest(cfg)
    sessionize.run(cfg)
    split.run(cfg)
    build_features.run(cfg)
    return cfg


@pytest.fixture(scope="module")
def raw(built):
    return prepare_splits(built, 5, "lightgbm")


def _fit(cfg, name, data, pre, seed=42):
    m = build_model(name, cfg, seed=seed, columns=pre.output_columns)
    return m.fit(data["train"].X, data["train"].y, data["val"].X, data["val"].y)


# ---------------------------------------------------------------- shared behaviour
@pytest.mark.parametrize("name", ["random_forest", "lightgbm"])
def test_learns_signal(built, raw, name):
    data, pre = raw
    m = _fit(built, name, data, pre)
    p = m.predict_proba(data["test"].X)
    assert p.shape == (len(data["test"].y),) and ((p >= 0) & (p <= 1)).all()
    assert average_precision_score(data["test"].y, p) > 1.5 * data["test"].y.mean()
    assert m.train_time_sec > 0


@pytest.mark.parametrize("name", ["random_forest", "lightgbm"])
def test_same_seed_same_model(built, raw, name):
    data, pre = raw
    a = _fit(built, name, data, pre, seed=3).predict_proba(data["test"].X)
    b = _fit(built, name, data, pre, seed=3).predict_proba(data["test"].X)
    # RF sums tree votes in thread order -> last-digit float noise only.
    assert np.allclose(a, b, rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", ["random_forest", "lightgbm"])
def test_named_importances_favour_cart_signals(built, raw, name):
    data, pre = raw
    imp = _fit(built, name, data, pre).feature_importances()
    assert set(imp) == set(pre.output_columns)
    top = sorted(imp, key=imp.get, reverse=True)[:5]
    cart = {"cart_count", "has_cart", "first_cart_position", "cart_to_view_ratio",
            "remove_to_cart_ratio", "last_event_type"}
    assert cart & set(top)                     # synthetic signal lives in cart behaviour


@pytest.mark.parametrize("name", ["random_forest", "lightgbm"])
def test_save_load_and_threads(built, raw, name, tmp_path):
    data, pre = raw
    m = _fit(built, name, data, pre)
    m.set_threshold(0.35)
    back = BaseModel.load(m.save(tmp_path / f"{name}.pkl"))
    assert back.threshold == 0.35
    assert np.allclose(back.predict_proba(data["test"].X), m.predict_proba(data["test"].X), rtol=0, atol=1e-12)
    back.set_n_jobs(1)                          # RQ4 single-thread latency
    assert np.allclose(back.predict_proba(data["test"].X), m.predict_proba(data["test"].X))


# ---------------------------------------------------------------- LightGBM specifics
def test_lightgbm_early_stops_on_val_pr_auc(built, raw):
    data, pre = raw
    m = _fit(built, "lightgbm", data, pre)
    assert 1 <= m.best_iteration_ < built["models"]["lightgbm"]["n_estimators"]
    curve = m.model.evals_result_["val"]["average_precision"]
    assert m.best_val_score_ == pytest.approx(max(curve))
    assert set(m.model.evals_result_["val"]) == {"average_precision"}   # logloss replaced


def test_lightgbm_uses_categorical_columns(built, raw):
    data, pre = raw
    m = _fit(built, "lightgbm", data, pre)
    assert m.categorical_features == [c for c in CFG["features"]["categorical"] if c in pre.output_columns]
    assert m.categorical_indices == pre.categorical_indices()
    dump = m.model.booster_.dump_model()
    assert any("==" in str(t) for t in dump["tree_info"])               # categorical splits exist


def test_lightgbm_requires_validation(raw, built):
    data, pre = raw
    m = build_model("lightgbm", built, seed=0, columns=pre.output_columns)
    with pytest.raises(ValueError, match="validation"):
        m.fit(data["train"].X, data["train"].y)


def test_lightgbm_rejects_unknown_categorical():
    from src.models.tree_models import LightGBMModel
    with pytest.raises(ValueError, match="not in columns"):
        LightGBMModel(CFG["models"]["lightgbm"], columns=["a", "b"], categorical_features=["c"])
