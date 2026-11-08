"""Tests for src/models/base.py and src/models/baselines.py."""

import warnings

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from src.config import load_config
from src.data import sessionize, split
from src.data.loader import ingest
from src.features import build_features
from src.features.preprocessing import prepare_splits
from src.models.base import BaseModel, build_model
from src.models.baselines import LogisticRegressionBaseline, RuleBasedBaseline

CFG = load_config()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("models")
    cfg = load_config(overrides={
        "data.mode": "synthetic",
        "data.synthetic.n_sessions": 6000,
        "paths.interim_dir": str(tmp / "interim"),
        "paths.processed_dir": str(tmp / "processed"),
    })
    ingest(cfg)
    sessionize.run(cfg)
    split.run(cfg)
    build_features.run(cfg)
    return cfg


# ---------------------------------------------------------------- rule
def test_rule_predicts_cart_presence():
    cols = ["view_count", "cart_count", "x"]
    X = np.array([[3, 0, 1.0], [2, 1, 0.0], [0, 3, 5.0]], dtype=np.float32)
    rule = RuleBasedBaseline(cols).fit(X, np.zeros(3))
    assert rule.predict_proba(X).tolist() == [0.0, 1.0, 1.0]
    assert rule.predict(X).tolist() == [0, 1, 1]
    rule.set_threshold(0.9)                       # ignored: rule threshold is fixed
    assert rule.threshold == 0.5


def test_rule_needs_cart_count_column():
    with pytest.raises(ValueError, match="cart_count"):
        RuleBasedBaseline(["view_count"])
    with pytest.raises(ValueError, match="Unknown rule"):
        RuleBasedBaseline(["cart_count"], rule="always_buy")


def test_rule_on_raw_profile_matches_has_cart(built):
    data, pre = prepare_splits(built, 5, "rule_based")
    rule = build_model("rule_based", built, seed=1, columns=pre.output_columns)
    rule.fit(data["train"].X, data["train"].y)
    has_cart = data["test"].X[:, pre.output_columns.index("has_cart")]
    assert np.array_equal(rule.predict(data["test"].X), has_cart.astype(np.int8))


# ---------------------------------------------------------------- logistic regression
def test_logreg_learns_signal_without_warnings(built):
    data, pre = prepare_splits(built, 5, "logistic_regression")
    lr = build_model("logistic_regression", built, seed=42, columns=pre.output_columns)
    with warnings.catch_warnings():
        warnings.simplefilter("error")            # no FutureWarning / ConvergenceWarning
        lr.fit(data["train"].X, data["train"].y)
    p = lr.predict_proba(data["val"].X)
    assert p.shape == (len(data["val"].y),) and ((p >= 0) & (p <= 1)).all()
    base_rate = data["val"].y.mean()
    assert average_precision_score(data["val"].y, p) > 1.5 * base_rate
    assert lr.converged_ and lr.train_time_sec > 0
    coefs = lr.coefficients()
    assert set(coefs) == set(pre.output_columns)
    assert coefs["cart_count"] > 0 or coefs["has_cart"] > 0   # carting raises purchase odds


def test_logreg_balanced_weights_raise_positive_rate(built):
    data, pre = prepare_splits(built, 5, "logistic_regression")
    params = CFG["models"]["logistic_regression"]
    balanced = LogisticRegressionBaseline(params, "balanced", 0).fit(data["train"].X, data["train"].y)
    plain = LogisticRegressionBaseline(params, None, 0).fit(data["train"].X, data["train"].y)
    assert balanced.predict(data["val"].X).mean() > plain.predict(data["val"].X).mean()


def test_logreg_reproducible_and_saveable(built, tmp_path):
    data, pre = prepare_splits(built, 3, "logistic_regression")
    a = build_model("logistic_regression", built, seed=7).fit(data["train"].X, data["train"].y)
    b = build_model("logistic_regression", built, seed=7).fit(data["train"].X, data["train"].y)
    assert np.allclose(a.predict_proba(data["test"].X), b.predict_proba(data["test"].X))
    a.set_threshold(0.3)
    back = BaseModel.load(a.save(tmp_path / "lr.pkl"))
    assert back.threshold == 0.3
    assert np.array_equal(back.predict(data["test"].X), a.predict(data["test"].X))


def test_penalty_translation():
    p = dict(CFG["models"]["logistic_regression"])
    assert LogisticRegressionBaseline({**p, "penalty": "l1", "solver": "saga"}).model.l1_ratio == 1.0
    assert LogisticRegressionBaseline({**p, "penalty": "none"}).model.C == float("inf")
    with pytest.raises(ValueError):
        LogisticRegressionBaseline({**p, "penalty": "l3"})


# ---------------------------------------------------------------- interface
def test_unfitted_model_raises():
    lr = build_model("logistic_regression", CFG, seed=0)
    with pytest.raises(RuntimeError, match="not fitted"):
        lr.predict_proba(np.zeros((1, 3)))


def test_threshold_controls_predictions(built):
    data, _ = prepare_splits(built, 5, "logistic_regression")
    lr = build_model("logistic_regression", built, seed=0).fit(data["train"].X, data["train"].y)
    low = lr.predict(data["val"].X, threshold=0.1).sum()
    high = lr.predict(data["val"].X, threshold=0.9).sum()
    assert low >= high
    with pytest.raises(ValueError):
        lr.set_threshold(1.5)


def test_factory_not_implemented_yet():
    with pytest.raises(NotImplementedError):
        build_model("gru", CFG, seed=0)
