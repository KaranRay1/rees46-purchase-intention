"""Tests for src/evaluation/metrics.py."""

import math

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.config import load_config
from src.data import sessionize, split
from src.data.loader import ingest
from src.evaluation.metrics import (
    classification_metrics,
    evaluate_model,
    metrics_by_group,
    pr_auc,
    pr_auc_trapezoid,
    tune_and_evaluate,
    tune_threshold,
)
from src.features import build_features
from src.features.preprocessing import prepare_splits
from src.models.base import build_model

Y = np.array([0, 0, 1, 1, 0, 1, 0, 0])
S = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.7, 0.6, 0.05])


def test_bundle_matches_sklearn_and_hand_counts():
    m = classification_metrics(Y, S, threshold=0.5)
    # predicted positive: 0.8, 0.7, 0.6 -> tp=2 (0.8, 0.7), fp=1 (0.6), fn=1 (0.35)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 4)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(f1_score(Y, S >= 0.5))
    assert m["macro_f1"] == pytest.approx(f1_score(Y, S >= 0.5, average="macro"))
    assert m["pr_auc"] == pytest.approx(average_precision_score(Y, S))
    assert m["roc_auc"] == pytest.approx(roc_auc_score(Y, S))
    assert m["base_rate"] == pytest.approx(3 / 8) and m["predicted_positive_rate"] == pytest.approx(3 / 8)
    assert m["n"] == 8 and m["n_positive"] == 3


def test_trapezoid_matches_definition_and_is_close_to_ap():
    from sklearn.metrics import auc, precision_recall_curve
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    p, r, _ = precision_recall_curve(y, s)
    assert pr_auc_trapezoid(y, s) == pytest.approx(auc(r, p))
    assert pr_auc_trapezoid(y, s) == pytest.approx(pr_auc(y, s), abs=0.02)   # different, but similar


def test_random_scores_pr_auc_near_base_rate():
    rng = np.random.default_rng(1)
    y = (rng.random(20000) < 0.1).astype(int)
    assert pr_auc(y, rng.random(20000)) == pytest.approx(0.1, abs=0.01)


def test_single_class_gives_nan_not_error():
    m = classification_metrics(np.zeros(5), np.linspace(0, 1, 5))
    assert math.isnan(m["pr_auc"]) and math.isnan(m["roc_auc"])
    assert m["precision"] == 0.0 and m["f1"] == 0.0


def test_tuned_threshold_maximises_f1_by_brute_force():
    rng = np.random.default_rng(2)
    y = (rng.random(400) < 0.2).astype(int)
    s = np.clip(0.4 * y + rng.random(400) * 0.6, 0, 1)
    t = tune_threshold(y, s)
    best = max(f1_score(y, s >= c, zero_division=0) for c in np.unique(s))
    assert f1_score(y, s >= t) == pytest.approx(best)


def test_threshold_edge_cases():
    assert tune_threshold(np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4])) == 0.5   # no positives
    assert tune_threshold(np.array([0, 1, 0, 1]), np.full(4, 0.7)) == 0.5        # constant scores
    t = tune_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))   # perfect split
    assert t == pytest.approx(0.8)
    with pytest.raises(ValueError):
        tune_threshold(Y, S, objective="accuracy")


def test_fbeta_prefers_recall_with_beta_above_one():
    rng = np.random.default_rng(3)
    y = (rng.random(1000) < 0.15).astype(int)
    s = np.clip(0.3 * y + rng.random(1000) * 0.7, 0, 1)
    assert tune_threshold(y, s, "fbeta", beta=2.0) <= tune_threshold(y, s, "f1")


def test_input_validation():
    with pytest.raises(ValueError, match="lengths"):
        classification_metrics(Y, S[:-1])
    with pytest.raises(ValueError, match="binary"):
        classification_metrics(Y + 1, S)
    with pytest.raises(ValueError, match="NaN"):
        classification_metrics(Y, np.where(Y == 1, np.nan, S))


def test_metrics_by_group():
    groups = np.array(["short", "short", "long", "long", "short", "long", "long", "short"])
    table = metrics_by_group(Y, S, groups, threshold=0.5)
    assert table["group"].to_list() == ["long", "short"]
    assert table["n"].sum() == len(Y)
    long_ = table.filter(table["group"] == "long").row(0, named=True)
    assert long_["base_rate"] == pytest.approx(3 / 4)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("metrics")
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


def test_tune_and_evaluate_uses_validation_threshold_on_test(built):
    data, pre = prepare_splits(built, 5, "logistic_regression")
    lr = build_model("logistic_regression", built, seed=0, columns=pre.output_columns)
    lr.fit(data["train"].X, data["train"].y)
    res = tune_and_evaluate(lr, data["val"], data["test"], built)
    expected = tune_threshold(data["val"].y, lr.predict_proba(data["val"].X))
    assert res["threshold"] == pytest.approx(expected) == pytest.approx(lr.threshold)
    assert res["test"]["threshold"] == res["threshold"]
    assert res["test"] == evaluate_model(lr, data["test"])
    assert len(res["test_scores"]) == len(data["test"].y)
    assert res["val"]["f1"] >= classification_metrics(data["val"].y, res["val_scores"], 0.5)["f1"] - 1e-12


def test_rule_threshold_stays_fixed(built):
    data, pre = prepare_splits(built, 5, "rule_based")
    rule = build_model("rule_based", built, seed=0, columns=pre.output_columns).fit(data["train"].X, data["train"].y)
    res = tune_and_evaluate(rule, data["val"], data["test"], built)
    assert res["threshold"] == 0.5
    assert res["test"]["recall"] > 0


def test_evaluate_accepts_sequence_data():
    class _Seq:  # minimal SequenceData stand-in
        cat = np.zeros((4, 3, 2), dtype=np.int64)
        num = np.zeros((4, 3, 2), dtype=np.float32)
        y = np.array([0, 1, 0, 1])

    class _Model:
        threshold = 0.5
        def predict_proba(self, X):
            assert hasattr(X, "cat")
            return np.array([0.1, 0.9, 0.2, 0.6])

    assert evaluate_model(_Model(), _Seq())["recall"] == 1.0
