"""Tests for src/explainability/shap_analysis.py (run in .venv)."""

import numpy as np
import pytest

shap = pytest.importorskip("shap")
pytest.importorskip("lightgbm", exc_type=ImportError)
torch = pytest.importorskip("torch")

from src.config import feature_list, load_config  # noqa: E402
from src.data import sessionize, split  # noqa: E402
from src.data.loader import ingest  # noqa: E402
from src.explainability.shap_analysis import (  # noqa: E402
    aggregate_onehot,
    compare_importance,
    error_groups,
    explain_model,
    global_importance,
    pick_examples,
    plot_beeswarm,
    plot_importance_bar,
    plot_importance_comparison,
    plot_waterfall,
    sample_rows,
    shap_by_error_group,
)
from src.features import build_features  # noqa: E402
from src.features.preprocessing import prepare_splits  # noqa: E402
from src.models.base import build_model  # noqa: E402

N_EXPLAIN = 300


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("shap")
    cfg = load_config(overrides={
        "data.mode": "synthetic",
        "data.synthetic.n_sessions": 8000,
        "paths.interim_dir": str(tmp / "interim"),
        "paths.processed_dir": str(tmp / "processed"),
        "models.random_forest.n_estimators": 60,
        "models.lightgbm.n_estimators": 200,
        "models.mlp.max_epochs": 6,
        "models.tabnet.max_epochs": 4,
        "models.tabnet.batch_size": 512,
    })
    ingest(cfg)
    sessionize.run(cfg)
    split.run(cfg)
    build_features.run(cfg)
    return cfg


@pytest.fixture(scope="module")
def explained(cfg):
    """{model: (attribution, model, data, preprocessor)} for every supported model."""
    out = {}
    for name in ("logistic_regression", "random_forest", "lightgbm", "mlp", "tabnet"):
        data, pre = prepare_splits(cfg, 5, name)
        extra = {"cat_dims": pre.categorical_dims()} if name == "tabnet" else {}
        m = build_model(name, cfg, seed=0, columns=pre.output_columns, **extra)
        m.fit(data["train"].X, data["train"].y, data["val"].X, data["val"].y)
        X = data["test"].X[:N_EXPLAIN]
        bg = data["train"].X[sample_rows(len(data["train"].X), 100, seed=1)]
        out[name] = (explain_model(m, X, pre.output_columns, bg), m, data, pre)
    return out


def _logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------- additivity: SHAP sums to the prediction
def test_random_forest_additive_in_probability(explained):
    a, m, data, _ = explained["random_forest"]
    assert a.method == "TreeSHAP" and a.scale == "probability"
    pred = m.predict_proba(data["test"].X[:N_EXPLAIN])
    assert np.allclose(a.values.sum(1) + a.base_value, pred, atol=1e-6)


@pytest.mark.parametrize("name", ["lightgbm", "logistic_regression"])
def test_log_odds_models_additive(explained, name):
    a, m, data, _ = explained[name]
    assert a.scale == "log-odds"
    pred = _logit(m.predict_proba(data["test"].X[:N_EXPLAIN]))
    assert np.allclose(a.values.sum(1) + a.base_value, pred, atol=1e-4)


def test_mlp_deepshap_additive_in_logit(explained):
    a, m, data, _ = explained["mlp"]
    assert a.method == "DeepSHAP"
    pred = _logit(m.predict_proba(data["test"].X[:N_EXPLAIN]))
    assert np.allclose(a.values.sum(1) + a.base_value, pred, atol=1e-3)


def test_tabnet_attention_shares(explained):
    a, _, _, pre = explained["tabnet"]
    assert a.values.shape == (N_EXPLAIN, len(pre.output_columns))
    assert np.allclose(a.values.sum(1), 1.0, atol=1e-6) and (a.values >= 0).all()


# ---------------------------------------------------------------- aggregation and importance
def test_onehot_aggregation_preserves_row_sums(explained, cfg):
    a, _, _, _ = explained["mlp"]
    agg = aggregate_onehot(a, feature_list(cfg))
    assert agg.features == feature_list(cfg)
    assert np.allclose(agg.values.sum(1), a.values.sum(1))
    with pytest.raises(KeyError):
        aggregate_onehot(a, ["not_a_feature"])


def test_global_importance_shares_and_ranks(explained):
    imp = global_importance(explained["lightgbm"][0])
    assert imp["share"].sum() == pytest.approx(1.0)
    assert imp["rank"].to_list() == list(range(1, imp.height + 1))
    assert imp["mean_abs"].is_sorted(descending=True)


def test_compare_importance_across_families(explained, cfg):
    feats = feature_list(cfg)
    attrs = {n: aggregate_onehot(explained[n][0], feats) for n in ("logistic_regression", "lightgbm", "mlp")}
    wide, corr = compare_importance(attrs)
    assert set(wide["feature"]) == set(feats)
    for n in attrs:
        assert wide[n].sum() == pytest.approx(1.0)
    assert corr.height == 3 and corr["spearman_rho"].is_between(-1, 1).all()
    cart = {"cart_count", "has_cart", "first_cart_position", "cart_to_view_ratio", "last_event_type"}
    assert cart & set(wide["feature"].head(5))                  # H5 holds on the synthetic signal
    same, c2 = compare_importance({"x": attrs["lightgbm"], "y": attrs["lightgbm"]})
    assert c2["spearman_rho"][0] == pytest.approx(1.0)


# ---------------------------------------------------------------- error groups
def test_error_groups_and_examples():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([1, 0, 1, 0, 1, 0])
    g = error_groups(y, p)
    assert g.tolist() == ["true_positive", "false_negative", "false_positive",
                          "true_negative", "true_positive", "true_negative"]
    scores = np.array([0.9, 0.2, 0.8, 0.1, 0.6, 0.3])
    ex = pick_examples(scores, g, n_per_group=1)
    assert ex == {"true_positive": [0], "false_positive": [2], "false_negative": [1], "true_negative": [3]}


def test_shap_by_error_group(explained):
    a, m, data, _ = explained["lightgbm"]
    y = data["test"].y[:N_EXPLAIN]
    g = error_groups(y, m.predict(data["test"].X[:N_EXPLAIN]))
    table = shap_by_error_group(a, g)
    assert set(table["group"]) <= {"true_positive", "false_positive", "false_negative", "true_negative"}
    assert table.group_by("group").len()["len"].unique().to_list() == [len(a.features)]
    tp = table.filter(table["group"] == "true_positive")["mean_attribution"].sum()
    tn = table.filter(table["group"] == "true_negative")["mean_attribution"].sum()
    assert tp > tn                                          # positives pushed up more than negatives


# ---------------------------------------------------------------- plots and errors
def test_plots_are_written(explained, cfg, tmp_path):
    a, _, _, _ = explained["lightgbm"]
    paths = [
        plot_beeswarm(a, tmp_path / "bee.png"),
        plot_importance_bar(a, tmp_path / "bar.png"),
        plot_waterfall(a, 0, tmp_path / "wf.png", title="session 0"),
    ]
    feats = feature_list(cfg)
    wide, _ = compare_importance({n: aggregate_onehot(explained[n][0], feats) for n in ("lightgbm", "mlp")})
    paths.append(plot_importance_comparison(wide, tmp_path / "compare.png"))
    for p in paths:
        assert p.exists() and p.stat().st_size > 5_000


def test_waterfall_rejects_attention(explained, tmp_path):
    with pytest.raises(ValueError, match="base value"):
        plot_waterfall(explained["tabnet"][0], 0, tmp_path / "x.png")


def test_unsupported_model_and_missing_background(cfg, explained):
    data, pre = prepare_splits(cfg, 5, "rule_based")
    rule = build_model("rule_based", cfg, seed=0, columns=pre.output_columns).fit(data["train"].X, data["train"].y)
    with pytest.raises(ValueError, match="No explainer"):
        explain_model(rule, data["test"].X[:5], pre.output_columns)
    _, lr, d, p = explained["logistic_regression"]
    with pytest.raises(ValueError, match="X_background"):
        explain_model(lr, d["test"].X[:5], p.output_columns)
