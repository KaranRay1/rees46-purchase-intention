"""Tests for src/features/preprocessing.py."""

import math

import numpy as np
import polars as pl
import pytest

from src.config import feature_list, load_config
from src.data import sessionize, split
from src.data.loader import ingest
from src.features import build_features
from src.features.preprocessing import OTHER, TabularPreprocessor, prepare_splits, preprocessor_for

CFG = load_config()


def _frame(n=6, cats=None, ratio=None, split_name="train"):
    cats = cats or ["electronics", "electronics", "computers", "kids", "electronics", "computers"][:n]
    return pl.DataFrame({
        "user_session": [f"s{i}" for i in range(n)],
        "split": [split_name] * n,
        "target_purchase": [i % 2 for i in range(n)],
        "x": [float(i) for i in range(n)],
        "ratio": ratio or [0.0, 1.0, 3.0, 1e6, 0.5, 0.0][:n],
        "const": [7.0] * n,
        "main_category": cats,
        "last_event_type": ["view", "cart", "view", "view", "remove_from_cart", "view"][:n],
    })


def _pre(**kw):
    base = dict(
        features=["x", "ratio", "const", "main_category", "last_event_type"],
        categorical=["main_category", "last_event_type"],
        log1p_features=["ratio"],
        top_n=2,
    )
    return TabularPreprocessor(**{**base, **kw})


def test_top_n_and_unseen_map_to_other():
    pre = _pre(categorical_encoding="ordinal").fit(_frame())
    assert pre.vocab["main_category"] == ["electronics", "computers"]   # kids dropped (top 2)
    test = _frame(n=3, cats=["kids", "electronics", "toys"], split_name="test")
    codes = pre.transform(test)["main_category"].to_list()
    assert codes == [0.0, 1.0, 0.0]                                       # kids/toys -> other (0)


def test_onehot_columns_and_rows_sum_to_one():
    pre = _pre(categorical_encoding="onehot").fit(_frame())
    out = pre.transform(_frame())
    assert pre.output_columns[:3] == ["x", "ratio", "const"]
    assert "main_category=other" in out.columns and "last_event_type=view" in out.columns
    mc = out.select(pl.col("^main_category=.*$")).to_numpy()
    assert (mc.sum(axis=1) == 1).all()


def test_log_then_standardise_on_train():
    train = _frame()
    out = _pre().fit_transform(train)
    for c in ("x", "ratio"):
        assert out[c].mean() == pytest.approx(0, abs=1e-6)
        assert out[c].std(ddof=0) == pytest.approx(1, abs=1e-5)
    assert (out["const"] == 0).all()                                      # std 0 -> divide by 1
    assert out["ratio"].max() < 3                                         # 1e6 tamed by log1p


def test_raw_profile_leaves_numbers_untouched():
    out = _pre(log1p=False, scale=False, categorical_encoding="ordinal").fit_transform(_frame())
    assert out["ratio"].to_list() == pytest.approx(_frame()["ratio"].to_list())


def test_statistics_come_only_from_fit_data():
    train = _frame()
    pre = _pre().fit(train)
    before = (dict(pre.means), dict(pre.stds), dict(pre.vocab))
    pre.transform(_frame(cats=["toys"] * 6, ratio=[1e9] * 6, split_name="test"))
    assert (pre.means, pre.stds, pre.vocab) == before


def test_save_load_round_trip(tmp_path):
    pre = _pre(categorical_encoding="ordinal").fit(_frame())
    back = TabularPreprocessor.load(pre.save(tmp_path / "pre.json"))
    assert back.transform(_frame()).equals(pre.transform(_frame()))
    assert back.categorical_dims() == {"main_category": 3, "last_event_type": 4}
    assert back.categorical_indices() == [3, 4]


def test_unfitted_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        _pre().transform(_frame())


def test_model_profiles_from_config():
    assert preprocessor_for(CFG, "mlp").categorical_encoding == "onehot"
    assert preprocessor_for(CFG, "tabnet").categorical_encoding == "ordinal"
    lgbm = preprocessor_for(CFG, "lightgbm")
    assert not lgbm.scale and not lgbm.log1p
    sub = preprocessor_for(CFG, "lightgbm", ["G1_activity", "G2_cart_signals"])
    assert sub.features == feature_list(CFG, ["G1_activity", "G2_cart_signals"])
    assert sub.categorical == ["last_event_type"]
    with pytest.raises(KeyError):
        preprocessor_for(CFG, "gru")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pre")
    cfg = load_config(overrides={
        "data.mode": "synthetic",
        "data.synthetic.n_sessions": 3000,
        "paths.interim_dir": str(tmp / "interim"),
        "paths.processed_dir": str(tmp / "processed"),
    })
    ingest(cfg)
    sessionize.run(cfg)
    split.run(cfg)
    build_features.run(cfg)
    return cfg


@pytest.mark.parametrize("model", ["logistic_regression", "mlp", "tabnet", "lightgbm", "random_forest"])
def test_prepare_splits_on_synthetic(built, model):
    data, pre = prepare_splits(built, 5, model)
    n_cols = len(pre.output_columns)
    for s in ("train", "val", "test"):
        d = data[s]
        assert d.X.dtype == np.float32 and d.X.shape == (len(d.y), n_cols)
        assert np.isfinite(d.X).all()
        assert set(np.unique(d.y)) <= {0, 1}
    assert data["train"].X.shape[0] > data["val"].X.shape[0]


def test_prepare_splits_fits_on_train_only(built):
    table = build_features.load_feature_table(built, 5)
    _, pre = prepare_splits(built, 5, "mlp", table=table)
    train_only = preprocessor_for(built, "mlp").fit(table.filter(pl.col("split") == "train"))
    assert pre.means == train_only.means and pre.stds == train_only.stds
    assert pre.vocab == train_only.vocab
    all_rows = preprocessor_for(built, "mlp").fit(table)
    assert any(not math.isclose(pre.means[c], all_rows.means[c]) for c in pre.numeric)
