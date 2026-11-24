"""Segment analysis: segment labels from prefix features and per-segment metrics."""

import numpy as np
import polars as pl
import pytest

from src.evaluation.segments import assign_segments, disparity, price_edges, segment_table


@pytest.fixture
def feats():
    rng = np.random.default_rng(0)
    n = 4000
    return pl.DataFrame({
        "user_session": [f"s{i}" for i in range(n)],
        "main_category": rng.choice(["electronics", "apparel", "unknown", "kids"], n),
        "price_mean": rng.lognormal(4, 1, n),
        "has_cart": rng.integers(0, 2, n).astype(np.int8),
        "hour_of_day": rng.integers(0, 24, n).astype(np.int8),
        "day_of_week": rng.integers(1, 8, n).astype(np.int8),
    })


def test_segments_cover_every_visit(feats):
    edges = price_edges(feats["price_mean"])
    ret = pl.DataFrame({"user_session": feats["user_session"], "uh_is_returning": [float(i % 2) for i in range(feats.height)]})
    s = assign_segments(feats, edges, top_categories=2, returning=ret)
    for c in ["seg_category", "seg_price", "seg_cart", "seg_time_of_day", "seg_day_type", "seg_user_type"]:
        assert s[c].null_count() == 0
    assert set(s["seg_category"]) <= {"electronics", "apparel", "kids", "unknown", "other"}
    assert s["seg_price"].n_unique() == 4
    wk = s.filter(pl.col("seg_day_type") == "weekend").join(feats, on="user_session")
    assert (wk["day_of_week"] >= 6).all()


def test_metrics_and_global_flag_cutoff(feats):
    rng = np.random.default_rng(1)
    y = (rng.random(feats.height) < 0.1 + 0.3 * feats["has_cart"].to_numpy()).astype(int)
    score = y * 0.5 + rng.random(feats.height)
    segs = assign_segments(feats, price_edges(feats["price_mean"]))
    t = segment_table(y, {"m": score}, segs, flag_share=0.2, min_size=100)
    assert {"pr_auc_lift", "tpr", "fpr", "flag_rate"} <= set(t.columns)
    # one global cutoff: the size-weighted flag rate over any dimension equals the overall share
    cart = t.filter(pl.col("dimension") == "cart")
    assert float((cart["flag_rate"] * cart["share"]).sum()) == pytest.approx(0.2, abs=0.01)
    d = disparity(t)
    assert d.filter(pl.col("dimension") == "cart")["lift_ratio"][0] <= 1
