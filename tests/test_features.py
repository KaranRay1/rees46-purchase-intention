"""Feature values are correct on hand-built and synthetic sessions."""

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from src.config import feature_list, load_config
from src.data import sessionize, split
from src.data.loader import clean_events, ingest
from src.data.sessionize import build_prefixes, index_events, session_stats
from src.features.build_features import (
    ID_COLUMNS,
    build_features,
    load_feature_table,
    run,
)

T0 = datetime(2019, 10, 1, 12, tzinfo=timezone.utc)  # a Tuesday
CFG = load_config()
EPS = CFG["features"]["epsilon"]
# The default config leaves out the remove features (REES46 multi-category data has no
# remove_from_cart events); this one adds them back to test that they are still computed right.
_G = CFG["features"]["groups"]
CFG_REMOVE = load_config(overrides={
    "features.groups.G2_cart_signals": [*_G["G2_cart_signals"], "remove_count"],
    "features.groups.G3_behavioural_ratios": [*_G["G3_behavioural_ratios"], "remove_to_cart_ratio"],
})

# (seconds, event_type, product, category_id, category_code, brand, price)
SESSIONS = {
    "a": [
        (0, "view", 1, 10, "electronics.smartphone", "samsung", 100.0),
        (10, "view", 2, 10, "electronics.smartphone", "apple", 300.0),
        (40, "cart", 2, 10, "electronics.smartphone", "apple", 300.0),
        (45, "view", 3, 20, "appliances.kitchen.washer", None, 200.0),
        (105, "remove_from_cart", 2, 10, "electronics.smartphone", "apple", 300.0),
        (200, "purchase", 2, 10, "electronics.smartphone", "apple", 300.0),  # after K=5
    ],
    "b": [(i * 10, "view", 10 + i, 30, None, "lg", 50.0) for i in range(6)],
    "c": [(i, "cart", 20 + i, 40, "kids.toys", "lego", 30.0) for i in range(5)]
    + [(9, "view", 99, 40, "kids.toys", "lego", 30.0)],
}


def _prefix(spec=SESSIONS, k=5):
    rows = [
        {"event_time": T0 + timedelta(seconds=s), "event_type": t, "product_id": p,
         "category_id": c, "category_code": code, "brand": b, "price": price,
         "user_id": 1, "user_session": sess}
        for sess, evts in spec.items()
        for s, t, p, c, code, b, price in evts
    ]
    df = pl.DataFrame(rows).with_columns(pl.col("event_time").cast(pl.Datetime("us", "UTC")))
    ev = clean_events(df.lazy(), CFG)[0]
    idx = index_events(ev)
    return build_prefixes(idx, session_stats(idx), k).prefix_events


@pytest.fixture(scope="module")
def feats():
    df = build_features(_prefix(), 5, CFG_REMOVE).collect()
    return {r["user_session"]: r for r in df.iter_rows(named=True)}


def test_columns_are_config_features_in_order():
    df = build_features(_prefix(), 5, CFG).collect()
    assert df.columns == ["user_session", *feature_list(CFG)]
    assert df.null_count().sum_horizontal().item() == 0


def test_remove_features_off_by_default():
    assert not {"remove_count", "remove_to_cart_ratio"} & set(feature_list(CFG))
    assert {"remove_count", "remove_to_cart_ratio"} <= set(feature_list(CFG_REMOVE))


def test_session_a_exact_values(feats):
    a = feats["a"]
    assert a["view_count"] == 3 and a["cart_count"] == 1 and a["remove_count"] == 1
    assert a["unique_products_count"] == 3
    assert a["session_duration_sec"] == 105.0
    assert a["max_gap_sec"] == 60.0
    assert a["has_cart"] == 1 and a["first_cart_position"] == 3
    assert a["last_event_type"] == "remove_from_cart"
    assert a["cart_to_view_ratio"] == pytest.approx(1 / (3 + EPS))
    assert a["remove_to_cart_ratio"] == pytest.approx(1 / (1 + EPS))
    assert a["unique_categories_count"] == 2
    assert a["unique_brands_count"] == 2           # "unknown" brand not counted
    assert a["main_category"] == "electronics"     # product 2 has 3 of the 5 events
    assert a["price_mean"] == 240.0 and a["price_max"] == 300.0 and a["price_sum"] == 1200.0
    assert a["price_std"] == pytest.approx(80.0)
    assert a["hour_of_day"] == 12 and a["day_of_week"] == 2


def test_no_cart_session(feats):
    b = feats["b"]
    assert b["cart_count"] == 0 and b["has_cart"] == 0
    assert b["first_cart_position"] == 6           # K + 1
    assert b["cart_to_view_ratio"] == 0.0 and b["remove_to_cart_ratio"] == 0.0
    assert b["main_category"] == "unknown"         # missing category_code
    assert b["price_std"] == 0.0


def test_zero_denominator_ratio_is_large_but_finite(feats):
    c = feats["c"]
    assert c["view_count"] == 0 and c["cart_count"] == 5
    assert c["cart_to_view_ratio"] == pytest.approx(5 / EPS)
    assert c["first_cart_position"] == 1
    assert c["main_category"] == "kids"


def test_k_one_has_zero_gap_and_duration():
    df = build_features(_prefix(k=1), 1, CFG).collect()
    assert (df["max_gap_sec"] == 0).all() and (df["session_duration_sec"] == 0).all()


def test_run_on_synthetic(tmp_path):
    cfg = load_config(overrides={
        "data.mode": "synthetic",
        "data.synthetic.n_sessions": 3000,
        "paths.interim_dir": str(tmp_path / "interim"),
        "paths.processed_dir": str(tmp_path / "processed"),
    })
    ingest(cfg)
    sessionize.run(cfg)
    split.run(cfg)
    written = run(cfg)
    assert sorted(written) == cfg["sessionize"]["prefix_k"]
    for k in written:
        table = load_feature_table(cfg, k)
        assert table.columns == [*ID_COLUMNS, *feature_list(cfg)]
        assert table["user_session"].is_unique().all()
        n_split = split.load_split_sessions(cfg, k).select(pl.len()).collect().item()
        assert table.height == n_split
        assert table.select(feature_list(cfg)).null_count().sum_horizontal().item() == 0
        assert (table["view_count"] + table["cart_count"] <= k).all()   # synthetic data has removals
    assert load_feature_table(cfg, 5, "test")["split"].cast(pl.String).unique().to_list() == ["test"]
