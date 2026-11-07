"""Tests for src/data/synthetic.py."""

from datetime import datetime, timezone

import polars as pl
import pytest

from src.config import load_config
from src.data.synthetic import (
    REES46_COLUMNS,
    generate_from_config,
    generate_synthetic_events,
    write_rees46_csv,
)


@pytest.fixture(scope="module")
def events() -> pl.DataFrame:
    return generate_synthetic_events(n_sessions=3000, seed=7)


def test_schema_matches_rees46(events):
    assert events.columns == REES46_COLUMNS
    assert events.schema["event_time"] == pl.Datetime("us", "UTC")
    assert events.schema["price"] == pl.Float64
    assert set(events["event_type"].unique()) == {"view", "cart", "remove_from_cart", "purchase"}


def test_same_seed_same_output():
    a = generate_synthetic_events(n_sessions=200, seed=3)
    b = generate_synthetic_events(n_sessions=200, seed=3)
    c = generate_synthetic_events(n_sessions=200, seed=4)
    assert a.equals(b) and not a.equals(c)


def test_purchase_rate_close_to_target(events):
    rate = (
        events.group_by("user_session")
        .agg(buy=(pl.col("event_type") == "purchase").any())["buy"]
        .mean()
    )
    assert 0.04 < rate < 0.08


def test_time_range_and_order(events):
    assert events["event_time"].is_sorted()
    start = datetime(2019, 10, 1, tzinfo=timezone.utc)
    assert events["event_time"].min() >= start
    assert events["event_time"].max() < datetime(2019, 11, 3, tzinfo=timezone.utc)


def test_every_purchase_follows_cart_of_same_product(events):
    carts = events.filter(pl.col("event_type") == "cart").group_by("user_session", "product_id").agg(
        first_cart=pl.col("event_time").min()
    )
    buys = events.filter(pl.col("event_type") == "purchase").join(
        carts, on=["user_session", "product_id"], how="left"
    )
    assert buys["first_cart"].null_count() == 0
    assert (buys["first_cart"] <= buys["event_time"]).all()


def test_early_cart_signal_is_learnable(events):
    """Sessions with a cart in their first 5 events should buy more often."""
    s = events.group_by("user_session", maintain_order=True).agg(
        early_cart=pl.col("event_type").head(5).eq("cart").any(),
        buy=pl.col("event_type").eq("purchase").any(),
    )
    rates = s.group_by("early_cart").agg(pl.col("buy").mean()).sort("early_cart")["buy"]
    assert rates[1] > 2 * rates[0]


def test_missing_category_and_brand_present(events):
    assert 0.1 < events["category_code"].null_count() / events.height < 0.5
    assert events["brand"].null_count() > 0


def test_clean_by_default(events):
    assert events.is_duplicated().sum() == 0
    assert (events["price"] > 0).all()
    assert events["user_session"].null_count() == 0


def test_noise_injection():
    noisy = generate_synthetic_events(n_sessions=1000, seed=1, inject_noise=True)
    assert noisy.is_duplicated().sum() > 0
    assert (noisy["price"] <= 0).any()
    assert noisy["user_session"].null_count() > 0


def test_csv_uses_rees46_text_format(tmp_path):
    df = generate_synthetic_events(n_sessions=50, seed=1)
    path = write_rees46_csv(df, tmp_path / "raw" / "synthetic.csv")
    header, first = path.read_text().splitlines()[:2]
    assert header.split(",") == REES46_COLUMNS
    assert first.split(",")[0].endswith(" UTC")
    back = pl.read_csv(path)
    assert back.height == df.height


def test_generate_from_config():
    cfg = load_config(overrides={"data.synthetic.n_sessions": 100})
    df = generate_from_config(cfg)
    assert df["user_session"].n_unique() == 100
