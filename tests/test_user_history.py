"""G6 user-history features: only earlier, finished sessions in the window may count."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from src.features.user_history import HISTORY_FEATURES, history_features, session_table

T0 = datetime(2020, 1, 1)


def _events(rows):
    return pl.LazyFrame(
        [{"user_session": s, "user_id": u, "event_time": T0 + timedelta(hours=h), "event_type": e} for s, u, h, e in rows]
    )


EVENTS = _events([
    # user 1: bought 40 days ago (outside window), carted 5 days ago, bought 2 days ago
    ("a", 1, -40 * 24, "view"), ("a", 1, -40 * 24 + 1, "purchase"),
    ("b", 1, -5 * 24, "view"), ("b", 1, -5 * 24 + 1, "cart"),
    ("c", 1, -2 * 24, "view"), ("c", 1, -2 * 24 + 1, "purchase"),
    # target session of user 1 starts at T0; a parallel session started before T0 but ends after it
    ("t1", 1, 0, "view"), ("t1", 1, 3, "view"),
    ("p", 1, -1, "view"), ("p", 1, 2, "purchase"),
    # a later session (future) must never count
    ("f", 1, 10, "purchase"),
    # user 2: brand new
    ("t2", 2, 0, "view"),
])


@pytest.fixture(scope="module")
def feats():
    sessions = session_table(EVENTS).collect()
    targets = sessions.filter(pl.col("user_session").is_in(["t1", "t2"]))
    return {r["user_session"]: r for r in history_features(targets, sessions, window_days=30).to_dicts()}


def test_columns(feats):
    assert set(HISTORY_FEATURES) <= set(feats["t1"])


def test_only_finished_earlier_sessions_in_window(feats):
    r = feats["t1"]
    assert r["uh_prior_sessions"] == 2           # b and c; not a (too old), p (overlaps), f (future)
    assert r["uh_prior_purchase_sessions"] == 1  # c
    assert r["uh_prior_cart_sessions"] == 1      # b
    assert r["uh_prior_purchase_rate"] == pytest.approx(1 / 3)
    assert r["uh_is_returning"] == 1
    assert r["uh_days_since_last_session"] == pytest.approx(2 - 1 / 24)
    assert r["uh_days_since_last_purchase"] == pytest.approx(2 - 1 / 24)


def test_new_user(feats):
    r = feats["t2"]
    assert r["uh_prior_sessions"] == 0 and r["uh_is_returning"] == 0 and r["uh_prior_purchase_rate"] == 0
    assert r["uh_days_since_last_session"] == 31 and r["uh_days_since_last_purchase"] == 31


def test_old_purchase_is_capped():
    ev = _events([("a", 1, -40 * 24, "purchase"), ("t", 1, 0, "view")])
    s = session_table(ev).collect()
    r = history_features(s.filter(pl.col("user_session") == "t"), s).row(0, named=True)
    assert r["uh_prior_sessions"] == 0 and r["uh_days_since_last_purchase"] == 31
