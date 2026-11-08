"""Tests for src/features/sequences.py."""

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from src.config import load_config
from src.data import sessionize, split
from src.data.loader import clean_events, ingest
from src.data.sessionize import TARGET, build_prefixes, index_events, session_stats
from src.features import build_features
from src.features.preprocessing import prepare_splits
from src.features.sequences import (
    SequenceEncoder,
    encoder_from_config,
    event_steps,
    prepare_sequence_splits,
)

CFG = load_config()
T0 = datetime(2019, 10, 1, 12, tzinfo=timezone.utc)


def _steps(k=3):
    """Two sessions, K=3: 'a' (train) and 'b' (test), with known gaps and prices."""
    spec = {
        "a": [(0, "view", "electronics.smartphone", 99.0), (10, "cart", "electronics.smartphone", 99.0),
              (70, "view", "kids.toys", 9.0), (80, "purchase", "electronics.smartphone", 99.0)],
        "b": [(0, "view", "sport.bicycle", 300.0), (5, "view", None, 20.0),
              (6, "remove_from_cart", "sport.bicycle", 300.0), (9, "view", "kids.toys", 9.0)],
    }
    rows = [
        {"event_time": T0 + timedelta(seconds=s), "event_type": t, "product_id": i,
         "category_id": 1, "category_code": c, "brand": "x", "price": p,
         "user_id": 1, "user_session": sess}
        for sess, ev in spec.items() for i, (s, t, c, p) in enumerate(ev)
    ]
    df = pl.DataFrame(rows).with_columns(pl.col("event_time").cast(pl.Datetime("us", "UTC")))
    idx = index_events(clean_events(df.lazy(), CFG)[0])
    sp = build_prefixes(idx, session_stats(idx), k)
    labels = sp.sessions.select("user_session", TARGET).with_columns(
        split=pl.when(pl.col("user_session") == "a").then(pl.lit("train")).otherwise(pl.lit("test"))
    )
    return event_steps(sp.prefix_events, labels, "unknown")


def test_event_steps_raw_values():
    s = _steps()
    a = s.filter(pl.col("user_session") == "a")
    assert a["pos"].to_list() == [1, 2, 3]
    assert a["event_type"].to_list() == ["view", "cart", "view"]
    assert a["main_category"].to_list() == ["electronics", "electronics", "kids"]
    assert a["log_gap_sec"].to_list() == pytest.approx([0.0, math.log1p(10), math.log1p(60)])
    assert a["log_price"][0] == pytest.approx(math.log1p(99.0))
    b = s.filter(pl.col("user_session") == "b")
    assert b["main_category"][1] == "unknown"               # missing category_code


def test_encoder_fit_on_train_and_unseen_to_zero():
    s = _steps()
    enc = encoder_from_config(CFG).fit(s.filter(pl.col("split") == "train"))
    assert enc.vocab["event_type"] == ["view", "cart"]      # remove_from_cart never seen in train
    cat, num = enc.transform(s.filter(pl.col("split") == "test"), 3)
    et = enc.cat_features.index("event_type")
    assert cat[0, :, et].tolist() == [1, 1, 0]              # view, view, unseen remove -> 0
    mc = enc.cat_features.index("main_category")
    assert cat[0, 0, mc] == 0                               # sport never seen in train
    assert num.shape == (1, 3, 2) and num.dtype == np.float32


def test_train_numbers_are_standardised():
    s = _steps().filter(pl.col("split") == "train")
    enc = encoder_from_config(CFG).fit(s)
    _, num = enc.transform(s, 3)
    flat = num.reshape(-1, num.shape[-1])
    assert np.allclose(flat.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(flat.std(axis=0), 1, atol=1e-5)


def test_embedding_spec_and_round_trip(tmp_path):
    s = _steps()
    enc = encoder_from_config(CFG).fit(s.filter(pl.col("split") == "train"))
    spec = enc.embedding_spec()
    assert spec["event_type"] == (3, CFG["sequences"]["categorical"]["event_type"]["embedding_dim"])
    back = SequenceEncoder.load(enc.save(tmp_path / "enc.json"))
    a, b = enc.transform(s, 3), back.transform(s, 3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_bad_step_count_raises():
    s = _steps()
    enc = encoder_from_config(CFG).fit(s)
    with pytest.raises(ValueError, match="multiple of K"):
        enc.transform(s.head(5), 3)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("seq")
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


@pytest.mark.parametrize("k", [3, 5, 10])
def test_sequences_match_tabular_sessions(built, k):
    seq, enc = prepare_sequence_splits(built, k)
    tab, _ = prepare_splits(built, k, "lightgbm")
    for s in ("train", "val", "test"):
        d = seq[s]
        assert d.cat.shape == (len(d.y), k, len(enc.cat_features))
        assert d.num.shape == (len(d.y), k, len(enc.num_features))
        assert np.isfinite(d.num).all()
        assert d.session_ids == tab[s].session_ids          # same sessions, same order
        assert np.array_equal(d.y, tab[s].y)


def test_cart_steps_match_tabular_cart_count(built):
    seq, enc = prepare_sequence_splits(built, 5)
    tab = build_features.load_feature_table(built, 5, "train").sort("user_session")
    cart_code = enc.vocab["event_type"].index("cart") + 1
    et = enc.cat_features.index("event_type")
    carts = (seq["train"].cat[:, :, et] == cart_code).sum(axis=1)
    assert np.array_equal(carts, tab["cart_count"].to_numpy())
