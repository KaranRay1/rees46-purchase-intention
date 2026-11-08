"""Leakage tests: no event after the K-th event, and no in-prefix purchase, reaches the features.

Covers src/data/sessionize.py. Feature-level leakage checks are added here
when src/features/ is built.
"""

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from src.config import load_config
from src.data.loader import clean_events, ingest
from src.data.sessionize import (
    ANALYSIS_ONLY_COLUMNS,
    TARGET,
    build_prefixes,
    index_events,
    load_prefixes,
    run,
    session_stats,
    sessionize,
    summarise,
)
from src.data.synthetic import generate_synthetic_events

T0 = datetime(2019, 10, 1, 12, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _events(spec: dict[str, list[tuple[int, str]]], cfg) -> pl.LazyFrame:
    """Build cleaned events from {session: [(seconds_after_T0, event_type), ...]}."""
    rows = []
    for i, (sess, evts) in enumerate(spec.items()):
        for j, (sec, etype) in enumerate(evts):
            rows.append({
                "event_time": T0 + timedelta(seconds=sec), "event_type": etype,
                "product_id": 100 + j, "category_id": 1, "category_code": "a.b",
                "brand": "x", "price": 10.0, "user_id": i, "user_session": sess,
            })
    df = pl.DataFrame(rows).with_columns(pl.col("event_time").cast(pl.Datetime("us", "UTC")))
    return clean_events(df.sample(fraction=1.0, shuffle=True, seed=0).lazy(), cfg)[0]  # shuffled on purpose


SPEC = {
    "short":   [(0, "view"), (10, "view"), (20, "view")],                                    # == K -> drop
    "early":   [(0, "view"), (10, "cart"), (20, "purchase"), (30, "view"), (40, "view")],    # buy at 3 -> drop
    "late":    [(0, "view"), (10, "view"), (20, "cart"), (30, "view"), (40, "purchase")],    # buy at 5 -> 1
    "never":   [(s, "view") for s in range(0, 60, 10)],                                       # -> 0
    # cart and purchase in the same second as event 3: funnel order puts cart first,
    # so the purchase is event 4 (after K=3) -> eligible, target 1.
    "tie":     [(0, "view"), (10, "view"), (20, "purchase"), (20, "cart")],
}


def _build(cfg, k=3, spec=SPEC):
    ev = _events(spec, cfg)
    idx = index_events(ev)
    return build_prefixes(idx, session_stats(idx), k)


def test_eligibility_and_targets(cfg):
    sp = _build(cfg)
    s = {r["user_session"]: r[TARGET] for r in sp.sessions.collect().iter_rows(named=True)}
    assert s == {"late": 1, "never": 0, "tie": 1}


def test_prefix_has_exactly_k_events_and_no_purchase(cfg):
    for k in (1, 2, 3, 4):
        pe = _build(cfg, k).prefix_events.collect()
        per_session = pe.group_by("user_session").agg(n=pl.len(), maxpos=pl.col("pos").max())
        assert (per_session["n"] == k).all() and (per_session["maxpos"] == k).all()
        assert (pe["event_type"] != "purchase").all()


def test_prefix_is_chronological_first_k(cfg):
    pe = _build(cfg).prefix_events.filter(pl.col("user_session") == "late").sort("pos").collect()
    assert pe["event_type"].to_list() == ["view", "view", "cart"]
    assert pe["event_time"].is_sorted()


def test_no_prefix_event_after_prediction_time(cfg):
    sp = _build(cfg)
    joined = sp.prefix_events.join(sp.sessions, on="user_session").collect()
    assert (joined["event_time"] <= joined["prefix_end_time"]).all()


def test_prediction_time_before_first_purchase():
    cfg = load_config()
    events = clean_events(generate_synthetic_events(n_sessions=3000, seed=11).lazy(), cfg)[0]
    idx = index_events(events)
    stats = session_stats(idx)
    purchases = idx.filter(pl.col("event_type") == "purchase").group_by("user_session").agg(
        first_buy=pl.col("event_time").min()
    )
    for k in cfg["sessionize"]["prefix_k"]:
        s = build_prefixes(idx, stats, k).sessions.join(purchases, on="user_session").collect()
        assert (s["prefix_end_time"] <= s["first_buy"]).all()
        assert (s[TARGET] == 1).all()  # every eligible session with a purchase is positive


def test_k_one_keeps_early_session(cfg):
    s = _build(cfg, k=1).sessions.collect()
    assert dict(zip(s["user_session"], s[TARGET]))["early"] == 1


def test_summary_counts(cfg):
    idx = index_events(_events(SPEC, cfg))
    r = summarise(session_stats(idx), 3)
    assert r == {
        "sessions_total": 5, "too_short": 1, "purchase_within_prefix": 1,
        "eligible": 3, "positives": 2, "k": 3, "positive_rate": pytest.approx(2 / 3),
    }


def test_analysis_only_columns_stay_out_of_prefix_events(cfg):
    pe_cols = set(_build(cfg).prefix_events.collect_schema().names())
    assert not pe_cols & set(ANALYSIS_ONLY_COLUMNS)


def test_leaky_setting_rejected(cfg):
    bad = load_config(overrides={"sessionize.drop_purchase_within_prefix": False})
    with pytest.raises(ValueError, match="leakage"):
        sessionize(_events(SPEC, cfg), bad)


def test_requires_enum_event_type(cfg):
    raw = generate_synthetic_events(n_sessions=10, seed=1).lazy()  # event_type is plain string
    with pytest.raises(TypeError, match="Enum"):
        index_events(raw)


def _pipeline_rank(cfg) -> pl.Expr:
    """Event position using the same ordering as index_events (time, funnel order, product)."""
    funnel = pl.col("event_type").cast(pl.Enum(cfg["data"]["event_types"]))
    return pl.int_range(1, pl.len() + 1).over(
        "user_session", order_by=[pl.col("event_time"), funnel, pl.col("product_id")]
    )


def test_features_ignore_everything_after_event_k(cfg):
    """Changing or adding events after position K must not change any feature."""
    from src.features.build_features import build_features

    base = generate_synthetic_events(n_sessions=1500, seed=21)
    k = cfg["sessionize"]["default_k"]

    def feats(events):
        idx = index_events(clean_events(events.lazy(), cfg)[0])
        sp = build_prefixes(idx, session_stats(idx), k)
        return build_features(sp.prefix_events, k, cfg).sort("user_session").collect()

    # Tamper with the future: inflate prices and turn views into carts for every
    # event that is not among the first K of its session.
    tampered = base.with_columns(_pipeline_rank(cfg).alias("_r")).with_columns(
        price=pl.when(pl.col("_r") > k).then(pl.col("price") * 10).otherwise(pl.col("price")),
        event_type=pl.when((pl.col("_r") > k) & (pl.col("event_type") == "view"))
        .then(pl.lit("cart"))
        .otherwise(pl.col("event_type")),
    ).drop("_r")

    a, b = feats(base), feats(tampered)
    # Sessions whose eligibility changes (e.g. a new early event) are excluded; compare the rest.
    common = a.join(b.select("user_session"), on="user_session", how="semi")
    assert common.height > 0.9 * a.height
    assert common.equals(b.join(a.select("user_session"), on="user_session", how="semi"))


def test_sequences_ignore_everything_after_event_k(cfg):
    """GRU input must not change when events after position K change."""
    from src.features.sequences import event_steps

    base = generate_synthetic_events(n_sessions=1500, seed=22)
    k = cfg["sessionize"]["default_k"]

    def steps(events):
        idx = index_events(clean_events(events.lazy(), cfg)[0])
        sp = build_prefixes(idx, session_stats(idx), k)
        labels = sp.sessions.select("user_session", TARGET).with_columns(split=pl.lit("train"))
        return event_steps(sp.prefix_events, labels, "unknown").drop(TARGET)

    tampered = base.with_columns(_pipeline_rank(cfg).alias("_r")).with_columns(
        price=pl.when(pl.col("_r") > k).then(pl.col("price") * 10).otherwise(pl.col("price")),
    ).drop("_r")

    a, b = steps(base), steps(tampered)
    assert a.height > 0 and a.equals(b)


def test_run_writes_every_k(tmp_path):
    cfg = load_config(overrides={
        "data.mode": "synthetic",
        "data.synthetic.n_sessions": 2000,
        "paths.interim_dir": str(tmp_path / "interim"),
        "paths.processed_dir": str(tmp_path / "processed"),
    })
    ingest(cfg)
    reports = run(cfg)
    assert sorted(reports) == cfg["sessionize"]["prefix_k"]
    for k, r in reports.items():
        sp = load_prefixes(cfg, k)
        sessions = sp.sessions.collect()
        assert sessions.height == r["eligible"]
        assert sessions[TARGET].sum() == r["positives"]
        assert sp.prefix_events.select(pl.len()).collect().item() == k * r["eligible"]
        assert (tmp_path / "processed" / f"k{k}" / "sessionize_report.json").exists()
