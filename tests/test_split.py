"""Train/val/test splits are chronological, disjoint and shared across K."""

import json
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from src.config import load_config
from src.data import sessionize
from src.data.loader import ingest
from src.data.split import (
    assign_splits,
    compute_cutoffs,
    cutoffs_path,
    load_split_sessions,
    run,
)

T0 = datetime(2019, 10, 1, tzinfo=timezone.utc)


def _sessions(starts_min, durations_min=None):
    durations_min = durations_min or [1] * len(starts_min)
    return pl.DataFrame({
        "user_session": [f"s{i}" for i in range(len(starts_min))],
        "session_start": [T0 + timedelta(minutes=m) for m in starts_min],
        "session_end": [T0 + timedelta(minutes=m + d) for m, d in zip(starts_min, durations_min)],
        "target_purchase": [i % 2 for i in range(len(starts_min))],
    })


def test_cutoffs_on_simple_sequence():
    starts = _sessions(list(range(100)))["session_start"].sample(fraction=1, shuffle=True, seed=0)
    c = compute_cutoffs(starts, 0.7, 0.15)
    assert c["val_start"] == T0 + timedelta(minutes=70)
    assert c["test_start"] == T0 + timedelta(minutes=85)


@pytest.mark.parametrize("bad", [(0.0, 0.15), (0.7, 0.3), (1.0, 0.1)])
def test_bad_fractions_raise(bad):
    with pytest.raises(ValueError):
        compute_cutoffs(_sessions(list(range(10)))["session_start"], *bad)


def test_splits_are_chronological_and_disjoint():
    df = _sessions(list(range(1000)), durations_min=[0] * 1000)  # no boundary crossings
    kept, purged = assign_splits(df.lazy(), compute_cutoffs(df["session_start"], 0.7, 0.15))
    kept = kept.collect()
    by = {s: kept.filter(pl.col("split") == s)["session_start"] for s in ("train", "val", "test")}
    assert by["train"].max() < by["val"].min() and by["val"].max() < by["test"].min()
    assert kept["user_session"].is_unique().all()
    assert [len(by[s]) for s in ("train", "val", "test")] == [700, 150, 150]


def test_boundary_crossing_sessions_are_purged():
    # s0 starts in train but ends after the val cut-off (minute 7); s7 is the first val session.
    df = _sessions(list(range(10)), durations_min=[30] + [0] * 9)
    cut = compute_cutoffs(df["session_start"], 0.7, 0.15)
    kept, purged = assign_splits(df.lazy(), cut, purge_boundary_overlap=True)
    assert purged.collect()["user_session"].to_list() == ["s0"]
    assert "s0" not in kept.collect()["user_session"].to_list()

    kept_all, purged_none = assign_splits(df.lazy(), cut, purge_boundary_overlap=False)
    assert kept_all.collect().height == 10 and purged_none.collect().height == 0


def test_test_sessions_are_never_purged():
    df = _sessions(list(range(10)), durations_min=[0] * 9 + [10_000])
    kept, purged = assign_splits(df.lazy(), compute_cutoffs(df["session_start"], 0.7, 0.15))
    assert purged.collect().height == 0 and kept.collect().height == 10


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("split")
    cfg = load_config(overrides={
        "data.mode": "synthetic",
        "data.synthetic.n_sessions": 4000,
        "paths.interim_dir": str(tmp / "interim"),
        "paths.processed_dir": str(tmp / "processed"),
    })
    ingest(cfg)
    sessionize.run(cfg)
    return cfg, run(cfg)


def test_run_uses_same_cutoffs_for_every_k(built):
    cfg, reports = built
    cut = json.loads(cutoffs_path(cfg).read_text())
    assert cut["computed_from_k"] == min(cfg["sessionize"]["prefix_k"])
    val_start = datetime.fromisoformat(cut["val_start"])
    test_start = datetime.fromisoformat(cut["test_start"])
    for k in cfg["sessionize"]["prefix_k"]:
        df = load_split_sessions(cfg, k).collect()
        assert (df.filter(pl.col("split") == "train")["session_start"] < val_start).all()
        assert (df.filter(pl.col("split") == "test")["session_start"] >= test_start).all()


def test_run_reports_match_files(built):
    cfg, reports = built
    for k, r in reports.items():
        df = load_split_sessions(cfg, k).collect()
        assert df.height == r["sessions_kept"]
        for s in ("train", "val", "test"):
            part = df.filter(pl.col("split") == s)
            assert part.height == r[s]["sessions"]
            assert part["target_purchase"].sum() == r[s]["positives"]
        assert 0.6 < r["train"]["fraction"] < 0.8


def test_load_split_sessions_filter(built):
    cfg, _ = built
    k = cfg["sessionize"]["default_k"]
    test = load_split_sessions(cfg, k, "test").collect()
    assert set(test["split"].cast(pl.String)) == {"test"}
    with pytest.raises(ValueError):
        load_split_sessions(cfg, k, "holdout")
