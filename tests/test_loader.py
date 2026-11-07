"""Tests for src/data/loader.py."""

import json
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from src.config import load_config
from src.data.loader import (
    EVENT_TIME_DTYPE,
    SchemaError,
    clean_events,
    csv_to_parquet,
    ingest,
    load_events,
    scan_rees46_csv,
)
from src.data.synthetic import REES46_COLUMNS, generate_synthetic_events, write_rees46_csv


def _cfg(tmp_path, **overrides):
    base = {
        "paths.raw_dir": str(tmp_path / "raw"),
        "paths.interim_dir": str(tmp_path / "interim"),
    }
    return load_config(overrides={**base, **overrides})


def _event(t, etype="view", session="s1", price=10.0, code="electronics.smartphone", brand="x"):
    return {
        "event_time": t, "event_type": etype, "product_id": 1, "category_id": 2,
        "category_code": code, "brand": brand, "price": price, "user_id": 3, "user_session": session,
    }


def _frame(rows):
    return pl.DataFrame(rows).with_columns(
        pl.col("event_time").cast(EVENT_TIME_DTYPE)
    ).select(REES46_COLUMNS)


T0 = datetime(2019, 10, 1, 12, tzinfo=timezone.utc)


# ---------------------------------------------------------------- schema / CSV
def test_csv_round_trip_keeps_values_and_types(tmp_path):
    df = generate_synthetic_events(n_sessions=200, seed=1)
    csv = write_rees46_csv(df, tmp_path / "raw" / "2019-Oct.csv")
    back = scan_rees46_csv(csv).collect()
    assert back.schema["event_time"] == EVENT_TIME_DTYPE
    assert back.equals(df)


def test_missing_column_raises(tmp_path):
    path = tmp_path / "bad.csv"
    pl.DataFrame({"event_time": ["2019-10-01 00:00:00 UTC"], "event_type": ["view"]}).write_csv(path)
    with pytest.raises(SchemaError, match="missing columns"):
        scan_rees46_csv(path)


def test_missing_raw_file_has_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="data.mode=synthetic"):
        scan_rees46_csv(tmp_path / "2019-Oct.csv")


def test_csv_to_parquet_is_cached(tmp_path):
    csv = write_rees46_csv(generate_synthetic_events(n_sessions=20, seed=1), tmp_path / "a.csv")
    out = csv_to_parquet(csv, tmp_path / "pq")
    mtime = out.stat().st_mtime_ns
    assert csv_to_parquet(csv, tmp_path / "pq") == out
    assert out.stat().st_mtime_ns == mtime
    assert not list((tmp_path / "pq").glob("*.tmp"))


# ---------------------------------------------------------------- cleaning rules
def test_each_cleaning_rule(tmp_path):
    cfg = _cfg(tmp_path)
    rows = [
        _event(T0),                                            # kept
        _event(T0),                                            # duplicate of the first
        _event(T0 + timedelta(seconds=5), price=0.0),          # bad price
        _event(T0, session=None),                              # null session
        _event(T0, etype="wishlist"),                          # unknown event type
        _event(T0, session="long"),                            # long session start ...
        _event(T0 + timedelta(hours=25), session="long"),      # ... ends 25 h later
        _event(T0, session="s2", code=None, brand=None),       # kept, nulls filled
    ]
    cleaned, report = clean_events(_frame(rows).lazy(), cfg, with_report=True)
    out = cleaned.collect()

    assert out.height == 2
    assert set(out["user_session"]) == {"s1", "s2"}
    assert out.schema["event_type"] == pl.Enum(cfg["data"]["event_types"])
    s2 = out.filter(pl.col("user_session") == "s2").row(0, named=True)
    assert s2["category_code"] == "unknown" and s2["brand"] == "unknown"

    removed = report["removed"]
    assert removed["duplicate_rows"] == 1
    assert removed["non_positive_or_null_price"] == 1
    assert removed["null_session"] == 1
    assert removed["unknown_event_type"] == 1
    assert removed["over_long_sessions"] == 2
    assert report["rows_in"] - sum(removed.values()) == report["rows_out"] == 2


def test_noisy_synthetic_is_fully_cleaned(tmp_path):
    cfg = _cfg(tmp_path)
    noisy = generate_synthetic_events(n_sessions=500, seed=2, inject_noise=True)
    out = clean_events(noisy.lazy(), cfg)[0].collect()
    assert out.is_duplicated().sum() == 0
    assert (out["price"] > 0).all()
    assert out["user_session"].null_count() == 0


def test_session_sampling_keeps_whole_sessions(tmp_path):
    cfg = _cfg(tmp_path, **{"data.session_sample_fraction": 0.3})
    events = generate_synthetic_events(n_sessions=2000, seed=3)
    a = clean_events(events.lazy(), cfg)[0].collect()
    b = clean_events(events.lazy(), cfg)[0].collect()
    kept = set(a["user_session"])
    assert kept == set(b["user_session"])                      # deterministic
    assert 0.2 < len(kept) / 2000 < 0.4                        # roughly the fraction
    full_len = events.filter(pl.col("user_session").is_in(list(kept))).height
    assert a.height == full_len                                # no partial sessions


# ---------------------------------------------------------------- orchestration
def test_ingest_synthetic_writes_events_and_report(tmp_path):
    cfg = _cfg(tmp_path, **{"data.mode": "synthetic", "data.synthetic.n_sessions": 300})
    out = ingest(cfg)
    assert out.name == "events_synthetic.parquet"
    report = json.loads((out.parent / "cleaning_report_synthetic.json").read_text())
    assert report["sessions_out"] == 300
    assert load_events(cfg).select(pl.len()).collect().item() == report["rows_out"]


def test_ingest_dev_from_csv(tmp_path):
    cfg = _cfg(tmp_path, **{"data.mode": "dev"})
    events = generate_synthetic_events(n_sessions=300, seed=4, inject_noise=True)
    write_rees46_csv(events, tmp_path / "raw" / cfg["data"]["dev_files"][0])
    out = ingest(cfg)
    assert (tmp_path / "interim" / "raw" / "2019-Oct.parquet").exists()
    df = pl.read_parquet(out)
    assert df.is_duplicated().sum() == 0 and df.height > 0


def test_load_events_without_build_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_events(_cfg(tmp_path), build_if_missing=False)
