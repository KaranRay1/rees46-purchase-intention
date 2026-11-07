"""Session prefixes at K events and leakage-safe purchase targets.

For each prefix length K this module produces two tables:

``prefix_events``  — the first K events of every eligible session, with their
                     position ``pos`` (1..K). **The only input features may use.**
``sessions``       — one row per eligible session: ids, timestamps, the target
                     and analysis-only columns. **Never a feature source.**

Leakage rules (docs/research_blueprint.md §4)
---------------------------------------------
1. Events are ordered by time; ties within the same second are broken in
   funnel order (view < cart < remove_from_cart < purchase), then product_id,
   so the order is deterministic and a same-second purchase never sneaks
   ahead of the event that led to it.
2. A session is eligible only if it has more than K events (there must be a
   "future" to predict).
3. Sessions with a purchase among events 1..K are dropped: the outcome is
   already known at prediction time.
4. ``target_purchase = 1`` iff a purchase happens after event K.

Prediction time is ``prefix_end_time`` (timestamp of event K). The temporal
split uses ``session_start``.

Usage::

    python -m src.data.sessionize               # all K in config, current data.mode
    python -m src.data.sessionize --mode synthetic
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from src.config import apply_overrides, load_config, processed_dir_for_k
from src.data.loader import load_events

TARGET = "target_purchase"

# Columns in `sessions` that describe the future of the session. They exist for
# cohort/error analysis only and must never be used as model inputs.
ANALYSIS_ONLY_COLUMNS: tuple[str, ...] = (
    TARGET,
    "n_events_total",
    "first_purchase_pos",
    "session_end",
)

SESSION_COLUMNS: list[str] = [
    "user_session",
    "user_id",
    "session_start",
    "prefix_end_time",
    "k",
    TARGET,
    "n_events_total",
    "first_purchase_pos",
    "session_end",
]


@dataclass
class SessionPrefixes:
    """Prefix events and session table for one K (both lazy)."""

    k: int
    prefix_events: pl.LazyFrame
    sessions: pl.LazyFrame


# -----------------------------------------------------------------------------
# Core
# -----------------------------------------------------------------------------
def index_events(events: pl.LazyFrame) -> pl.LazyFrame:
    """Order events within each session and add 1-based position ``pos``.

    ``event_type`` must be the Enum produced by the loader (its physical order
    is the funnel order used for tie-breaking).
    """
    if not isinstance(events.collect_schema()["event_type"], pl.Enum):
        raise TypeError("event_type must be a pl.Enum (use events from src.data.loader).")
    return events.sort(
        ["user_session", "event_time", "event_type", "product_id"], maintain_order=True
    ).with_columns(pos=pl.int_range(1, pl.len() + 1, dtype=pl.UInt32).over("user_session"))


def session_stats(indexed: pl.LazyFrame) -> pl.LazyFrame:
    """One row per session with length, timing and first-purchase position."""
    return indexed.group_by("user_session").agg(
        user_id=pl.col("user_id").first(),
        session_start=pl.col("event_time").min(),
        session_end=pl.col("event_time").max(),
        n_events_total=pl.len().cast(pl.UInt32),
        first_purchase_pos=pl.col("pos").filter(pl.col("event_type") == "purchase").min(),
    )


def build_prefixes(
    indexed: pl.LazyFrame,
    stats: pl.LazyFrame,
    k: int,
    require_more_than_k: bool = True,
) -> SessionPrefixes:
    """Apply the leakage rules for one K.

    Args:
        indexed: output of ``index_events``.
        stats: output of ``session_stats`` on the same events.
        k: prefix length.
        require_more_than_k: keep sessions with > K events (default) instead of >= K.
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    min_len = k + 1 if require_more_than_k else k
    eligible = stats.filter(
        (pl.col("n_events_total") >= min_len)
        & (pl.col("first_purchase_pos").is_null() | (pl.col("first_purchase_pos") > k))
    ).with_columns(
        pl.col("first_purchase_pos").is_not_null().cast(pl.Int8).alias(TARGET),
        pl.lit(k, dtype=pl.UInt8).alias("k"),
    )

    prefix_events = indexed.filter(pl.col("pos") <= k).join(
        eligible.select("user_session"), on="user_session", how="semi"
    )
    prefix_end = prefix_events.group_by("user_session").agg(
        prefix_end_time=pl.col("event_time").max()
    )
    sessions = (
        eligible.join(prefix_end, on="user_session", how="inner")
        .select(SESSION_COLUMNS)
        .sort(["session_start", "user_session"])
    )
    return SessionPrefixes(k=k, prefix_events=prefix_events, sessions=sessions)


def _check_settings(sz: Mapping[str, Any]) -> bool:
    """Reject leaky settings; return ``require_more_than_k_events``."""
    if not sz.get("drop_purchase_within_prefix", True):
        raise ValueError(
            "sessionize.drop_purchase_within_prefix=false would put the outcome inside "
            "the features (target leakage) and is not supported."
        )
    return sz.get("require_more_than_k_events", True)


def sessionize(events: pl.LazyFrame, cfg: Mapping[str, Any]) -> dict[int, SessionPrefixes]:
    """Build prefixes for every K in ``cfg["sessionize"]["prefix_k"]``."""
    require_more = _check_settings(cfg["sessionize"])
    indexed = index_events(events)
    stats = session_stats(indexed)
    return {k: build_prefixes(indexed, stats, k, require_more) for k in cfg["sessionize"]["prefix_k"]}


# -----------------------------------------------------------------------------
# Reporting and persistence
# -----------------------------------------------------------------------------
def summarise(stats: pl.LazyFrame, k: int, require_more_than_k: bool = True) -> dict[str, Any]:
    """Counts behind the eligibility funnel for one K (for the report / thesis table)."""
    min_len = k + 1 if require_more_than_k else k
    s = stats.select(
        sessions_total=pl.len(),
        too_short=(pl.col("n_events_total") < min_len).sum(),
        purchase_within_prefix=(
            (pl.col("n_events_total") >= min_len) & (pl.col("first_purchase_pos") <= k)
        ).sum(),
        eligible=(
            (pl.col("n_events_total") >= min_len)
            & (pl.col("first_purchase_pos").is_null() | (pl.col("first_purchase_pos") > k))
        ).sum(),
        positives=(
            (pl.col("n_events_total") >= min_len) & (pl.col("first_purchase_pos") > k)
        ).sum(),
    ).collect(engine="streaming").row(0, named=True)
    s = {key: int(v) for key, v in s.items()}
    s["k"] = k
    s["positive_rate"] = s["positives"] / s["eligible"] if s["eligible"] else 0.0
    return s


def write_prefixes(sp: SessionPrefixes, out_dir: str | Path) -> dict[str, Path]:
    """Write ``prefix_events.parquet`` and ``sessions.parquet`` for one K."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "prefix_events": out_dir / "prefix_events.parquet",
        "sessions": out_dir / "sessions.parquet",
    }
    sp.prefix_events.sink_parquet(paths["prefix_events"])
    sp.sessions.sink_parquet(paths["sessions"])
    return paths


def run(cfg: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    """Load cleaned events, build and write prefixes for every K, return reports."""
    require_more = _check_settings(cfg["sessionize"])
    indexed = index_events(load_events(cfg))
    stats = session_stats(indexed)

    reports: dict[int, dict[str, Any]] = {}
    for k in cfg["sessionize"]["prefix_k"]:
        out_dir = processed_dir_for_k(cfg, k)
        write_prefixes(build_prefixes(indexed, stats, k, require_more), out_dir)
        report = summarise(stats, k, require_more)
        report["mode"] = cfg["data"]["mode"]
        (out_dir / "sessionize_report.json").write_text(json.dumps(report, indent=2))
        reports[k] = report
    return reports


def load_prefixes(cfg: Mapping[str, Any], k: int) -> SessionPrefixes:
    """Lazy scans of the written prefix tables for one K."""
    d = processed_dir_for_k(cfg, k)
    paths = [d / "prefix_events.parquet", d / "sessions.parquet"]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{missing} not found; run `python -m src.data.sessionize` first.")
    return SessionPrefixes(k=k, prefix_events=pl.scan_parquet(paths[0]), sessions=pl.scan_parquet(paths[1]))


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe session prefixes for each K.")
    parser.add_argument("--mode", choices=["synthetic", "dev", "full"], help="override data.mode")
    args = parser.parse_args()
    cfg = load_config()
    if args.mode:
        cfg = apply_overrides(cfg, {"data.mode": args.mode})

    print(f"{'K':>3} {'sessions':>10} {'too short':>10} {'early buy':>10} {'eligible':>10} {'positive %':>11}")
    for k, r in run(cfg).items():
        print(
            f"{k:>3} {r['sessions_total']:>10,} {r['too_short']:>10,} "
            f"{r['purchase_within_prefix']:>10,} {r['eligible']:>10,} {r['positive_rate']:>10.2%}"
        )


if __name__ == "__main__":
    _main()
