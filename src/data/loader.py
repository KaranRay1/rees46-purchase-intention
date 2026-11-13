"""REES46 ingestion: schema checks, CSV -> Parquet, cleaning (Polars lazy).

Pipeline (one call: ``ingest(cfg)``)::

    data/raw/2019-Oct.csv ──csv_to_parquet──► data/interim/raw/2019-Oct.parquet   (typed, 1 pass)
                                                   │  (all months scanned together)
                                                   ▼
                                    clean_events ──► data/interim/events_<mode>.parquet
                                                   + data/interim/cleaning_report_<mode>.json

Why two stages: parsing a multi-GB CSV is the slow part, so it is done once
per month and cached as typed Parquet. Cleaning then runs on Parquet, which
is fast enough to rerun whenever a cleaning rule changes, and lets sessions
that cross a month boundary be cleaned as one session.

Everything stays lazy and is written with ``sink_parquet`` so the full
~285M-row dataset never has to fit in memory at once.

Usage::

    python -m src.data.loader                 # uses data.mode from config.yaml
    python -m src.data.loader --mode synthetic --overwrite
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from src.config import apply_overrides, load_config, raw_files, resolve_path
from src.data.synthetic import REES46_COLUMNS, generate_from_config

# Types used when reading the raw CSV. event_time is parsed separately because
# REES46 writes it as text with a literal " UTC" suffix.
RAW_SCHEMA: dict[str, pl.DataType] = {
    "event_time": pl.String(),
    "event_type": pl.String(),
    "product_id": pl.Int64(),
    "category_id": pl.Int64(),
    "category_code": pl.String(),
    "brand": pl.String(),
    "price": pl.Float64(),
    "user_id": pl.Int64(),
    "user_session": pl.String(),
}
EVENT_TIME_DTYPE = pl.Datetime("us", "UTC")


class SchemaError(ValueError):
    """Raised when an input file does not have the REES46 columns."""


# -----------------------------------------------------------------------------
# Schema
# -----------------------------------------------------------------------------
def validate_columns(columns: list[str], source: str = "input") -> None:
    """Raise ``SchemaError`` if any REES46 column is missing or unexpected."""
    missing = [c for c in REES46_COLUMNS if c not in columns]
    extra = [c for c in columns if c not in REES46_COLUMNS]
    if missing or extra:
        raise SchemaError(f"{source}: missing columns {missing}, unexpected columns {extra}")


def _parse_event_time(col: str = "event_time") -> pl.Expr:
    """'2019-10-01 00:00:00 UTC' -> Datetime(us, UTC). Unparseable values become null."""
    return (
        pl.col(col)
        .str.strip_suffix(" UTC")
        .str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S", strict=False)
        .dt.replace_time_zone("UTC")
        .alias(col)
    )


def scan_rees46_csv(path: str | Path) -> pl.LazyFrame:
    """Lazily read one REES46 CSV with explicit types and parsed timestamps."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Raw file not found: {path}\n"
            "Download the REES46 multi-category logs into data/raw/, "
            "or run with data.mode=synthetic."
        )
    header = pl.read_csv(path, n_rows=0).columns
    validate_columns(header, source=str(path))
    return (
        pl.scan_csv(path, schema_overrides=RAW_SCHEMA)
        .with_columns(_parse_event_time())
        .select(REES46_COLUMNS)
    )


# -----------------------------------------------------------------------------
# Stage 1: CSV -> typed Parquet (cached per month)
# -----------------------------------------------------------------------------
def csv_to_parquet(csv_path: str | Path, out_dir: str | Path, overwrite: bool = False) -> Path:
    """Convert one REES46 CSV to typed Parquet in a single streaming pass.

    Returns the Parquet path. Skips work if it already exists (unless ``overwrite``).
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{csv_path.stem}.parquet"
    if out.exists() and not overwrite:
        return out
    tmp = out.with_suffix(".parquet.tmp")
    scan_rees46_csv(csv_path).sink_parquet(tmp)
    tmp.replace(out)  # only a complete file ever has the final name
    return out


# -----------------------------------------------------------------------------
# Stage 2: cleaning
# -----------------------------------------------------------------------------
def _cleaning_steps(cfg: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Ordered cleaning steps as (name, LazyFrame -> LazyFrame)."""
    data = cfg["data"]
    rules = data["cleaning"]
    event_types: list[str] = data["event_types"]
    frac = data.get("session_sample_fraction")
    seed = cfg["project"]["seeds"][0]
    max_secs = rules["max_session_duration_hours"] * 3600

    def drop_bad_time(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.filter(pl.col("event_time").is_not_null())

    def drop_null_session(lf: pl.LazyFrame) -> pl.LazyFrame:
        if not rules.get("drop_null_session", True):
            return lf
        return lf.filter(pl.col("user_session").is_not_null() & (pl.col("user_session") != ""))

    def drop_unknown_event_type(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.filter(pl.col("event_type").is_in(event_types))

    def drop_bad_price(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.filter(pl.col("price").is_not_null() & (pl.col("price") > rules["min_price"]))

    def drop_duplicates(lf: pl.LazyFrame) -> pl.LazyFrame:
        if not rules.get("drop_duplicate_events", True):
            return lf
        # Rows are identical across all columns, so which copy survives does not matter.
        return lf.unique(subset=REES46_COLUMNS, keep="any", maintain_order=False)

    def sample_sessions(lf: pl.LazyFrame) -> pl.LazyFrame:
        # Hash-based: deterministic, keeps whole sessions, works lazily at any scale.
        if frac is None or frac >= 1:
            return lf
        bucket = pl.col("user_session").hash(seed=seed) % 1_000_000
        return lf.filter(bucket < int(frac * 1_000_000))

    def drop_long_sessions(lf: pl.LazyFrame) -> pl.LazyFrame:
        long = (
            lf.group_by("user_session")
            .agg(dur=(pl.col("event_time").max() - pl.col("event_time").min()).dt.total_seconds())
            .filter(pl.col("dur") > max_secs)
            .select("user_session")
        )
        return lf.join(long, on="user_session", how="anti")

    return [
        ("unparseable_event_time", drop_bad_time),
        ("null_session", drop_null_session),
        # Sampling whole sessions early keeps the multi-month run small; duplicates share a
        # session id, so sampling before de-duplicating removes exactly the same rows.
        ("session_sampling", sample_sessions),
        ("unknown_event_type", drop_unknown_event_type),
        ("non_positive_or_null_price", drop_bad_price),
        ("duplicate_rows", drop_duplicates),
        ("over_long_sessions", drop_long_sessions),
    ]


def _finalise(lf: pl.LazyFrame, cfg: Mapping[str, Any]) -> pl.LazyFrame:
    """Fill missing category/brand tokens, use compact dtypes, sort by session and time.

    Sorting keeps each session's events together in time order: Parquet
    compresses far better, and sessionising can rely on the order.
    """
    rules = cfg["data"]["cleaning"]
    return (
        lf.with_columns(
            pl.col("event_type").cast(pl.Enum(cfg["data"]["event_types"])),
            pl.col("category_code").fill_null(rules["missing_category_token"]),
            pl.col("brand").fill_null(rules["missing_brand_token"]),
        )
        .select(REES46_COLUMNS)
        .sort(["user_session", "event_time"], maintain_order=True)
    )


def clean_events(
    lf: pl.LazyFrame, cfg: Mapping[str, Any], with_report: bool = False
) -> tuple[pl.LazyFrame, dict[str, Any] | None]:
    """Apply all cleaning rules from ``cfg["data"]["cleaning"]``.

    Args:
        lf: raw events with REES46 columns (``event_time`` already parsed).
        with_report: also count rows removed by each step. Each count is a
            separate streaming query over the source, so it costs time
            (one pass per step) but not memory.

    Returns:
        (cleaned LazyFrame, report dict or None)
    """
    validate_columns(lf.collect_schema().names(), source="events")
    report: dict[str, Any] | None = None
    if with_report:
        report = {"rows_in": _count(lf), "removed": {}}
        sessions_in = _n_sessions(lf)

    before = report["rows_in"] if report is not None else 0
    for name, step in _cleaning_steps(cfg):
        lf = step(lf)
        if report is not None:
            after = _count(lf)  # streaming count; re-runs the lazy chain, never holds it in memory
            report["removed"][name] = before - after
            before = after

    lf = _finalise(lf, cfg)
    if with_report:
        report["rows_out"] = _count(lf)
        report["sessions_in"] = sessions_in
        report["sessions_out"] = _n_sessions(lf)
        report["purchase_sessions_out"] = (
            lf.filter(pl.col("event_type") == "purchase")
            .select(pl.col("user_session").n_unique())
            .collect()
            .item()
        )
    return lf, report


def _count(lf: pl.LazyFrame) -> int:
    return lf.select(pl.len()).collect(engine="streaming").item()


def _n_sessions(lf: pl.LazyFrame) -> int:
    return lf.select(pl.col("user_session").n_unique()).collect(engine="streaming").item()


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def cleaned_events_path(cfg: Mapping[str, Any]) -> Path:
    """Where ``ingest`` writes the cleaned events for the current ``data.mode``."""
    return resolve_path(cfg, "interim_dir") / f"events_{cfg['data']['mode']}.parquet"


def ingest(cfg: Mapping[str, Any], overwrite: bool = False, with_report: bool = True) -> Path:
    """Run raw -> typed Parquet -> cleaned Parquet for ``cfg["data"]["mode"]``.

    * ``synthetic``: generate events in memory (no CSV needed).
    * ``dev`` / ``full``: convert each configured CSV once, then clean all together.

    ``with_report`` re-scans the data once per cleaning step to count removals;
    pass ``with_report=False`` on the full dataset to save time.
    Cached files are reused: after changing a cleaning rule, pass ``overwrite=True``.

    Returns:
        Path to ``data/interim/events_<mode>.parquet``.
    """
    out = cleaned_events_path(cfg)
    if out.exists() and not overwrite:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = cfg["data"]["mode"]

    if mode == "synthetic":
        raw = generate_from_config(cfg).lazy()
        sources = ["synthetic"]
    else:
        parquet_dir = resolve_path(cfg, "interim_dir") / "raw"
        parts = [csv_to_parquet(p, parquet_dir, overwrite=overwrite) for p in raw_files(cfg)]
        raw = pl.scan_parquet(parts)
        sources = [p.name for p in parts]

    cleaned, report = clean_events(raw, cfg, with_report=with_report)
    tmp = out.with_suffix(".parquet.tmp")
    cleaned.sink_parquet(tmp)
    tmp.replace(out)

    if report is not None:
        report = {"mode": mode, "sources": sources, **report}
        report_path = out.parent / f"cleaning_report_{mode}.json"
        report_path.write_text(json.dumps(report, indent=2))
    return out


def load_events(cfg: Mapping[str, Any], build_if_missing: bool = True) -> pl.LazyFrame:
    """Lazy scan of the cleaned events for the current mode (runs ``ingest`` if needed)."""
    path = cleaned_events_path(cfg)
    if not path.exists():
        if not build_if_missing:
            raise FileNotFoundError(f"{path} not found; run `python -m src.data.loader` first.")
        ingest(cfg)
    return pl.scan_parquet(path)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Ingest and clean REES46 event logs.")
    parser.add_argument("--mode", choices=["synthetic", "dev", "full"], help="override data.mode")
    parser.add_argument("--overwrite", action="store_true", help="rebuild cached Parquet files")
    parser.add_argument("--no-report", action="store_true", help="skip per-step counts (faster)")
    args = parser.parse_args()

    cfg = load_config()
    if args.mode:
        cfg = apply_overrides(cfg, {"data.mode": args.mode})

    out = ingest(cfg, overwrite=args.overwrite, with_report=not args.no_report)
    print(f"Cleaned events -> {out}")
    report_path = out.parent / f"cleaning_report_{cfg['data']['mode']}.json"
    if report_path.exists() and not args.no_report:
        print(report_path.read_text())


if __name__ == "__main__":
    _main()
