"""Chronological train / validation / test split by session start time.

Design choices
--------------
* **Time-based, no shuffling.** Sessions are ordered by ``session_start``;
  the earliest 70% go to train, the next 15% to validation, the last 15% to
  test (fractions from ``config.yaml``). This mimics deployment: train on the
  past, predict the future, and avoids look-ahead bias.
* **One set of cut-off times for every K.** Cut-offs are computed once from
  the sessions eligible at the smallest K (a superset of every larger K's
  sessions) and reused, so K = 3, 5, 10 are evaluated on the *same calendar
  periods* — the RQ2 comparison then varies only K, not the time window.
* **Boundary purge.** A train session that starts before the validation
  cut-off but ends after it carries label information from the validation
  period (same for validation → test). With ``purge_boundary_overlap: true``
  such sessions are dropped, and the report says how many.

Outputs (per K, next to the sessionize outputs)::

    data/processed/split_cutoffs.json          shared cut-off timestamps
    data/processed/k{K}/splits.parquet         user_session, split
    data/processed/k{K}/split_report.json      sizes and positive rates per split

Usage::

    python -m src.data.split                    # all K, current data.mode
    python -m src.data.split --mode synthetic
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from src.config import apply_overrides, load_config, processed_dir_for_k, resolve_path
from src.data.sessionize import TARGET, load_prefixes

SPLITS: list[str] = ["train", "val", "test"]
SPLIT_DTYPE = pl.Enum(SPLITS)


# -----------------------------------------------------------------------------
# Core
# -----------------------------------------------------------------------------
def compute_cutoffs(session_starts: pl.Series, train: float, val: float) -> dict[str, datetime]:
    """Timestamps where validation and test begin.

    The validation cut-off is the start time of the session at rank
    ``floor(n * train)`` in time order; the test cut-off likewise at
    ``floor(n * (train + val))``. Sessions starting exactly on a cut-off go
    to the later split, so the realised fractions can differ slightly.
    """
    n = session_starts.len()
    if n < 3:
        raise ValueError(f"Need at least 3 sessions to split, got {n}.")
    if not (0 < train < 1 and 0 < val < 1 and train + val < 1):
        raise ValueError("train and val must be in (0, 1) with train + val < 1.")
    s = session_starts.sort()
    i_val = min(max(int(n * train), 1), n - 2)
    i_test = min(max(int(n * (train + val)), i_val + 1), n - 1)
    return {"val_start": s[i_val], "test_start": s[i_test]}


def assign_splits(
    sessions: pl.LazyFrame, cutoffs: Mapping[str, datetime], purge_boundary_overlap: bool = True
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Label each session train/val/test.

    Args:
        sessions: needs ``user_session``, ``session_start`` and ``session_end``.
        cutoffs: output of ``compute_cutoffs``.
        purge_boundary_overlap: drop sessions whose events cross into the next split.

    Returns:
        (kept sessions with a ``split`` column, purged sessions with their would-be split)
    """
    val_start, test_start = cutoffs["val_start"], cutoffs["test_start"]
    labelled = sessions.with_columns(
        pl.when(pl.col("session_start") < val_start)
        .then(pl.lit("train"))
        .when(pl.col("session_start") < test_start)
        .then(pl.lit("val"))
        .otherwise(pl.lit("test"))
        .cast(SPLIT_DTYPE)
        .alias("split")
    )
    if not purge_boundary_overlap:
        return labelled, labelled.head(0)

    crosses = ((pl.col("split") == "train") & (pl.col("session_end") >= val_start)) | (
        (pl.col("split") == "val") & (pl.col("session_end") >= test_start)
    )
    return labelled.filter(~crosses), labelled.filter(crosses)


def split_report(kept: pl.DataFrame, purged: pl.DataFrame) -> dict[str, Any]:
    """Sizes, positive rates and time ranges per split."""
    per_split = (
        kept.group_by("split")
        .agg(
            sessions=pl.len(),
            positives=pl.col(TARGET).sum(),
            first_start=pl.col("session_start").min(),
            last_start=pl.col("session_start").max(),
        )
        .with_columns(positive_rate=pl.col("positives") / pl.col("sessions"))
        .sort("split")
    )
    total = kept.height
    report: dict[str, Any] = {"sessions_kept": total, "purged_boundary_sessions": purged.height}
    for row in per_split.iter_rows(named=True):
        report[str(row["split"])] = {
            "sessions": row["sessions"],
            "fraction": row["sessions"] / total if total else 0.0,
            "positives": int(row["positives"]),
            "positive_rate": row["positive_rate"],
            "first_start": row["first_start"].isoformat(),
            "last_start": row["last_start"].isoformat(),
        }
    return report


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def cutoffs_path(cfg: Mapping[str, Any]) -> Path:
    return resolve_path(cfg, "processed_dir") / "split_cutoffs.json"


def run(cfg: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    """Compute shared cut-offs, split every K, write splits and reports."""
    sp_cfg = cfg["split"]
    if sp_cfg.get("method", "chronological") != "chronological":
        raise ValueError("Only split.method = chronological is supported.")
    ks = sorted(cfg["sessionize"]["prefix_k"])
    purge = sp_cfg.get("purge_boundary_overlap", True)

    # Smallest K has the largest eligible set (superset of larger K) -> shared cut-offs.
    starts = load_prefixes(cfg, ks[0]).sessions.select("session_start").collect()["session_start"]
    cutoffs = compute_cutoffs(starts, sp_cfg["train"], sp_cfg["val"])
    path = cutoffs_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"computed_from_k": ks[0], "n_sessions": starts.len(),
         **{k: v.isoformat() for k, v in cutoffs.items()}},
        indent=2,
    ))

    reports: dict[int, dict[str, Any]] = {}
    for k in ks:
        sessions = load_prefixes(cfg, k).sessions
        kept, purged = assign_splits(sessions, cutoffs, purge)
        kept_df, purged_df = kept.collect(), purged.collect()
        out_dir = processed_dir_for_k(cfg, k)
        kept_df.select("user_session", "split").write_parquet(out_dir / "splits.parquet")
        report = {"k": k, **split_report(kept_df, purged_df)}
        (out_dir / "split_report.json").write_text(json.dumps(report, indent=2))
        reports[k] = report
    return reports


def load_split_sessions(cfg: Mapping[str, Any], k: int, split: str | None = None) -> pl.LazyFrame:
    """Sessions table for K joined with its split (purged sessions excluded).

    Args:
        split: ``"train"``, ``"val"``, ``"test"`` or ``None`` for all.
    """
    path = processed_dir_for_k(cfg, k) / "splits.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m src.data.split` first.")
    lf = load_prefixes(cfg, k).sessions.join(pl.scan_parquet(path), on="user_session", how="inner")
    if split is not None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        lf = lf.filter(pl.col("split") == split)
    return lf


def _main() -> None:
    parser = argparse.ArgumentParser(description="Chronological train/val/test split for every K.")
    parser.add_argument("--mode", choices=["synthetic", "dev", "full"], help="override data.mode")
    args = parser.parse_args()
    cfg = load_config()
    if args.mode:
        cfg = apply_overrides(cfg, {"data.mode": args.mode})

    reports = run(cfg)
    print(json.loads(cutoffs_path(cfg).read_text()))
    print(f"{'K':>3} {'split':>6} {'sessions':>10} {'share':>7} {'positive %':>11}")
    for k, r in reports.items():
        for s in SPLITS:
            if s in r:
                x = r[s]
                print(f"{k:>3} {s:>6} {x['sessions']:>10,} {x['fraction']:>7.1%} {x['positive_rate']:>10.2%}")
        print(f"{k:>3} purged {r['purged_boundary_sessions']:>10,}")


if __name__ == "__main__":
    _main()
