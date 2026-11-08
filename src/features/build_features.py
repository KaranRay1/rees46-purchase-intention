"""Aggregated session-prefix features (groups G1–G5) for the tabular models.

Input is **only** ``prefix_events`` (the first K events of each eligible
session, from ``src.data.sessionize``). Nothing from the ``sessions`` table
except the target and split labels is attached, and those are added after
the features are computed, so no future information can enter a feature.

Feature groups (names must match ``config.yaml → features.groups``)
--------------------------------------------------------------------
G1 activity          unique_products_count, session_duration_sec, max_gap_sec
G2 cart signals      view_count, cart_count, remove_count, has_cart, first_cart_position, last_event_type
                     (view_count is here, not in G1: with K fixed, view = K − cart − remove,
                     so it carries cart information; G1 stays cart-free for the ablation)
G3 behavioural ratios cart_to_view_ratio, remove_to_cart_ratio          (ε from config)
G4 category hierarchy unique_categories_count, unique_brands_count, main_category
G5 price & time      price_mean, price_max, price_sum, price_std, hour_of_day, day_of_week

Notes
-----
* ``session_duration_sec`` = time of event K − time of event 1 (prefix only).
* ``first_cart_position`` = position (1..K) of the first cart, K+1 if none.
* ``main_category`` = top level of ``category_code`` (e.g. "electronics") of the
  product with the most events in the prefix (ties → earliest seen). It is
  kept as a raw string here; mapping rare values to "other" (top-N) is fitted
  on the training split in ``preprocessing.py``.
* ``hour_of_day`` / ``day_of_week`` are taken at the prediction time (event K), UTC.
  ``day_of_week``: Monday = 1 … Sunday = 7.
* Ratios follow the proposal formula ``a / (b + ε)``. When ``b = 0`` they become
  very large (e.g. 1e6); tree models are unaffected, and ``preprocessing.py``
  applies a log transform for the linear and neural models.

Usage::

    python -m src.features.build_features            # all K, current data.mode
    python -m src.features.build_features --mode synthetic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from src.config import apply_overrides, feature_list, load_config, processed_dir_for_k
from src.data.sessionize import ANALYSIS_ONLY_COLUMNS, TARGET, load_prefixes
from src.data.split import load_split_sessions

ID_COLUMNS: list[str] = ["user_session", "split", TARGET]


def _main_category(prefix_events: pl.LazyFrame, unknown: str) -> pl.LazyFrame:
    """Top-level category of the most frequent product in each prefix."""
    per_product = prefix_events.group_by("user_session", "product_id").agg(
        n=pl.len(),
        first_pos=pl.col("pos").min(),
        code=pl.col("category_code").first(),
    )
    top = per_product.sort(
        ["user_session", "n", "first_pos"], descending=[False, True, False]
    ).group_by("user_session", maintain_order=True).first()
    return top.select(
        "user_session",
        main_category=pl.when(pl.col("code") == unknown)
        .then(pl.lit(unknown))
        .otherwise(pl.col("code").str.split(".").list.first())
        .fill_null(unknown),
    )


def build_features(prefix_events: pl.LazyFrame, k: int, cfg: Mapping[str, Any]) -> pl.LazyFrame:
    """Compute all G1–G5 features, one row per ``user_session``.

    Args:
        prefix_events: first K events per session with a ``pos`` column (1..K).
        k: prefix length (used for ``first_cart_position`` when no cart).
        cfg: loaded config (epsilon, missing-value tokens, feature names).

    Returns:
        LazyFrame with ``user_session`` followed by the configured features, in config order.
    """
    eps = float(cfg["features"]["epsilon"])
    unknown_cat = cfg["data"]["cleaning"]["missing_category_token"]
    unknown_brand = cfg["data"]["cleaning"]["missing_brand_token"]

    etype = pl.col("event_type").cast(pl.String)
    by_pos = pl.col("event_time").sort_by("pos")

    agg = prefix_events.group_by("user_session").agg(
        # G1 activity
        view_count=(etype == "view").sum().cast(pl.Int32),
        unique_products_count=pl.col("product_id").n_unique().cast(pl.Int32),
        session_duration_sec=(pl.col("event_time").max() - pl.col("event_time").min())
        .dt.total_seconds()
        .cast(pl.Float64),
        max_gap_sec=by_pos.diff().dt.total_seconds().max().cast(pl.Float64),
        # G2 cart signals
        cart_count=(etype == "cart").sum().cast(pl.Int32),
        remove_count=(etype == "remove_from_cart").sum().cast(pl.Int32),
        first_cart_position=pl.col("pos").filter(etype == "cart").min().cast(pl.Int32),
        last_event_type=etype.sort_by(pl.col("pos")).last(),
        # G4 category hierarchy
        unique_categories_count=pl.col("category_id").n_unique().cast(pl.Int32),
        unique_brands_count=pl.col("brand").filter(pl.col("brand") != unknown_brand).n_unique().cast(pl.Int32),
        # G5 price & time
        price_mean=pl.col("price").mean(),
        price_max=pl.col("price").max(),
        price_sum=pl.col("price").sum(),
        price_std=pl.col("price").std(ddof=0),
        prediction_time=pl.col("event_time").max(),
    )

    features = (
        agg.with_columns(
            max_gap_sec=pl.col("max_gap_sec").fill_null(0.0),  # K = 1 has no gaps
            has_cart=(pl.col("cart_count") > 0).cast(pl.Int8),
            first_cart_position=pl.col("first_cart_position").fill_null(k + 1),
            # G3 behavioural ratios (proposal formula)
            cart_to_view_ratio=pl.col("cart_count") / (pl.col("view_count") + eps),
            remove_to_cart_ratio=pl.col("remove_count") / (pl.col("cart_count") + eps),
            hour_of_day=pl.col("prediction_time").dt.hour().cast(pl.Int8),
            day_of_week=pl.col("prediction_time").dt.weekday().cast(pl.Int8),
        )
        .join(_main_category(prefix_events, unknown_cat), on="user_session", how="left")
    )
    return features.select(["user_session", *feature_list(cfg)])


def build_feature_table(cfg: Mapping[str, Any], k: int) -> pl.DataFrame:
    """Features for K joined with split and target (purged sessions excluded)."""
    sp = load_prefixes(cfg, k)
    feats = build_features(sp.prefix_events, k, cfg)
    labels = load_split_sessions(cfg, k).select(ID_COLUMNS)
    table = labels.join(feats, on="user_session", how="inner").sort("user_session").collect()

    leaked = set(table.columns) & (set(ANALYSIS_ONLY_COLUMNS) - {TARGET})
    if leaked:  # defensive: analysis-only columns must never reach a feature table
        raise RuntimeError(f"Analysis-only columns in feature table: {sorted(leaked)}")
    return table


def feature_summary(table: pl.DataFrame, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Per-feature mean / null count on the training split (for checks and the report)."""
    train = table.filter(pl.col("split") == "train")
    out: dict[str, Any] = {"rows": table.height, "train_rows": train.height, "features": {}}
    categorical = set(cfg["features"]["categorical"])
    for f in feature_list(cfg):
        col = train[f]
        info: dict[str, Any] = {"nulls": col.null_count()}
        if f in categorical:
            info["n_unique"] = col.n_unique()
        else:
            info["mean"] = float(col.mean()) if col.len() else None
            info["max"] = float(col.max()) if col.len() else None
        out["features"][f] = info
    return out


def run(cfg: Mapping[str, Any]) -> dict[int, Path]:
    """Build and write ``features.parquet`` (+ summary) for every K."""
    written: dict[int, Path] = {}
    for k in cfg["sessionize"]["prefix_k"]:
        table = build_feature_table(cfg, k)
        out_dir = processed_dir_for_k(cfg, k)
        path = out_dir / "features.parquet"
        table.write_parquet(path)
        (out_dir / "features_summary.json").write_text(
            json.dumps({"k": k, **feature_summary(table, cfg)}, indent=2)
        )
        written[k] = path
    return written


def load_feature_table(cfg: Mapping[str, Any], k: int, split: str | None = None) -> pl.DataFrame:
    """Read ``features.parquet`` for K, optionally one split."""
    path = processed_dir_for_k(cfg, k) / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m src.features.build_features` first.")
    df = pl.read_parquet(path)
    return df if split is None else df.filter(pl.col("split") == split)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build G1–G5 prefix features for every K.")
    parser.add_argument("--mode", choices=["synthetic", "dev", "full"], help="override data.mode")
    args = parser.parse_args()
    cfg = load_config()
    if args.mode:
        cfg = apply_overrides(cfg, {"data.mode": args.mode})
    for k, path in run(cfg).items():
        df = pl.read_parquet(path)
        print(f"K={k}: {df.height:,} sessions x {len(feature_list(cfg))} features -> {path}")


if __name__ == "__main__":
    _main()
