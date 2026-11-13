"""Shop-side aggregates for the dashboard: per day, and per product and month.

Both are computed once from the cleaned events (``data/interim/events_<mode>.parquet``)
and cached:

* ``daily_profile_<mode>.csv`` (small) next to the result tables;
* ``product_month_<mode>.parquet`` (one row per product and month) next to the events.

Category labels: REES46 switched to a new category tree in December 2019 (new
``category_id``s whose ``category_code`` text no longer matches the products, e.g.
iPhones under ``construction.tools.light``). Each product therefore gets one label
for all months: its Oct/Nov label when it had one, otherwise the label most of the
products in its new category had in Oct/Nov (see :func:`product_categories`).

All counts come from the cleaned events. In ``full`` mode those events are a random
sample of whole sessions, so counts are the sample's, while rates (conversion,
cart rate) are representative of the whole shop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import polars as pl

from src.config import resolve_path
from src.serving.results import ResultStore

_ET = pl.col("event_type")


def _events(cfg: Mapping[str, Any]) -> pl.LazyFrame | None:
    from src.data.loader import cleaned_events_path

    src = cleaned_events_path(cfg)
    return pl.scan_parquet(src) if src.exists() else None


def daily_profile(cfg: Mapping[str, Any]) -> pl.DataFrame | None:
    """One row per calendar day: events, visits, buying visits, purchases, revenue, active products."""
    store = ResultStore(cfg)
    cached = store.csv("daily_profile")
    if cached is not None:
        return cached.with_columns(pl.col("date").str.to_date())
    ev = _events(cfg)
    if ev is None:
        return None
    out = (
        ev.with_columns(date=pl.col("event_time").dt.date())
        .group_by("date")
        .agg(
            events=pl.len(),
            sessions=pl.col("user_session").n_unique(),
            buying_sessions=pl.col("user_session").filter(_ET == "purchase").n_unique(),
            views=(_ET == "view").sum(),
            carts=(_ET == "cart").sum(),
            purchases=(_ET == "purchase").sum(),
            revenue=pl.col("price").filter(_ET == "purchase").sum(),
            products=pl.col("product_id").n_unique(),
        )
        .sort("date")
        .collect(engine="streaming")
        .with_columns(month=pl.col("date").dt.strftime("%Y-%m"))
    )
    store.tables_dir.mkdir(parents=True, exist_ok=True)
    out.write_csv(store.path("daily_profile", "csv"))
    return out


TREE_CHANGE = "2019-12-01"     # first day of the new REES46 category tree
_KNOWN = pl.col("category_code").is_not_null() & (pl.col("category_code") != "unknown")


def product_categories(ev: pl.LazyFrame, min_share: float = 0.6, min_products: int = 3) -> pl.DataFrame:
    """``product_id -> category_code`` on the Oct/Nov 2019 tree, for products in any month.

    A product seen before December keeps its own (most frequent) Oct/Nov label. A
    product seen only later is labelled through its new ``category_id``: products
    seen in both periods vote for that id's old label, which is used when at least
    ``min_products`` of them voted and ``min_share`` agree. Otherwise ``unknown``.
    """
    from datetime import datetime, timezone

    cut = datetime.fromisoformat(TREE_CHANGE).replace(tzinfo=timezone.utc)
    pc = (ev.with_columns(early=pl.col("event_time") < cut)
          .group_by("product_id", "early", "category_id", "category_code").agg(n=pl.len())
          .collect(engine="streaming"))
    old = (pc.filter(pl.col("early") & _KNOWN).sort("n", descending=True)
           .group_by("product_id").agg(old=pl.col("category_code").first()))
    new_id = (pc.filter(~pl.col("early")).sort("n", descending=True)
              .group_by("product_id").agg(cid=pl.col("category_id").first()))
    votes = new_id.join(old, on="product_id").group_by("cid", "old").agg(n=pl.len())
    bridge = (votes.sort("n", descending=True).group_by("cid")
              .agg(bridged=pl.col("old").first(), top=pl.col("n").first(), total=pl.col("n").sum())
              .filter((pl.col("total") >= min_products) & (pl.col("top") / pl.col("total") >= min_share))
              .select("cid", "bridged"))
    return (pc.select("product_id").unique()
            .join(old, on="product_id", how="left")
            .join(new_id, on="product_id", how="left")
            .join(bridge, on="cid", how="left")
            .select("product_id", category_code=pl.coalesce("old", "bridged", pl.lit("unknown"))))


def product_month_path(cfg: Mapping[str, Any]) -> Path:
    return resolve_path(cfg, "interim_dir") / f"product_month_{cfg['data']['mode']}.parquet"


def product_month(cfg: Mapping[str, Any]) -> pl.DataFrame | None:
    """One row per (month, product): category, brand, price and funnel counts."""
    path = product_month_path(cfg)
    if path.exists():
        return pl.read_parquet(path)
    ev = _events(cfg)
    if ev is None:
        return None
    cats = product_categories(ev)
    out = (
        ev.with_columns(month=pl.col("event_time").dt.strftime("%Y-%m"))
        .group_by("month", "product_id")
        .agg(
            brand=pl.col("brand").drop_nulls().first(),
            price=pl.col("price").median(),
            views=(_ET == "view").sum().cast(pl.Int64),
            carts=(_ET == "cart").sum().cast(pl.Int64),
            purchases=(_ET == "purchase").sum().cast(pl.Int64),
            sessions=pl.col("user_session").n_unique().cast(pl.Int64),
            revenue=pl.col("price").filter(_ET == "purchase").sum(),
        )
        .collect(engine="streaming")
        .join(cats, on="product_id", how="left")
        .with_columns(
            pl.col("category_code").fill_null("unknown"),
            pl.col("brand").fill_null("unknown"),
        )
        .with_columns(department=pl.col("category_code").str.split(".").list.first())
        .sort("month", "product_id")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path)
    return out


def products(pm: pl.DataFrame) -> pl.DataFrame:
    """Collapse the (month, product) rows to one row per product with funnel rates."""
    return with_rates(
        pm.group_by("product_id")
        .agg(
            category_code=pl.col("category_code").first(),
            department=pl.col("department").first(),
            brand=pl.col("brand").mode().first(),
            price=pl.col("price").median(),
            views=pl.col("views").sum(),
            carts=pl.col("carts").sum(),
            purchases=pl.col("purchases").sum(),
            sessions=pl.col("sessions").sum(),
            revenue=pl.col("revenue").sum(),
            months=pl.col("month").n_unique(),
        )
    )


def with_rates(df: pl.DataFrame) -> pl.DataFrame:
    """Add view→cart and view→purchase rates (null when a product has no views)."""
    v = pl.when(pl.col("views") > 0).then(pl.col("views"))
    return df.with_columns(cart_rate=pl.col("carts") / v, conversion=pl.col("purchases") / v)
