"""Shop-side aggregates for the dashboard: per day, and per product and month.

Both are computed once from the cleaned events (``data/interim/events_<mode>.parquet``)
and cached:

* ``daily_profile_<mode>.csv`` (small) next to the result tables;
* ``product_month_<mode>.parquet`` (one row per product and month) next to the events.

Category labels: REES46 switched to a new category tree in December 2019 (new
``category_id``s whose ``category_code`` text no longer matches the products, e.g.
iPhones under ``construction.tools.light``). Each product therefore gets one label
for all months on the Oct/Nov tree (``src/data/categories.py``). Events cleaned with
``harmonise_categories`` already carry these labels; mapping again is then a no-op.

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


def product_month_path(cfg: Mapping[str, Any]) -> Path:
    return resolve_path(cfg, "interim_dir") / f"product_month_{cfg['data']['mode']}.parquet"


def product_month(cfg: Mapping[str, Any]) -> pl.DataFrame | None:
    """One row per (month, product): category, brand, price and funnel counts."""
    path = product_month_path(cfg)
    if path.exists():
        return pl.read_parquet(path)
    ev = _events(cfg)
    if ev is None:
        from src.serving.results import demo_dir

        demo = demo_dir(cfg) / "product_month.parquet"        # hosted demo: no raw events
        return pl.read_parquet(demo) if demo.exists() else None
    from src.data.categories import product_categories

    c = cfg["data"]["cleaning"]
    h = c.get("harmonise_categories") or {}
    cats = product_categories(ev, h.get("tree_change", "2019-12-01"), h.get("min_share", 0.6),
                              h.get("min_products", 3), c["missing_category_token"])
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
