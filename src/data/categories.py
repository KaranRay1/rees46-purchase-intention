"""Harmonise REES46 category labels across the December 2019 category-tree change.

From 1 December 2019 the REES46 log uses a new category tree: products get new
``category_id``s whose ``category_code`` text no longer describes them (iPhones appear
as ``construction.tools.light``, Apple headphones as ``sport.bicycle``). Oct/Nov 2019
labels are correct.

:func:`product_categories` gives every product one label on the Oct/Nov tree:

1. its own most frequent Oct/Nov label, when it was seen then with a known label;
2. otherwise, through its most frequent ``category_id`` after the change:
   * an id that already existed in Oct/Nov keeps that id's Oct/Nov label;
   * a new id takes the label most of its products had in Oct/Nov, if at least
     ``min_products`` products voted and ``min_share`` of them agree;
3. otherwise the missing-category token.

Only the label text changes; ``category_id`` stays as logged. The mapping uses product
identity, never purchases or targets, so it cannot leak the label.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

import polars as pl


def _known(unknown: str) -> pl.Expr:
    return pl.col("category_code").is_not_null() & (pl.col("category_code") != unknown)


def product_categories(
    events: pl.LazyFrame,
    tree_change: str = "2019-12-01",
    min_share: float = 0.6,
    min_products: int = 3,
    unknown: str = "unknown",
) -> pl.DataFrame:
    """``product_id -> category_code`` on the pre-change tree for every product in ``events``."""
    cut = datetime.fromisoformat(tree_change).replace(tzinfo=timezone.utc)
    pc = (events.with_columns(early=pl.col("event_time") < cut)
          .group_by("product_id", "early", "category_id", "category_code").agg(n=pl.len())
          .collect(engine="streaming"))
    early = pc.filter(pl.col("early") & _known(unknown))
    own = (early.sort("n", descending=True).group_by("product_id")
           .agg(own=pl.col("category_code").first()))
    old_ids = (early.group_by("category_id", "category_code").agg(pl.col("n").sum())
               .sort("n", descending=True).group_by("category_id").agg(by_id=pl.col("category_code").first()))
    late_id = (pc.filter(~pl.col("early")).sort("n", descending=True)
               .group_by("product_id").agg(category_id=pl.col("category_id").first()))
    votes = (late_id.join(own, on="product_id").group_by("category_id", "own").agg(n=pl.len())
             .sort("n", descending=True).group_by("category_id")
             .agg(voted=pl.col("own").first(), top=pl.col("n").first(), total=pl.col("n").sum())
             .filter((pl.col("total") >= min_products) & (pl.col("top") / pl.col("total") >= min_share))
             .select("category_id", "voted"))
    return (pc.select("product_id").unique()
            .join(own, on="product_id", how="left")
            .join(late_id, on="product_id", how="left")
            .join(old_ids, on="category_id", how="left")
            .join(votes, on="category_id", how="left")
            .select("product_id", category_code=pl.coalesce("own", "by_id", "voted", pl.lit(unknown))))


def harmonise(lf: pl.LazyFrame, cfg: Mapping[str, Any]) -> tuple[pl.LazyFrame, pl.DataFrame | None]:
    """Replace ``category_code`` with the harmonised label when enabled and the data spans the change.

    Returns (events, mapping or None). The mapping is computed eagerly (one streaming pass).
    """
    rules = cfg["data"]["cleaning"]
    opts = rules.get("harmonise_categories") or {}
    if not opts.get("enabled", False):
        return lf, None
    cut = datetime.fromisoformat(opts.get("tree_change", "2019-12-01")).replace(tzinfo=timezone.utc)
    spans = lf.select((pl.col("event_time").max() >= cut) & (pl.col("event_time").min() < cut)).collect().item()
    if not spans:
        return lf, None
    mapping = product_categories(lf, opts.get("tree_change", "2019-12-01"), opts.get("min_share", 0.6),
                                 opts.get("min_products", 3), rules["missing_category_token"])
    out = lf.drop("category_code").join(mapping.lazy(), on="product_id", how="left")
    return out.with_columns(pl.col("category_code").fill_null(rules["missing_category_token"])), mapping
