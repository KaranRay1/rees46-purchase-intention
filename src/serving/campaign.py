"""Campaign planner: what a purchase-intent ranking is worth to a discount campaign.

The ranking comes from real test-period predictions. The marketing inputs (discount,
the extra conversion the offer causes, cost per contact) are assumptions the user sets.

The offer is assumed to work equally on every visitor who would not have bought
(``uplift``), and visitors who would have bought anyway still buy and redeem the
discount. Under that assumption, sending the offer to the *highest*-scoring visitors
is worse than sending it at random: they are the ones who buy anyway. The model's
value is the opposite move, **suppression**: send the offer to everyone except the
top ``share`` of visits by score, so the discount is not given away to sure buyers.

For a set of targeted visits with ``b`` buyers-anyway and ``n`` non-buyers:

``net = n * uplift * aov * (1 - discount) - b * aov * discount - (b + n) * contact_cost``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class Offer:
    aov: float                 # average order value of a buying visit
    discount: float            # share of the order given away, e.g. 0.10
    uplift: float              # share of targeted non-buyers the offer converts, e.g. 0.02
    contact_cost: float = 0.0  # cost per targeted visit


def gain_table(y: np.ndarray, score: np.ndarray, shares: np.ndarray | None = None) -> pl.DataFrame:
    """Buyers captured when targeting the top ``share`` of visits by ``score``.

    Ties are broken by a fixed random order so equal scores (e.g. many zeros) are not
    ordered by how the file happened to be sorted.
    """
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    n = len(y)
    if shares is None:
        shares = np.round(np.arange(0.01, 1.0001, 0.01), 2)
    tie = np.random.default_rng(0).random(n)
    order = np.lexsort((tie, -score))
    cum = np.concatenate([[0], np.cumsum(y[order])])
    sizes = np.clip(np.round(np.asarray(shares) * n).astype(int), 0, n)
    buyers = cum[sizes]
    total = int(y.sum())
    ranked = score[order]
    return pl.DataFrame({
        "share": np.asarray(shares, dtype=float), "targeted": sizes, "buyers": buyers,
        "non_buyers": sizes - buyers,
        "precision": np.where(sizes > 0, buyers / np.maximum(sizes, 1), np.nan),
        "recall": buyers / max(total, 1),
        "score_cutoff": [float(ranked[s - 1]) if s > 0 else float("inf") for s in sizes],
        "n_total": np.full(len(sizes), n), "buyers_total": np.full(len(sizes), total),
    })


def _net(buyers, non, offer: "Offer"):
    extra = non * offer.uplift
    return extra, extra * offer.aov * (1 - offer.discount), buyers * offer.aov * offer.discount + (buyers + non) * offer.contact_cost


def suppression_value(gain: pl.DataFrame, offer: Offer) -> pl.DataFrame:
    """Net value of offering to everyone except the top ``share`` of visits, per row of :func:`gain_table`.

    Adds ``net`` (skip the top by score), ``net_random`` (skip the same number at random)
    and ``net_blanket`` (offer to everyone), plus the counts behind ``net``.
    """
    n_all = int(gain["n_total"][0])
    b_all = int(gain["buyers_total"][0])
    base = b_all / max(n_all, 1)
    out = gain.rename({"targeted": "skipped", "buyers": "skipped_buyers", "non_buyers": "skipped_non_buyers"})
    out = out.with_columns(
        offered=n_all - pl.col("skipped"),
        offered_buyers=b_all - pl.col("skipped_buyers"),
    ).with_columns(offered_non_buyers=pl.col("offered") - pl.col("offered_buyers"))
    extra, revenue, cost = _net(pl.col("offered_buyers"), pl.col("offered_non_buyers"), offer)
    rb = pl.col("offered") * base
    _, r_rev, r_cost = _net(rb, pl.col("offered") - rb, offer)
    _, bl_rev, bl_cost = _net(b_all, n_all - b_all, offer)
    return out.with_columns(
        extra_orders=extra, extra_revenue=revenue, cost=cost, net=revenue - cost,
        net_random=r_rev - r_cost, net_blanket=pl.lit(bl_rev - bl_cost),
    )


def best_share(valued: pl.DataFrame) -> dict:
    """The row with the highest net value (the first one on ties)."""
    return valued.sort("net", descending=True, maintain_order=True).row(0, named=True)
