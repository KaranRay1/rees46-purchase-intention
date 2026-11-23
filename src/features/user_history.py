"""User-level history features (group G6): what did this user do in earlier sessions?

The G1-G5 features describe only the current session prefix. A returning user who bought
last week is a different prospect from a first-time visitor, so G6 summarises the user's
**earlier sessions in a trailing window** (default 30 days).

Leakage rules
-------------
* Only sessions that **ended before the current session started** count. Overlapping
  (parallel) sessions and the current session itself are excluded, so everything in G6
  was already known when the visit began, well before the prediction time (event K).
* History is read from the **full** cleaned event log (all sessions), not from the 10 %
  modelling sample: sampling is by session, so the sample would miss most of a user's
  earlier visits. Using all sessions is not leakage, because the time rule above still holds.
* A fixed trailing window keeps the features comparable over time. With all-time counts,
  test-period users (April) would have six months of history and train-period users far
  less, which is a drift the model would learn by accident. Only the first 30 days of data
  (early October 2019) have a shortened window.

Features (all numeric, prefix ``uh_``)
--------------------------------------
uh_prior_sessions            earlier sessions in the window
uh_prior_purchase_sessions   ... of which ended with a purchase
uh_prior_cart_sessions       ... of which had a cart event
uh_prior_purchase_rate       purchase sessions / (sessions + 1)   (smoothed, 0 for new users)
uh_days_since_last_session   days since the last earlier session ended, capped at the window
                             (window + 1 when there is none in the window)
uh_days_since_last_purchase  same for the last purchase session
uh_is_returning              1 if there is any earlier session in the window
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

HISTORY_FEATURES: list[str] = [
    "uh_prior_sessions",
    "uh_prior_purchase_sessions",
    "uh_prior_cart_sessions",
    "uh_prior_purchase_rate",
    "uh_days_since_last_session",
    "uh_days_since_last_purchase",
    "uh_is_returning",
]
_CUM = ["cum_sessions", "cum_purchase", "cum_cart"]


def session_table(events: pl.LazyFrame) -> pl.LazyFrame:
    """One row per session: user, start, end and whether it had a cart / purchase."""
    etype = pl.col("event_type").cast(pl.String)
    return events.group_by("user_session").agg(
        user_id=pl.col("user_id").first(),
        session_start=pl.col("event_time").min(),
        session_end=pl.col("event_time").max(),
        has_purchase=(etype == "purchase").any().cast(pl.Int32),
        has_cart=(etype == "cart").any().cast(pl.Int32),
    )


def _cumulative(sessions: pl.DataFrame) -> pl.DataFrame:
    """Running totals per user in order of session end (the as-of lookup table)."""
    s = sessions.select("user_id", "session_end", "has_purchase", "has_cart").sort("user_id", "session_end")
    s = s.with_columns(
        cum_sessions=pl.int_range(1, pl.len() + 1, dtype=pl.Int64).over("user_id"),
        cum_purchase=pl.col("has_purchase").cum_sum().over("user_id").cast(pl.Int64),
        cum_cart=pl.col("has_cart").cum_sum().over("user_id").cast(pl.Int64),
        last_end=pl.col("session_end"),
        last_purchase_end=pl.when(pl.col("has_purchase") == 1).then(pl.col("session_end"))
        .forward_fill().over("user_id"),
    )
    return s.drop("has_purchase", "has_cart").sort("session_end")


def history_features(targets: pl.DataFrame, sessions: pl.DataFrame, window_days: int = 30) -> pl.DataFrame:
    """G6 features for each target session.

    Args:
        targets: ``user_session``, ``user_id``, ``session_start`` of the sessions to describe.
        sessions: output of ``session_table`` (collected), covering all sessions of these users.
        window_days: trailing window length.

    Returns:
        ``user_session`` plus ``HISTORY_FEATURES``.
    """
    cum = _cumulative(sessions)
    tick = timedelta(microseconds=1)
    window = timedelta(days=window_days)
    t = targets.select("user_session", "user_id", "session_start").with_columns(
        upto=pl.col("session_start") - tick,                  # strictly before the visit starts
        upto_old=pl.col("session_start") - tick - window,
    )
    now = t.sort("upto").join_asof(
        cum.rename({"session_end": "upto"}), on="upto", by="user_id", strategy="backward", check_sortedness=False
    )
    old = t.sort("upto_old").join_asof(
        cum.select("user_id", "session_end", *_CUM).rename({"session_end": "upto_old", **{c: f"{c}_old" for c in _CUM}}),
        on="upto_old", by="user_id", strategy="backward", check_sortedness=False,
    ).select("user_session", *[f"{c}_old" for c in _CUM])
    df = now.join(old, on="user_session", how="left")

    cap = float(window_days + 1)

    def days_since(col: str) -> pl.Expr:
        d = (pl.col("session_start") - pl.col(col)).dt.total_seconds() / 86400.0
        return pl.when(d.is_null() | (d > window_days)).then(cap).otherwise(d)

    n = {c: (pl.col(c).fill_null(0) - pl.col(f"{c}_old").fill_null(0)) for c in _CUM}
    out = df.select(
        "user_session",
        uh_prior_sessions=n["cum_sessions"].cast(pl.Float64),
        uh_prior_purchase_sessions=n["cum_purchase"].cast(pl.Float64),
        uh_prior_cart_sessions=n["cum_cart"].cast(pl.Float64),
        uh_prior_purchase_rate=n["cum_purchase"] / (n["cum_sessions"] + 1.0),
        uh_days_since_last_session=days_since("last_end"),
        uh_days_since_last_purchase=days_since("last_purchase_end"),
        uh_is_returning=(n["cum_sessions"] > 0).cast(pl.Float64),
    )
    return targets.select("user_session").join(out, on="user_session", how="left")
