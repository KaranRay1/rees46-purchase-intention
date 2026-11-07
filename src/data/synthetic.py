"""Synthetic REES46-like clickstream for unit tests and quick end-to-end runs.

The output has exactly the REES46 schema (event_time, event_type, product_id,
category_id, category_code, brand, price, user_id, user_session), so every
downstream module can run without the real dataset.

How sessions are generated
--------------------------
* Each session gets a latent **intent** in [0, 1] (Beta-distributed). Buyers
  (``purchase_rate`` of sessions) draw from a higher-intent Beta, so the
  signal is learnable but noisy: some non-buyers cart a lot, some buyers
  barely cart at all.
* Intent raises the chance of ``cart`` events, lowers ``remove_from_cart``,
  and buyers get longer sessions. Buyers end with 1-2 ``purchase`` events of
  products already in their cart, so most purchases fall *after* the first K
  events (the target the models predict); a few short buyer sessions purchase
  early, which exercises the leakage filter in ``sessionize.py``.
* Products come from a catalogue with REES46-style dotted ``category_code``
  values (with missing codes/brands) and category-specific log-normal prices.
* Session starts follow a daily traffic curve over a date range, so the
  chronological split has something to split on.
* ``inject_noise=True`` adds duplicate rows, non-positive prices and missing
  session ids, so the cleaning rules in ``loader.py`` can be tested.

Usage::

    python -m src.data.synthetic --n-sessions 5000 --out data/raw/synthetic.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl

REES46_COLUMNS: list[str] = [
    "event_time", "event_type", "product_id", "category_id", "category_code",
    "brand", "price", "user_id", "user_session",
]
REES46_TIME_FORMAT = "%Y-%m-%d %H:%M:%S UTC"

# (category_code, median price) — a small REES46-like category tree.
_CATEGORIES: list[tuple[str, float]] = [
    ("electronics.smartphone", 300.0),
    ("electronics.audio.headphone", 60.0),
    ("electronics.video.tv", 450.0),
    ("electronics.clocks", 150.0),
    ("computers.notebook", 700.0),
    ("computers.desktop", 550.0),
    ("computers.peripherals.mouse", 25.0),
    ("appliances.kitchen.refrigerators", 500.0),
    ("appliances.kitchen.washer", 380.0),
    ("appliances.environment.vacuum", 150.0),
    ("apparel.shoes", 80.0),
    ("apparel.shoes.keds", 60.0),
    ("furniture.living_room.sofa", 400.0),
    ("furniture.bedroom.bed", 350.0),
    ("auto.accessories.player", 90.0),
    ("construction.tools.drill", 110.0),
    ("kids.toys", 30.0),
    ("sport.bicycle", 320.0),
    ("accessories.bag", 45.0),
    ("country_yard.lawn_mower", 260.0),
    ("medicine.tools.tonometer", 40.0),
    ("stationery.cartrige", 20.0),
]
_BRANDS: list[str] = [
    "samsung", "apple", "xiaomi", "huawei", "lenovo", "acer", "hp", "sony",
    "lg", "bosch", "philips", "nike", "adidas", "ikea", "makita", "lego",
    "indesit", "beko", "jbl", "asus", "canon", "redmond", "polaris", "vitek",
]

# Relative traffic per hour of day (UTC) — busy evenings, quiet nights.
_HOURLY_TRAFFIC = np.array(
    [2, 1, 1, 1, 1, 2, 4, 6, 8, 9, 10, 10, 10, 10, 10, 10, 11, 12, 13, 14, 13, 10, 6, 3],
    dtype=float,
)


def _build_catalogue(rng: np.random.Generator, n_products: int) -> dict[str, np.ndarray]:
    """Products with category, brand and price. ~30% miss category_code, ~15% brand."""
    n_cat = len(_CATEGORIES)
    cat_idx = rng.integers(0, n_cat, size=n_products)
    median_price = np.array([p for _, p in _CATEGORIES])[cat_idx]
    price = np.round(median_price * rng.lognormal(0.0, 0.5, size=n_products), 2)

    codes = np.array([c for c, _ in _CATEGORIES], dtype=object)[cat_idx]
    codes[rng.random(n_products) < 0.30] = None
    brands = np.array(_BRANDS, dtype=object)[rng.integers(0, len(_BRANDS), size=n_products)]
    brands[rng.random(n_products) < 0.15] = None

    # REES46 ids are large integers; keep that look.
    category_ids = 2_053_013_550_000_000_000 + np.arange(n_cat, dtype=np.int64) * 7_919
    return {
        "product_id": 1_000_000 + rng.choice(9_000_000, size=n_products, replace=False),
        "cat_idx": cat_idx,
        "category_id": category_ids[cat_idx],
        "category_code": codes,
        "brand": brands,
        "price": price,
    }


def _session_id(rng: np.random.Generator) -> str:
    """UUID-shaped hex string like REES46 ``user_session``."""
    h = rng.bytes(16).hex()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def generate_synthetic_events(
    n_sessions: int = 1000,
    purchase_rate: float = 0.06,
    seed: int = 42,
    start_date: str = "2019-10-01",
    n_days: int = 31,
    n_products: int = 500,
    max_events: int = 60,
    inject_noise: bool = False,
) -> pl.DataFrame:
    """Generate a REES46-schema event log.

    Args:
        n_sessions: number of sessions to simulate.
        purchase_rate: share of sessions that end in a purchase.
        seed: random seed (same seed -> identical output).
        start_date: first day of the simulated period (UTC, ``YYYY-MM-DD``).
        n_days: length of the period in days.
        n_products: catalogue size.
        max_events: cap on events per session.
        inject_noise: add duplicates, bad prices and null sessions for cleaning tests.

    Returns:
        Polars DataFrame with the REES46 columns, sorted by ``event_time``;
        ``event_time`` is ``Datetime("us", "UTC")``.
    """
    if n_sessions < 1:
        raise ValueError("n_sessions must be >= 1")
    if not 0 <= purchase_rate <= 1:
        raise ValueError("purchase_rate must be in [0, 1]")

    rng = np.random.default_rng(seed)
    cat = _build_catalogue(rng, n_products)
    products_by_cat = [np.flatnonzero(cat["cat_idx"] == c) for c in range(len(_CATEGORIES))]
    users = 500_000_000 + rng.choice(100_000_000, size=max(1, int(n_sessions * 0.6)), replace=False)

    t0 = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc).timestamp()
    hour_p = _HOURLY_TRAFFIC / _HOURLY_TRAFFIC.sum()

    rows: dict[str, list[Any]] = {k: [] for k in ("t", "event_type", "prod", "user_id", "user_session")}

    for _ in range(n_sessions):
        is_buyer = rng.random() < purchase_rate
        intent = rng.beta(4, 3) if is_buyer else rng.beta(1.2, 6)

        n_browse = (3 + rng.geometric(0.09)) if is_buyer else rng.geometric(0.13)
        n_browse = int(min(n_browse, max_events - 2))

        start = (
            t0
            + rng.integers(0, n_days) * 86_400
            + rng.choice(24, p=hour_p) * 3_600
            + rng.integers(0, 3_600)
        )
        user = int(rng.choice(users))
        sid = _session_id(rng)

        # Sessions focus on one category, with occasional cross-category browsing.
        home_cat = int(rng.integers(0, len(_CATEGORIES)))
        p_cart = 0.04 + 0.40 * intent
        p_remove = 0.10 * (1 - intent)

        t = float(start)
        cart: list[int] = []
        current = None
        events: list[tuple[float, str, int]] = []
        for _step in range(n_browse):
            u = rng.random()
            if current is not None and u < p_cart:
                etype = "cart"
                cart.append(current)
            elif cart and u < p_cart + p_remove:
                etype = "remove_from_cart"
                current = cart.pop(int(rng.integers(0, len(cart))))
            else:
                etype = "view"
                pool = products_by_cat[home_cat] if rng.random() < 0.8 else np.arange(n_products)
                if pool.size == 0:
                    pool = np.arange(n_products)
                current = int(rng.choice(pool))
            events.append((t, etype, current))
            t += float(rng.lognormal(np.log(25), 1.0))  # seconds between events

        if is_buyer:
            if not cart:  # buyers who never carted: cart the last viewed item first
                cart.append(current if current is not None else int(rng.integers(0, n_products)))
                events.append((t, "cart", cart[-1]))
                t += float(rng.lognormal(np.log(40), 0.8))
            n_buy = min(len(cart), 1 + int(rng.random() < 0.3))
            for p in rng.choice(cart, size=n_buy, replace=False):
                events.append((t, "purchase", int(p)))
                t += float(rng.lognormal(np.log(5), 0.5))

        for et, etype, p in events:
            rows["t"].append(et)
            rows["event_type"].append(etype)
            rows["prod"].append(p)
            rows["user_id"].append(user)
            rows["user_session"].append(sid)

    prod = np.asarray(rows["prod"], dtype=np.int64)
    df = pl.DataFrame(
        {
            "event_time": (np.asarray(rows["t"]) * 1e6).astype("int64"),
            "event_type": rows["event_type"],
            "product_id": cat["product_id"][prod].astype(np.int64),
            "category_id": cat["category_id"][prod].astype(np.int64),
            "category_code": pl.Series(cat["category_code"][prod].tolist(), dtype=pl.String),
            "brand": pl.Series(cat["brand"][prod].tolist(), dtype=pl.String),
            "price": cat["price"][prod].astype(np.float64),
            "user_id": np.asarray(rows["user_id"], dtype=np.int64),
            "user_session": pl.Series(rows["user_session"], dtype=pl.String),
        }
    ).with_columns(
        pl.col("event_time").cast(pl.Datetime("us")).dt.replace_time_zone("UTC").dt.truncate("1s")
    )

    if inject_noise:
        df = _inject_noise(df, rng)

    return df.select(REES46_COLUMNS).sort("event_time", maintain_order=True)


def _inject_noise(df: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """Add ~1% duplicate rows, ~0.5% non-positive prices, ~0.2% null sessions."""
    n = df.height
    dupes = df[rng.choice(n, size=max(1, n // 100), replace=False).tolist()]
    bad_price = rng.random(n) < 0.005
    null_sess = rng.random(n) < 0.002
    df = df.with_columns(
        pl.when(pl.Series(bad_price)).then(0.0).otherwise(pl.col("price")).alias("price"),
        pl.when(pl.Series(null_sess)).then(None).otherwise(pl.col("user_session")).alias("user_session"),
    )
    return pl.concat([df, dupes])


def generate_from_config(cfg: Mapping[str, Any]) -> pl.DataFrame:
    """Generate events using ``cfg["data"]["synthetic"]`` and the first project seed."""
    syn = cfg["data"]["synthetic"]
    return generate_synthetic_events(
        n_sessions=syn["n_sessions"],
        purchase_rate=syn["purchase_rate"],
        seed=cfg["project"]["seeds"][0],
        start_date=str(syn.get("start_date", "2019-10-01")),
        n_days=syn.get("n_days", 31),
        inject_noise=syn.get("inject_noise", False),
    )


def write_rees46_csv(df: pl.DataFrame, path: str | Path) -> Path:
    """Write events as a CSV in the exact REES46 text format (``... UTC`` timestamps)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.with_columns(pl.col("event_time").dt.strftime(REES46_TIME_FORMAT)).write_csv(path)
    return path


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic REES46-like CSV.")
    parser.add_argument("--n-sessions", type=int, default=1000)
    parser.add_argument("--purchase-rate", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise", action="store_true", help="inject cleaning-test noise")
    parser.add_argument("--out", type=Path, default=Path("data/raw/synthetic.csv"))
    args = parser.parse_args()

    df = generate_synthetic_events(
        n_sessions=args.n_sessions,
        purchase_rate=args.purchase_rate,
        seed=args.seed,
        inject_noise=args.noise,
    )
    out = write_rees46_csv(df, args.out)
    n_sess = df["user_session"].n_unique()
    buy = df.filter(pl.col("event_type") == "purchase")["user_session"].n_unique()
    print(f"Wrote {df.height:,} events / {n_sess:,} sessions ({buy / n_sess:.1%} with purchase) -> {out}")


if __name__ == "__main__":
    _main()
