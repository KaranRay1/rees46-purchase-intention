"""Category harmonisation across the REES46 December 2019 tree change."""

from datetime import datetime, timezone

import polars as pl

from src.config import load_config
from src.data.categories import harmonise, product_categories

OCT = datetime(2019, 11, 20, tzinfo=timezone.utc)
DEC = datetime(2019, 12, 10, tzinfo=timezone.utc)


def _events(rows):
    return pl.DataFrame(rows, schema=["event_time", "product_id", "category_id", "category_code"], orient="row").lazy()


def test_products_keep_their_old_label_and_new_ids_are_bridged():
    rows = [
        # products 1-3: smartphones in Nov (id 10), moved to new id 99 with a wrong label in Dec
        *[(OCT, p, 10, "electronics.smartphone") for p in (1, 2, 3)],
        *[(DEC, p, 99, "construction.tools.light") for p in (1, 2, 3)],
        (DEC, 4, 99, "construction.tools.light"),          # new product on new id -> bridged
        (DEC, 5, 10, "electronics.smartphone"),             # new product on an old id -> old id's label
        (DEC, 6, 77, "sport.bicycle"),                      # new id nobody voted for -> unknown
        (OCT, 7, 11, "unknown"), (DEC, 7, 99, "construction.tools.light"),  # unknown in Nov -> bridged
    ]
    m = dict(product_categories(_events(rows)).iter_rows())
    assert m[1] == m[2] == m[3] == "electronics.smartphone"
    assert m[4] == "electronics.smartphone"
    assert m[5] == "electronics.smartphone"
    assert m[6] == "unknown"
    assert m[7] == "electronics.smartphone"


def test_weak_votes_are_not_used():
    rows = [(OCT, 1, 10, "a.x"), (OCT, 2, 11, "b.y"), (OCT, 3, 12, "c.z"),
            *[(DEC, p, 99, "wrong") for p in (1, 2, 3, 4)]]
    m = dict(product_categories(_events(rows), min_share=0.6).iter_rows())
    assert m[4] == "unknown"                                # 1/3 agreement is below 60 %


def test_harmonise_is_a_no_op_inside_one_tree():
    cfg = load_config(overrides=["data.mode=synthetic"])
    lf = _events([(OCT, 1, 10, "electronics.smartphone")])
    out, mapping = harmonise(lf, cfg)
    assert mapping is None and out.collect()["category_code"].to_list() == ["electronics.smartphone"]


def test_harmonise_can_be_switched_off():
    cfg = load_config(overrides=["data.mode=synthetic", "data.cleaning.harmonise_categories.enabled=false"])
    _, mapping = harmonise(_events([(OCT, 1, 10, "a.b"), (DEC, 1, 99, "c.d")]), cfg)
    assert mapping is None
