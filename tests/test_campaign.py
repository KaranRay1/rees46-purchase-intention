"""Campaign planner maths (src/serving/campaign.py)."""

import numpy as np
import polars as pl
import pytest

from src.serving.campaign import Offer, best_share, gain_table, suppression_value


def test_gain_table_counts_buyers_in_the_top_share():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([.9, .8, .7, .1, .1, .1, .1, .1, .1, .1])
    g = gain_table(y, s, np.array([0.1, 0.3, 1.0]))
    assert g["targeted"].to_list() == [1, 3, 10]
    assert g["buyers"].to_list() == [1, 2, 2]
    assert g["precision"][1] == pytest.approx(2 / 3)
    assert g["recall"].to_list() == [0.5, 1.0, 1.0]


def test_suppression_formula():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([.9, .8, .2, .2, .2, .2, .2, .2, .2, .2])
    offer = Offer(aov=100, discount=0.1, uplift=0.1, contact_cost=1)
    v = suppression_value(gain_table(y, s, np.array([0.0, 0.2, 1.0])), offer)
    part = suppression_value(gain_table(y, s, np.array([0.0, 0.2])), offer)   # totals do not depend on the grid
    assert part["net"].to_list() == v["net"].to_list()[:2]
    blanket = 8 * 0.1 * 90 - (2 * 10 + 10 * 1)          # everyone: 72 - 30 = 42
    assert v["net_blanket"][0] == pytest.approx(blanket)
    assert v["net"][0] == pytest.approx(blanket)         # skipping nobody = blanket
    assert v["net"][1] == pytest.approx(72 - 8)          # skip both sure buyers: no wasted discount
    assert v["net"][2] == 0                              # skip everyone
    assert v["net_random"][1] == pytest.approx(0.8 * blanket)


def test_a_good_ranking_beats_random_suppression():
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.1).astype(int)
    s = y * 0.5 + rng.random(5000) * 0.6
    v = suppression_value(gain_table(y, s), Offer(aov=300, discount=0.1, uplift=0.03, contact_cost=0.05))
    assert (v["net"] >= v["net_random"] - 1e-6).all()
    assert best_share(v)["net"] == pytest.approx(v["net"].max())
    assert isinstance(v, pl.DataFrame)
