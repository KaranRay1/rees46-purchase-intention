"""Profit comparison: honest cutoffs, tie handling and paired bootstrap."""

import numpy as np
import pytest

from src.evaluation.business_value import _parts, choose_cutoff, compare_models, net_value, paired_diff, split_halves
from src.serving.campaign import Offer

OFFER = Offer(aov=100.0, discount=0.10, uplift=0.02)


def _data(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(int)
    good = y * 0.6 + rng.random(n) * 0.5          # informative score
    return y, good, rng.random(n)                 # and a random one


def test_halves_are_disjoint_and_complete():
    sp = split_halves(101, seed=3)
    assert len(np.intersect1d(sp.choose, sp.measure)) == 0
    assert len(sp.choose) + len(sp.measure) == 101


def test_net_formula():
    # 10 offered buyers, 90 offered non-buyers: 90*0.02*100*0.9 - 10*100*0.1 = 162 - 100
    assert net_value(10, 90, OFFER) == pytest.approx(62.0)


def test_tied_scores_keep_the_chosen_share():
    y = np.r_[np.ones(20), np.zeros(80)].astype(int)
    score = np.r_[np.full(50, 1.0), np.full(50, 0.0)]       # two big ties
    cut, tie = choose_cutoff(y, score, OFFER)
    b, n = _parts(y, score, (cut, tie))
    skipped = len(y) - (b.sum() + n.sum())
    assert 0 < skipped < len(y)                               # never "skip everyone" because of a tie


def test_informative_ranking_earns_more_than_random():
    y, good, rand = _data()
    r = compare_models(y, {"good": good, "random": rand}, OFFER, n_resamples=200, seed=0)
    net = {x["model"]: x["net"] for x in r["rows"]}
    assert net["good"] > net["random"] and net["good"] > r["blanket"]
    t = paired_diff(r["draws"]["good"], r["draws"]["random"], net["good"], net["random"])
    assert t["lower"] > 0 and t["p_value"] < 0.05
    for x in r["rows"]:
        assert x["net_low"] <= x["net"] <= x["net_high"]
