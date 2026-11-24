"""Advanced ML extensions: conformal sets, stacking, embeddings, retraining, Transformer, tuning (steps 12-16)."""

import importlib.util
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from src.config import PROJECT_ROOT
from src.evaluation.conformal import MondrianConformal, weekly_coverage
from src.evaluation.stacking import error_overlap, fit_meta, logit, mean_rank
from src.features.embeddings import item2vec, session_features
from src.serving.results import ResultStore
from tests.test_product_scripts import _run
from tests.test_serving import trained  # noqa: F401  (module-scoped fixture: synthetic data + models)

pytest.importorskip("torch")


def _scores(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(int)
    s = np.clip(0.35 + 0.25 * y + rng.normal(0, 0.15, n), 0.001, 0.999)
    return y, s


def test_conformal_coverage_holds_per_class():
    y, s = _scores()
    yc, sc = _scores(seed=1)
    for alpha in (0.05, 0.1, 0.2):
        cp = MondrianConformal.fit(yc, sc, alpha)
        r = cp.evaluate(y, s)
        assert r["coverage_buyers"] >= 1 - alpha - 0.03 and r["coverage_non_buyers"] >= 1 - alpha - 0.01
        assert abs(r["share_confident_buyer"] + r["share_confident_non_buyer"] + r["share_unsure"] + r["share_empty"] - 1) < 1e-9
    tight, loose = MondrianConformal.fit(yc, sc, 0.05).evaluate(y, s), MondrianConformal.fit(yc, sc, 0.2).evaluate(y, s)
    assert tight["avg_set_size"] >= loose["avg_set_size"]             # stronger guarantee -> bigger sets
    w = weekly_coverage(MondrianConformal.fit(yc, sc, 0.1), y, s, np.repeat(np.arange(4), len(y) // 4))
    assert w.height == 4 and w["coverage"].min() > 0.8


def test_stacking_helpers():
    y, s1 = _scores()
    _, s2 = _scores(seed=3)
    X = np.column_stack([logit(s1), logit(s2)])
    meta, C, cv = fit_meta(X, y, blocks=3, Cs=(0.1, 1.0))
    assert C in (0.1, 1.0) and set(cv) == {0.1, 1.0} and meta.coef_.shape == (1, 2)
    assert meta.coef_[0][0] > 0                                         # the informative model gets weight
    r = mean_rank(np.column_stack([s1, s2]))
    assert r.shape == y.shape and 0 < r.min() and r.max() <= 1
    ov = error_overlap(y, {"a": s1 > 0.5, "b": s1 > 0.5, "c": s2 > 0.5})
    ab = next(o for o in ov if (o["model_a"], o["model_b"]) == ("a", "b"))
    assert ab["agreement"] == 1.0 and ab["missed_by_both"] == 1.0


def test_item2vec_puts_co_viewed_products_together():
    t0 = datetime(2019, 10, 1, tzinfo=timezone.utc)
    rows = []
    rng = np.random.default_rng(0)
    for sess in range(600):                     # two "aisles": products 1-5 and 6-10 are viewed together
        aisle = [1, 2, 3, 4, 5] if sess % 2 else [6, 7, 8, 9, 10]
        for i, p in enumerate(rng.choice(aisle, 6)):
            rows.append({"user_session": f"s{sess}", "event_time": t0 + timedelta(minutes=sess, seconds=i),
                         "product_id": int(p)})
    vec = item2vec(pl.DataFrame(rows).lazy(), t0 + timedelta(days=30), dim=4, min_count=5)
    v = {r[0]: np.array(r[1:]) for r in vec.iter_rows()}
    assert v[1] @ v[2] > v[1] @ v[7]
    prefix = pl.DataFrame({"user_session": ["a", "a", "b", "b"], "pos": [1, 2, 1, 2], "product_id": [1, 2, 1, 7]})
    sf = session_features(prefix, vec, ["product_id"], "i2v_")
    coh = dict(zip(sf["user_session"], sf["i2v_coherence"]))
    assert coh["a"] > coh["b"] and set(sf.columns) >= {"i2v_0", "i2v_known"}


def test_retraining_policies():
    spec = importlib.util.spec_from_file_location("drift", PROJECT_ROOT / "scripts" / "13_drift_retraining.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    months = ["10", "11", "12", "01", "02"]
    assert mod.training_months("frozen", months, 4) == ["10", "11", "12"]
    assert mod.training_months("expanding", months, 4) == ["10", "11", "12", "01"]
    assert mod.training_months("window_1", months, 4) == ["01"]
    assert mod.training_months("window_3", months, 3) == ["10", "11", "12"]


def test_conformal_stacking_transformer_and_tuning_scripts(trained):  # noqa: F811
    store = ResultStore(trained)
    _run("12_conformal_stacking.py", trained, ["--n-resamples", "20"])
    conf = store.csv("conformal")
    assert conf is not None and {"coverage", "share_unsure"} <= set(conf.columns)
    assert set(store.csv("stacking")["kind"].to_list()) >= {"single", "average"}
    assert store.csv("stacking_tests")["p_holm"].is_not_null().all()

    fast = ["--set", "models.transformer.max_epochs=2", "--set", "models.transformer.d_model=16"]
    _run("16_transformer.py", trained, ["--ks", "5", "--seeds", "42", "--n-resamples", "20", *fast])
    tr = store.csv("transformer")
    assert set(tr["model"].to_list()) == {"transformer", "gru"}
    att = store.csv("transformer_attention")
    assert att.height == 5 and abs(att["attention"].sum() - 1) < 1e-3

    _run("15_tune.py", trained, ["--models", "lightgbm", "--trials", "3", "--k", "5", "--n-resamples", "20",
                                 "--set", "models.lightgbm.n_estimators=50"])
    tu = store.csv("tuning")
    assert tu.height == 1 and tu["tuned_test_mean"][0] > 0
    assert store.csv("tuning_trials").height == 3


def test_embeddings_script(trained):  # noqa: F811
    _run("14_embeddings.py", trained, ["--dim", "4", "--min-count", "3", "--n-resamples", "20", "--ks", "5",
                                       "--set", "models.lightgbm.n_estimators=50"])
    ab = ResultStore(trained).csv("embeddings_ablation")
    assert {"base", "item2vec"} <= set(ab["feature_set"].to_list())


def test_user_history_script(trained):  # noqa: F811
    _run("17_user_history.py", trained, ["--n-resamples", "20", "--ks", "5",
                                         "--set", "models.lightgbm.n_estimators=50"])
    s = ResultStore(trained)
    assert set(s.csv("user_history_ablation")["feature_set"].to_list()) == {"base", "history"}
    assert set(s.csv("user_history_segments")["segment"].to_list()) <= {"new", "returning"}
    assert s.json("user_history_summary")["window_days"] == 30


def test_business_value_script(trained):  # noqa: F811
    _run("18_business_value.py", trained, ["--n-resamples", "50", "--ks", "5"])
    s = ResultStore(trained)
    bv = s.csv("business_value")
    assert bv["best"].sum() == 1 and "gain_vs_blanket" in bv.columns
    assert s.csv("business_value_sensitivity").height == 25
    assert "offer_to_everyone" in s.csv("business_value_tests")["vs"].to_list()


def test_segments_script(trained):  # noqa: F811
    _run("19_segments.py", trained, ["--n-resamples", "20", "--ks", "5", "--min-size", "20"])
    s = ResultStore(trained)
    m = s.csv("segment_metrics")
    assert {"category", "price", "cart"} <= set(m["dimension"].to_list())
    assert s.csv("segment_disparity").height > 0
