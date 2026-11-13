"""Tests for src/serving (live predictor + result store) and a smoke test of the dashboard pages."""

import importlib.util
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from src.config import PROJECT_ROOT, load_config
from src.serving.predictor import SessionPredictor, available_ks, prepare_events
from src.serving.results import ResultStore, modes_with_results

pytest.importorskip("lightgbm", exc_type=ImportError)
pytest.importorskip("torch")

MODELS = ["rule_based", "logistic_regression", "lightgbm", "gru"]
FAST = ["--set", "models.lightgbm.n_estimators=50", "--set", "models.lightgbm.early_stopping_rounds=10",
        "--set", "models.gru.max_epochs=2"]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name[:-3], PROJECT_ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _overrides(tmp):
    return {
        "paths.interim_dir": tmp / "interim", "paths.processed_dir": tmp / "processed",
        "paths.tables_dir": tmp / "tables", "paths.runs_dir": tmp / "runs", "paths.figures_dir": tmp / "figures",
        "paths.all_runs_csv": tmp / "tables" / "all_runs.csv", "data.synthetic.n_sessions": 1500,
    }


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Synthetic data + four quickly trained models in a temporary folder; returns the config."""
    tmp = tmp_path_factory.mktemp("serving")
    ov = _overrides(tmp)
    args = ["--mode", "synthetic", *FAST]
    for k, v in ov.items():
        args += ["--set", f"{k}={v}"]
    _script("01_prepare_data.py").main(args + ["--no-report"])
    _script("02_build_features.py").main(args)
    _script("03_run_experiments.py").main(args + ["--seeds", "42", "--models", *MODELS, "--n-resamples", "20"])
    return load_config(overrides=[f"{k}={v}" for k, v in ov.items()] + ["data.mode=synthetic"])


def _events(k: int, carts: tuple[int, ...] = (2,)) -> pl.DataFrame:
    t0 = datetime(2019, 10, 28, 14, tzinfo=timezone.utc)
    return pl.DataFrame({
        "event_time": [t0 + timedelta(seconds=30 * i) for i in range(k)],
        "event_type": ["cart" if i in carts else "view" for i in range(k)],
        "product_id": [1 + i % 2 for i in range(k)],
        "category_id": [10] * k,
        "category_code": ["electronics.smartphone"] * k,
        "brand": ["apple"] * k,
        "price": [400.0 + i for i in range(k)],
    })


def test_predictor_scores_every_model(trained):
    k = trained["sessionize"]["prefix_k"][0]
    assert k in available_ks(trained)
    pred = SessionPredictor(trained, k)
    assert pred.models == MODELS
    scores = pred.predict(_events(k))
    assert [s.model for s in scores] == MODELS
    assert all(0.0 <= s.score <= 1.0 and s.decision == (s.score >= s.threshold) for s in scores)
    rule = next(s for s in scores if s.model == "rule_based")
    assert rule.score == 1.0  # the prefix contains a cart
    assert pred.predict(_events(k, carts=()))[0].score == 0.0


def test_predictor_matches_training_pipeline(trained):
    """A test session scored live must get the same score the training run logged for it."""
    from src.data.sessionize import load_prefixes
    from src.serving.predictor import available_runs
    from src.config import resolve_path

    k = trained["sessionize"]["prefix_k"][0]
    pred = SessionPredictor(trained, k, models=["lightgbm", "gru"])
    runs = available_runs(trained).filter(pl.col("k") == k)
    for name in ("lightgbm", "gru"):
        run_id = runs.filter(pl.col("model") == name)["run_id"][0]
        logged = pl.read_parquet(resolve_path(trained, "runs_dir") / run_id / "predictions.parquet").filter(
            pl.col("split") == "test").head(5)
        prefixes = load_prefixes(trained, k).prefix_events
        for sid, score in zip(logged["user_session"], logged["score"]):
            ev = prefixes.filter(pl.col("user_session") == sid).sort("pos").collect()
            live = next(s for s in pred.predict(ev.with_columns(pl.col("event_type").cast(pl.String))) if s.model == name)
            assert live.score == pytest.approx(score, abs=1e-5), (name, sid)


def test_contributions_add_up_to_lightgbm_score(trained):
    k = trained["sessionize"]["prefix_k"][0]
    pred = SessionPredictor(trained, k, models=["lightgbm"])
    ev = _events(k)
    contrib = pred.contributions(ev)
    score = pred.predict(ev)[0].score
    logit = float(contrib["contribution"].sum())
    assert 1 / (1 + np.exp(-logit)) == pytest.approx(score, abs=1e-6)
    assert contrib["feature"][-1] == "(base)"


def test_prepare_events_validates():
    with pytest.raises(ValueError, match="at least K=5"):
        prepare_events(_events(3), 5)
    with pytest.raises(ValueError, match="already contains a purchase"):
        prepare_events(_events(5).with_columns(event_type=pl.lit("purchase")), 5)
    with pytest.raises(ValueError, match="unknown event types"):
        prepare_events(_events(5).with_columns(event_type=pl.lit("click")), 5)
    with pytest.raises(ValueError, match="missing columns"):
        prepare_events(_events(5).drop("price"), 5)
    out = prepare_events(_events(7), 5)
    assert out.height == 5 and out["pos"].to_list() == [1, 2, 3, 4, 5]


def test_predictor_without_runs_explains_what_to_run(tmp_path):
    cfg = load_config(overrides=[f"paths.all_runs_csv={tmp_path / 'none.csv'}", "data.mode=synthetic"])
    with pytest.raises(FileNotFoundError, match="03_run_experiments.py --mode synthetic"):
        SessionPredictor(cfg, 5)


def test_result_store(tmp_path):
    (tmp_path / "rq1_model_comparison_dev.csv").write_text("k,model\n5,mlp\n")
    (tmp_path / "data_summary_synthetic.json").write_text('{"mode": "synthetic"}')
    assert modes_with_results(tmp_path) == ["dev", "synthetic"]
    store = ResultStore(load_config(overrides=[f"paths.tables_dir={tmp_path}", "data.mode=dev"]))
    assert store.csv("rq1_model_comparison")["model"].to_list() == ["mlp"]
    assert store.json("data_summary") is None and store.csv("missing") is None


def test_dashboard_pages_render(trained, monkeypatch):
    """Every page runs without an exception on synthetic results (Streamlit's headless test runner)."""
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("PYTHONPATH", str(PROJECT_ROOT))
    import src.config as config

    real = config.load_config
    tmp_over = [f"paths.{p}={trained['paths'][p]}" for p in
                ("interim_dir", "processed_dir", "tables_dir", "runs_dir", "figures_dir", "all_runs_csv")]
    monkeypatch.setattr(config, "load_config", lambda overrides=None, **kw: real(overrides=[*tmp_over, *(overrides or [])], **kw))
    for page in ("overview", "data", "ml", "dl", "compare", "predict"):
        at = AppTest.from_file(str(PROJECT_ROOT / "app" / "main.py"), default_timeout=120)
        at.session_state["mode"] = "synthetic"
        at.run()
        assert not at.exception, (page, at.exception)
        if page != "overview":
            at.switch_page(f"views/{page}.py").run()
            assert not at.exception, (page, at.exception)
