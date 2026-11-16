"""Scripts 08 (model and data cards) and 09 (demo bundle) on the synthetic test models."""

import importlib.util
from pathlib import Path

import polars as pl

from src.config import PROJECT_ROOT, load_config
from src.serving.results import demo_dir, demo_overrides
from src.serving.results import test_predictions as load_test_predictions
from tests.test_serving import trained  # noqa: F401  (module-scoped fixture: synthetic data + models)

PATHS = ("interim_dir", "processed_dir", "tables_dir", "runs_dir", "figures_dir", "all_runs_csv")


def _run(name: str, trained_cfg, extra=()):  # noqa: F811
    spec = importlib.util.spec_from_file_location(name[:-3], PROJECT_ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = ["--mode", "synthetic", *extra]
    for p in PATHS:
        args += ["--set", f"paths.{p}={trained_cfg['paths'][p]}"]
    mod.main(args)


def test_model_and_data_cards(trained, monkeypatch, tmp_path):  # noqa: F811
    spec = importlib.util.spec_from_file_location("cards", PROJECT_ROOT / "scripts" / "08_model_cards.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.PROJECT_ROOT = tmp_path                                       # write the cards to a temporary docs/
    args = ["--mode", "synthetic"]
    for p in PATHS:
        args += ["--set", f"paths.{p}={trained['paths'][p]}"]
    mod.main(args)
    card = (tmp_path / "docs" / "model_card_synthetic.md").read_text()
    assert "## Performance" in card and "(engine)" in card and "## Limitations" in card
    assert (tmp_path / "docs" / "data_card_synthetic.md").read_text().startswith("# Data card")


def test_demo_bundle_serves_without_run_folders(trained, monkeypatch):  # noqa: F811
    _run("09_build_demo.py", trained)
    d = demo_dir(trained)
    assert (d / "runs.csv").exists() and (d / "test_predictions_k5.parquet").exists()
    assert (d / "product_month.parquet").exists()
    monkeypatch.setenv("PIE_DEMO", "1")
    over = demo_overrides(trained)
    cfg = load_config(overrides=[f"paths.{p}={trained['paths'][p]}" for p in PATHS] + ["data.mode=synthetic", *over])
    assert Path(cfg["paths"]["runs_dir"]).parts[-2:] == ("synthetic", "runs")
    p = load_test_predictions(cfg, 5, "lightgbm")
    assert p is not None and p.height and set(p.columns) >= {"y_true", "score"}
    from src.serving.predictor import SessionPredictor

    assert "lightgbm" in SessionPredictor(cfg, 5).models                # models load from the bundle
    assert pl.read_csv(d / "runs.csv")["run_id"].n_unique() >= 1
