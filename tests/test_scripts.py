"""Smoke tests for the numbered scripts in scripts/ (synthetic data, temp folders)."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    """Import a script whose file name starts with a digit."""
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _args(tmp_path, n_sessions=2000):
    return [
        "--mode", "synthetic",
        "--set", f"data.synthetic.n_sessions={n_sessions}",
        "--set", f"paths.interim_dir={tmp_path / 'interim'}",
        "--set", f"paths.processed_dir={tmp_path / 'processed'}",
        "--set", f"paths.tables_dir={tmp_path / 'tables'}",
        "--set", f"paths.runs_dir={tmp_path / 'runs'}",
        "--set", f"paths.figures_dir={tmp_path / 'figures'}",
        "--set", f"paths.all_runs_csv={tmp_path / 'tables' / 'all_runs.csv'}",
    ]


def test_prepare_data_end_to_end(tmp_path, capsys):
    summary = _load("01_prepare_data.py").main(_args(tmp_path))
    out = capsys.readouterr().out
    assert "1/3" in out and "2/3" in out and "3/3" in out
    for k in ("3", "5", "10"):
        assert summary["sessionize"][k]["eligible"] > 0
        assert set(summary["splits"][k]) >= {"train", "val", "test"}
        assert (tmp_path / "processed" / f"k{k}" / "splits.parquet").exists()
    saved = json.loads((tmp_path / "tables" / "data_summary_synthetic.json").read_text())
    assert saved["cleaning"]["sessions_out"] == 2000
    assert saved["split_cutoffs"]["computed_from_k"] == 3


def test_prepare_data_no_report_and_single_k(tmp_path):
    summary = _load("01_prepare_data.py").main(_args(tmp_path) + ["--no-report", "--set", "sessionize.prefix_k=[5]",
                                                                    "--set", "sessionize.default_k=5"])
    assert "cleaning" not in summary and list(summary["sessionize"]) == ["5"]


def test_script_runs_from_command_line(tmp_path):
    """Plain `python scripts/01_prepare_data.py ...` must find the src package."""
    res = subprocess.run(
        [sys.executable, "scripts/01_prepare_data.py", *_args(tmp_path, 500), "--no-report"],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert res.returncode == 0, res.stderr[-2000:]
    assert "Summary for the report" in res.stdout


def test_build_features_end_to_end(tmp_path, capsys):
    args = _args(tmp_path, 3000)
    _load("01_prepare_data.py").main(args + ["--no-report"])
    summary = _load("02_build_features.py").main(args)
    out = capsys.readouterr().out
    assert "aligned" in out and "strongest differences" in out
    for k in ("3", "5", "10"):
        entry = summary["k"][k]
        assert entry["checks"] == "passed" and entry["sequence_shape_train"][1] == int(k)
        assert entry["sessions"]["train"] > entry["sessions"]["test"] > 0
        assert (tmp_path / "tables" / f"feature_profile_synthetic_k{k}.csv").exists()
    top = [r["feature"] for r in summary["k"]["5"]["top_smd"]]
    assert {"cart_count", "has_cart", "view_count", "cart_to_view_ratio", "first_cart_position"} & set(top)


def test_build_features_detects_broken_table():
    mod = _load("02_build_features.py")
    import polars as pl
    from src.config import feature_list, load_config
    cfg = load_config()
    row = {f: 0 for f in feature_list(cfg)}
    row.update({"user_session": "a", "split": "train", "target_purchase": 0,
                "view_count": 1, "cart_count": 1, "remove_count": 0, "n_events_total": 9})
    problems = mod.check_table(pl.DataFrame([row, row]), cfg, k=5)
    assert any("duplicate" in p for p in problems)
    assert any("analysis-only" in p for p in problems)
    assert any("!= K" in p for p in problems)


FAST = [
    "--set", "models.random_forest.n_estimators=30",
    "--set", "models.lightgbm.n_estimators=100",
    "--set", "models.lightgbm.early_stopping_rounds=10",
    "--set", "models.mlp.max_epochs=3",
    "--set", "models.tabnet.max_epochs=2",
    "--set", "models.tabnet.batch_size=512",
    "--set", "models.gru.max_epochs=3",
]


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    pytest.importorskip("lightgbm", exc_type=ImportError)
    pytest.importorskip("torch")
    tmp = tmp_path_factory.mktemp("rq1")
    args = _args(tmp, 3000) + FAST
    _load("01_prepare_data.py").main(args + ["--no-report"])
    _load("02_build_features.py").main(args + ["--skip-sequences"])
    res = _load("03_run_experiments.py").main(
        args + ["--seeds", "42", "7", "--n-resamples", "50"]
    )
    return tmp, args, res


def test_experiment_logs_every_run(experiment):
    from src.config import load_config
    tmp, _, _ = experiment
    cfg = load_config(overrides={"paths.all_runs_csv": str(tmp / "tables" / "all_runs.csv")})
    from src.utils.results_logger import load_runs
    runs = load_runs(cfg, "rq1_models")
    # 6 trained models x 3 K x 2 seeds + rule x 3 K x 1 seed
    assert runs.height == 6 * 3 * 2 + 3
    assert runs["status"].unique().to_list() == ["completed"]
    for run_id in runs["run_id"].head(3):
        files = {p.name for p in (tmp / "runs" / run_id).iterdir()}
        assert {"predictions.parquet", "model.pkl", "metrics.json", "artifacts"} <= files


def test_experiment_tables_and_figures(experiment):
    tmp, _, res = experiment
    t = res["table"]
    assert t.height == 7 * 3 and set(t["k"]) == {3, 5, 10}
    assert (t["pr_auc_ci_low"] <= t["pr_auc_primary"]).all() and (t["pr_auc_primary"] <= t["pr_auc_ci_high"]).all()
    assert t.filter(t["model"] == "rule_based")["n_seeds"].to_list() == [1, 1, 1]
    tests = res["tests"]
    assert tests.height == 21 * 3                                  # all pairs of 7 models per K
    assert (tests["p_bootstrap_holm"] >= tests["p_bootstrap"]).all()
    conf = tests.filter(tests["family"] == "confirmatory")
    assert conf.height == 4 * 3 + 1                                # 4 models vs LR at each K + GRU vs best at K=10
    no_gru = conf.filter((conf["model_a"] != "gru") & (conf["model_b"] != "gru"))   # GRU's opponent may be LR
    assert no_gru.height == 12
    assert ((no_gru["model_a"] == "logistic_regression") | (no_gru["model_b"] == "logistic_regression")).all()
    gru = conf.filter((conf["model_a"] == "gru") | (conf["model_b"] == "gru"))
    assert gru.height == 1 and gru["k"][0] == 10
    assert res["hypotheses"]["confirmatory_tests"] == 13
    for name in ("rq1_model_comparison", "rq1_pairwise_tests", "rq2_by_k"):
        assert (tmp / "tables" / f"{name}_synthetic.csv").exists()
    assert (tmp / "figures" / "rq2_metrics_vs_k_synthetic.png").exists()
    assert (tmp / "figures" / "rq1_pr_curves_k5_synthetic.png").exists()
    hyp = res["hypotheses"]
    for key in ("H1a_trees_beat_logistic_regression", "H1b_deep_tabular_beat_logistic_regression",
                "H1c_gru_matches_best_aggregated_at_max_k", "H2_performance_rises_with_k_diminishing"):
        assert hyp[key]["verdict"] in ("supported", "partly supported", "not supported")


def test_analyse_only_reproduces_tables(experiment):
    tmp, args, res = experiment
    again = _load("03_run_experiments.py").main(args + ["--analyse-only", "--n-resamples", "50"])
    assert again["table"].equals(res["table"])
    assert again["tests"].equals(res["tests"])


def test_experiment_rejects_unknown_model(tmp_path):
    with pytest.raises(SystemExit):
        _load("03_run_experiments.py").main(_args(tmp_path) + ["--models", "xgboost"])


def test_bad_override_fails_clearly(tmp_path):
    with pytest.raises(Exception, match="Unknown config key"):
        _load("01_prepare_data.py").main(_args(tmp_path) + ["--set", "split.trian=0.7"])
