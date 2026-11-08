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


@pytest.fixture(scope="module")
def ablation(experiment):
    tmp, args, _ = experiment                       # reuses prepared data from the RQ1 fixture
    res = _load("04_run_ablation.py").main(args + ["--seeds", "42", "7", "--n-resamples", "50"])
    return tmp, args, res


def test_ablation_tables(ablation):
    from src.config import feature_list, load_config
    tmp, _, res = ablation
    cfg = load_config()
    t = res["table"]
    assert t.height == 3 * 2 * (5 + 1)                               # K x models x (ladder steps + variant)
    assert set(t["n_seeds"]) == {2}
    ladder = t.filter(~t["is_variant"])
    sizes = ladder.filter((ladder["k"] == 5) & (ladder["model"] == "lightgbm")).sort("step")["n_inputs"].to_list()
    assert sizes == sorted(sizes) and sizes[-1] == len(feature_list(cfg))  # raw profile: 1 column per feature
    assert ladder.filter(ladder["step"] == 1)["pr_auc_gain"].null_count() == 6  # no gain for the first step
    var = t.filter(t["is_variant"])
    assert var.height == 6 and set(var["step"]) == {2} and set(var["feature_set"]) == {"V:ratios_instead_of_counts"}
    g1 = len(cfg["features"]["groups"]["G1_activity"])
    g3 = len(cfg["features"]["groups"]["G3_behavioural_ratios"])
    assert set(var.filter(var["model"] == "lightgbm")["n_inputs"]) == {g1 + g3}
    tests = res["tests"]
    assert tests.height == 3 * 2 * (4 + 2)                           # consecutive steps + variant vs S2 and S1
    conf = tests.filter(tests["family"] == "confirmatory")
    assert conf.height == 6 and set(conf["added"]) == {"G3"}
    vt = tests.filter(tests["family"] == "variant")
    assert vt.height == 12 and set(vt["step"]) == {1, 2}
    assert "exploratory_ratios_instead_of_counts" in res["hypothesis"]
    for name in ("rq3_ablation", "rq3_step_tests"):
        assert (tmp / "tables" / f"{name}_synthetic.csv").exists()
    assert (tmp / "figures" / "rq3_ablation_synthetic.png").exists()
    assert res["hypothesis"]["H3_behavioural_ratios_add_pr_auc"]["verdict"] in (
        "supported", "partly supported", "not supported")


def test_ablation_cart_step_helps(ablation):
    """Synthetic signal lives in cart behaviour: adding G2 should raise PR-AUC for both models."""
    _, _, res = ablation
    g2 = res["tests"].filter(res["tests"]["added"] == "G2")
    assert (g2["pr_auc_gain"] > 0).mean() >= 0.8


def test_ablation_analyse_only_matches(ablation):
    _, args, res = ablation
    again = _load("04_run_ablation.py").main(args + ["--analyse-only", "--n-resamples", "50"])
    assert again["table"].equals(res["table"]) and again["tests"].equals(res["tests"])


def test_ablation_rejects_model_outside_ladder(tmp_path):
    with pytest.raises(SystemExit):
        _load("04_run_ablation.py").main(_args(tmp_path) + ["--models", "gru"])


@pytest.fixture(scope="module")
def benchmark(experiment):
    tmp, args, _ = experiment                       # benchmarks the models saved by the RQ1 fixture
    fast = ["--set", "efficiency.n_repeats=5", "--set", "efficiency.n_warmup=1",
            "--set", "efficiency.batch_sizes=[1, 64]"]
    return tmp, _load("05_benchmark_latency.py").main(args + fast)


def test_benchmark_table_and_outputs(benchmark):
    tmp, res = benchmark
    t = res["table"]
    assert t.height == 7 * 3
    assert (t["latency_b1_p50_ms"] <= t["latency_b1_p95_ms"]).all()
    assert (t["model_size_mb"] > 0).all() and (t["n_threads"] == 1).all()
    assert t.filter(t["model"] == "gru")["n_parameters"][0] > 0
    for name in ("rq4_efficiency_synthetic.csv", "rq4_hardware_synthetic.json", "rq4_hypothesis_synthetic.json"):
        assert (tmp / "tables" / name).exists()
    assert (tmp / "figures" / "rq4_pareto_synthetic.png").exists()
    a = res["assessment"]
    assert set(a["per_k"]) == {"3", "5", "10"}
    assert a["H4_trees_dominate_accuracy_latency_front"]["verdict"] in ("supported", "partly supported", "not supported")
    for info in a["per_k"].values():
        assert set(info["pareto_front_non_baseline"]) <= {"random_forest", "lightgbm", "mlp", "tabnet", "gru"}
        assert info["pareto_front_all_models"]


def test_benchmark_logs_runs(benchmark):
    from src.config import load_config
    from src.utils.results_logger import load_runs
    tmp, res = benchmark
    cfg = load_config(overrides={"paths.all_runs_csv": str(tmp / "tables" / "all_runs.csv")})
    runs = load_runs(cfg, "rq4_efficiency")
    assert runs.height == res["table"].height
    assert runs["efficiency_latency_b1_p95_ms"].is_not_null().all()


def test_pareto_front_logic():
    import polars as pl
    mod = _load("05_benchmark_latency.py")
    pts = pl.DataFrame({"model": ["fast_bad", "slow_good", "dominated", "best_both"],
                        "latency_b1_p95_ms": [0.1, 5.0, 6.0, 0.05],
                        "test_pr_auc_mean": [0.20, 0.40, 0.30, 0.25]})
    assert set(mod.pareto_front(pts)) == {"slow_good", "best_both"}


def test_experiment_rejects_unknown_model(tmp_path):
    with pytest.raises(SystemExit):
        _load("03_run_experiments.py").main(_args(tmp_path) + ["--models", "xgboost"])


def test_bad_override_fails_clearly(tmp_path):
    with pytest.raises(Exception, match="Unknown config key"):
        _load("01_prepare_data.py").main(_args(tmp_path) + ["--set", "split.trian=0.7"])
