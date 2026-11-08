"""Tests for src/utils/results_logger.py."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

from src.config import load_config  # noqa: E402
from src.utils.results_logger import (  # noqa: E402
    flatten,
    git_state,
    load_predictions,
    load_runs,
    start_run,
)


@pytest.fixture
def cfg(tmp_path):
    return load_config(overrides={
        "paths.runs_dir": str(tmp_path / "runs"),
        "paths.all_runs_csv": str(tmp_path / "tables" / "all_runs.csv"),
    })


class _Model:
    name = "fake"

    def get_params(self):
        return {"seed": 1, "n_estimators": 10, "hidden": [8, 4], "nested": {"lr": 0.1}}

    def save(self, path):
        path.write_bytes(b"model")
        return path


def _metrics(f1=0.4):
    return {"threshold": 0.3, "val": {"f1": f1, "pr_auc": 0.5}, "test": {"f1": f1 - 0.05, "pr_auc": 0.45},
            "test_scores": np.zeros(3)}


def test_run_writes_folder_and_csv_row(cfg):
    with start_run(cfg, "rq1_models", "fake", k=5, seed=42) as run:
        run.log_params(_Model().get_params())
        run.log_metrics(_metrics())
        run.log_model(_Model())
    files = {p.name for p in run.dir.iterdir()}
    assert {"run.json", "config_snapshot.yaml", "params.json", "metrics.json", "model.pkl"} <= files
    info = json.loads((run.dir / "run.json").read_text())
    assert info["status"] == "completed" and info["k"] == 5 and info["duration_sec"] >= 0
    assert "test_scores" not in json.loads((run.dir / "metrics.json").read_text())
    assert yaml.safe_load((run.dir / "config_snapshot.yaml").read_text()) == cfg

    df = load_runs(cfg)
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["run_id"] == run.run_id and row["k"] == 5 and row["seed"] == 42
    assert row["test_pr_auc"] == pytest.approx(0.45) and row["threshold"] == pytest.approx(0.3)
    assert row["param_n_estimators"] == 10 and row["param_nested_lr"] == pytest.approx(0.1)


def test_schema_grows_and_rows_accumulate(cfg):
    with start_run(cfg, "rq1_models", "a", k=3, seed=1) as r1:
        r1.log_metrics({"test": {"pr_auc": 0.3}})
    with start_run(cfg, "rq3_ablation", "b", k=5, seed=1, feature_set="G1+G2") as r2:
        r2.log_metrics({"test": {"pr_auc": 0.35, "roc_auc": 0.7}})
    df = load_runs(cfg)
    assert df.height == 2 and "test_roc_auc" in df.columns
    assert df.filter(pl.col("model") == "a")["test_roc_auc"][0] is None
    assert load_runs(cfg, experiment="rq3_ablation")["feature_set"].to_list() == ["G1+G2"]
    assert r1.run_id != r2.run_id


def test_failed_run_is_recorded_and_reraised(cfg):
    with pytest.raises(RuntimeError, match="boom"):
        with start_run(cfg, "rq1_models", "bad", k=5, seed=0) as run:
            run.log_metrics({"val": {"f1": 0.1}})
            raise RuntimeError("boom")
    assert load_runs(cfg).height == 0                    # default filter: completed only
    failed = load_runs(cfg, status="failed")
    assert failed.height == 1 and "boom" in failed["error"][0]
    assert json.loads((run.dir / "run.json").read_text())["status"] == "failed"


def test_predictions_per_split(cfg):
    with start_run(cfg, "rq1_models", "fake", k=5, seed=1) as run:
        run.log_predictions("val", ["a", "b"], [0, 1], [0.2, 0.8], threshold=0.5)
        run.log_predictions("test", ["c", "d", "e"], [1, 0, 1], [0.9, 0.6, 0.1], threshold=0.5)
        run.log_predictions("test", ["c", "d", "e"], [1, 0, 1], [0.9, 0.4, 0.1], threshold=0.5)  # overwrite
    preds = load_predictions(run.dir)
    assert preds.height == 5
    test = load_predictions(run.dir, "test")
    assert test["y_pred"].to_list() == [1, 0, 0] and test["user_session"].to_list() == ["c", "d", "e"]


def test_saving_switches(tmp_path):
    cfg = load_config(overrides={
        "paths.runs_dir": str(tmp_path / "runs"),
        "paths.all_runs_csv": str(tmp_path / "all_runs.csv"),
        "logging.save_models": False,
        "logging.save_predictions": False,
    })
    with start_run(cfg, "x", "fake") as run:
        assert run.log_model(_Model()) is None
        assert run.log_predictions("test", ["a"], [1], [0.9], 0.5) is None
    assert not (run.dir / "model.pkl").exists() and not (run.dir / "predictions.parquet").exists()


def test_artifacts_of_each_kind(cfg, tmp_path):
    src = tmp_path / "note.txt"
    src.write_text("hello")
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    with start_run(cfg, "rq5_shap", "lightgbm", k=5, seed=1) as run:
        paths = [
            run.log_artifact("summary", {"a": np.float32(1.5), "b": [np.int64(2)], "c": float("nan")}),
            run.log_artifact("importance", pl.DataFrame({"feature": ["x"], "share": [1.0]})),
            run.log_artifact("curve", fig),
            run.log_artifact("note", src),
        ]
    plt.close(fig)
    assert [p.suffix for p in paths] == [".json", ".parquet", ".png", ".txt"]
    assert json.loads(paths[0].read_text()) == {"a": 1.5, "b": [2], "c": None}
    with pytest.raises(TypeError):
        run.log_artifact("bad", object())


def test_flatten_keeps_scalars_only():
    flat = flatten({"a": 1, "b": {"c": 2.5, "d": np.float64(3)}, "e": [1, 2], "f": np.zeros(2), "g": None})
    assert flat == {"a": 1, "b_c": 2.5, "b_d": 3.0, "g": None}


def test_git_state_in_this_repo():
    commit, dirty = git_state()
    assert commit is None or (len(commit) == 40 and isinstance(dirty, bool))


def test_load_runs_empty(cfg):
    assert load_runs(cfg).height == 0
