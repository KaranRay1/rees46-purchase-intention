"""Tests for src/utils/notebook.py and end-to-end execution of the analysis notebooks."""

import importlib.util
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from src.config import PROJECT_ROOT, processed_dir_for_k  # noqa: E402
from src.utils.notebook import (  # noqa: E402
    ensure_features,
    ensure_prepared,
    features_status,
    load_result,
    model_colours,
    notebook_config,
    prepared_data_status,
    save_figure,
)


def _paths(tmp_path):
    return {
        "paths.interim_dir": tmp_path / "interim",
        "paths.processed_dir": tmp_path / "processed",
        "paths.tables_dir": tmp_path / "tables",
        "paths.runs_dir": tmp_path / "runs",
        "paths.figures_dir": tmp_path / "figures",
        "paths.all_runs_csv": tmp_path / "tables" / "all_runs.csv",
    }


def _env(monkeypatch, tmp_path, mode="synthetic", n_sessions=1500):
    overrides = ";".join([
        *(f"{k}={v}" for k, v in _paths(tmp_path).items()),
        f"data.synthetic.n_sessions={n_sessions}",
    ])
    monkeypatch.setenv("NB_MODE", mode)
    monkeypatch.setenv("NB_OVERRIDES", overrides)


def test_notebook_config_reads_environment(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    cfg = notebook_config()
    assert cfg["data"]["mode"] == "synthetic"
    assert cfg["paths"]["figures_dir"] == str(tmp_path / "figures")


def test_notebook_config_defaults_to_file(monkeypatch):
    monkeypatch.delenv("NB_MODE", raising=False)
    monkeypatch.delenv("NB_OVERRIDES", raising=False)
    assert notebook_config()["data"]["mode"] == "dev"


def test_ensure_prepared_builds_synthetic(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    cfg = notebook_config()
    assert set(prepared_data_status(cfg).values()) == {None}
    ensure_prepared(cfg)
    assert set(prepared_data_status(cfg).values()) == {"synthetic"}


def test_ensure_prepared_refuses_other_modes(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path, mode="dev")
    with pytest.raises(RuntimeError, match="01_prepare_data.py --mode dev"):
        ensure_prepared(notebook_config())


def test_ensure_prepared_detects_mode_mismatch(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    cfg = notebook_config()
    ensure_prepared(cfg)
    k = cfg["sessionize"]["prefix_k"][0]
    report = processed_dir_for_k(cfg, k) / "sessionize_report.json"
    report.write_text(json.dumps({**json.loads(report.read_text()), "mode": "dev"}))
    assert prepared_data_status(cfg)[k] == "dev"


def test_save_figure_names_by_mode(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    cfg = notebook_config()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    path = save_figure(fig, cfg, "demo")
    plt.close(fig)
    assert path.endswith("demo_synthetic.png") and (tmp_path / "figures" / "demo_synthetic.png").exists()


def test_ensure_features_builds_and_detects_stale(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    cfg = notebook_config()
    ensure_features(cfg)
    assert all(features_status(cfg).values())
    k = cfg["sessionize"]["prefix_k"][0]
    (processed_dir_for_k(cfg, k) / "splits.parquet").touch()  # split rebuilt after features
    assert not features_status(cfg)[k]


def test_ensure_features_refuses_other_modes(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    ensure_prepared(notebook_config())  # prepared data but no features...
    for k in notebook_config()["sessionize"]["prefix_k"]:
        report = processed_dir_for_k(notebook_config(), k) / "sessionize_report.json"
        report.write_text(json.dumps({**json.loads(report.read_text()), "mode": "dev"}))
    monkeypatch.setenv("NB_MODE", "dev")  # ...labelled as dev
    with pytest.raises(RuntimeError, match="02_build_features.py --mode dev"):
        ensure_features(notebook_config())


def test_model_colours_follow_config_order(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    cfg = notebook_config()
    colours = model_colours(cfg)
    assert list(colours) == cfg["models"]["enabled"] and len(set(colours.values())) == len(colours)


def test_load_result_reads_csv_json_and_reports_missing(monkeypatch, tmp_path, capsys):
    _env(monkeypatch, tmp_path)
    cfg = notebook_config()
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "a_synthetic.csv").write_text("x,y\n1,2\n")
    (tmp_path / "tables" / "b_synthetic.json").write_text(json.dumps({"v": 1}))
    assert load_result(cfg, "a", "s.py")["y"].to_list() == [2]
    assert load_result(cfg, "b", "s.py") == {"v": 1}
    assert load_result(cfg, "c", "03_x.py") is None
    assert "python scripts/03_x.py --mode synthetic" in capsys.readouterr().out


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name[:-3], PROJECT_ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FAST = [
    "--set", "models.random_forest.n_estimators=20",
    "--set", "models.lightgbm.n_estimators=50",
    "--set", "models.lightgbm.early_stopping_rounds=10",
    "--set", "models.mlp.max_epochs=2",
    "--set", "models.tabnet.max_epochs=1",
    "--set", "models.tabnet.batch_size=512",
    "--set", "models.gru.max_epochs=2",
]


@pytest.fixture(scope="module")
def run_notebook(tmp_path_factory):
    """Execute a notebook on synthetic data in a temporary folder; returns that folder."""
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    tmp = tmp_path_factory.mktemp("notebooks")
    mp = pytest.MonkeyPatch()
    _env(mp, tmp, n_sessions=2000)
    produced: set[str] = set()

    def results():
        """Run scripts 03-05 once (fast settings) so the results notebook has something to read."""
        if "results" in produced:
            return
        pytest.importorskip("lightgbm", exc_type=ImportError)
        pytest.importorskip("torch")
        args = ["--mode", "synthetic", "--set", "data.synthetic.n_sessions=2000", *FAST]
        for k, v in _paths(tmp).items():
            args += ["--set", f"{k}={v}"]
        _script("01_prepare_data.py").main(args + ["--no-report"])
        _script("02_build_features.py").main(args + ["--skip-sequences"])
        _script("03_run_experiments.py").main(args + ["--seeds", "42", "7", "--n-resamples", "30"])
        _script("04_run_ablation.py").main(args + ["--seeds", "42", "--n-resamples", "30"])
        _script("05_benchmark_latency.py").main(args + ["--set", "efficiency.n_repeats=3", "--set", "efficiency.n_warmup=1",
                                                        "--set", "efficiency.batch_sizes=[1, 32]"])
        _script("06_explain.py").main(args + ["--set", "explainability.n_explain_samples=200",
                                              "--set", "explainability.n_background=30"])
        produced.add("results")

    def run(name: str):
        if name[:2] in ("03", "04"):
            results()
        nb = nbformat.read(PROJECT_ROOT / "notebooks" / name, as_version=4)
        assert not any(c.get("outputs") for c in nb.cells if c.cell_type == "code"), "commit notebooks without outputs"
        nbclient.NotebookClient(
            nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(PROJECT_ROOT / "notebooks")}}
        ).execute()
        return tmp

    yield run
    mp.undo()


def test_eda_notebook_runs_on_synthetic(run_notebook):
    tmp = run_notebook("01_eda.ipynb")
    summary = json.loads((tmp / "tables" / "eda_summary_synthetic.json").read_text())
    assert summary["sessions"]["n"] > 0 and set(summary["eligibility"]) == {"3", "5", "10"}
    assert len(list((tmp / "figures").glob("eda_*_synthetic.png"))) == 11


def test_feature_notebook_runs_on_synthetic(run_notebook):
    tmp = run_notebook("02_feature_analysis.ipynb")
    summary = json.loads((tmp / "tables" / "feature_analysis_synthetic.json").read_text())
    assert summary["integrity_ok"] is True
    assert summary["ratios_determined_by_counts"] is True
    assert len(list((tmp / "figures").glob("feat_*_synthetic.png"))) == 8


def test_results_notebook_runs_on_synthetic(run_notebook):
    tmp = run_notebook("03_results_comparison.ipynb")
    board = (tmp / "tables" / "hypothesis_scoreboard_synthetic.csv").read_text()
    for h in ("H1a", "H1b", "H1c", "H2", "H3", "H4", "H5"):
        assert f"\n{h}," in board
    assert len(list((tmp / "figures").glob("res_*_synthetic.png"))) == 9


@pytest.mark.parametrize("name, expected", [
    ("03_results_comparison.ipynb", "[missing] rq1_model_comparison_synthetic"),
    ("04_explainability_errors.ipynb", "No RQ5 results yet"),
])
def test_results_notebooks_skip_missing_results(monkeypatch, tmp_path, name, expected):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    _env(monkeypatch, tmp_path)
    nb = nbformat.read(PROJECT_ROOT / "notebooks" / name, as_version=4)
    nbclient.NotebookClient(nb, timeout=120, kernel_name="python3",
                            resources={"metadata": {"path": str(PROJECT_ROOT / "notebooks")}}).execute()
    text = "".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []))
    assert expected in text


def test_explain_notebook_runs_on_synthetic(run_notebook):
    tmp = run_notebook("03_results_comparison.ipynb")   # writes the scoreboard that 04 completes
    tmp = run_notebook("04_explainability_errors.ipynb")
    summary = json.loads((tmp / "tables" / "rq5_explain_summary_synthetic.json").read_text())
    assert summary["H5"]["verdict"] in {"supported", "partly supported", "not supported"}
    assert summary["buyers_missed_by_all"]["of"] > 0
    assert len(list((tmp / "figures").glob("xai_*_synthetic.png"))) == 7
    board = (tmp / "tables" / "hypothesis_scoreboard_synthetic.csv").read_text()
    assert "see notebook 04" not in board
