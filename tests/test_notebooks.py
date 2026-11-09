"""Tests for src/utils/notebook.py and end-to-end execution of the analysis notebooks."""

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
    notebook_config,
    prepared_data_status,
    save_figure,
)


def _env(monkeypatch, tmp_path, mode="synthetic", n_sessions=1500):
    overrides = ";".join([
        f"paths.interim_dir={tmp_path / 'interim'}",
        f"paths.processed_dir={tmp_path / 'processed'}",
        f"paths.tables_dir={tmp_path / 'tables'}",
        f"paths.figures_dir={tmp_path / 'figures'}",
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


@pytest.fixture(scope="module")
def run_notebook(tmp_path_factory):
    """Execute a notebook on synthetic data in a temporary folder; returns that folder."""
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    tmp = tmp_path_factory.mktemp("notebooks")
    mp = pytest.MonkeyPatch()
    _env(mp, tmp, n_sessions=2000)

    def run(name: str):
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
