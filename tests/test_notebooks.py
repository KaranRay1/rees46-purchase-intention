"""Tests for src/utils/notebook.py and end-to-end execution of the analysis notebooks."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from src.config import PROJECT_ROOT, processed_dir_for_k  # noqa: E402
from src.utils.notebook import (  # noqa: E402
    ensure_prepared,
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


def test_eda_notebook_runs_on_synthetic(monkeypatch, tmp_path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    _env(monkeypatch, tmp_path, n_sessions=2000)
    nb = nbformat.read(PROJECT_ROOT / "notebooks" / "01_eda.ipynb", as_version=4)
    assert not any(c.get("outputs") for c in nb.cells if c.cell_type == "code"), "commit notebooks without outputs"
    nbclient.NotebookClient(
        nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(PROJECT_ROOT / "notebooks")}}
    ).execute()
    summary = json.loads((tmp_path / "tables" / "eda_summary_synthetic.json").read_text())
    assert summary["sessions"]["n"] > 0 and set(summary["eligibility"]) == {"3", "5", "10"}
    assert len(list((tmp_path / "figures").glob("eda_*_synthetic.png"))) == 11
