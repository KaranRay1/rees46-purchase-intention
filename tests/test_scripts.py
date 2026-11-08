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


def test_bad_override_fails_clearly(tmp_path):
    with pytest.raises(Exception, match="Unknown config key"):
        _load("01_prepare_data.py").main(_args(tmp_path) + ["--set", "split.trian=0.7"])
