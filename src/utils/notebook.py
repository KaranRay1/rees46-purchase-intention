"""Shared set-up for the analysis notebooks in ``notebooks/``.

Every notebook starts with::

    from src.utils.notebook import notebook_config, use_style, save_figure
    cfg = notebook_config()

The data mode and any config overrides come from environment variables, so
the same notebook runs on the real data, on synthetic data, or in a
temporary folder (tests) without editing a cell:

``NB_MODE``       ``synthetic`` | ``dev`` | ``full`` (default: ``data.mode`` in config.yaml)
``NB_OVERRIDES``  ``;``-separated ``key.path=value`` pairs, e.g.
                  ``paths.processed_dir=/tmp/x/processed;sessionize.prefix_k=[5]``
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

import matplotlib.pyplot as plt

from src.config import load_config, processed_dir_for_k, resolve_path, set_seed

# Categorical slots in fixed order (colour-blind-validated reference palette).
PALETTE: list[str] = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
EVENT_COLOURS: dict[str, str] = {
    "view": PALETTE[0], "cart": PALETTE[1], "remove_from_cart": PALETTE[2], "purchase": PALETTE[3],
}
TARGET_COLOURS: dict[int, str] = {0: PALETTE[0], 1: PALETTE[1]}  # non-buyer, buyer
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def notebook_config() -> dict[str, Any]:
    """Config with ``NB_MODE`` / ``NB_OVERRIDES`` applied, seeded with the first project seed."""
    overrides = [o.strip() for o in os.environ.get("NB_OVERRIDES", "").split(";") if o.strip()]
    if os.environ.get("NB_MODE"):
        overrides.append(f"data.mode={os.environ['NB_MODE']}")
    cfg = load_config(overrides=overrides)
    set_seed(cfg["project"]["seeds"][0])
    return cfg


def prepared_data_status(cfg: Mapping[str, Any]) -> dict[int, str | None]:
    """For each K, the mode the processed data was built with (``None`` if missing)."""
    out: dict[int, str | None] = {}
    for k in cfg["sessionize"]["prefix_k"]:
        d = processed_dir_for_k(cfg, k)
        report = d / "sessionize_report.json"
        if report.exists() and (d / "splits.parquet").exists():
            out[k] = json.loads(report.read_text()).get("mode")
        else:
            out[k] = None
    return out


def ensure_prepared(cfg: Mapping[str, Any]) -> None:
    """Make sure step 1 (data preparation) has run for the current mode.

    Synthetic data is cheap, so it is built on the spot. For ``dev`` / ``full``
    the notebook stops with the command to run, because preparing a month of
    REES46 takes minutes and belongs in the script, not a notebook cell.
    """
    mode = cfg["data"]["mode"]
    status = prepared_data_status(cfg)
    if all(m == mode for m in status.values()):
        return
    if mode == "synthetic":
        from src.data import sessionize, split
        from src.data.loader import ingest

        ingest(cfg)
        sessionize.run(cfg)
        split.run(cfg)
        return
    found = {k: m for k, m in status.items() if m != mode}
    raise RuntimeError(
        f"Processed data for mode={mode!r} not found (per K: {found}). "
        f"Run `python scripts/01_prepare_data.py --mode {mode}` first."
    )


def features_status(cfg: Mapping[str, Any]) -> dict[int, bool]:
    """For each K, whether ``features.parquet`` exists and is newer than the split it was built from."""
    out: dict[int, bool] = {}
    for k in cfg["sessionize"]["prefix_k"]:
        d = processed_dir_for_k(cfg, k)
        feats, splits = d / "features.parquet", d / "splits.parquet"
        out[k] = feats.exists() and splits.exists() and feats.stat().st_mtime >= splits.stat().st_mtime
    return out


def ensure_features(cfg: Mapping[str, Any]) -> None:
    """Like ``ensure_prepared``, for step 2 (feature tables)."""
    ensure_prepared(cfg)
    if all(features_status(cfg).values()):
        return
    mode = cfg["data"]["mode"]
    if mode == "synthetic":
        from src.features import build_features

        build_features.run(cfg)
        return
    stale = [k for k, ok in features_status(cfg).items() if not ok]
    raise RuntimeError(
        f"Feature tables for K={stale} are missing or older than the data split. "
        f"Run `python scripts/02_build_features.py --mode {mode}` first."
    )


def use_style() -> None:
    """Quiet, report-ready matplotlib defaults: recessive axes and grid, fixed colour order."""
    plt.rcParams.update({
        "figure.figsize": (8, 4.2),
        "figure.dpi": 110,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "font.size": 10,
    })


def save_figure(fig: plt.Figure, cfg: Mapping[str, Any], name: str) -> str:
    """Save ``fig`` to ``figures_dir/<name>_<mode>.<format>`` and return the path."""
    log = cfg["logging"]
    out = resolve_path(cfg, "figures_dir") / f"{name}_{cfg['data']['mode']}.{log.get('figure_format', 'png')}"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=log.get("figure_dpi", 300), bbox_inches="tight")
    return str(out)
