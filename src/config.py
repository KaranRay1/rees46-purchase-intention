"""Configuration loading, validation, seeding and snapshotting.

Every script starts with::

    from src.config import load_config, set_seed

    cfg = load_config(overrides=["data.mode=synthetic"])
    set_seed(cfg["project"]["seeds"][0])

`load_config` returns a plain ``dict`` (easy to save, print and diff). It is
validated on load so a bad setting fails at start-up, not hours into a run.
"""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "config.yaml"

DATA_MODES = {"synthetic", "dev", "full"}
KNOWN_MODELS = {
    "rule_based",
    "logistic_regression",
    "random_forest",
    "lightgbm",
    "mlp",
    "tabnet",
    "gru",
}


class ConfigError(ValueError):
    """Raised when config.yaml contains an invalid or inconsistent setting."""


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
def load_config(
    path: str | Path | None = None,
    overrides: Iterable[str] | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load, override and validate the experiment configuration.

    Args:
        path: YAML file to load. Defaults to ``config/config.yaml``.
        overrides: Either ``["data.mode=synthetic", "sessionize.default_k=3"]``
            (CLI style; values parsed as YAML, so ``3`` is an int and
            ``[3, 5]`` a list) or a mapping ``{"data.mode": "synthetic"}``.

    Returns:
        The validated configuration as a nested dict.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping at the top level.")

    if overrides:
        cfg = apply_overrides(cfg, overrides)

    validate_config(cfg)
    return cfg


def apply_overrides(
    cfg: dict[str, Any], overrides: Iterable[str] | Mapping[str, Any]
) -> dict[str, Any]:
    """Return a copy of ``cfg`` with dotted-key overrides applied.

    Only existing keys can be overridden, so a typo raises instead of being
    silently ignored.
    """
    if isinstance(overrides, Mapping):
        items = list(overrides.items())
    else:
        items = []
        for item in overrides:
            if "=" not in item:
                raise ConfigError(f"Override must look like key.path=value, got {item!r}")
            key, raw = item.split("=", 1)
            items.append((key.strip(), yaml.safe_load(raw)))

    out = copy.deepcopy(cfg)
    for dotted_key, value in items:
        parts = dotted_key.split(".")
        node = out
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise ConfigError(f"Unknown config key in override: {dotted_key!r}")
            node = node[part]
        if not isinstance(node, dict) or parts[-1] not in node:
            raise ConfigError(f"Unknown config key in override: {dotted_key!r}")
        node[parts[-1]] = value
    return out


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
def validate_config(cfg: Mapping[str, Any]) -> None:
    """Check the configuration for missing sections and inconsistent values.

    Raises:
        ConfigError: listing every problem found, not just the first.
    """
    errors: list[str] = []

    required = [
        "project", "paths", "data", "sessionize", "split", "features",
        "sequences", "imbalance", "models", "evaluation", "efficiency",
        "explainability", "logging",
    ]
    missing = [s for s in required if s not in cfg]
    if missing:
        raise ConfigError(f"Missing config sections: {missing}")

    # project
    seeds = cfg["project"].get("seeds")
    if not seeds or not all(isinstance(s, int) for s in seeds):
        errors.append("project.seeds must be a non-empty list of integers.")
    elif len(set(seeds)) != len(seeds):
        errors.append("project.seeds contains duplicates.")

    # data
    mode = cfg["data"].get("mode")
    if mode not in DATA_MODES:
        errors.append(f"data.mode must be one of {sorted(DATA_MODES)}, got {mode!r}.")
    frac = cfg["data"].get("session_sample_fraction")
    if frac is not None and not (0 < frac <= 1):
        errors.append("data.session_sample_fraction must be null or in (0, 1].")

    # sessionize
    ks = cfg["sessionize"].get("prefix_k", [])
    if not ks or not all(isinstance(k, int) and k >= 1 for k in ks):
        errors.append("sessionize.prefix_k must be a non-empty list of positive integers.")
    elif cfg["sessionize"].get("default_k") not in ks:
        errors.append("sessionize.default_k must be one of sessionize.prefix_k.")

    # split
    split = cfg["split"]
    fractions = [split.get(name) for name in ("train", "val", "test")]
    if any(not isinstance(f, (int, float)) or f <= 0 for f in fractions):
        errors.append("split.train/val/test must all be positive numbers.")
    elif abs(sum(fractions) - 1.0) > 1e-9:
        errors.append(f"split.train + val + test must equal 1.0, got {sum(fractions):.4f}.")

    # features
    feats = cfg["features"]
    groups: dict[str, list[str]] = feats.get("groups", {})
    all_features = [f for names in groups.values() for f in names]
    dupes = sorted({f for f in all_features if all_features.count(f) > 1})
    if dupes:
        errors.append(f"Features listed in more than one group: {dupes}")
    unknown_cat = [f for f in feats.get("categorical", []) if f not in all_features]
    if unknown_cat:
        errors.append(f"features.categorical not defined in any group: {unknown_cat}")
    for i, step in enumerate(feats.get("ablation_ladder", [])):
        bad = [g for g in step if g not in groups]
        if bad:
            errors.append(f"features.ablation_ladder[{i}] uses unknown groups: {bad}")
    if feats.get("epsilon", 0) <= 0:
        errors.append("features.epsilon must be > 0.")

    # models
    enabled = cfg["models"].get("enabled", [])
    unknown_models = [m for m in enabled if m not in KNOWN_MODELS]
    if unknown_models:
        errors.append(f"models.enabled has unknown models: {unknown_models}")
    no_params = [m for m in enabled if m not in cfg["models"]]
    if no_params:
        errors.append(f"models.enabled without a settings block: {no_params}")
    for key, models in (
        ("features.ablation_models", feats.get("ablation_models", [])),
        ("explainability.models", cfg["explainability"].get("models", [])),
    ):
        off = [m for m in models if m not in enabled]
        if off:
            errors.append(f"{key} uses models not in models.enabled: {off}")

    # evaluation
    ev = cfg["evaluation"]
    if not 0 < ev.get("bootstrap", {}).get("confidence", 0) < 1:
        errors.append("evaluation.bootstrap.confidence must be in (0, 1).")
    if not 0 < ev.get("alpha", 0) < 1:
        errors.append("evaluation.alpha must be in (0, 1).")
    bins = ev.get("cohorts", {}).get("session_length_bins", [])
    if bins != sorted(bins):
        errors.append("evaluation.cohorts.session_length_bins must be increasing.")

    if errors:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))


# -----------------------------------------------------------------------------
# Helpers used by the pipeline
# -----------------------------------------------------------------------------
def resolve_path(cfg: Mapping[str, Any], key: str) -> Path:
    """Absolute path for ``cfg["paths"][key]``, relative to the project root."""
    try:
        rel = cfg["paths"][key]
    except KeyError as exc:
        raise ConfigError(f"Unknown path key: {key!r}") from exc
    p = Path(rel)
    return p if p.is_absolute() else PROJECT_ROOT / p


def processed_dir_for_k(cfg: Mapping[str, Any], k: int) -> Path:
    """Folder holding the train/val/test tables for prefix length ``k``."""
    return resolve_path(cfg, "processed_dir") / f"k{k}"


def raw_files(cfg: Mapping[str, Any]) -> list[Path]:
    """Raw CSV files for the current ``data.mode`` (empty list in synthetic mode)."""
    mode = cfg["data"]["mode"]
    if mode == "synthetic":
        return []
    names = cfg["data"]["dev_files"] if mode == "dev" else cfg["data"]["full_files"]
    raw_dir = resolve_path(cfg, "raw_dir")
    return [raw_dir / name for name in names]


def feature_list(cfg: Mapping[str, Any], groups: Iterable[str] | str | None = None) -> list[str]:
    """Ordered feature names for the given groups.

    Args:
        groups: group names, or ``None`` / ``"all"`` for every group in config order.
    """
    all_groups: dict[str, list[str]] = cfg["features"]["groups"]
    if groups is None or groups == "all":
        selected = list(all_groups)
    elif isinstance(groups, str):
        selected = [groups]
    else:
        selected = list(groups)
    unknown = [g for g in selected if g not in all_groups]
    if unknown:
        raise ConfigError(f"Unknown feature groups: {unknown}")
    return [f for g in selected for f in all_groups[g]]


def model_params(cfg: Mapping[str, Any], model_name: str) -> dict[str, Any]:
    """Copy of a model's settings without the ``tuning_grid`` entry."""
    if model_name not in cfg["models"]:
        raise ConfigError(f"No settings for model {model_name!r}")
    params = copy.deepcopy(cfg["models"][model_name])
    params.pop("tuning_grid", None)
    return params


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy and (if installed) PyTorch.

    Note: ``PYTHONHASHSEED`` only affects child processes started after this
    call; the current interpreter's hash seed is fixed at start-up.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def save_config_snapshot(cfg: Mapping[str, Any], out_dir: str | Path) -> Path:
    """Write the exact config used by a run to ``out_dir/config_snapshot.yaml``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "config_snapshot.yaml"
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(_to_plain(cfg), fh, sort_keys=False)
    return path


def _to_plain(obj: Any) -> Any:
    """Convert values ``yaml.safe_dump`` rejects (Path, tuple) to plain types."""
    if isinstance(obj, Mapping):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj
