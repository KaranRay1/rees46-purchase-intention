"""Tests for src/config.py: loading, overrides, validation, helpers, seeding."""

import random

import numpy as np
import pytest
import yaml

from src.config import (
    PROJECT_ROOT,
    ConfigError,
    feature_list,
    load_config,
    model_params,
    processed_dir_for_k,
    raw_files,
    save_config_snapshot,
    set_seed,
    validate_config,
)


@pytest.fixture
def cfg():
    return load_config()


def test_default_config_is_valid(cfg):
    assert cfg["sessionize"]["default_k"] in cfg["sessionize"]["prefix_k"]


def test_cli_overrides_are_parsed_as_yaml():
    cfg = load_config(overrides=["data.mode=synthetic", "sessionize.prefix_k=[3, 5]"])
    assert cfg["data"]["mode"] == "synthetic"
    assert cfg["sessionize"]["prefix_k"] == [3, 5]


def test_override_does_not_mutate_file_defaults():
    load_config(overrides={"data.mode": "synthetic"})
    assert load_config()["data"]["mode"] == "dev"


def test_unknown_override_key_raises():
    with pytest.raises(ConfigError, match="Unknown config key"):
        load_config(overrides=["data.moed=synthetic"])


@pytest.mark.parametrize(
    "override, message",
    [
        ({"split.test": 0.2}, "must equal 1.0"),
        ({"sessionize.default_k": 7}, "default_k"),
        ({"data.mode": "prod"}, "data.mode"),
        ({"models.enabled": ["lightgbm", "xgboost"]}, "unknown models"),
        ({"features.categorical": ["nonexistent"]}, "not defined in any group"),
    ],
)
def test_invalid_settings_raise(override, message):
    with pytest.raises(ConfigError, match=message):
        load_config(overrides=override)


def test_validation_reports_all_errors(cfg):
    cfg["split"]["test"] = 0.5
    cfg["data"]["mode"] = "prod"
    with pytest.raises(ConfigError) as exc:
        validate_config(cfg)
    assert "split" in str(exc.value) and "data.mode" in str(exc.value)


def test_feature_list_follows_ablation_ladder(cfg):
    ladder = cfg["features"]["ablation_ladder"]
    sizes = [len(feature_list(cfg, step)) for step in ladder]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)
    assert feature_list(cfg, ladder[-1]) == feature_list(cfg, "all")


def test_no_purchase_based_features(cfg):
    assert not any("purchase" in f for f in feature_list(cfg))


def test_paths_and_files(cfg):
    assert processed_dir_for_k(cfg, 5) == PROJECT_ROOT / "data" / "processed" / "k5"
    assert [p.name for p in raw_files(cfg)] == cfg["data"]["dev_files"]
    assert raw_files(load_config(overrides={"data.mode": "synthetic"})) == []


def test_model_params_drop_tuning_grid(cfg):
    params = model_params(cfg, "lightgbm")
    assert "tuning_grid" not in params and "tuning_grid" in cfg["models"]["lightgbm"]


def test_set_seed_is_reproducible():
    set_seed(123)
    a = (random.random(), np.random.rand())
    set_seed(123)
    assert (random.random(), np.random.rand()) == a


def test_snapshot_round_trip(cfg, tmp_path):
    path = save_config_snapshot(cfg, tmp_path / "run")
    assert yaml.safe_load(path.read_text()) == cfg
