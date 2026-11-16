"""Tests for src/models/torch_base.py and src/models/tabular_nn.py (run in .venv)."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_tabnet")

from src.config import load_config  # noqa: E402
from src.data import sessionize, split  # noqa: E402
from src.data.loader import ingest  # noqa: E402
from src.features import build_features  # noqa: E402
from src.features.preprocessing import prepare_splits  # noqa: E402
from src.models.base import BaseModel, build_model  # noqa: E402


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("nn")
    cfg = load_config(overrides={
        "data.mode": "synthetic",
        "data.synthetic.n_sessions": 8000,
        "paths.interim_dir": str(tmp / "interim"),
        "paths.processed_dir": str(tmp / "processed"),
        # small and fast for tests
        "models.mlp.max_epochs": 15,
        "models.mlp.early_stopping_patience": 3,
        "models.tabnet.max_epochs": 8,
        "models.tabnet.early_stopping_patience": 3,
        "models.tabnet.batch_size": 512,
        "models.tabnet.virtual_batch_size": 128,
    })
    ingest(cfg)
    sessionize.run(cfg)
    split.run(cfg)
    build_features.run(cfg)
    return cfg


def _fit(cfg, name, seed=42):
    data, pre = prepare_splits(cfg, 5, name)
    extra = {"cat_dims": pre.categorical_dims()} if name == "tabnet" else {}
    m = build_model(name, cfg, seed=seed, columns=pre.output_columns, **extra)
    m.fit(data["train"].X, data["train"].y, data["val"].X, data["val"].y)
    return m, data, pre


@pytest.fixture(scope="module")
def mlp(built):
    return _fit(built, "mlp")


@pytest.fixture(scope="module")
def tabnet(built):
    return _fit(built, "tabnet")


# ---------------------------------------------------------------- shared behaviour
@pytest.mark.parametrize("which", ["mlp", "tabnet"])
def test_learns_signal(which, request):
    m, data, _ = request.getfixturevalue(which)
    p = m.predict_proba(data["test"].X)
    assert p.shape == (len(data["test"].y),) and ((p >= 0) & (p <= 1)).all()
    assert average_precision_score(data["test"].y, p) > 1.5 * data["test"].y.mean()
    assert m.train_time_sec > 0 and m.n_parameters() > 0


@pytest.mark.parametrize("which", ["mlp", "tabnet"])
def test_best_epoch_restored(which, request):
    m, data, _ = request.getfixturevalue(which)
    assert m.best_epoch_ >= 1
    val_ap = average_precision_score(data["val"].y, m.predict_proba(data["val"].X))
    assert val_ap == pytest.approx(m.best_val_score_, abs=1e-6)


@pytest.mark.parametrize("which", ["mlp", "tabnet"])
def test_save_load(which, request, tmp_path):
    m, data, _ = request.getfixturevalue(which)
    m.set_threshold(0.4)
    back = BaseModel.load(m.save(tmp_path / f"{which}.pkl"))
    assert back.threshold == 0.4
    assert np.allclose(back.predict_proba(data["test"].X), m.predict_proba(data["test"].X), atol=1e-6)


@pytest.mark.parametrize("name", ["mlp", "tabnet"])
def test_needs_validation(built, name):
    data, pre = prepare_splits(built, 5, name)
    extra = {"cat_dims": pre.categorical_dims()} if name == "tabnet" else {}
    m = build_model(name, built, seed=0, columns=pre.output_columns, **extra)
    with pytest.raises(ValueError, match="validation"):
        m.fit(data["train"].X[:200], data["train"].y[:200])


# ---------------------------------------------------------------- MLP specifics
def test_mlp_same_seed_same_model(built, mlp):
    a, data, _ = mlp
    b, _, _ = _fit(built, "mlp")
    assert np.allclose(a.predict_proba(data["test"].X), b.predict_proba(data["test"].X), atol=1e-6)


def test_mlp_architecture_follows_config(mlp, built):
    m, _, pre = mlp
    linears = [layer for layer in m.module if isinstance(layer, torch.nn.Linear)]
    hidden = built["models"]["mlp"]["hidden_layers"]
    assert [layer.out_features for layer in linears] == [*hidden, 1]
    assert linears[0].in_features == len(pre.output_columns)
    assert any(isinstance(layer, torch.nn.BatchNorm1d) for layer in m.module)
    assert any(isinstance(layer, torch.nn.Dropout) for layer in m.module)


def test_mlp_history_and_early_stop(mlp, built):
    m, _, _ = mlp
    h = m.history_
    assert [r["epoch"] for r in h] == list(range(1, len(h) + 1))
    assert len(h) <= built["models"]["mlp"]["max_epochs"]
    assert m.best_val_score_ == pytest.approx(max(r["val_pr_auc"] for r in h))


def test_mlp_bad_activation():
    from src.models.tabular_nn import MLPModel
    m = MLPModel({"activation": "swishy", "max_epochs": 1})
    X = np.random.rand(20, 3).astype(np.float32)
    y = np.array([0, 1] * 10)
    with pytest.raises(ValueError, match="activation"):
        m.fit(X, y, X, y)


# ---------------------------------------------------------------- TabNet specifics
def test_tabnet_uses_categorical_embeddings(tabnet):
    m, _, pre = tabnet
    assert m.cat_idxs == pre.categorical_indices()
    assert m.cat_dims == list(pre.categorical_dims().values())


def test_tabnet_importances_and_masks(tabnet):
    m, data, pre = tabnet
    imp = m.feature_importances()
    assert set(imp) == set(pre.output_columns)
    assert sum(imp.values()) == pytest.approx(1.0, abs=1e-6)
    masks = m.explain(data["test"].X[:50])
    assert masks.shape == (50, len(pre.output_columns))


def test_tabnet_requires_cat_dims(built):
    with pytest.raises(ValueError, match="cat_dims"):
        build_model("tabnet", built, seed=0, columns=["a"])
