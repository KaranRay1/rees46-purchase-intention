"""Tests for src/models/sequence_models.py (run in .venv)."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

torch = pytest.importorskip("torch")

from src.config import load_config  # noqa: E402
from src.data import sessionize, split  # noqa: E402
from src.data.loader import ingest  # noqa: E402
from src.features import build_features  # noqa: E402
from src.features.sequences import prepare_sequence_splits  # noqa: E402
from src.models.base import BaseModel, build_model  # noqa: E402


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gru")
    cfg = load_config(overrides={
        "data.mode": "synthetic",
        "data.synthetic.n_sessions": 8000,
        "paths.interim_dir": str(tmp / "interim"),
        "paths.processed_dir": str(tmp / "processed"),
        "models.gru.max_epochs": 12,
        "models.gru.early_stopping_patience": 3,
    })
    ingest(cfg)
    sessionize.run(cfg)
    split.run(cfg)
    build_features.run(cfg)
    return cfg


def _fit(cfg, k=5, seed=42, **param_overrides):
    data, enc = prepare_sequence_splits(cfg, k)
    if param_overrides:
        cfg = load_config(overrides={**{f"models.gru.{p}": v for p, v in param_overrides.items()},
                                     "paths.processed_dir": cfg["paths"]["processed_dir"]})
    m = build_model("gru", cfg, seed=seed, embedding_spec=enc.embedding_spec(),
                    num_features=enc.num_features)
    m.fit(data["train"], data["train"].y, data["val"], data["val"].y)
    return m, data, enc


@pytest.fixture(scope="module")
def gru(built):
    return _fit(built)


def test_learns_signal(gru):
    m, data, _ = gru
    p = m.predict_proba(data["test"])
    assert p.shape == (len(data["test"].y),) and ((p >= 0) & (p <= 1)).all()
    assert average_precision_score(data["test"].y, p) > 1.5 * data["test"].y.mean()
    assert m.input_kind == "sequence" and m.n_parameters() > 0


def test_accepts_tuple_or_sequence_data(gru):
    m, data, _ = gru
    d = data["test"]
    assert np.allclose(m.predict_proba(d), m.predict_proba((d.cat, d.num)))


def test_best_epoch_restored(gru):
    m, data, _ = gru
    val_ap = average_precision_score(data["val"].y, m.predict_proba(data["val"]))
    assert val_ap == pytest.approx(m.best_val_score_, abs=1e-6)
    assert m.best_val_score_ == pytest.approx(max(r["val_pr_auc"] for r in m.history_))


def test_same_seed_same_model(built, gru):
    a, data, _ = gru
    b, _, _ = _fit(built)
    assert np.allclose(a.predict_proba(data["test"]), b.predict_proba(data["test"]), atol=1e-6)


def test_architecture_follows_config(gru, built):
    m, _, enc = gru
    spec = enc.embedding_spec()
    embs = list(m.module.embeddings)
    assert [(e.num_embeddings, e.embedding_dim) for e in embs] == list(spec.values())
    assert m.module.gru.hidden_size == built["models"]["gru"]["hidden_size"]
    assert m.module.gru.input_size == sum(d for _, d in spec.values()) + len(enc.num_features)


@pytest.mark.parametrize("k", [3, 10])
def test_other_prefix_lengths(built, k):
    m, data, _ = _fit(built, k=k)
    assert data["test"].cat.shape[1] == k
    assert np.isfinite(m.predict_proba(data["test"])).all()


def test_bidirectional_two_layer_variant(built):
    m, data, _ = _fit(built, bidirectional=True, num_layers=2, max_epochs=2)
    assert m.module.gru.bidirectional and m.module.gru.num_layers == 2
    assert m.module.head[-1].in_features == 2 * m.module.gru.hidden_size
    assert np.isfinite(m.predict_proba(data["test"])).all()


def test_save_load(gru, tmp_path):
    m, data, _ = gru
    m.set_threshold(0.45)
    back = BaseModel.load(m.save(tmp_path / "gru.pkl"))
    assert back.threshold == 0.45
    assert np.allclose(back.predict_proba(data["test"]), m.predict_proba(data["test"]), atol=1e-6)


def test_input_checks(gru):
    m, data, _ = gru
    d = data["test"]
    with pytest.raises(ValueError, match="shape"):
        m.predict_proba((d.cat[:, :, 0], d.num))
    with pytest.raises(ValueError, match="numeric"):
        m.predict_proba((d.cat, d.num[:, :, :1]))
    bad = d.cat.copy()
    bad[0, 0, 0] = 999
    with pytest.raises(ValueError, match="embedding range"):
        m.predict_proba((bad, d.num))


def test_requires_validation_and_spec(built):
    data, enc = prepare_sequence_splits(built, 5)
    m = build_model("gru", built, seed=0, embedding_spec=enc.embedding_spec())
    with pytest.raises(ValueError, match="validation"):
        m.fit(data["train"], data["train"].y)
    with pytest.raises(ValueError, match="embedding_spec"):
        build_model("gru", built, seed=0)
