"""REST API (src/api/app.py) on quickly trained synthetic models."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("lightgbm", exc_type=ImportError)
pytest.importorskip("torch")

from fastapi.testclient import TestClient  # noqa: E402

import src.api.app as api  # noqa: E402
from src.config import load_config  # noqa: E402
from tests.test_serving import trained  # noqa: E402,F401  (module-scoped fixture: synthetic data + models)


@pytest.fixture(scope="module")
def client(trained):  # noqa: F811
    original = api.config
    api.config = lambda: trained          # serve the synthetic test models
    api.predictor.cache_clear()
    yield TestClient(api.app)
    api.config = original
    api.predictor.cache_clear()


def _events(k: int, cart: bool = True):
    return [{"event_time": f"2019-10-28T14:00:{10 * i:02d}Z", "event_type": "cart" if cart and i == 1 else "view",
             "product_id": 1 + i % 2, "category_code": "electronics.smartphone", "brand": "apple",
             "price": 400.0 + i} for i in range(k)]


def test_health_and_models(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["mode"] == "synthetic" and 5 in h["ks"]
    m = client.get("/models", params={"k": 5}).json()
    assert {x["model"] for x in m["models"]} >= {"lightgbm", "gru"}


def test_score_one_model_with_explanation(client):
    r = client.post("/score", json={"k": 5, "model": "lightgbm", "explain": True, "events": _events(6)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "lightgbm" and 0 <= body["score"] <= 1
    assert body["intent"] in ("high", "warm", "browsing") and body["action"]
    assert body["buyer"] == (body["score"] >= body["threshold"])
    assert body["explanation"][-1]["feature"] == "(base)"


def test_score_all_models(client):
    r = client.post("/score/all", json={"k": 5, "events": _events(5)})
    assert r.status_code == 200, r.text
    assert {x["model"] for x in r.json()} >= {"rule_based", "lightgbm", "gru"}


@pytest.mark.parametrize("payload, code", [
    ({"k": 5, "events": _events(3)}, 422),                         # fewer than K events
    ({"k": 4, "events": _events(5)}, 404),                         # K without models
    ({"k": 5, "model": "nope", "events": _events(5)}, 404),        # unknown model
    ({"k": 5, "events": [{**_events(1)[0], "event_type": "purchase"}] * 5}, 422),  # purchase in the input
    ({"k": 5, "events": [{**_events(1)[0], "price": -1}] * 5}, 422),
])
def test_bad_requests(client, payload, code):
    assert client.post("/score", json=payload).status_code == code


def test_config_reads_the_mode_from_the_environment(monkeypatch):
    monkeypatch.setenv("PIE_MODE", "dev")
    assert api._cfg()["data"]["mode"] == "dev"
    assert load_config(overrides=["data.mode=dev"])["data"]["mode"] == "dev"
