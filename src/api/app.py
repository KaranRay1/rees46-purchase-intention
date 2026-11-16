"""REST API for the purchase-intent engine.

Run from the project root::

    uvicorn src.api.app:app --port 8000

Interactive docs at http://localhost:8000/docs. Settings come from environment variables:

* ``PIE_MODE``: which trained runs to serve (``full``, ``dev`` or ``synthetic``; default ``full``)
* ``PIE_CONFIG_OVERRIDES``: extra ``key=value`` config overrides, separated by ``;``

Endpoints:

* ``GET  /health``           liveness, mode and which K have trained models
* ``GET  /models?k=5``       models, thresholds and the recommended model at K
* ``POST /score``            score one visit with one model (default: the recommended one)
* ``POST /score/all``        score one visit with every model

The models are the primary-seed runs saved by ``scripts/03_run_experiments.py``; scoring
uses the same feature code as training (``src.serving.predictor.SessionPredictor``).
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from functools import lru_cache
from typing import Any, Literal

import polars as pl
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.config import load_config
from src.serving.predictor import EVENT_COLUMNS, SessionPredictor, available_ks
from src.serving.recommend import intent, recommended
from src.serving.results import ResultStore, demo_overrides

VERSION = "1.0.0"


# --------------------------------------------------------------------------- schemas
class Event(BaseModel):
    event_time: datetime = Field(description="ISO 8601; assumed UTC when no offset is given")
    event_type: Literal["view", "cart"] = Field(description="Purchases cannot be in the input")
    product_id: int
    category_id: int = 0
    category_code: str = "unknown"
    brand: str = "unknown"
    price: float = Field(gt=0)


class ScoreRequest(BaseModel):
    events: list[Event] = Field(min_length=1, description="The visit's clicks so far; the first K are used")
    k: int = Field(5, description="Clicks seen before scoring (a K with trained models)")
    model: str | None = Field(None, description="Model name; default is the recommended model at K")
    explain: bool = Field(False, description="Add LightGBM per-feature contributions (log-odds)")

    model_config = {"json_schema_extra": {"examples": [{
        "k": 3,
        "events": [
            {"event_time": "2020-04-10T14:00:00Z", "event_type": "view", "product_id": 1005115,
             "category_code": "electronics.smartphone", "brand": "apple", "price": 900.9},
            {"event_time": "2020-04-10T14:00:40Z", "event_type": "view", "product_id": 1005115,
             "category_code": "electronics.smartphone", "brand": "apple", "price": 900.9},
            {"event_time": "2020-04-10T14:01:30Z", "event_type": "cart", "product_id": 1005115,
             "category_code": "electronics.smartphone", "brand": "apple", "price": 900.9},
        ],
    }]}}


class Score(BaseModel):
    model: str
    probability: float | None = Field(None, description="calibrated purchase probability (null if not calibrated)")
    score: float = Field(description="ranking score (class-balanced, not a probability)")
    threshold: float
    buyer: bool
    intent: Literal["high", "warm", "browsing"]
    action: str
    latency_ms: float
    run_id: str


class ScoreResponse(Score):
    k: int
    explanation: list[dict[str, Any]] | None = None


# --------------------------------------------------------------------------- state
def _cfg() -> dict[str, Any]:
    mode = os.environ.get("PIE_MODE", "full")
    extra = [o for o in os.environ.get("PIE_CONFIG_OVERRIDES", "").split(";") if o.strip()]
    cfg = load_config(overrides=[f"data.mode={mode}", *extra])
    demo = demo_overrides(cfg)                 # no trained runs on this machine: serve the demo bundle
    return load_config(overrides=[f"data.mode={mode}", *extra, *demo]) if demo else cfg


@lru_cache(maxsize=1)
def config() -> dict[str, Any]:
    return _cfg()


@lru_cache(maxsize=8)
def predictor(k: int) -> SessionPredictor:
    return SessionPredictor(config(), k)


def recommended_model(k: int) -> str | None:
    store = ResultStore(config())
    rq1 = store.csv("rq1_model_comparison")
    if rq1 is None or not rq1.filter(pl.col("k") == k).height:
        return None
    ship, _ = recommended(rq1, store.csv("rq1_pairwise_tests"), store.csv("rq4_efficiency"), k)
    return ship


def _events_frame(events: list[Event]) -> pl.DataFrame:
    rows = [e.model_dump() for e in events]
    for r in rows:
        t = r["event_time"]
        r["event_time"] = t.replace(tzinfo=None) if t.tzinfo is None else t
    df = pl.DataFrame(rows)
    ts = df["event_time"]
    df = df.with_columns(event_time=(ts.dt.replace_time_zone("UTC") if ts.dtype.time_zone is None  # type: ignore[union-attr]
                                     else ts.dt.convert_time_zone("UTC")))
    return df.select([pl.col(c).cast(t) for c, t in EVENT_COLUMNS.items()])


def _check_k(k: int) -> SessionPredictor:
    ks = available_ks(config())
    if k not in ks:
        raise HTTPException(404, f"No trained models for K={k}; available: {ks}")
    return predictor(k)


def _to_score(s) -> Score:
    it = intent(s.score, s.threshold)
    prob = round(s.probability, 6) if s.probability is not None else None
    return Score(model=s.model, probability=prob, score=round(s.score, 6), threshold=round(s.threshold, 6), buyer=bool(s.decision),
                 intent=it.level, action=it.action, latency_ms=round(s.latency_ms, 3), run_id=s.run_id)


# --------------------------------------------------------------------------- app
app = FastAPI(
    title="Purchase Intent Engine",
    version=VERSION,
    summary="Scores an online-shop visit from its first clicks: purchase probability, intent level and next action.",
)


@app.get("/health")
def health() -> dict[str, Any]:
    cfg = config()
    return {"status": "ok", "version": VERSION, "mode": cfg["data"]["mode"], "ks": available_ks(cfg)}


@app.get("/models")
def models(k: int = Query(5, description="Clicks seen before scoring")) -> dict[str, Any]:
    p = _check_k(k)
    return {
        "k": k,
        "recommended": recommended_model(k),
        "models": [{"model": lm.name, "threshold": float(lm.model.threshold), "run_id": lm.run_id}
                   for lm in p.loaded.values()],
    }


def _score(req: ScoreRequest) -> tuple[SessionPredictor, pl.DataFrame]:
    p = _check_k(req.k)
    if len(req.events) < req.k:
        raise HTTPException(422, f"Need at least K={req.k} events, got {len(req.events)}.")
    return p, _events_frame(req.events)


@app.post("/score", response_model=ScoreResponse, response_model_exclude_none=True)
def score(req: ScoreRequest) -> ScoreResponse:
    p, events = _score(req)
    name = req.model or recommended_model(req.k) or p.models[-1]
    if name not in p.loaded:
        raise HTTPException(404, f"Model {name!r} not available at K={req.k}; available: {p.models}")
    start = time.perf_counter()
    try:
        s = p.predict(events, models=[name])[0]
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    out = ScoreResponse(k=req.k, **_to_score(s).model_dump())
    if req.explain:
        c = p.contributions(events)
        out.explanation = None if c is None else c.to_dicts()
    out.latency_ms = round((time.perf_counter() - start) * 1000, 3)
    return out


@app.post("/score/all", response_model=list[Score])
def score_all(req: ScoreRequest) -> list[Score]:
    p, events = _score(req)
    try:
        return [_to_score(s) for s in p.predict(events)]
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
