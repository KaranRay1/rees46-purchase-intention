"""Turn model scores into purchase probabilities (Platt scaling).

The models were trained with class balancing, so their scores rank visits well but
are not probabilities: the average score is about 0.42 while about 11 % of visits buy.
Platt scaling fits ``p = sigmoid(a * logit(score) + b)`` on the **validation**
predictions (never the test period). It is monotone, so the ranking, PR-AUC and the
decision threshold's meaning are unchanged; only the number shown to people is
corrected.

The fit is saved next to each run as ``calibration.json`` and applied by
:class:`src.serving.predictor.SessionPredictor` when present.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

FILE = "calibration.json"
_CLIP = 1e-6


def _logit(s: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(s, dtype=float), _CLIP, 1 - _CLIP)
    return np.log(s / (1 - s))


@dataclass(frozen=True)
class Calibrator:
    a: float
    b: float
    method: str = "platt"

    def __call__(self, score):
        z = self.a * _logit(score) + self.b
        out = 1.0 / (1.0 + np.exp(-z))
        return float(out) if np.ndim(out) == 0 else out

    def save(self, run_dir: Path, **extra) -> None:
        (Path(run_dir) / FILE).write_text(json.dumps({**asdict(self), **extra}, indent=2))

    @classmethod
    def load(cls, run_dir: Path) -> "Calibrator | None":
        p = Path(run_dir) / FILE
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        return cls(float(d["a"]), float(d["b"]), d.get("method", "platt"))


def fit_platt(y: np.ndarray, score: np.ndarray, max_iter: int = 100) -> Calibrator:
    """Maximum-likelihood fit of ``a`` and ``b`` by Newton's method (two parameters, so no library needed)."""
    x, y = _logit(score), np.asarray(y, dtype=float)

    def nll(a: float, b: float) -> float:          # negative log-likelihood, numerically stable
        z = a * x + b
        return float(np.sum(np.logaddexp(0.0, z) - y * z))

    rate = float(np.clip(y.mean(), _CLIP, 1 - _CLIP))
    a, b = 0.0, float(np.log(rate / (1 - rate)))      # start at "everyone gets the base rate": well-conditioned
    cur = nll(a, b)
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-(a * x + b)))
        w = np.clip(p * (1 - p), 1e-12, None)
        g = np.array([np.sum((p - y) * x), np.sum(p - y)])
        h = np.array([[np.sum(w * x * x), np.sum(w * x)], [np.sum(w * x), np.sum(w)]]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(h, g)
        t = 1.0                                      # halve the step until the fit improves (two-valued scores
        while t > 1e-8 and nll(a - t * step[0], b - t * step[1]) > cur:   # such as the rule's overshoot)
            t /= 2
        a, b = a - t * step[0], b - t * step[1]
        new = nll(a, b)
        if abs(cur - new) < 1e-10 * max(1.0, abs(cur)) or t <= 1e-8:
            break
        cur = new
    return Calibrator(float(a), float(b))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error over equal-width probability bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for i in range(bins):
        m = idx == i
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    """Mean predicted vs observed buyer rate per probability bin (for a reliability chart)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return [{"bin": i, "predicted": float(p[idx == i].mean()), "observed": float(y[idx == i].mean()),
             "n": int((idx == i).sum())} for i in range(bins) if (idx == i).any()]
