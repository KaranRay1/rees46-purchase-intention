"""Retrieval for the RAG assistant: BM25 (sparse), embeddings (dense) and a hybrid of both.

* :class:`BM25Retriever`: Okapi BM25 (k1 = 1.5, b = 0.75) over a light tokeniser that
  keeps numbers and splits ``random_forest`` / ``K=5`` into words, implemented here so it
  needs no extra package.
* :class:`DenseRetriever`: cosine similarity of Ollama embeddings; the chunk embeddings
  are cached under ``data/interim/rag/`` and rebuilt only when the corpus changes.
* :class:`HybridRetriever`: reciprocal rank fusion (Cormack et al., 2009) of the two
  rankings, ``score = sum 1 / (60 + rank)``, so neither score scale dominates.

All three return ``[(chunk, score), ...]`` best first. The retrieval evaluation
(``scripts/11_rag_eval.py``) compares them on a labelled question set.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from src.llm.corpus import Chunk

STOP = set("""a an and are as at be by for from has have how in is it its of on or that the this to was were what
when which who why will with does do did than then there their these those into about over after before can
could should would our we you your i me my""".split())
_SYN = {"rf": "random forest", "lgbm": "lightgbm", "lr": "logistic regression", "dl": "deep learning",
        "ml": "machine learning", "prauc": "pr auc", "auprc": "pr auc"}


def tokenize(text: str) -> list[str]:
    text = text.lower().replace("_", " ").replace("pr-auc", "pr auc").replace("roc-auc", "roc auc")
    toks = []
    for t in re.findall(r"[a-z]+|\d+(?:\.\d+)?", text):
        toks.extend(_SYN.get(t, t).split())
    return [t for t in toks if t not in STOP]


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]: ...


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks, self.k1, self.b = list(chunks), k1, b
        self.docs = [Counter(tokenize(c.body)) for c in self.chunks]
        self.lens = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg = float(self.lens.mean()) if len(self.lens) else 0.0
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        s = np.zeros(len(self.docs))
        for i, d in enumerate(self.docs):
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / max(self.avg, 1e-9))
            for t in q:
                f = d.get(t, 0)
                if f:
                    s[i] += self.idf[t] * f * (self.k1 + 1) / (f + norm)
        return s

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        s = self.scores(query)
        order = np.argsort(-s)[:k]
        return [(self.chunks[i], float(s[i])) for i in order if s[i] > 0]


class DenseRetriever:
    name = "dense"

    def __init__(self, chunks: Sequence[Chunk], embedder, cache_dir: Path | None = None) -> None:
        self.chunks, self.embedder = list(chunks), embedder
        key = hashlib.sha1(("\n\x00".join(c.body for c in self.chunks) + embedder.model).encode()).hexdigest()[:16]
        path = (cache_dir / f"{key}.npy") if cache_dir else None
        if path is not None and path.exists():
            self.matrix = np.load(path)
        else:
            self.matrix = embedder.embed([c.body for c in self.chunks], kind="document")
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, self.matrix)

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        q = self.embedder.embed([query], kind="query")[0]
        s = self.matrix @ q
        order = np.argsort(-s)[:k]
        return [(self.chunks[i], float(s[i])) for i in order]


class HybridRetriever:
    name = "hybrid"

    def __init__(self, retrievers: Sequence[Retriever], rrf_k: int = 60, depth: int = 30) -> None:
        self.retrievers, self.rrf_k, self.depth = list(retrievers), rrf_k, depth

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for r in self.retrievers:
            for rank, (c, _) in enumerate(r.search(query, self.depth)):
                fused[c.id] = fused.get(c.id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                by_id[c.id] = c
        best = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [(by_id[i], s) for i, s in best]
