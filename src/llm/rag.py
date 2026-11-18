"""Retrieval-augmented analyst assistant and grounded per-visit explanations.

:class:`Analyst` answers questions about the project (models, results, data, monitoring)
from the knowledge base in :mod:`src.llm.corpus`: retrieve the top chunks, ask the
language model to answer **only** from them with ``[S1]``-style citations, then check
the answer automatically:

* **citation validity**: every ``[Sn]`` refers to a source that was shown;
* **numeric grounding**: every number in the answer (0.336, 33.6 %, 5.41 ms ...) appears
  in the sources, allowing for rounding and percent/fraction forms. Ungrounded numbers
  are the usual sign of a hallucinated figure.

:func:`explain_visit` turns a model's calibrated probability and the LightGBM SHAP
contributions for one visit into a short plain-English explanation and next step, and
checks that it mentions only features that were actually given to it.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.config import resolve_path
from src.llm.corpus import Chunk, build_corpus
from src.llm.providers import Completion, LLMError, OllamaEmbedder, Provider
from src.llm.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, Retriever

SYSTEM = """You are the analyst assistant for a purchase-intent prediction project (REES46 e-commerce \
clickstream; machine-learning models by Merve Kara, deep-learning models by Karan Ray).

Answer the question using ONLY the numbered sources provided. Rules:
- Cite every factual claim with its source number in square brackets, e.g. [S2].
- Copy numbers exactly as they appear in the sources; never compute, round differently or invent figures.
- When sources disagree, prefer sources titled [Results ...] (generated from the latest result tables); \
the [Report] contains older single-seed numbers.
- If the sources do not contain the answer, say so plainly instead of guessing.
- Be concise: at most about 150 words, plain prose or a short list."""

_NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)*(?:\.\d+)?\s*%?")


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Chunk]
    scores: list[float]
    provider: str | None
    model: str | None
    seconds: float
    retrieval_seconds: float
    invalid_citations: list[str] = field(default_factory=list)
    ungrounded_numbers: list[str] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    error: str | None = None
    usage: dict = field(default_factory=dict)

    @property
    def grounded(self) -> bool:
        return not self.invalid_citations and not self.ungrounded_numbers


# --------------------------------------------------------------------------- checks
def _parse(token: str) -> tuple[float, int] | None:
    """(value as a fraction-or-plain number, decimals shown); "33.6%" -> (33.6, 1) with percent handled by callers."""
    t = token.strip().replace(",", "").rstrip("%").strip()
    try:
        v = float(t)
    except ValueError:
        return None
    return v, (len(t.split(".")[1]) if "." in t else 0)


def number_matches(quoted: str, source: str) -> bool:
    """True if ``quoted`` is ``source`` shown at the quoted precision (also across % and fraction forms)."""
    q, s = _parse(quoted), _parse(source)
    if q is None or s is None:
        return False
    (qv, qd), (sv, _) = q, s
    for scale in (1.0, 100.0, 0.01):
        if round(sv * scale, qd) == round(qv, qd) and abs(sv * scale - qv) < 10 ** (-qd) + 1e-9:
            return True
    return False


def numbers_in(text: str) -> list[str]:
    out = []
    for m in _NUM.finditer(text):
        tok = m.group(0).strip()
        bare = tok.rstrip("%").replace(",", "").strip().lstrip("+-")
        if not bare or re.fullmatch(r"\d", bare):          # single digits (K=3, "2 models") are not checked
            continue
        out.append(tok)
    return out


def ungrounded(text: str, context: str) -> list[str]:
    """Numbers in ``text`` that do not match any number in ``context`` at the precision quoted."""
    ctx_nums = [m.group(0) for m in _NUM.finditer(context.replace(",", ""))]
    body = re.sub(r"\[S\d+\]", "", text)
    return [tok for tok in numbers_in(body) if not any(number_matches(tok, c) for c in ctx_nums)]


def citations(text: str, n_sources: int) -> tuple[list[int], list[str]]:
    found = [int(x) for x in re.findall(r"\[S(\d+)\]", text)]
    return sorted(set(found)), sorted({f"S{i}" for i in found if not 1 <= i <= n_sources})


# --------------------------------------------------------------------------- analyst
class Analyst:
    """Question answering over the project's knowledge base.

    Args:
        cfg: loaded config (its ``data.mode`` picks the result tables).
        provider: language model (``None``: retrieval only, the answer lists the sources).
        retriever: ``"hybrid"`` (default), ``"bm25"`` or ``"dense"``. Dense and hybrid need
            the Ollama embedding model; without it the analyst falls back to BM25.
    """

    def __init__(self, cfg: Mapping[str, Any], provider: Provider | None = None, retriever: str = "hybrid",
                 k: int = 5, embedder: OllamaEmbedder | None = None, chunks: Sequence[Chunk] | None = None) -> None:
        self.cfg, self.provider, self.k = cfg, provider, k
        self.chunks = list(chunks) if chunks is not None else build_corpus(cfg)
        self.retrievers: dict[str, Retriever] = {"bm25": BM25Retriever(self.chunks)}
        self.notes: list[str] = []
        emb = embedder or OllamaEmbedder()
        ok, why = emb.available()
        if ok:
            cache = Path(resolve_path(cfg, "interim_dir")) / "rag"
            self.retrievers["dense"] = DenseRetriever(self.chunks, emb, cache)
            self.retrievers["hybrid"] = HybridRetriever([self.retrievers["bm25"], self.retrievers["dense"]])
        else:
            self.notes.append(f"Dense retrieval off: {why}. Using BM25 only.")
        self.retriever_name = retriever if retriever in self.retrievers else "bm25"

    @property
    def retriever(self) -> Retriever:
        return self.retrievers[self.retriever_name]

    def retrieve(self, question: str, k: int | None = None) -> list[tuple[Chunk, float]]:
        return self.retriever.search(question, k or self.k)

    @staticmethod
    def context(sources: Sequence[Chunk]) -> str:
        return "\n\n".join(f"[S{i}] {c.title}\n{c.text}" for i, c in enumerate(sources, 1))

    def ask(self, question: str, k: int | None = None, max_tokens: int = 900) -> Answer:
        t0 = time.perf_counter()
        hits = self.retrieve(question, k)
        t_ret = time.perf_counter() - t0
        sources, scores = [c for c, _ in hits], [s for _, s in hits]
        ctx = self.context(sources)
        if self.provider is None:
            text = "No language model selected; the most relevant sources are listed below."
            return Answer(question, text, sources, scores, None, None, 0.0, t_ret)
        prompt = f"Sources:\n\n{ctx}\n\nQuestion: {question}"
        try:
            c: Completion = self.provider.complete(SYSTEM, prompt, max_tokens=max_tokens)
        except LLMError as e:
            return Answer(question, f"The language model is unavailable: {e}", sources, scores,
                          self.provider.name, getattr(self.provider, "model", None), 0.0, t_ret, error=str(e))
        cited, invalid = citations(c.text, len(sources))
        return Answer(question, c.text, sources, scores, c.provider, c.model, c.seconds, t_ret,
                      invalid_citations=invalid, ungrounded_numbers=ungrounded(c.text, ctx), cited=cited,
                      usage={"input_tokens": c.input_tokens, "output_tokens": c.output_tokens})


# --------------------------------------------------------------------------- visit explanations
EXPLAIN_SYSTEM = """You explain one prediction of a purchase-intent model to a shop manager.
Use ONLY the facts given. Mention at most three features, by their exact names, and only features listed.
Positive contributions push towards a purchase, negative ones away. Do not invent numbers.
Write two short paragraphs: (1) why the visit got this probability, (2) one concrete next step for the shop."""


@dataclass
class Explanation:
    text: str
    unknown_features: list[str]
    ungrounded_numbers: list[str]
    provider: str
    seconds: float

    @property
    def grounded(self) -> bool:
        return not self.unknown_features and not self.ungrounded_numbers


def explain_visit(provider: Provider, model_label: str, probability: float, threshold: float, intent: str,
                  contributions: Sequence[tuple[str, str, float]], k: int, all_features: Sequence[str]) -> Explanation:
    """``contributions``: (feature, value, log-odds contribution), most important first."""
    lines = [f"Model: {model_label}, scored after the first {k} clicks.",
             f"Purchase probability: {probability:.0%} (buyer threshold {threshold:.0%}); intent level: {intent}.",
             "Top feature contributions from LightGBM TreeSHAP (log-odds):"]
    lines += [f"- {f} = {v}: {c:+.2f}" for f, v, c in contributions]
    facts = "\n".join(lines)
    c = provider.complete(EXPLAIN_SYSTEM, facts, max_tokens=400)
    shown = {f for f, _, _ in contributions}
    mentioned = {f for f in all_features if re.search(rf"\b{re.escape(f)}\b", c.text)}
    return Explanation(c.text, sorted(mentioned - shown), ungrounded(c.text, facts), c.provider, c.seconds)
