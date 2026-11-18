"""Evaluation of the RAG analyst: retrieval quality and answer quality.

Questions come from two places:

* **generated** from the result tables (every model x K for PR-AUC and latency, the engine
  choice and calibration per K), so the gold source and the exact expected figure are known;
* **hand-written** in ``config/rag_eval_questions.yaml``: conceptual questions with gold
  sources and expected phrases, plus questions the sources cannot answer.

Retrieval metrics (per retriever): recall@1/3/5 (a relevant chunk in the top k) and MRR@10.
Answer metrics (per language model): correctness (every expected phrase present; numbers
match in any rounding or percent form), correct abstention on unanswerable questions,
citation validity, numeric grounding, latency and tokens.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import polars as pl
import yaml

from src.config import PROJECT_ROOT, resolve_path
from src.llm.corpus import LABEL, Chunk
from src.llm.rag import Analyst, number_matches, numbers_in

ABSTAIN = re.compile(r"(do(es)? not (contain|include|mention|provide|cover|have)|don't|no (information|data|mention)|"
                     r"not (available|provided|covered|mentioned|included|in the sources)|cannot|can't|unable|"
                     r"isn't|is not (given|stated|available))", re.I)


@dataclass
class Question:
    id: str
    question: str
    gold: list[dict]
    expect: list[list[str]]
    answerable: bool = True
    origin: str = "manual"
    tags: list[str] = field(default_factory=list)


def is_relevant(chunk: Chunk, gold: Sequence[dict]) -> bool:
    for g in gold:
        if chunk.source == g["source"] and (not g.get("contains") or g["contains"].lower() in chunk.body.lower()):
            return True
    return False


def generated_questions(cfg: Mapping[str, Any]) -> list[Question]:
    tables = resolve_path(cfg, "tables_dir")
    mode = cfg["data"]["mode"]
    out: list[Question] = []

    def read(stem):
        p = Path(tables) / f"{stem}_{mode}.csv"
        return pl.read_csv(p) if p.exists() else None

    rq1, eff, cal = read("rq1_model_comparison"), read("rq4_efficiency"), read("calibration")
    if rq1 is not None:
        for r in rq1.iter_rows(named=True):
            m, k = r["model"], r["k"]
            out.append(Question(f"g_auc_{m}_k{k}", f"What test PR-AUC does {LABEL[m]} reach after {k} clicks?",
                                [{"source": "rq1_model_comparison", "contains": f"K={k} clicks"}],
                                [[f"{r['pr_auc_mean']:.3f}", f"{r['pr_auc_mean']:.4f}"]], origin="generated",
                                tags=["accuracy"]))
    if eff is not None:
        for r in eff.iter_rows(named=True):
            m, k = r["model"], r["k"]
            out.append(Question(f"g_lat_{m}_k{k}", f"How long does {LABEL[m]} take to score one visit (p95 latency) at K={k}?",
                                [{"source": "rq4_efficiency", "contains": f"K={k}"}],
                                [[f"{r['latency_b1_p95_ms']:.3f}"]], origin="generated", tags=["latency"]))
    if rq1 is not None and eff is not None:
        from src.serving.recommend import recommended

        pairs = read("rq1_pairwise_tests")
        for k in sorted(set(rq1["k"].to_list())):
            ship, _ = recommended(rq1, pairs, eff, k)
            out.append(Question(f"g_engine_k{k}", f"Which model should the engine run when scoring after {k} clicks?",
                                [{"source": "rq1_ml_vs_dl", "contains": f"K={k}"}],
                                [[LABEL[ship].split(" (")[0]]], origin="generated", tags=["engine"]))
    if cal is not None:
        for r in cal.filter(pl.col("model") == "gru").iter_rows(named=True):
            out.append(Question(f"g_ece_gru_k{r['k']}",
                                f"What is the GRU's expected calibration error after calibration at K={r['k']}?",
                                [{"source": "calibration", "contains": f"K={r['k']}"}],
                                [[f"{r['ece']:.3f}"]], origin="generated", tags=["calibration"]))
    return out


def manual_questions(path: Path = PROJECT_ROOT / "config" / "rag_eval_questions.yaml") -> list[Question]:
    rows = yaml.safe_load(path.read_text()) or []
    return [Question(r["id"], r["question"], r.get("gold", []),
                     [[str(x) for x in group] for group in r.get("expect", [])],   # YAML reads bare no/yes as booleans
                     r.get("answerable", True), "manual") for r in rows]


def question_set(cfg: Mapping[str, Any]) -> list[Question]:
    return manual_questions() + generated_questions(cfg)


# --------------------------------------------------------------------------- retrieval
def retrieval_metrics(analyst: Analyst, questions: Sequence[Question], ks=(1, 3, 5), depth: int = 10) -> pl.DataFrame:
    rows = []
    for name, ret in analyst.retrievers.items():
        for q in questions:
            if not q.answerable or not q.gold:
                continue
            hits = ret.search(q.question, depth)
            ranks = [i for i, (c, _) in enumerate(hits, 1) if is_relevant(c, q.gold)]
            first = ranks[0] if ranks else None
            rows.append({"retriever": name, "id": q.id, "origin": q.origin,
                         **{f"recall@{k}": float(first is not None and first <= k) for k in ks},
                         "rr@10": 1.0 / first if first else 0.0, "first_relevant_rank": first})
    return pl.DataFrame(rows)


def summarise_retrieval(per_q: pl.DataFrame) -> pl.DataFrame:
    cols = [c for c in per_q.columns if c.startswith("recall@")] + ["rr@10"]
    overall = per_q.group_by("retriever").agg([pl.col(c).mean() for c in cols], n=pl.len()).with_columns(
        origin=pl.lit("all"))
    by = per_q.group_by("retriever", "origin").agg([pl.col(c).mean() for c in cols], n=pl.len())
    return pl.concat([overall, by], how="diagonal_relaxed").rename({"rr@10": "mrr@10"}).sort("retriever", "origin")


# --------------------------------------------------------------------------- answers
def _contains(answer: str, phrase: str) -> bool:
    if re.fullmatch(r"[-+]?\d[\d.,]*%?", phrase.strip()):
        return any(number_matches(tok, phrase) for tok in numbers_in(answer))
    return phrase.lower() in answer.lower()


def score_answer(q: Question, text: str) -> dict:
    abstained = bool(ABSTAIN.search(text))
    if not q.answerable:
        return {"correct": float(abstained), "abstained": float(abstained)}
    ok = all(any(_contains(text, p) for p in group) for group in q.expect) if q.expect else True
    return {"correct": float(ok), "abstained": float(abstained)}


def answer_eval(analyst: Analyst, questions: Sequence[Question], label: str) -> pl.DataFrame:
    rows = []
    for q in questions:
        a = analyst.ask(q.question)
        s = score_answer(q, a.text) if not a.error else {"correct": 0.0, "abstained": 0.0}
        rows.append({"provider": label, "retriever": analyst.retriever_name, "id": q.id, "origin": q.origin,
                     "answerable": q.answerable, **s, "error": a.error or "",
                     "citations_valid": float(not a.invalid_citations), "cited_any": float(bool(a.cited)),
                     "grounded_numbers": float(not a.ungrounded_numbers),
                     "ungrounded": ", ".join(a.ungrounded_numbers), "seconds": a.seconds,
                     "input_tokens": a.usage.get("input_tokens", 0), "output_tokens": a.usage.get("output_tokens", 0),
                     "answer": a.text, "sources": ", ".join(c.id for c in a.sources)})
    return pl.DataFrame(rows)


def summarise_answers(per_q: pl.DataFrame) -> pl.DataFrame:
    ok = per_q.filter(pl.col("error") == "")
    return ok.group_by("provider", "retriever").agg(
        n=pl.len(),
        accuracy=pl.col("correct").filter(pl.col("answerable")).mean(),
        abstention=pl.col("correct").filter(~pl.col("answerable")).mean(),
        citation_validity=pl.col("citations_valid").mean(),
        cites_sources=pl.col("cited_any").filter(pl.col("answerable")).mean(),
        numeric_grounding=pl.col("grounded_numbers").mean(),
        seconds_mean=pl.col("seconds").mean(),
        seconds_p95=pl.col("seconds").quantile(0.95),
        input_tokens=pl.col("input_tokens").sum(),
        output_tokens=pl.col("output_tokens").sum(),
    ).join(per_q.group_by("provider", "retriever").agg(errors=(pl.col("error") != "").sum()),
           on=["provider", "retriever"], how="left").sort("provider", "retriever")


def sample(questions: Sequence[Question], n: int | None, seed: int = 0) -> list[Question]:
    """All manual questions plus a stratified sample of generated ones (n in total)."""
    if n is None or n >= len(questions):
        return list(questions)
    manual = [q for q in questions if q.origin == "manual"]
    gen = [q for q in questions if q.origin != "manual"]
    rng = random.Random(seed)
    by_tag: dict[str, list[Question]] = {}
    for q in gen:
        by_tag.setdefault(q.tags[0] if q.tags else "", []).append(q)
    picked, quota = [], max(n - len(manual), 0)
    while len(picked) < quota and any(by_tag.values()):
        for tag in sorted(by_tag):
            if by_tag[tag] and len(picked) < quota:
                picked.append(by_tag[tag].pop(rng.randrange(len(by_tag[tag]))))
    return manual + picked
