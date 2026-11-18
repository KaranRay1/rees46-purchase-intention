"""Step 11: evaluate the RAG analyst assistant (retrieval and answers).

Needs the result tables (steps 03-07, 10). Retrieval is evaluated for every available
retriever (BM25 always; dense and hybrid when the Ollama embedding model is pulled). Answers
are evaluated for each local Ollama model in ``--models`` that is pulled (default: the
``OLLAMA_MODEL`` setting, ``llama3.1:8b``), so open models can be compared on the same
questions. Everything runs locally and costs nothing.

Writes to ``results/tables``: ``rag_retrieval_<mode>.csv`` (summary per retriever),
``rag_retrieval_questions_<mode>.csv``, ``rag_answers_<mode>.csv`` (every answer with its
scores), ``rag_answer_summary_<mode>.csv`` and ``rag_eval_<mode>.json``.

Examples::

    python scripts/11_rag_eval.py --mode full
    python scripts/11_rag_eval.py --mode full --models llama3.1:8b qwen3:14b --n 40
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_parser, print_table, setup, stage  # noqa: E402

import polars as pl  # noqa: E402

from src.llm.evaluation import (answer_eval, question_set, retrieval_metrics, sample,  # noqa: E402
                                summarise_answers, summarise_retrieval)
from src.llm.providers import OllamaProvider, setting  # noqa: E402
from src.llm.rag import Analyst  # noqa: E402
from src.serving.results import ResultStore  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--models", nargs="*", default=None,
                   help="local Ollama models to evaluate (default: OLLAMA_MODEL); `--models` alone = retrieval only")
    p.add_argument("--retriever", default="hybrid", help="retriever for the answer evaluation (falls back to bm25)")
    p.add_argument("--n", type=int, default=None, help="questions for the answer evaluation (default: all)")
    p.add_argument("--k", type=int, default=5, help="sources given to the model")
    args = p.parse_args(argv)
    cfg = setup(args)
    store = ResultStore(cfg)
    store.tables_dir.mkdir(parents=True, exist_ok=True)

    analyst = Analyst(cfg, None, retriever=args.retriever, k=args.k)
    for n in analyst.notes:
        print(f"note: {n}")
    questions = question_set(cfg)
    summary: dict = {"mode": cfg["data"]["mode"], "questions": len(questions), "k": args.k,
                     "retrievers": list(analyst.retrievers), "notes": analyst.notes}

    with stage(f"Retrieval ({', '.join(analyst.retrievers)}) on {len(questions)} questions"):
        per_q = retrieval_metrics(analyst, questions)
        ret = summarise_retrieval(per_q)
        per_q.write_csv(store.path("rag_retrieval_questions", "csv"))
        ret.write_csv(store.path("rag_retrieval", "csv"))
        print_table(ret.with_columns(pl.col(pl.Float64).round(3)).to_dicts())
        summary["retrieval"] = ret.to_dicts()

    answers, sums = [], []
    for name in (setting("OLLAMA_MODEL"),) if args.models is None else args.models:
        prov = OllamaProvider(model=name)
        ok, why = prov.available()
        if not ok:
            print(f"skip {name}: {why}")
            summary.setdefault("skipped", {})[name] = why
            continue
        analyst.provider = prov
        qs = sample(questions, args.n)
        with stage(f"Answers · {why} · {analyst.retriever_name} retrieval · {len(qs)} questions"):
            a = answer_eval(analyst, qs, f"ollama:{prov.model}")
            answers.append(a)
            s = summarise_answers(a)
            sums.append(s)
            print_table(s.with_columns(pl.col(pl.Float64).round(3)).to_dicts())
    if answers:
        pl.concat(answers, how="diagonal_relaxed").write_csv(store.path("rag_answers", "csv"))
        allsum = pl.concat(sums, how="diagonal_relaxed")
        allsum.write_csv(store.path("rag_answer_summary", "csv"))
        summary["answers"] = allsum.to_dicts()
    store.path("rag_eval", "json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote rag_retrieval, rag_answers and rag_eval tables for mode {cfg['data']['mode']}")


if __name__ == "__main__":
    main()
