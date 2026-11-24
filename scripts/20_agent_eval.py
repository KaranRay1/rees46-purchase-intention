"""Step 20: does a tool-using agent answer better than plain RAG?

Needs Ollama with the chosen models, and the result tables of the earlier steps. Both systems use
the same local model and are scored on two question sets:

* **standard**: the RAG evaluation questions (step 11: hand-written, including unanswerable
  ones, plus questions generated from the result tables), a stratified sample of ``--n``;
* **tool**: questions that need a table lookup, a calculation or a model run (segment analysis,
  user-history gain, campaign value under new settings, significance after Holm), generated
  with their expected answers by ``src.llm.evaluation.tool_questions``.

Metrics per system x model x question set: accuracy, correct abstention, numeric grounding
(every number found in the sources or tool outputs), seconds per answer, tool calls per answer
and the share of tool calls that failed.

Writes ``agent_eval_answers`` and ``agent_eval_summary`` (CSV) and ``agent_eval`` (JSON).

Examples::

    python scripts/20_agent_eval.py --mode full --models llama3.1:8b qwen3:14b
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_parser, print_table, setup, stage  # noqa: E402

import polars as pl  # noqa: E402

from src.llm.agent import Agent, Tools  # noqa: E402
from src.llm.evaluation import question_set, sample, score_answer, tool_questions  # noqa: E402
from src.llm.providers import OllamaProvider, setting  # noqa: E402
from src.llm.rag import Analyst  # noqa: E402
from src.serving.results import ResultStore  # noqa: E402


def run(system: str, ask, qs, label: str, qset: str) -> list[dict]:
    rows = []
    for i, q in enumerate(qs, 1):
        a = ask(q.question)
        s = score_answer(q, a.text) if not a.error else {"correct": 0.0, "abstained": 0.0}
        steps = getattr(a, "steps", [])
        rows.append({"system": system, "model": label, "question_set": qset, "id": q.id, "answerable": q.answerable,
                     **s, "error": a.error or "", "grounded_numbers": float(not a.ungrounded_numbers),
                     "ungrounded": ", ".join(a.ungrounded_numbers), "seconds": a.seconds,
                     "tool_calls": len(steps), "tool_errors": sum(st.error for st in steps),
                     "tools": ", ".join(st.tool for st in steps), "answer": a.text})
        print(f"   {i:>3}/{len(qs)} {q.id:<28} {'ok ' if s['correct'] else 'BAD'} {a.seconds:5.1f}s"
              + (f" tools={len(steps)}" if system == "agent" else ""), flush=True)
    return rows


def main(argv: list[str] | None = None) -> None:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+", default=None, help="Ollama models (default: OLLAMA_MODEL)")
    p.add_argument("--n", type=int, default=30, help="standard questions (stratified sample; default 30)")
    p.add_argument("--max-steps", type=int, default=5)
    p.add_argument("--retriever", default="hybrid")
    args = p.parse_args(argv)
    cfg = setup(args)
    store = ResultStore(cfg)
    analyst = Analyst(cfg, None, retriever=args.retriever, k=5)
    tools = Tools(cfg, analyst=Analyst(cfg, None, retriever=args.retriever, k=4))
    sets = {"standard": sample(question_set(cfg), args.n), "tool": tool_questions(cfg)}
    print("questions: " + ", ".join(f"{k} {len(v)}" for k, v in sets.items()))

    rows, skipped = [], {}
    for name in args.models or [setting("OLLAMA_MODEL")]:
        prov = OllamaProvider(model=name)
        ok, why = prov.available()
        if not ok:
            print(f"skip {name}: {why}")
            skipped[name] = why
            continue
        analyst.provider = prov
        agent = Agent(cfg, prov, tools=tools, max_steps=args.max_steps)
        for qset, qs in sets.items():
            with stage(f"{name} · RAG · {qset} ({len(qs)})"):
                rows += run("rag", analyst.ask, qs, name, qset)
            with stage(f"{name} · agent · {qset} ({len(qs)})"):
                rows += run("agent", agent.ask, qs, name, qset)
    if not rows:
        raise SystemExit("No model could be evaluated: " + json.dumps(skipped))

    per = pl.DataFrame(rows)
    per.write_csv(store.path("agent_eval_answers", "csv"))
    ok = per.filter(pl.col("error") == "")
    summ = ok.group_by("system", "model", "question_set").agg(
        n=pl.len(),
        accuracy=pl.col("correct").filter(pl.col("answerable")).mean(),
        abstention=pl.col("correct").filter(~pl.col("answerable")).mean(),
        numeric_grounding=pl.col("grounded_numbers").mean(),
        seconds_mean=pl.col("seconds").mean(),
        tool_calls_mean=pl.col("tool_calls").mean(),
        tool_error_rate=pl.col("tool_errors").sum() / pl.col("tool_calls").sum().clip(1),
    ).sort("model", "question_set", "system")
    summ.write_csv(store.path("agent_eval_summary", "csv"))
    store.path("agent_eval", "json").write_text(json.dumps(
        {"mode": cfg["data"]["mode"], "questions": {k: len(v) for k, v in sets.items()}, "max_steps": args.max_steps,
         "retriever": analyst.retriever_name, "skipped": skipped, "summary": summ.to_dicts()}, indent=2, default=str))
    print()
    print_table(summ.with_columns(pl.col(pl.Float64).round(3)).to_dicts())


if __name__ == "__main__":
    main()
