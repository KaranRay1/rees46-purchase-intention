"""Tool-using analyst agent: a local LLM that looks things up and runs the project's code.

The RAG :class:`~src.llm.rag.Analyst` can only quote passages it retrieved. The agent can also
**act**: read a result table, compare two models, score a session described in words, or value a
discount campaign under new assumptions. It uses Ollama's function calling, so everything still
runs locally.

Loop: the model gets the question and the tool definitions; whenever it asks for tools they are
run and their output is sent back; after at most ``max_steps`` rounds it must answer. The answer
is checked like the RAG answers: every number must appear in a tool output (or the question),
otherwise it is flagged as ungrounded.

Tools
-----
search_docs     retrieve passages from the knowledge base (report, cards, README, result fact sheets)
list_tables     the result tables that can be read, with their columns
read_table      rows of one result table, filtered by K / model / any column
compare_models  PR-AUC, paired test and latency for two models at one K
recommend_engine which model the engine should run at one K (fastest of those tied for best), and why
score_session   score a visit described as a list of clicks with the trained models
campaign_value  profit of the discount campaign for one model under chosen settings
"""

from __future__ import annotations

import ast
import inspect
import json
import re
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import polars as pl

from src.config import resolve_path
from src.llm.corpus import LABEL
from src.llm.providers import LLMError
from src.llm.rag import Analyst, ungrounded

SYSTEM = """You are the analyst agent for a purchase-intent prediction project (REES46 e-commerce \
clickstream, Oct 2019 - Apr 2020). Models predict, after the first K clicks of a visit (K = 3, 5 or 10), \
whether the visit ends in a purchase. Machine-learning models (rule baseline, logistic regression, random \
forest, LightGBM) are by Merve Kara; deep-learning models (MLP, TabNet, GRU) by Karan Ray. The GRU is the \
deployed engine.

Use the tools to get facts; never answer a factual or numeric question from memory.
- Exact results (PR-AUC, latency, tests, profit, segments): read_table or compare_models.
- Which model to deploy / the engine at some K: recommend_engine.
- Concepts, methods, decisions, limitations: search_docs.
- "What would the model say about a visit ...": score_session.
- "What is a campaign worth with X% discount ...": campaign_value.
Main tables for read_table:
{tables}
Copy numbers exactly as the tools return them. If the tools cannot answer, say so plainly.
Final answer: concise (about 150 words), plain prose or a short list, no tool syntax."""

MODEL_ALIASES = {**{v.lower(): k for k, v in LABEL.items()}, **{k: k for k in LABEL},
                 "gru": "gru", "rf": "random_forest", "random forest": "random_forest", "lr": "logistic_regression",
                 "logistic": "logistic_regression", "rule": "rule_based", "rules": "rule_based",
                 "rule-based": "rule_based", "lgbm": "lightgbm", "light gbm": "lightgbm"}


TABLE_GUIDE = {
    "rq1_model_comparison": "PR-AUC/ROC-AUC/F1 of every model at every K (main results)",
    "rq1_pairwise_tests": "paired bootstrap + McNemar tests between models",
    "rq2_by_k": "how results change with K",
    "rq3_ablation": "feature-group ablation for LightGBM and GRU (which feature groups matter)",
    "rq4_efficiency": "latency, model size, training time",
    "calibration": "calibrated probabilities, ECE, Brier per model",
    "hypothesis_scoreboard": "which hypotheses were supported",
    "transformer": "Transformer vs GRU", "tuning": "Optuna tuning results",
    "stacking": "stacked ensembles", "embeddings_ablation": "item2vec / text embedding features",
    "user_history_ablation": "user-history (G6) features vs session-only, test PR-AUC per seed",
    "user_history_tests": "gain (diff) in PR-AUC from user-history features per K, with CI and Holm p",
    "user_history_segments": "user-history gain for new vs returning users",
    "business_value": "profit per month of every model's campaign",
    "business_value_sensitivity": "profit gain by discount x uplift",
    "segment_metrics": "PR-AUC, lift, flag rate, TPR per segment (category, price, cart, time, user type)",
    "segment_disparity": "per dimension: worst_segment and best_segment by PR-AUC lift, gaps",
    "conformal": "conformal prediction coverage", "retraining": "drift and retraining policies",
    "monitoring_drift": "input drift (PSI) per feature",
}


ID_COLUMNS = ["k", "model", "model_a", "model_b", "feature_set", "dimension", "segment", "group", "policy", "system"]
DEFAULT_COLUMNS = {   # wide tables: what read_table shows unless columns are asked for
    "rq1_model_comparison": ["k", "model", "n_seeds", "base_rate", "pr_auc_mean", "pr_auc_std", "roc_auc_mean",
                             "f1_mean", "precision_mean", "recall_mean", "train_time_sec_mean"],
    "rq4_efficiency": ["k", "model", "test_pr_auc_mean", "latency_b1_p50_ms", "latency_b1_p95_ms", "latency_b1_p99_ms",
                       "throughput_b1_per_sec", "model_size_mb", "n_parameters", "train_time_sec_mean"],
    "rq1_pairwise_tests": ["k", "model_a", "model_b", "pr_auc_a", "pr_auc_b", "pr_auc_diff", "diff_ci_low",
                           "diff_ci_high", "p_bootstrap_holm"],
    "segment_metrics": ["k", "model", "dimension", "segment", "n", "base_rate", "pr_auc", "pr_auc_lift",
                        "calibration_gap", "flag_rate", "tpr"],
}


def _coerce(v: Any) -> Any:
    """Small models often send lists/objects as JSON text; turn them back into Python values."""
    if isinstance(v, str) and v.strip()[:1] in "[{":
        for parse in (json.loads, ast.literal_eval):
            try:
                return parse(v)
            except (ValueError, SyntaxError):
                continue
    if isinstance(v, list):
        return [_coerce(x) for x in v]
    if isinstance(v, dict):
        return {k: _coerce(x) for k, x in v.items()}
    return v


_TEXT_CALL = re.compile(r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"(?:parameters|arguments)"\s*:\s*(\{.*\})\s*\}', re.S)


def text_tool_call(content: str) -> dict | None:
    """A tool call the model wrote as text instead of using function calling, if any."""
    m = _TEXT_CALL.search(content or "")
    if not m:
        return None
    try:
        return {"function": {"name": m.group(1), "arguments": json.loads(m.group(2))}}
    except json.JSONDecodeError:
        return None


def model_key(name: str) -> str:
    n = str(name).strip().lower().replace("_", " ")
    for cand in (n, n.replace(" ", "_"), n.split(" (")[0]):
        if cand in MODEL_ALIASES:
            return MODEL_ALIASES[cand]
    raise ValueError(f"unknown model {name!r}; use one of {sorted(LABEL)}")


def _fmt(df: pl.DataFrame, limit: int = 25) -> str:
    """Compact text table; floats to 6 decimals (enough to quote any rounding exactly)."""
    if df.height == 0:
        return "(no rows)"
    df = df.head(limit).with_columns(
        [pl.col(c).round(6) for c, t in df.schema.items() if t in (pl.Float32, pl.Float64)])
    cols = df.columns
    lines = [" | ".join(cols)] + [" | ".join("" if v is None else str(v) for v in row) for row in df.iter_rows()]
    return "\n".join(lines)


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": properties,
                                                            "required": required}}}


@dataclass
class Step:
    tool: str
    arguments: dict
    output: str
    seconds: float
    error: bool = False


@dataclass
class AgentAnswer:
    question: str
    text: str
    steps: list[Step]
    model: str | None
    seconds: float
    ungrounded_numbers: list[str] = field(default_factory=list)
    error: str | None = None
    usage: dict = field(default_factory=dict)

    @property
    def grounded(self) -> bool:
        return not self.ungrounded_numbers

    @property
    def tools_used(self) -> list[str]:
        return [s.tool for s in self.steps]


class Tools:
    """The agent's tools, bound to one config (mode, result tables, trained runs)."""

    def __init__(self, cfg: Mapping[str, Any], analyst: Analyst | None = None) -> None:
        self.cfg = cfg
        self.mode = cfg["data"]["mode"]
        self.tables_dir = Path(resolve_path(cfg, "tables_dir"))
        self._analyst = analyst
        self._predictors: dict[int, Any] = {}

    # ------------------------------------------------------------------ registry
    def specs(self) -> list[dict]:
        k_prop = {"type": "integer", "description": "clicks seen before predicting: 3, 5 or 10"}
        model_prop = {"type": "string", "description": "model: " + ", ".join(sorted(LABEL))}
        return [
            _fn("search_docs", "Search the project documents (report, model/data cards, README, result fact sheets) "
                "for passages about a topic.", {"query": {"type": "string"}}, ["query"]),
            _fn("list_tables", "List the result tables that read_table can open, with their columns.", {}, []),
            _fn("read_table", "Read rows of one result table. Filter by k, model, or other columns.",
                {"table": {"type": "string", "description": "table name from list_tables, e.g. rq1_model_comparison"},
                 "k": k_prop, "model": model_prop,
                 "where": {"type": "object", "description": "extra equality filters, e.g. {\"dimension\": \"price\"}"},
                 "columns": {"type": "array", "items": {"type": "string"}, "description": "columns to keep"},
                 "limit": {"type": "integer"}}, ["table"]),
            _fn("compare_models", "Compare two models at one K: test PR-AUC (mean of seeds), the paired bootstrap "
                "test (Holm-corrected) and scoring latency.",
                {"k": k_prop, "model_a": model_prop, "model_b": model_prop}, ["k", "model_a", "model_b"]),
            _fn("recommend_engine", "Which model the engine should run after K clicks: the fastest of the models "
                "statistically tied with the most accurate one.", {"k": k_prop}, ["k"]),
            _fn("score_session", "Score one visit with the trained models. Describe the first K clicks in order; "
                "K is the number of clicks (3, 5 or 10).",
                {"clicks": {"type": "array", "description": "clicks in time order", "items": {
                    "type": "object", "properties": {
                        "event_type": {"type": "string", "enum": ["view", "cart"]},
                        "price": {"type": "number"},
                        "category": {"type": "string", "description": "e.g. electronics.smartphone"},
                        "brand": {"type": "string"},
                        "product": {"type": "integer", "description": "same number = same product"},
                        "minutes_after_previous": {"type": "number"}},
                    "required": ["event_type", "price"]}},
                 "model": {"type": "string", "description": "optional: one model; default all"}}, ["clicks"]),
            _fn("campaign_value", "Profit per month (whole shop) of a discount campaign that skips the visits a "
                "model scores as most likely to buy, and the gain over offering the discount to everyone.",
                {"k": k_prop, "model": model_prop,
                 "discount": {"type": "number", "description": "fraction, e.g. 0.1 for 10%"},
                 "uplift": {"type": "number", "description": "share of non-buyers the offer converts, e.g. 0.02"},
                 "order_value": {"type": "number", "description": "average order value in $ (optional)"}},
                ["k", "model"]),
        ]

    def run(self, name: str, args: Mapping[str, Any]) -> str:
        fn: Callable[..., str] | None = {"search_docs": self.search_docs, "list_tables": self.list_tables,
                                         "read_table": self.read_table, "compare_models": self.compare_models,
                                         "recommend_engine": self.recommend_engine,
                                         "score_session": self.score_session,
                                         "campaign_value": self.campaign_value}.get(name)
        if fn is None:
            raise ValueError(f"unknown tool {name!r}")
        args = {k: _coerce(v) for k, v in dict(args or {}).items()}
        params = inspect.signature(fn).parameters
        args = {k: v for k, v in args.items() if k in params and v not in ("", None, [], {})}
        return fn(**args)

    # ------------------------------------------------------------------ tools
    def search_docs(self, query: str) -> str:
        if self._analyst is None:
            self._analyst = Analyst(self.cfg, provider=None)
        hits = self._analyst.retrieve(query, 4)
        return "\n\n".join(f"[D{i}] {c.title}\n{c.text[:1200]}" for i, (c, _) in enumerate(hits, 1))

    def _tables(self) -> dict[str, Path]:
        suf = f"_{self.mode}.csv"
        return {p.name[: -len(suf)]: p for p in sorted(self.tables_dir.glob(f"*{suf}"))
                if not p.name.startswith(("rag_", "daily_profile", "tuning_trials"))}

    def list_tables(self) -> str:
        out = []
        for name, p in self._tables().items():
            cols = pl.read_csv(p, n_rows=1).columns
            out.append(f"{name}: {', '.join(cols[:18])}{' ...' if len(cols) > 18 else ''}")
        return "\n".join(out)

    def read_table(self, table: str, k: int | None = None, model: str | list | None = None,
                   where: dict | str | None = None, columns: list[str] | str | None = None,
                   limit: int | str | None = 20) -> str:
        tables = self._tables()
        name = str(table).removesuffix(".csv").removesuffix(f"_{self.mode}")
        if name not in tables:
            close = [t for t in tables if name.lower() in t or t in name.lower()]
            raise ValueError(f"unknown table {table!r}" + (f"; did you mean {close}?" if close else "; call list_tables"))
        df = pl.read_csv(tables[name], infer_schema_length=10000)
        notes = []
        if k is not None and "k" in df.columns:
            df = df.filter(pl.col("k") == int(k))
        mcols = [c for c in ("model", "model_a", "model_b", "engine") if c in df.columns]
        if model is not None and mcols:
            wanted = [model_key(m) for m in (model if isinstance(model, list) else [model])
                      if str(m).lower() not in ("all", "every", "*")]
            if wanted:
                sub = df.filter(pl.any_horizontal([pl.col(c).is_in(wanted) for c in mcols]))
                if sub.height == 0:
                    notes.append(f"no rows for model {wanted}; models in this table: "
                                 f"{sorted(set(df[mcols[0]].drop_nulls().to_list()))}, so all are shown")
                else:
                    df = sub
        if isinstance(where, str):                      # e.g. "dimension='price', k=5"
            where = dict(re.findall(r"(\w+)\s*[=:]\s*['\"]?([^,'\"]+)['\"]?", where))
        for c, v in (where or {}).items():
            if c in df.columns:
                df = df.filter(pl.col(c).cast(pl.String).str.to_lowercase() == str(v).lower())
            else:
                notes.append(f"ignored filter on unknown column {c!r}")
        if columns:
            asked = [columns] if isinstance(columns, str) else list(columns)
            keys = [c for c in ID_COLUMNS if c in df.columns]
            keep: list[str] = []
            for a in asked:
                a_l = str(a).lower()
                words = [w for w in re.split(r"[\s_\-]+", a_l) if w]
                hits = ([c for c in df.columns if c.lower() == a_l] or [c for c in df.columns if a_l in c.lower()]
                        or [c for c in df.columns if words and all(w in c.lower() for w in words)])
                if not hits:
                    notes.append(f"no column like {a!r}")
                keep += hits
            df = df.select(list(dict.fromkeys(keys + keep))) if keep else df
            if not keep:
                notes.append("all columns shown")
        elif name in DEFAULT_COLUMNS:
            df = df.select([c for c in DEFAULT_COLUMNS[name] if c in df.columns])
        n, lim = df.height, int(limit or 20)
        head = f"{name}: {n} matching rows" + (f" (first {lim} shown)" if n > lim else "")
        if n == 0 and "k" in df.columns:
            notes.append("try without some filters")
        return head + ("\nNote: " + "; ".join(notes) if notes else "") + "\n" + _fmt(df, lim)

    def compare_models(self, k: int, model_a: str, model_b: str) -> str:
        a, b = model_key(model_a), model_key(model_b)
        t = self._tables()
        out = []
        if "rq1_model_comparison" in t:
            r = pl.read_csv(t["rq1_model_comparison"]).filter((pl.col("k") == int(k)) & pl.col("model").is_in([a, b]))
            out.append("PR-AUC on the test weeks (mean over seeds):\n" + _fmt(
                r.select("model", "n_seeds", "pr_auc_mean", "pr_auc_std", "roc_auc_mean", "f1_mean")))
        if "rq1_pairwise_tests" in t:
            p = pl.read_csv(t["rq1_pairwise_tests"]).filter(
                (pl.col("k") == int(k)) & (((pl.col("model_a") == a) & (pl.col("model_b") == b))
                                           | ((pl.col("model_a") == b) & (pl.col("model_b") == a))))
            if p.height:
                r = p.row(0, named=True)
                sig = r["p_bootstrap_holm"] < float(self.cfg["evaluation"]["alpha"])
                out.append(f"Paired bootstrap ({r['model_a']} minus {r['model_b']}): PR-AUC difference "
                           f"{r['pr_auc_diff']:.4f}, 95% CI {r['diff_ci_low']:.4f} to {r['diff_ci_high']:.4f}, "
                           f"Holm p {r['p_bootstrap_holm']:.4f} -> {'significant' if sig else 'not significant (a tie)'}.")
            else:
                out.append("No paired test for this pair at this K.")
        if "rq4_efficiency" in t:
            e = pl.read_csv(t["rq4_efficiency"]).filter((pl.col("k") == int(k)) & pl.col("model").is_in([a, b]))
            keep = [c for c in ("model", "latency_b1_p95_ms", "train_time_sec", "model_size_mb") if c in e.columns]
            out.append("Efficiency:\n" + _fmt(e.select(keep)))
        return "\n\n".join(out) or "No result tables for this comparison."

    def recommend_engine(self, k: int) -> str:
        from src.serving.recommend import recommended

        t = self._tables()
        if "rq1_model_comparison" not in t:
            raise ValueError("no model comparison table")
        rq1 = pl.read_csv(t["rq1_model_comparison"])
        pairs = pl.read_csv(t["rq1_pairwise_tests"]) if "rq1_pairwise_tests" in t else None
        eff = pl.read_csv(t["rq4_efficiency"]) if "rq4_efficiency" in t else None
        ship, tied = recommended(rq1, pairs, eff, int(k))
        part = rq1.filter(pl.col("k") == int(k)).sort("pr_auc_mean", descending=True)
        lat = {} if eff is None else dict(zip(*eff.filter(pl.col("k") == int(k)).select("model", "latency_b1_p95_ms")))
        lines = [f"Engine model after K={k} clicks: {LABEL[ship]}.",
                 f"Most accurate: {LABEL[part['model'][0]]} (PR-AUC {part['pr_auc_mean'][0]:.4f}). Statistically tied "
                 f"with it (paired bootstrap, Holm): {', '.join(LABEL[m] for m in tied)}.",
                 "Among the tied models the fastest is chosen. p95 latency per visit: "
                 + ", ".join(f"{LABEL[m]} {lat[m]:.4f} ms" for m in tied if m in lat) + "."]
        return "\n".join(lines)

    def score_session(self, clicks: list[dict], model: str | None = None) -> str:
        from src.serving.predictor import SessionPredictor

        k = len(clicks)
        if k not in (3, 5, 10):
            raise ValueError(f"give exactly 3, 5 or 10 clicks (got {k}); the models were trained for those K")
        t0 = datetime(2020, 4, 15, 14, 0, tzinfo=timezone.utc)
        rows, t = [], t0
        for c in clicks:
            t = t + timedelta(minutes=float(c.get("minutes_after_previous", 1) or 1))
            cat = str(c.get("category") or "unknown")
            brand = str(c.get("brand") or "unknown")
            pid = int(c.get("product") or zlib.crc32(f"{cat}|{brand}|{c.get('price')}".encode()) % 10**8)
            rows.append({"event_time": t, "event_type": str(c["event_type"]), "product_id": pid,
                         "category_id": zlib.crc32(cat.encode()) % 10**8, "category_code": cat, "brand": brand,
                         "price": float(c["price"])})
        if k not in self._predictors:
            self._predictors[k] = SessionPredictor(self.cfg, k)
        pred = self._predictors[k]
        only = [model_key(model)] if model and str(model).lower() not in ("all", "every", "any", "*") else None
        scores = pred.predict(pl.DataFrame(rows), models=only)
        lines = [f"Scored after K={k} clicks ({sum(c['event_type'] == 'cart' for c in clicks)} cart events):"]
        for s in scores:
            prob = f"{s.probability:.1%}" if s.probability is not None else "n/a"
            lines.append(f"- {LABEL.get(s.model, s.model)}: purchase probability {prob}, "
                         f"{'likely buyer' if s.decision else 'not flagged'} (raw score {s.score:.3f}, "
                         f"threshold {s.threshold:.3f})")
        return "\n".join(lines)

    def campaign_value(self, k: int, model: str, discount: float = 0.10, uplift: float = 0.02,
                       order_value: float | None = None) -> str:
        from src.evaluation.business_value import monthly_scale
        from src.serving.campaign import Offer, best_share, gain_table, suppression_value
        from src.serving.results import ResultStore, test_predictions

        m = model_key(model)
        discount = float(discount) / 100 if float(discount) > 1 else float(discount)
        uplift = float(uplift) / 100 if float(uplift) > 1 else float(uplift)
        pr = test_predictions(self.cfg, int(k), m)
        if pr is None or not pr.height:
            raise ValueError(f"no saved test predictions for {m} at K={k}")
        store = ResultStore(self.cfg)
        if order_value is None:
            info = store.json("business_value_summary") or {}
            order_value = float((info.get("offer") or {}).get("aov") or 300.0)
        offer = Offer(aov=float(order_value), discount=discount, uplift=uplift)
        v = suppression_value(gain_table(pr["y_true"].to_numpy(), pr["score"].to_numpy(),
                                         np.round(np.arange(0, 1.0001, 0.01), 2)), offer)
        best = best_share(v)
        scale, days, sample = monthly_scale(store.json("data_summary") or {}, int(k))
        net, blanket = best["net"] * scale, float(v["net_blanket"][0]) * scale
        rule = ("offer the discount to everyone" if best["share"] == 0 else
                f"skip the top {best['share']:.0%} of visits by score (no discount when the score is at least "
                f"{best['score_cutoff']:.3f})")
        return (f"{LABEL[m]} at K={k}, {discount:.0%} discount, {uplift:.1%} of non-buyers converted, order value "
                f"${order_value:,.0f}:\n- best rule: {rule}\n- net value per month (whole shop): ${net:,.0f}\n"
                f"- offering to everyone: ${blanket:,.0f}\n- gain from using the model: ${net - blanket:,.0f}\n"
                f"(Projection: {days:.0f} test days scaled to 30 days and a {sample:.0%} sample to the whole shop.)")


class Agent:
    """Tool-calling analyst. ``provider`` must be an :class:`~src.llm.providers.OllamaProvider`."""

    def __init__(self, cfg: Mapping[str, Any], provider, tools: Tools | None = None, max_steps: int = 5,
                 prefetch_docs: bool = False) -> None:
        self.provider, self.max_steps, self.prefetch_docs = provider, max_steps, prefetch_docs
        self.tools = tools or Tools(cfg)

    def ask(self, question: str, max_tokens: int = 900) -> AgentAnswer:
        t0 = time.perf_counter()
        msgs: list[dict] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
        steps: list[Step] = []
        usage = {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0}
        specs = self.tools.specs()
        nudged = False
        avail = self.tools._tables()
        guide = "\n".join(f"- {t}: {d}" for t, d in TABLE_GUIDE.items() if t in avail)
        msgs[0]["content"] = SYSTEM.replace("{tables}", guide)
        model = getattr(self.provider, "model", None)
        if self.prefetch_docs:       # start from the passages RAG would use, then let the model act
            s0 = time.perf_counter()
            try:
                docs, err = self.tools.search_docs(question), False
            except Exception as e:  # noqa: BLE001
                docs, err = f"Error: {e}", True
            steps.append(Step("search_docs", {"query": question}, docs, time.perf_counter() - s0, err))
            msgs.append({"role": "assistant", "content": "",
                         "tool_calls": [{"function": {"name": "search_docs", "arguments": {"query": question}}}]})
            msgs.append({"role": "tool", "content": docs[:6000], "tool_name": "search_docs"})
        try:
            for i in range(self.max_steps + 1):
                last = i == self.max_steps
                r = self.provider.chat(msgs, None if last else specs, max_tokens=max_tokens)
                usage["input_tokens"] += r["input_tokens"]
                usage["output_tokens"] += r["output_tokens"]
                usage["llm_calls"] += 1
                calls = [] if last else r["tool_calls"]
                if not calls and not _clean(r["content"]) and not nudged:
                    nudged = True                          # some models return nothing once; ask again
                    msgs.append({"role": "user", "content": "Please answer the question now, using the tools if needed."})
                    continue
                if not calls and not last:
                    tc = text_tool_call(r["content"])
                    calls = [tc] if tc else []
                if not calls:
                    text = _clean(r["content"]) or "The agent did not produce an answer."
                    if text_tool_call(text):
                        text = "The agent could not finish within its step limit."
                    context = question + "\n" + "\n".join(s.output for s in steps)
                    context += "\n" + re.sub(r"(?<![\w.])-(?=\$?\d)", "", context)   # a loss may be quoted unsigned
                    return AgentAnswer(question, text, steps, model, time.perf_counter() - t0,
                                       ungrounded(text, context), usage=usage)
                msgs.append({"role": "assistant", "content": r["content"], "tool_calls": calls})
                for c in calls:
                    fn = c.get("function", {})
                    name, args = fn.get("name", ""), fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    args = {k: _coerce(v) for k, v in args.items()}
                    s0 = time.perf_counter()
                    try:
                        out, err = self.tools.run(name, args), False
                    except Exception as e:  # noqa: BLE001 - the model sees the error and can retry
                        out, err = f"Error: {e}", True
                    steps.append(Step(name, dict(args), out, time.perf_counter() - s0, err))
                    msgs.append({"role": "tool", "content": out[:6000], "tool_name": name})
        except LLMError as e:
            return AgentAnswer(question, f"The language model is unavailable: {e}", steps, model,
                               time.perf_counter() - t0, error=str(e), usage=usage)
        raise AssertionError("unreachable")


def _clean(text: str) -> str:
    """Drop any <think>...</think> block a reasoning model may still emit."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
