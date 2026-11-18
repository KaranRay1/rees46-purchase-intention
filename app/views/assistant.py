"""Analyst assistant: retrieval-augmented answers about the project, with citations and checks."""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
import streamlit as st

import ui
from src.llm.providers import OllamaEmbedder, OllamaProvider, setting
from src.llm.rag import Analyst

s = ui.store()
ink, pal = ui.INK[ui.theme()], ui.SERIES[ui.theme()]
ui.page_header("Analyst assistant",
               "Ask about the models, results, data or monitoring. Answers come only from the project's own "
               "documents and result tables, with numbered citations and automatic fact checks.", rq="AI")

NONE = "none"


@st.cache_data(ttl=30, show_spinner=False)
def status() -> dict:
    return {"models": OllamaProvider().chat_models(), "ollama": OllamaProvider().available(),
            "embed": OllamaEmbedder().available()}


@st.cache_resource(show_spinner="Indexing the knowledge base…")
def analyst(mode: str, retriever: str, k: int) -> Analyst:
    return Analyst(ui.cfg(), None, retriever=retriever, k=k)


ask_tab, eval_tab, how_tab = st.tabs(["Ask", "How well it works", "How it works"])

# --------------------------------------------------------------------------- ask
with ask_tab:
    stat = status()
    c1, c2, c3 = st.columns([1.4, 1, 0.8])
    choices = [*stat["models"], NONE]
    preferred = setting("OLLAMA_MODEL")
    default = preferred if preferred in choices else choices[0]
    model = c1.selectbox("Local language model (Ollama)", choices, index=choices.index(default), key="ai_model",
                         format_func=lambda m: "Sources only (no model)" if m == NONE else m,
                         help="Open models running on this computer through Ollama. Pull more with `ollama pull`.")
    rets = ["hybrid", "dense", "bm25"] if stat["embed"][0] else ["bm25"]
    ret_name = c2.selectbox("Retrieval", rets, key="ai_retriever",
                            format_func={"hybrid": "Hybrid (BM25 + embeddings)", "dense": "Embeddings",
                                         "bm25": "Keyword (BM25)"}.get,
                            help="Hybrid fuses keyword and embedding rankings with reciprocal rank fusion.")
    k_src = c3.slider("Sources", 3, 8, 5, key="ai_k")
    off = [] if stat["models"] else [f"Language model: {stat['ollama'][1]}"]
    if not stat["embed"][0]:
        off.append(f"Embeddings: {stat['embed'][1]}")
    if off:
        st.caption(":material/info: " + " · ".join(off))

    a = analyst(st.session_state.get("mode", "dev"), ret_name, k_src)
    a.provider = None if model == NONE else OllamaProvider(model=model)

    st.write("")
    examples = ["Which model should the engine run after 5 clicks, and why?",
                "Do machine learning or deep learning models win at K=10?",
                "Why are raw scores not probabilities, and how was that fixed?",
                "Is there any input drift after training?",
                "Which feature group matters most?"]
    cols = st.columns(len(examples))
    picked = None
    for col, q in zip(cols, examples):
        if col.button(q, key=f"ex_{hash(q)}", width="stretch"):
            picked = q
    history = st.session_state.setdefault("ai_history", [])
    question = st.chat_input("Ask about the models, results, data or monitoring…") or picked
    if question:
        asking = "" if model == NONE else f" and asking {model}"
        with st.spinner(f"Searching {len(a.chunks)} passages{asking}…"):
            history.append(a.ask(question))
    if history and st.button("Clear conversation", type="tertiary", icon=":material/delete:"):
        history.clear()
        st.rerun()

    for ans in reversed(history):
        with st.chat_message("user"):
            st.write(ans.question)
        with st.chat_message("assistant", avatar=":material/neurology:"):
            st.markdown(ans.text)
            badges = []
            if ans.provider:
                badges.append(f"{ans.model} · {ans.seconds:.1f} s")
                if ans.error:
                    badges.append(":red-badge[:material/error: model unavailable]")
                elif ans.grounded:
                    badges.append(":green-badge[:material/verified: every number found in the sources]")
                else:
                    if ans.ungrounded_numbers:
                        badges.append(f":orange-badge[:material/warning: not in sources: "
                                      f"{', '.join(ans.ungrounded_numbers[:4])}]")
                    if ans.invalid_citations:
                        badges.append(f":orange-badge[:material/warning: unknown citation "
                                      f"{', '.join(ans.invalid_citations)}]")
            badges.append(f"retrieval {ans.retrieval_seconds * 1000:.0f} ms")
            st.caption(" · ".join(badges))
            with st.expander(f"Sources ({len(ans.sources)})", expanded=ans.provider is None):
                for i, (c, sc) in enumerate(zip(ans.sources, ans.scores), 1):
                    mark = "**cited** · " if i in ans.cited else ""
                    st.markdown(f"**[S{i}] {c.title}**  \n{mark}`{c.id}` · score {sc:.3f}")
                    st.text(c.text[:900] + ("…" if len(c.text) > 900 else ""))

# --------------------------------------------------------------------------- evaluation
with eval_tab:
    ret = s.csv("rag_retrieval")
    ans_sum = s.csv("rag_answer_summary")
    answers = s.csv("rag_answers")
    if ret is None:
        ui.missing("RAG evaluation", "11_rag_eval.py")
    else:
        info = s.json("rag_eval") or {}
        st.markdown(f"**{info.get('questions', '?')} test questions**: hand-written conceptual questions (including "
                    "questions the sources cannot answer) plus factual questions generated from the result tables, "
                    "so the right source and the exact figure are known.")
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            st.subheader("Retrieval: is the right passage found?", anchor=False)
            r = ret.filter(pl.col("origin") == "all")
            names = {"bm25": "Keyword (BM25)", "dense": "Embeddings", "hybrid": "Hybrid (RRF)"}
            fig = go.Figure()
            for i, m in enumerate(["recall@1", "recall@3", "recall@5", "mrr@10"]):
                fig.add_trace(go.Bar(x=[names.get(x, x) for x in r["retriever"]], y=r[m], name=m.replace("@", " @"),
                                     marker=dict(color=pal[i], cornerradius=4),
                                     text=[f"{v:.2f}" for v in r[m]], textposition="outside", cliponaxis=False,
                                     textfont=dict(color=ink["text"], size=11)))
            ui.style(fig, height=330, top=50)
            fig.update_yaxes(range=[0, 1.15], tickformat=".0%")
            ui.show(fig, key="rag_ret")
            st.caption("Recall@k: share of questions with a relevant passage in the top k. MRR: mean of "
                       "1 / rank of the first relevant passage.")
        with right:
            st.subheader("Answers: are they right and grounded?", anchor=False)
            if ans_sum is None:
                st.info("No answer evaluation yet. Run `python scripts/11_rag_eval.py --models llama3.1:8b qwen3:14b`.")
            else:
                st.dataframe(ans_sum.select(
                    pl.col("provider").alias("Model"), pl.col("retriever").alias("Retrieval"), pl.col("n").alias("Questions"),
                    (pl.col("accuracy") * 100).round(1).alias("Correct %"),
                    (pl.col("abstention") * 100).round(1).alias("Abstains when it should %"),
                    (pl.col("numeric_grounding") * 100).round(1).alias("Numbers grounded %"),
                    (pl.col("citation_validity") * 100).round(1).alias("Valid citations %"),
                    pl.col("seconds_mean").round(1).alias("Seconds / answer")),
                    hide_index=True, width="stretch")
                st.caption("Correct: every expected fact present (numbers in any rounding). Grounded: every number "
                           "in the answer appears in the retrieved sources.")
        if answers is not None:
            with st.expander("Every evaluated answer"):
                only_wrong = st.toggle("Only wrong answers", value=True, key="rag_wrong")
                d = answers.filter(pl.col("correct") == 0) if only_wrong else answers
                st.dataframe(d.select("provider", "id", "correct", "grounded_numbers", "seconds", "answer", "sources"),
                             hide_index=True, width="stretch")

# --------------------------------------------------------------------------- how it works
with how_tab:
    st.markdown("""
1. **Knowledge base.** The report, model and data cards, report updates and README are split at headings into
   passages; every result table (accuracy, ML vs DL tests, latency, ablation, SHAP groups, calibration,
   monitoring, monthly traffic) is turned into a short fact sheet per decision point.
2. **Retrieval.** Keyword search (BM25), embedding search (`nomic-embed-text` via Ollama, cosine similarity) and a
   hybrid that fuses both rankings with reciprocal rank fusion.
3. **Generation.** The top passages go to an open model running locally in Ollama (Llama 3.1 8B by default)
   with instructions to answer only from them and cite every claim as [S1], [S2]…
4. **Checks.** Every citation must point to a shown source, and every number in the answer must appear in the
   sources (allowing for rounding and % forms). Failures are flagged on the answer.
5. **Evaluation.** `scripts/11_rag_eval.py` measures retrieval (recall@k, MRR) and answers (correctness,
   abstention on unanswerable questions, grounding, latency) for each model.
""")
