"""RAG analyst, retrieval, checks and evaluation (src/llm), with a fake language model (no network)."""

import numpy as np
import polars as pl

from src.llm.corpus import Chunk, _split_markdown, build_corpus
from src.llm.evaluation import Question, is_relevant, question_set, score_answer, summarise_answers, answer_eval
from src.llm.providers import Completion, LLMError, OllamaProvider, get_provider, setting
from src.llm.rag import Analyst, citations, explain_visit, ungrounded
from src.llm.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, tokenize
from tests.test_product_scripts import _run
from tests.test_serving import trained  # noqa: F401  (module-scoped fixture: synthetic data + models)

CHUNKS = [
    Chunk("a#0", "rq1_model_comparison", "[Results] Model accuracy at K=5", "- LightGBM: PR-AUC 0.3331 at K=5.", "facts"),
    Chunk("b#0", "rq4_efficiency", "[Results] Latency at K=5", "- Random forest: p95 latency 5.412 ms per visit.", "facts"),
    Chunk("c#0", "report", "[Report] Data", "Sessions are split chronologically 70/15/15 into train, val and test.", "document"),
]


class FakeLLM:
    name, model = "fake", "fake-1"

    def __init__(self, reply="LightGBM reaches a PR-AUC of 0.3331 [S1].", fail=False):
        self.reply, self.fail, self.prompts = reply, fail, []

    def available(self):
        return True, "fake"

    def complete(self, system, prompt, max_tokens=1500):
        self.prompts.append(prompt)
        if self.fail:
            raise LLMError("offline")
        return Completion(self.reply, self.name, self.model, 0.01, 100, 20)


class FakeEmbedder:
    model = "fake-embed"

    def available(self):
        return True, "fake"

    def embed(self, texts, kind="document"):
        vocab = ["lightgbm", "latency", "split", "pr", "auc", "random", "forest", "chronologically"]
        v = np.array([[t.lower().count(w) for w in vocab] for t in texts], dtype=np.float32) + 1e-3
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class NoEmbedder:
    model = "none"

    def available(self):
        return False, "not pulled"


def test_tokenize_and_markdown_split():
    assert tokenize("random_forest PR-AUC at K=5") == ["random", "forest", "pr", "auc", "k", "5"]
    parts = _split_markdown("# T\nintro\n## A\ntext a\n### B\ntext b")
    assert [h for h, _ in parts] == ["T", "T › A", "T › A › B"]


def test_bm25_dense_and_hybrid_rank_the_right_chunk(tmp_path):
    bm = BM25Retriever(CHUNKS)
    assert bm.search("What is the latency of the random forest?", 1)[0][0].id == "b#0"
    dense = DenseRetriever(CHUNKS, FakeEmbedder(), tmp_path)
    assert dense.search("how is the data split", 1)[0][0].id == "c#0"
    assert any(tmp_path.glob("*.npy"))                                  # cached
    hyb = HybridRetriever([bm, dense])
    assert hyb.search("LightGBM PR-AUC", 1)[0][0].id == "a#0"


def test_grounding_and_citation_checks():
    ctx = "PR-AUC 0.3331, lift 3.28x, conversion 5.94%, 646,060 orders"
    assert ungrounded("PR-AUC 0.333 [S1] and 33.31% and 5.94% and 646060 orders", ctx) == []
    assert ungrounded("PR-AUC 0.412 [S1]", ctx) == ["0.412"]
    assert citations("x [S1] y [S3] [S1]", 2) == ([1, 3], ["S3"])


def test_analyst_answers_with_checks_and_fallbacks():
    a = Analyst({"data": {"mode": "x"}}, FakeLLM(), retriever="hybrid", chunks=CHUNKS, embedder=NoEmbedder())
    assert a.retriever_name == "bm25" and a.notes                      # no embeddings: BM25 only
    ans = a.ask("What PR-AUC does LightGBM reach at K=5?")
    assert ans.cited == [1] and ans.grounded and ans.sources[0].id == "a#0"
    bad = Analyst({"data": {"mode": "x"}}, FakeLLM("It is 0.9999 [S7]."), chunks=CHUNKS, embedder=NoEmbedder())
    b = bad.ask("LightGBM PR-AUC?")
    assert not b.grounded and b.invalid_citations == ["S7"] and b.ungrounded_numbers == ["0.9999"]
    off = Analyst({"data": {"mode": "x"}}, FakeLLM(fail=True), chunks=CHUNKS, embedder=NoEmbedder()).ask("latency?")
    assert off.error == "offline" and off.sources
    none = Analyst({"data": {"mode": "x"}}, None, chunks=CHUNKS, embedder=NoEmbedder()).ask("latency?")
    assert none.provider is None and none.sources


def test_answer_scoring():
    q = Question("q", "?", [], [["0.3331"], ["lightgbm"]])
    assert score_answer(q, "LightGBM: 0.333 [S1]")["correct"] == 1.0
    assert score_answer(q, "Random forest: 0.333")["correct"] == 0.0
    u = Question("u", "?", [], [], answerable=False)
    assert score_answer(u, "The sources do not contain this.")["correct"] == 1.0
    assert score_answer(u, "It was $3M.")["correct"] == 0.0


def test_explain_visit_flags_unknown_features():
    contrib = [("cart_count", "2", 0.5), ("view_count", "3", 1.89)]
    ok = explain_visit(FakeLLM("cart_count and view_count push towards a purchase."), "GRU", 0.45, 0.23, "High",
                       contrib, 5, ["cart_count", "view_count", "price_std"])
    assert ok.grounded
    bad = explain_visit(FakeLLM("price_std is very high (0.97)."), "GRU", 0.45, 0.23, "High", contrib, 5,
                        ["cart_count", "view_count", "price_std"])
    assert bad.unknown_features == ["price_std"] and "0.97" in bad.ungrounded_numbers


def test_settings_and_provider_factory(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    assert get_provider() is None
    assert get_provider("ollama").name == "ollama" and get_provider("claude") is None
    monkeypatch.setenv("OLLAMA_MODEL", "tiny")
    assert setting("OLLAMA_MODEL") == "tiny" and get_provider("ollama").model == "tiny"
    assert get_provider("ollama", "qwen3:14b").model == "qwen3:14b"
    monkeypatch.setattr(OllamaProvider, "models", lambda self: ["llama3.1:8b", "nomic-embed-text:latest"])
    assert OllamaProvider().chat_models() == ["llama3.1:8b"]
    ok, why = OllamaProvider(model="tiny").available()
    assert not ok and "ollama pull tiny" in why


def test_corpus_questions_and_eval_script_on_synthetic(trained, monkeypatch):  # noqa: F811
    monkeypatch.setattr("src.llm.rag.OllamaEmbedder", lambda: NoEmbedder())
    chunks = build_corpus(trained)
    facts = [c for c in chunks if c.kind == "facts"]
    assert {"rq1_model_comparison", "rq1_ml_vs_dl"} <= {c.source for c in facts}
    qs = question_set(trained)
    gen = [q for q in qs if q.origin == "generated"]
    assert gen and all(any(is_relevant(c, q.gold) for c in chunks) for q in gen)
    a = Analyst(trained, FakeLLM(), chunks=chunks)
    s = summarise_answers(answer_eval(a, gen[:3], "fake"))
    assert s["n"][0] == 3 and 0 <= s["accuracy"][0] <= 1
    _run("11_rag_eval.py", trained, ["--models"])                    # retrieval only, no model
    from src.serving.results import ResultStore

    r = ResultStore(trained).csv("rag_retrieval")
    assert r is not None and "bm25" in r["retriever"].to_list()
    assert pl.Series(r["recall@5"]).max() > 0
