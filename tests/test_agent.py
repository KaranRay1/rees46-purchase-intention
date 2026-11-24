"""Tool-using agent: tool dispatch, argument repair, grounding and the loop (scripted fake model)."""

import polars as pl
import pytest

from src.llm.agent import Agent, Tools, _coerce, model_key, text_tool_call
from src.llm.corpus import Chunk
from src.llm.rag import Analyst
from tests.test_serving import trained  # noqa: F401  (module-scoped fixture: synthetic data + models)


class FakeEmbedder:
    def available(self):
        return False, "off in tests"


class Scripted:
    """Replays chat turns: each turn is (content, tool_calls)."""

    name, model = "fake", "fake-1"

    def __init__(self, turns):
        self.turns, self.seen = list(turns), []

    def chat(self, messages, tools=None, max_tokens=900):
        self.seen.append((list(messages), tools))
        content, calls = self.turns.pop(0)
        return {"content": content, "tool_calls": calls, "seconds": 0.01, "input_tokens": 10, "output_tokens": 5}


def call(name, **args):
    return {"function": {"name": name, "arguments": args}}


@pytest.fixture(scope="module")
def tools(trained):  # noqa: F811
    chunks = [Chunk("c1", "readme", "README > Split", "Sessions were split chronologically 70/15/15.", "document")]
    return Tools(trained, analyst=Analyst(trained, provider=None, chunks=chunks, embedder=FakeEmbedder()))


def test_model_aliases():
    assert model_key("GRU") == "gru" and model_key("Random forest") == "random_forest"
    assert model_key("GRU (sequence)") == "gru" and model_key("lgbm") == "lightgbm"
    with pytest.raises(ValueError):
        model_key("xgboost")


def test_argument_repair_and_text_calls():
    assert _coerce("['a', 'b']") == ["a", "b"] and _coerce('{"x": 1}') == {"x": 1} and _coerce("plain") == "plain"
    tc = text_tool_call('Let me try {"name": "read_table", "parameters": {"table": "rq1_model_comparison"}}')
    assert tc == {"function": {"name": "read_table", "arguments": {"table": "rq1_model_comparison"}}}
    assert text_tool_call("no call here") is None


def test_tools_read_results(tools, trained):  # noqa: F811
    k = trained["sessionize"]["prefix_k"][0]
    assert "rq1_model_comparison" in tools.list_tables()
    out = tools.read_table("rq1_model_comparison", k=k, columns=["model", "pr_auc_mean"])
    assert "pr_auc_mean" in out and "matching rows" in out
    # string-encoded arguments from a small model still work, unknown arguments are dropped
    out2 = tools.run("read_table", {"table": "rq1_model_comparison", "k": str(k), "columns": "['model']", "bogus": 1})
    assert out2.splitlines()[1] == "k | model"                 # key columns are always kept
    out3 = tools.run("read_table", {"table": "rq1_model_comparison", "columns": "pr_auc", "limit": "2",
                                    "where": "model='gru'"})
    assert "pr_auc_mean" in out3 and "gru" in out3              # fuzzy column + string filter + string limit
    assert "models in this table" in tools.read_table("rq1_model_comparison", model="tabnet")
    out4 = tools.read_table("rq1_model_comparison", columns=["mean auc pr"])   # word-based column match
    assert "pr_auc_mean" in out4
    assert "Sessions were split" in tools.search_docs("how was the data split")
    with pytest.raises(ValueError):
        tools.read_table("not_a_table")


def test_score_session_tool(tools, trained):  # noqa: F811
    k = trained["sessionize"]["prefix_k"][0]
    clicks = [{"event_type": "view", "price": 100 + i, "category": "electronics.smartphone"} for i in range(k)]
    clicks[-1]["event_type"] = "cart"
    out = tools.score_session(clicks)
    assert f"K={k}" in out and "purchase probability" in out
    with pytest.raises(ValueError):
        tools.score_session(clicks[:2] if k != 2 else clicks[:1])


def test_agent_loop_grounds_numbers(tools, trained):  # noqa: F811
    k = trained["sessionize"]["prefix_k"][0]
    table = pl.read_csv(tools._tables()["rq1_model_comparison"]).filter(pl.col("k") == k)
    value = f"{table['pr_auc_mean'][0]:.4f}"
    fake = Scripted([("", [call("read_table", table="rq1_model_comparison", k=k)]),
                     (f"The PR-AUC is {value}, and accuracy is 0.9876.", [])])
    a = Agent(trained, fake, tools=tools, prefetch_docs=False).ask("What is the PR-AUC?")
    assert a.tools_used == ["read_table"] and not a.steps[0].error
    assert a.ungrounded_numbers == ["0.9876"]                  # the invented number is caught
    assert fake.seen[1][0][-1]["role"] == "tool"               # tool output went back to the model


def test_agent_recovers_from_tool_errors_and_stops(tools, trained):  # noqa: F811
    turns = [("", [call("read_table", table="nope")])] * 2 + [("I could not find it.", [])]
    a = Agent(trained, Scripted(turns), tools=tools, max_steps=2, prefetch_docs=False).ask("x")
    assert all(s.error for s in a.steps) and a.text == "I could not find it."
    assert a.usage["llm_calls"] == 3


def test_tool_questions_have_expected_answers(trained):  # noqa: F811
    from src.llm.evaluation import tool_questions

    qs = tool_questions(trained)
    assert qs and all(q.origin == "tool" and q.expect for q in qs)
    assert any(q.id.startswith("t_campaign_") for q in qs)


def test_agent_prefetches_documents(tools, trained):  # noqa: F811
    fake = Scripted([("Sessions were split chronologically 70/15/15.", [])])
    a = Agent(trained, fake, tools=tools, prefetch_docs=True).ask("How was the data split?")
    assert a.tools_used == ["search_docs"] and a.grounded
    assert fake.seen[0][0][-1]["role"] == "tool"               # the passages reach the model before its first turn


def test_recommend_engine_and_empty_answer_retry(tools, trained):  # noqa: F811
    k = trained["sessionize"]["prefix_k"][0]
    assert f"Engine model after K={k} clicks" in tools.recommend_engine(k)
    fake = Scripted([("", []), ("Done.", [])])                 # first reply empty: the agent asks once more
    a = Agent(trained, fake, tools=tools).ask("x")
    assert a.text == "Done." and a.usage["llm_calls"] == 2


def test_scorer_accepts_more_precise_figures():
    from src.llm.evaluation import Question, score_answer

    q = Question("q", "?", [], [["0.010"]])
    assert score_answer(q, "The ECE is 0.010292.")["correct"] == 1.0
    assert score_answer(q, "The ECE is 0.0142.")["correct"] == 0.0
