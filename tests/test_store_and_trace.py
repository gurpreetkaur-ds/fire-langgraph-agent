import logging

import pytest

from evaluation import store
from evaluation.db import session_scope
from evaluation.models import AgentStep, AgentTrace, EvaluationRun, ToolCall
from evaluation.tracing import TraceCollector, get_current_trace, new_trace_id, set_current_trace, setup_logging, traced_invoke
from tests.conftest import FakeLLM
from tests.helpers import make_trace


def test_trace_ids_are_unique_and_prefixed():
    ids = {new_trace_id() for _ in range(500)}
    assert len(ids) == 500 and all(i.startswith("req_") for i in ids)


def test_persist_trace_aggregates_tokens_cost_and_creates_pending_run(db):
    tid = make_trace()
    with session_scope() as s:
        t = s.get(AgentTrace, tid)
        assert (t.input_tokens, t.output_tokens, t.total_tokens) == (300, 150, 450)
        assert t.estimated_cost == pytest.approx(3 * (100 * 1.0 + 50 * 2.0) / 1e6)
        assert t.agents_executed == ["decomposer", "answer_agent", "synthesizer"]
        assert t.agent_name == "synthesizer" and t.success and t.prompt_version == "pv1"
        assert [x.seq for x in t.steps] == [1, 2, 3]
        run = s.query(EvaluationRun).one()
        assert run.status == "pending" and run.trace_id == tid and run.agent_name == "synthesizer"


def test_unknown_model_price_gives_null_cost_not_zero(db):
    tid = make_trace(model="unpriced-model")
    with session_scope() as s:
        assert s.get(AgentTrace, tid).estimated_cost is None


def test_failed_request_is_recorded_as_failure(db):
    tid = make_trace(answer=None, error="Fire API returned HTTP 500", agents=("decomposer",))
    with session_scope() as s:
        t = s.get(AgentTrace, tid)
        assert not t.success and t.agent_name == "decomposer" and "HTTP 500" in t.error


def test_sensitive_data_is_redacted_before_persisting(db):
    tid = make_trace(question="my key is sk-abcdefghijklmnop1234, mail bob@example.com", answer="Authorization: Bearer abcdefghijkl1234")
    with session_scope() as s:
        t = s.get(AgentTrace, tid)
        blob = f"{t.question} {t.answer}"
        assert "sk-abcdefghijklmnop1234" not in blob and "bob@example.com" not in blob and "abcdefghijkl1234" not in blob
        assert s.query(EvaluationRun).one().question == t.question


def test_disabled_evaluation_still_stores_trace_without_run(db, monkeypatch):
    monkeypatch.setenv("EVAL_ENABLED", "false")
    make_trace()
    with session_scope() as s:
        assert s.query(AgentTrace).count() == 1 and s.query(EvaluationRun).count() == 0


def test_tool_calls_are_stored_redacted(db):
    c = TraceCollector(question="q")
    c.record_tool_call("search", agent_name="a", tool_input={"query": "x", "api_key": "sk-abcdefghijklmnop1234"}, tool_output="token=abcd1234efgh")
    store.persist_trace(c, answer="a", error=None, context=[])
    with session_scope() as s:
        tc = s.query(ToolCall).one()
        assert tc.tool_input["api_key"] == "[REDACTED]" and "abcd1234efgh" not in str(tc.tool_output)
        assert s.get(AgentTrace, c.trace_id).tool_call_count == 1


async def test_traced_invoke_records_step_and_sets_log_trace_id(caplog, db):
    setup_logging()
    c = TraceCollector(question="q")
    set_current_trace(c)
    try:
        with caplog.at_level(logging.INFO, logger="fire_agent.trace"):
            await traced_invoke("answer_agent", "answer", FakeLLM("test-model", lambda m: "hi"), [])
        step = c.steps[0]
        assert (step.agent_name, step.input_tokens, step.output_tokens, step.call_id) == ("answer_agent", 100, 50, "call_test-model")
        assert step.estimated_cost == (100 * 1 + 50 * 2) / 1e6
        assert any("event=llm_call" in r.getMessage() and "agent=answer_agent" in r.getMessage() for r in caplog.records)
    finally:
        set_current_trace(None)
    assert get_current_trace() is None


async def test_traced_invoke_records_failures_and_reraises():
    class Boom:
        species_name = "m"

        async def ainvoke(self, messages):
            raise RuntimeError("boom sk-abcdefghijklmnop1234")

    c = TraceCollector(question="q")
    set_current_trace(c)
    try:
        try:
            await traced_invoke("a", "n", Boom(), [])
            raise AssertionError("should have raised")
        except RuntimeError:
            pass
    finally:
        set_current_trace(None)
    assert c.steps[0].success is False and "sk-abcdefghijklmnop1234" not in c.steps[0].error
