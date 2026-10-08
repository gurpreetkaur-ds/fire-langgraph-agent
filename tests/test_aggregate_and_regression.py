from datetime import date, datetime, timedelta, timezone

import pytest

from evaluation import aggregate
from evaluation.aggregate import Filters
from evaluation.db import session_scope
from evaluation.engine import EvaluationEngine
from evaluation.regression import compare, render_text
from evaluation.schemas import Expected
from evaluation.worker import EvaluationWorker
from tests.conftest import GOOD_JUDGE, ScriptedJudge
from tests.helpers import make_trace


async def evaluate_all(responses=None):
    await EvaluationWorker(EvaluationEngine(judge=ScriptedJudge({**GOOD_JUDGE, **(responses or {})}))).drain()


async def test_summary_metrics_counts_and_reliability(db):
    make_trace()
    make_trace(answer=None, error="boom", agents=("decomposer",))
    await evaluate_all()
    with session_scope() as s:
        out = aggregate.summary(s, Filters())
    assert out["evaluations_total"] == 2 and out["evaluations_completed"] == 2
    assert out["failed_evaluations"] == 1  # the failed request
    assert out["request_error_rate"] == 0.5
    assert out["metrics"]["faithfulness"] == {"avg": 1.0, "n": 1}  # failed request has no score -> excluded, not zeroed
    assert out["metrics"]["citation_accuracy"] == {"avg": None, "n": 0}
    assert out["reliability_score"] == pytest.approx(1.0)
    assert out["avg_cost"] is not None and out["total_tokens"] == 450 + 150


async def test_filters_agent_model_status_trace_date_and_failed_metric(db):
    t1 = make_trace(model="test-model")
    make_trace(model="other-model", agents=("decomposer",))
    old = make_trace(created_at=datetime.now(timezone.utc) - timedelta(days=10))
    await evaluate_all({"_Safety": {"violations": [{"category": "other", "severity": "medium"}], "refused_or_declined": False, "reason": "x", "confidence": 0.9}})
    with session_scope() as s:
        n = lambda **kw: aggregate.summary(s, Filters(**kw))["evaluations_total"]
        assert n() == 3
        assert n(agent="synthesizer") == 2 and n(agent="decomposer") == 3
        assert n(model="other-model") == 1
        assert n(trace_id=t1) == 1
        assert n(status="completed") == 3 and n(status="pending") == 0
        assert n(date_from=date.today() - timedelta(days=2)) == 2
        assert n(date_to=date.today() - timedelta(days=5)) == 1
        assert n(failed_metric="safety") == 3 and n(failed_metric="faithfulness") == 0
        assert aggregate.traces(s, Filters(trace_id=old), 10, 0)["items"][0]["trace_id"] == old


async def test_timeseries_buckets_by_day(db):
    make_trace()
    make_trace(created_at=datetime.now(timezone.utc) - timedelta(days=1))
    await evaluate_all()
    with session_scope() as s:
        pts = aggregate.timeseries(s, Filters())
    assert len(pts) == 2 and pts[0]["date"] < pts[1]["date"]
    assert pts[1]["total_tokens"] == 450 and pts[1]["scores"]["faithfulness"] == 1.0 and pts[1]["failure_rate"] == 0


async def test_agents_list_only_contains_agents_that_ran_and_detail(db):
    make_trace()
    await evaluate_all()
    with session_scope() as s:
        names = [a["agent"] for a in aggregate.agents(s, Filters())]
        assert names == ["answer_agent", "decomposer", "synthesizer"]
        d = aggregate.agent_detail(s, Filters(), "synthesizer")
        assert d["executions"] == 1 and d["success_rate"] == 1.0 and d["avg_latency_ms"] == 100.0 and d["avg_tokens"] == 150
        assert d["recent_traces"] and aggregate.agent_detail(s, Filters(), "ghost_agent") is None


async def test_trace_detail_contains_steps_and_metrics(db):
    tid = make_trace()
    await evaluate_all()
    with session_scope() as s:
        d = aggregate.trace_detail(s, tid)
    assert [x["agent_name"] for x in d["steps"]] == ["decomposer", "answer_agent", "synthesizer"]
    assert {m["metric"] for m in d["evaluations"][0]["metrics"]} == {"faithfulness", "relevance", "groundedness", "citation_accuracy", "tool_accuracy", "hallucination", "safety"}


async def _suite(suite_id, safety_violation=False, faithful=True, cases=("c1", "c2")):
    for c in cases:
        make_trace(suite=suite_id, case=c, claim=False, expected=Expected(refusal=False))
    resp = {}
    if safety_violation:
        resp["_Safety"] = {"violations": [{"category": "other", "severity": "low"}], "refused_or_declined": False, "reason": "x", "confidence": 0.9}
    if not faithful:
        resp["_Verification"] = {"verdicts": [{"claim": "a", "verdict": "contradicted", "kind": "specific_fact"}, {"claim": "b", "verdict": "supported", "kind": "general_knowledge"}], "confidence": 0.9}
    await evaluate_all(resp)
    with session_scope() as s:
        return aggregate.build_suite_report(s, suite_id)


async def test_regression_flags_degraded_improved_unchanged_and_newly_failing(db):
    base = await _suite("suite_a")
    cand = await _suite("suite_b", safety_violation=True, faithful=False)
    rep = compare(base, cand)
    status = {m.metric: m.status for m in rep.metrics}
    assert status["safety"] == "degraded" and status["faithfulness"] == "degraded"
    assert status["hallucination"] == "degraded"  # higher hallucination = worse
    assert status["citation_accuracy"] == "not_comparable" and status["relevance"] == "unchanged"
    assert rep.newly_failing == ["c1", "c2"] and rep.has_regression
    assert any("Safety" in w and "decreased" in w for w in rep.warnings)
    assert "Degraded:" in render_text(rep)


async def test_regression_reports_improvement_and_no_regression(db):
    bad = await _suite("suite_a", faithful=False)
    good = await _suite("suite_b")
    rep = compare(bad, good)
    assert {m.metric for m in rep.by_status("improved")} >= {"faithfulness", "hallucination", "groundedness"}
    assert rep.newly_passing == ["c1", "c2"] and not rep.has_regression


async def test_regression_tolerance_is_configurable(db, monkeypatch):
    base = await _suite("suite_a")
    cand = await _suite("suite_b", safety_violation=True)  # safety 0.85 vs 1.0
    monkeypatch.setenv("EVAL_REGRESSION_TOLERANCE_SAFETY", "0.5")
    assert {m.metric: m.status for m in compare(base, cand).metrics}["safety"] == "unchanged"


async def test_suite_report_counts(db):
    r = await _suite("suite_a")
    assert (r.passed, r.failed, len(r.cases)) == (2, 0, 2) and r.prompt_versions == ["pv1"]
