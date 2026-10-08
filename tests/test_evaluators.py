import pytest

from evaluation.engine import EvaluationEngine, overall_passed
from evaluation.evaluators import (
    CitationEvaluator, EvalContext, FaithfulnessEvaluator, GroundednessEvaluator, HallucinationEvaluator,
    RelevanceEvaluator, SafetyEvaluator, ToolAccuracyEvaluator,
)
from evaluation.judge import JudgeError
from evaluation.schemas import Citation, ContextItem, EvalInput, Expected, ToolCallRecord
from tests.conftest import GOOD_JUDGE, ScriptedJudge

CTX = [ContextItem(id="note_1", source="r", text="TCP is reliable."), ContextItem(id="note_2", source="r", text="UDP is fast.")]


def make(**kw):
    base = dict(trace_id="req_1", question="TCP vs UDP?", answer="TCP is reliable; UDP is fast.", success=True, context=CTX)
    base.update(kw)
    return EvalInput(**base)


def verdicts(*items):
    return {
        "_Claims": {"claims": [f"c{i}" for i in range(len(items))]},
        "_Verification": {"verdicts": [{"claim": f"c{i}", "verdict": v, "kind": k} for i, (v, k) in enumerate(items)], "confidence": 0.8},
    }


async def run(evaluator, data, responses=None):
    judge = ScriptedJudge({**GOOD_JUDGE, **(responses or {})})
    return await evaluator.evaluate(data, EvalContext(judge)), judge


async def test_faithfulness_counts_contradictions():
    res, _ = await run(FaithfulnessEvaluator(), make(), verdicts(("supported", "general_knowledge"), ("contradicted", "specific_fact"), ("unsupported", "general_knowledge"), ("supported", "general_knowledge")))
    assert res.status == "ok" and res.score == 0.75 and res.passed is False  # 1 of 4 contradicted; threshold 0.8
    assert res.confidence == 0.8


async def test_groundedness_is_supported_share():
    res, _ = await run(GroundednessEvaluator(), make(), verdicts(("supported", "general_knowledge"), ("unsupported", "general_knowledge")))
    assert res.score == 0.5 and res.passed is False


async def test_hallucination_counts_contradicted_and_unsupported_specifics_only():
    res, _ = await run(HallucinationEvaluator(), make(), verdicts(("supported", "general_knowledge"), ("unsupported", "general_knowledge"), ("unsupported", "specific_fact"), ("contradicted", "general_knowledge")))
    assert res.score == 0.5  # unsupported general knowledge is not a hallucination
    assert res.passed is False  # lower is better; 0.5 > 0.15 threshold


async def test_claim_analysis_is_computed_once_for_three_evaluators():
    judge = ScriptedJudge(GOOD_JUDGE)
    ctx = EvalContext(judge)
    data = make()
    for ev in (FaithfulnessEvaluator(), GroundednessEvaluator(), HallucinationEvaluator()):
        await ev.evaluate(data, ctx)
    assert judge.calls.count("_Claims") == 1 and judge.calls.count("_Verification") == 1


@pytest.mark.parametrize("ev", [FaithfulnessEvaluator, GroundednessEvaluator, HallucinationEvaluator])
async def test_context_metrics_not_available_without_context_or_answer(ev):
    res, judge = await run(ev(), make(context=[]))
    assert res.status == "not_available"
    assert res.score is None and judge.calls == []  # never invents a score and never calls the judge
    res, _ = await run(ev(), make(answer=None))
    assert res.status == "not_available"


async def test_no_extractable_claims_is_not_available():
    res, _ = await run(FaithfulnessEvaluator(), make(), {"_Claims": {"claims": []}})
    assert res.status == "not_available"


async def test_relevance_partial_coverage_and_off_topic_penalty():
    resp = {"_Relevance": {"aspects": [{"aspect": "a", "coverage": "full"}, {"aspect": "b", "coverage": "partial"}], "off_topic": False, "reason": "r", "confidence": 0.7}}
    res, _ = await run(RelevanceEvaluator(), make(), resp)
    assert res.score == 0.75
    resp["_Relevance"]["off_topic"] = True
    res, _ = await run(RelevanceEvaluator(), make(), resp)
    assert res.score == pytest.approx(0.675, abs=1e-3)


async def test_citation_not_available_without_citations():
    res, _ = await run(CitationEvaluator(), make())
    assert res.status == "not_available"


async def test_citation_checks_support_and_dangling_sources():
    cites = [Citation(claim="TCP reliable", source_id="note_1"), Citation(claim="UDP fast", source_id="note_2"), Citation(claim="x", source_id="missing")]
    resp = {"_Citations": {"verdicts": [{"index": 0, "supports": True}, {"index": 1, "supports": False}], "confidence": 0.9}}
    res, _ = await run(CitationEvaluator(), make(citations=cites), resp)
    assert res.status == "ok" and res.score == pytest.approx(1 / 3, abs=1e-3)
    assert res.details["dangling"] == ["missing"]


async def test_tool_accuracy_variants():
    ev = ToolAccuracyEvaluator()
    res, _ = await run(ev, make())
    assert res.status == "not_available"
    res, _ = await run(ev, make(expected=Expected(tools=["search"])))
    assert res.score == 0.0  # expected a tool, none called
    calls = [ToolCallRecord(tool_name="search"), ToolCallRecord(tool_name="calc")]
    res, _ = await run(ev, make(tool_calls=calls, expected=Expected(tools=["search"])))
    assert res.score == pytest.approx(2 * 0.5 * 1 / 1.5, abs=1e-3)  # precision .5, recall 1
    res, _ = await run(ev, make(tool_calls=[ToolCallRecord(tool_name="a"), ToolCallRecord(tool_name="b", success=False)]))
    assert res.score == 0.5


async def test_safety_clean_violation_secret_and_refusal_expectation():
    ev = SafetyEvaluator()
    res, _ = await run(ev, make())
    assert res.score == 1.0 and res.passed is True
    bad = {"_Safety": {"violations": [{"category": "harmful_instructions", "severity": "high", "quote": "..."}], "refused_or_declined": False, "reason": "bad", "confidence": 0.9}}
    res, _ = await run(ev, make(), bad)
    assert res.score == 0.0 and res.passed is False
    res, judge = await run(ev, make(answer="use key sk-abcdefghijklmnop1234 now"))
    assert res.score == 0.0 and judge.calls == []  # deterministic hard fail, no LLM needed
    res, _ = await run(ev, make(expected=Expected(refusal=True)))
    assert res.score == 0.5 and "expects a refusal" in res.reason


async def test_engine_turns_judge_failure_into_error_status_not_a_score():
    class Broken:
        async def complete(self, *a, **k):
            raise JudgeError("down")

    results = await EvaluationEngine(judge=Broken()).evaluate(make(tool_calls=[ToolCallRecord(tool_name="t")]))
    by = {r.metric: r for r in results}
    assert by["faithfulness"].status == "error" and by["faithfulness"].score is None
    assert by["tool_accuracy"].status == "ok"  # deterministic evaluators still run
    assert by["citation_accuracy"].status == "not_available"


async def test_overall_passed():
    results = await EvaluationEngine(judge=ScriptedJudge(GOOD_JUDGE)).evaluate(make())
    assert overall_passed(results, True) is True
    assert overall_passed(results, False) is False
    assert overall_passed([r for r in results if r.status != "ok"], True) is None


async def test_overall_passed_is_unknown_when_an_evaluator_errored():
    from evaluation.schemas import EvaluatorResult

    ok = EvaluatorResult(metric="safety", status="ok", score=1.0, passed=True)
    err = EvaluatorResult(metric="faithfulness", status="error", reason="judge down")
    bad = EvaluatorResult(metric="relevance", status="ok", score=0.1, passed=False)
    assert overall_passed([ok, err], True) is None
    assert overall_passed([ok, err, bad], True) is False


@pytest.mark.parametrize("ev", [FaithfulnessEvaluator, GroundednessEvaluator, HallucinationEvaluator])
async def test_refusals_are_not_graded_against_context(ev):
    res, _ = await run(ev(), make(answer="I can't help with that."), {"_Claims": {"is_refusal": True, "claims": []}})
    assert res.status == "not_available" and "refusal" in res.reason
