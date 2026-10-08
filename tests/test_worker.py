import asyncio

from evaluation import store
from evaluation.db import session_scope
from evaluation.engine import EvaluationEngine
from evaluation.models import EvaluationMetric, EvaluationRun, utcnow
from evaluation.worker import EvaluationWorker
from tests.conftest import GOOD_JUDGE, ScriptedJudge
from tests.helpers import make_trace


def engine():
    return EvaluationEngine(judge=ScriptedJudge(GOOD_JUDGE))


async def test_worker_evaluates_pending_runs_and_persists_metrics(db):
    make_trace()
    assert await EvaluationWorker(engine()).drain() == 1
    with session_scope() as s:
        run = s.query(EvaluationRun).one()
        assert run.status == "completed" and run.passed is True
        assert run.faithfulness_score == 1.0 and run.relevance_score == 1.0 and run.safety_score == 1.0 and run.hallucination_score == 0.0
        assert run.citation_accuracy_score is None and run.tool_accuracy_score is None  # honestly not available
        m = {x.metric: x for x in s.query(EvaluationMetric).filter_by(evaluation_id=run.evaluation_id)}
        assert m["citation_accuracy"].status == "not_available" and m["citation_accuracy"].score is None
        assert m["faithfulness"].reason is not None and m["faithfulness"].confidence == 0.9
        assert run.completed_at is not None


async def test_evaluation_runs_in_background_after_request_data_is_stored(db):
    """Persisting the trace never waits for the judge; the evaluation happens later, in the worker."""
    make_trace()
    with session_scope() as s:
        assert s.query(EvaluationRun).one().status == "pending"
    worker = EvaluationWorker(engine())
    worker.start()
    worker.notify()
    for _ in range(100):
        with session_scope() as s:
            if s.query(EvaluationRun).one().status == "completed":
                break
        await asyncio.sleep(0.05)
    await worker.stop()
    with session_scope() as s:
        assert s.query(EvaluationRun).one().status == "completed"


def test_claim_is_atomic_one_claim_per_run(db):
    make_trace()
    first, second = store.claim_next_pending(), store.claim_next_pending()
    assert first is not None and second is None


def test_stale_running_run_is_requeued(db, monkeypatch):
    make_trace()
    eid = store.claim_next_pending()
    monkeypatch.setenv("EVAL_STALE_RUNNING_SECONDS", "0")
    assert store.claim_next_pending() == eid


async def test_worker_failure_marks_run_failed_and_continues(db):
    make_trace()
    make_trace()

    class Exploding:
        calls = 0

        async def evaluate(self, data):
            Exploding.calls += 1
            if Exploding.calls == 1:
                raise RuntimeError("db exploded")
            return []

    w = EvaluationWorker(Exploding())
    assert await w.drain() == 2
    with session_scope() as s:
        statuses = sorted(r.status for r in s.query(EvaluationRun))
        assert statuses == ["completed", "failed"]


def test_claimed_suite_runs_are_not_picked_up_by_workers(db):
    make_trace(claim=True)
    assert store.claim_next_pending() is None
