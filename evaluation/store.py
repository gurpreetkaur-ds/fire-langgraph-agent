"""Persistence for traces and evaluation runs (all functions are synchronous; async callers use to_thread)."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import select, update

from .config import get_settings
from .db import session_scope
from .engine import overall_passed
from .models import AgentStep, AgentTrace, EvaluationMetric, EvaluationRun, ToolCall, utcnow
from .redaction import redact_text, truncate
from .schemas import Citation, ContextItem, EvalInput, EvaluatorResult, Expected, ToolCallRecord
from .tracing import TraceCollector

logger = logging.getLogger("fire_agent.eval.store")

SCORE_COLUMNS = {
    "faithfulness": "faithfulness_score",
    "relevance": "relevance_score",
    "groundedness": "groundedness_score",
    "citation_accuracy": "citation_accuracy_score",
    "tool_accuracy": "tool_accuracy_score",
    "hallucination": "hallucination_score",
    "safety": "safety_score",
}


def _sum(values: Sequence[Optional[int | float]]) -> Optional[int | float]:
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def persist_trace(
    collector: TraceCollector,
    *,
    answer: Optional[str],
    error: Optional[str],
    context: Sequence[dict],
    citations: Sequence[dict] = (),
    expected: Optional[Expected] = None,
    suite_run_id: Optional[str] = None,
    case_id: Optional[str] = None,
    enqueue_evaluation: bool = True,
    claim: bool = False,
) -> Optional[str]:
    """Store the finished trace (redacted) and, if evaluation is enabled, a `pending` evaluation run.

    `claim=True` creates the run already `running` so the caller (the dataset runner) evaluates it itself
    and background workers don't pick it up.

    Returns the evaluation_id (or None when evaluation is disabled).
    """
    settings = get_settings()
    limit = settings.max_text_chars
    steps = collector.steps
    success = error is None and bool(answer and answer.strip())
    agents = list(dict.fromkeys(s.agent_name for s in steps))
    # The synthesizer owns the final answer; if the run died earlier, the last agent that ran is responsible.
    responsible = "synthesizer" if "synthesizer" in agents else (agents[-1] if agents else "workflow")
    resp_steps = [s for s in steps if s.agent_name == responsible]
    model = resp_steps[-1].model_name if resp_steps else None
    in_tok = _sum([s.input_tokens for s in steps])
    out_tok = _sum([s.output_tokens for s in steps])
    costs = [s.estimated_cost for s in steps]
    total_cost = None if (not costs or any(c is None for c in costs)) else float(sum(costs))
    clean_answer = truncate(redact_text(answer), limit) if answer else None
    clean_question = truncate(redact_text(collector.question), limit)
    clean_error = redact_text(error) if error else None

    with session_scope() as db:
        trace = AgentTrace(
            trace_id=collector.trace_id,
            session_id=collector.session_id,
            user_id=collector.user_id,
            created_at=collector.created_at,
            question=clean_question,
            answer=clean_answer,
            agent_name=responsible,
            agents_executed=agents,
            model_name=model,
            models_used=list(dict.fromkeys(s.model_name for s in steps if s.model_name)),
            prompt_version=collector.prompt_version,
            context=[{**c, "text": truncate(redact_text(c["text"]), limit)} for c in context],
            citations=[{"claim": redact_text(c["claim"]), "source_id": c["source_id"]} for c in citations],
            duration_ms=collector.elapsed_ms(),
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=(in_tok or 0) + (out_tok or 0) if (in_tok is not None or out_tok is not None) else None,
            estimated_cost=total_cost,
            tool_call_count=len(collector.tool_calls),
            success=success,
            error=clean_error,
        )
        trace.steps = [
            AgentStep(
                seq=s.seq, agent_name=s.agent_name, node=s.node, model_name=s.model_name, call_id=s.call_id,
                started_at=s.started_at, latency_ms=s.latency_ms, input_tokens=s.input_tokens,
                output_tokens=s.output_tokens, estimated_cost=s.estimated_cost, success=s.success,
                error=s.error, input_preview=s.input_preview, output_preview=s.output_preview,
            )
            for s in steps
        ]
        trace.tool_calls = [
            ToolCall(
                seq=t.seq, agent_name=t.agent_name, tool_name=t.tool_name, tool_input=t.tool_input,
                tool_output=t.tool_output, latency_ms=t.latency_ms, success=t.success, error=t.error,
            )
            for t in collector.tool_calls
        ]
        db.add(trace)
        evaluation_id = None
        if settings.enabled and enqueue_evaluation:
            evaluation_id = "eval_" + uuid.uuid4().hex[:12]
            db.add(
                EvaluationRun(
                    evaluation_id=evaluation_id, trace_id=collector.trace_id, suite_run_id=suite_run_id,
                    created_at=collector.created_at, case_id=case_id, expected=expected.model_dump() if expected else None,
                    status="running" if claim else "pending", started_at=utcnow() if claim else None, agent_name=responsible, model_name=model,
                    question=clean_question, answer=clean_answer, latency_ms=trace.duration_ms,
                    input_tokens=in_tok, output_tokens=out_tok, total_tokens=trace.total_tokens,
                    estimated_cost=total_cost,
                )
            )
    return evaluation_id


def claim_next_pending() -> Optional[str]:
    """Atomically move one pending evaluation to `running` and return its id (safe with several workers)."""
    settings = get_settings()
    stale_before = utcnow() - timedelta(seconds=settings.stale_running_seconds)
    with session_scope() as db:
        # A worker that died mid-evaluation leaves a stale `running` row; put it back in the queue.
        db.execute(
            update(EvaluationRun)
            .where(EvaluationRun.status == "running", EvaluationRun.started_at < stale_before)
            .values(status="pending")
        )
        candidates = db.scalars(
            select(EvaluationRun.evaluation_id).where(EvaluationRun.status == "pending").order_by(EvaluationRun.created_at).limit(5)
        ).all()
        for evaluation_id in candidates:
            res = db.execute(
                update(EvaluationRun)
                .where(EvaluationRun.evaluation_id == evaluation_id, EvaluationRun.status == "pending")
                .values(status="running", started_at=utcnow())
            )
            if res.rowcount == 1:
                return evaluation_id
    return None


def load_eval_input(evaluation_id: str) -> EvalInput:
    with session_scope() as db:
        run = db.get(EvaluationRun, evaluation_id)
        if run is None:
            raise LookupError(evaluation_id)
        trace = run.trace
        return EvalInput(
            trace_id=trace.trace_id,
            question=trace.question,
            answer=trace.answer,
            success=trace.success,
            context=[ContextItem(**c) for c in trace.context or []],
            citations=[Citation(**c) for c in trace.citations or []],
            tool_calls=[
                ToolCallRecord(tool_name=t.tool_name, tool_input=t.tool_input, tool_output=t.tool_output, success=t.success, error=t.error)
                for t in trace.tool_calls
            ],
            expected=Expected(**(run.expected or {})),
        )


def save_results(evaluation_id: str, results: Sequence[EvaluatorResult], *, trace_success: bool, error: Optional[str] = None) -> None:
    with session_scope() as db:
        run = db.get(EvaluationRun, evaluation_id)
        if run is None:
            raise LookupError(evaluation_id)
        run.metrics = [
            EvaluationMetric(
                metric=r.metric, status=r.status, score=r.score, passed=r.passed, reason=redact_text(r.reason),
                confidence=r.confidence, details=r.details,
            )
            for r in results
        ]
        for r in results:
            col = SCORE_COLUMNS.get(r.metric)
            if col:
                setattr(run, col, r.score)
        run.passed = overall_passed(results, trace_success)
        run.status = "failed" if error else "completed"
        run.error = redact_text(error) if error else None
        run.completed_at = utcnow()


def mark_failed(evaluation_id: str, error: str) -> None:
    with session_scope() as db:
        run = db.get(EvaluationRun, evaluation_id)
        if run is not None:
            run.status = "failed"
            run.error = redact_text(error)
            run.completed_at = utcnow()
