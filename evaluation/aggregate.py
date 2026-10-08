"""Read-side queries for the dashboard, agent views and suite reports. All functions take an open Session."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel
from sqlalchemy import and_, exists, func, select
from sqlalchemy.orm import Session

from .config import ALL_METRICS, LOWER_IS_BETTER, QUALITY_METRICS
from .models import AgentStep, AgentTrace, EvaluationMetric, EvaluationRun
from .store import SCORE_COLUMNS

STATUSES = ("pending", "running", "completed", "failed")


@dataclass
class Filters:
    agent: Optional[str] = None
    model: Optional[str] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    status: Optional[str] = None
    trace_id: Optional[str] = None
    failed_metric: Optional[str] = None
    suite_run_id: Optional[str] = None


def _utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    dt = _utc(dt)
    return dt.isoformat() if dt else None


def _conditions(f: Filters) -> list:
    conds: list = []
    if f.agent:
        conds.append(exists().where(and_(AgentStep.trace_id == EvaluationRun.trace_id, AgentStep.agent_name == f.agent)))
    if f.model:
        conds.append(exists().where(and_(AgentStep.trace_id == EvaluationRun.trace_id, AgentStep.model_name == f.model)))
    if f.date_from:
        conds.append(EvaluationRun.created_at >= datetime.combine(f.date_from, datetime.min.time(), tzinfo=timezone.utc))
    if f.date_to:
        conds.append(EvaluationRun.created_at < datetime.combine(f.date_to + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc))
    if f.status:
        conds.append(EvaluationRun.status == f.status)
    if f.trace_id:
        conds.append(EvaluationRun.trace_id == f.trace_id)
    if f.suite_run_id:
        conds.append(EvaluationRun.suite_run_id == f.suite_run_id)
    if f.failed_metric:
        conds.append(
            exists().where(
                and_(
                    EvaluationMetric.evaluation_id == EvaluationRun.evaluation_id,
                    EvaluationMetric.metric == f.failed_metric,
                    EvaluationMetric.status == "ok",
                    EvaluationMetric.passed.is_(False),
                )
            )
        )
    return conds


def _avg(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def metric_averages(runs: Iterable[EvaluationRun]) -> Dict[str, Dict[str, Any]]:
    """Mean per metric over runs where it was actually computed; `n` shows how many that was."""
    runs = list(runs)
    out: Dict[str, Dict[str, Any]] = {}
    for metric in ALL_METRICS:
        vals = [getattr(r, SCORE_COLUMNS[metric]) for r in runs]
        present = [v for v in vals if v is not None]
        out[metric] = {"avg": (sum(present) / len(present)) if present else None, "n": len(present)}
    return out


def reliability_score(averages: Dict[str, Dict[str, Any]]) -> Optional[float]:
    """Unweighted mean of the computed quality metrics, with hallucination inverted (1 - rate)."""
    parts = [averages[m]["avg"] for m in QUALITY_METRICS if averages[m]["avg"] is not None]
    h = averages["hallucination"]["avg"]
    if h is not None:
        parts.append(1 - h)
    return sum(parts) / len(parts) if parts else None


def _runs(db: Session, f: Filters) -> List[EvaluationRun]:
    return list(db.scalars(select(EvaluationRun).where(*_conditions(f)).order_by(EvaluationRun.created_at)))


def summary(db: Session, f: Filters) -> Dict[str, Any]:
    runs = _runs(db, f)
    done = [r for r in runs if r.status == "completed"]
    averages = metric_averages(done)
    trace_ids = {r.trace_id for r in runs}
    traces = list(db.scalars(select(AgentTrace).where(AgentTrace.trace_id.in_(trace_ids)))) if trace_ids else []
    costs = [r.estimated_cost for r in runs]
    return {
        "reliability_score": reliability_score(averages),
        "metrics": averages,
        "evaluations_total": len(runs),
        "evaluations_completed": len(done),
        "evaluations_pending": sum(1 for r in runs if r.status in ("pending", "running")),
        "failed_evaluations": sum(1 for r in runs if r.passed is False),
        "evaluation_errors": sum(1 for r in runs if r.status == "failed"),
        "request_error_rate": (sum(1 for t in traces if not t.success) / len(traces)) if traces else None,
        "avg_latency_ms": _avg(r.latency_ms for r in runs),
        "avg_cost": _avg(costs),
        "cost_known_for": sum(1 for c in costs if c is not None),
        "total_cost": sum(c for c in costs if c is not None) if any(c is not None for c in costs) else None,
        "total_tokens": sum(r.total_tokens for r in runs if r.total_tokens is not None) if any(r.total_tokens is not None for r in runs) else None,
        "avg_tool_calls": _avg(t.tool_call_count for t in traces),
    }


def timeseries(db: Session, f: Filters) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[EvaluationRun]] = defaultdict(list)
    for r in _runs(db, f):
        buckets[_utc(r.created_at).date().isoformat()].append(r)
    points = []
    for day in sorted(buckets):
        rs = buckets[day]
        done = [r for r in rs if r.status == "completed"]
        avgs = metric_averages(done)
        costs = [r.estimated_cost for r in rs if r.estimated_cost is not None]
        points.append(
            {
                "date": day,
                "evaluations": len(rs),
                "scores": {m: avgs[m]["avg"] for m in ALL_METRICS},
                "avg_latency_ms": _avg(r.latency_ms for r in rs),
                "input_tokens": sum(r.input_tokens or 0 for r in rs),
                "output_tokens": sum(r.output_tokens or 0 for r in rs),
                "total_tokens": sum(r.total_tokens or 0 for r in rs),
                "total_cost": sum(costs) if costs else None,
                "failure_rate": sum(1 for r in rs if r.passed is False) / len(rs),
                "hallucination_rate": avgs["hallucination"]["avg"],
            }
        )
    return points


def _agent_stats(db: Session, f: Filters, agent: str) -> Dict[str, Any]:
    q = select(AgentStep).join(AgentTrace, AgentTrace.trace_id == AgentStep.trace_id).where(AgentStep.agent_name == agent)
    if f.date_from:
        q = q.where(AgentTrace.created_at >= datetime.combine(f.date_from, datetime.min.time(), tzinfo=timezone.utc))
    if f.date_to:
        q = q.where(AgentTrace.created_at < datetime.combine(f.date_to + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc))
    if f.model:
        q = q.where(AgentStep.model_name == f.model)
    steps = list(db.scalars(q))
    toks = [(s.input_tokens or 0) + (s.output_tokens or 0) for s in steps if s.input_tokens is not None or s.output_tokens is not None]
    costs = [s.estimated_cost for s in steps if s.estimated_cost is not None]
    trace_ids = {s.trace_id for s in steps}
    runs = _runs(db, Filters(date_from=f.date_from, date_to=f.date_to, status="completed", model=f.model, agent=agent))
    avgs = metric_averages(runs)
    return {
        "agent": agent,
        "executions": len(steps),
        "traces": len(trace_ids),
        "success_rate": (sum(1 for s in steps if s.success) / len(steps)) if steps else None,
        "avg_latency_ms": _avg(s.latency_ms for s in steps),
        "avg_tokens": (sum(toks) / len(toks)) if toks else None,
        "avg_cost": (sum(costs) / len(costs)) if costs else None,
        "models": sorted({s.model_name for s in steps if s.model_name}),
        # Evaluation is per request, so these are scores of requests this agent took part in.
        "scores": {m: avgs[m]["avg"] for m in ALL_METRICS},
        "reliability_score": reliability_score(avgs),
        "evaluated_traces": len(runs),
        "failed_evaluations": sum(1 for r in runs if r.passed is False),
        "is_final_answer_agent": db.scalar(select(func.count()).select_from(AgentTrace).where(AgentTrace.agent_name == agent)) > 0,
    }


def agents(db: Session, f: Filters) -> List[Dict[str, Any]]:
    names = list(db.scalars(select(AgentStep.agent_name).distinct().order_by(AgentStep.agent_name)))
    return [_agent_stats(db, f, n) for n in names]


def agent_detail(db: Session, f: Filters, agent: str) -> Optional[Dict[str, Any]]:
    if db.scalar(select(func.count()).select_from(AgentStep).where(AgentStep.agent_name == agent)) == 0:
        return None
    stats = _agent_stats(db, f, agent)
    ids = select(AgentStep.trace_id).where(AgentStep.agent_name == agent)
    recent = db.scalars(
        select(AgentTrace).where(AgentTrace.trace_id.in_(ids)).order_by(AgentTrace.created_at.desc()).limit(10)
    )
    failed = db.scalars(
        select(EvaluationRun).where(EvaluationRun.trace_id.in_(ids), EvaluationRun.passed.is_(False)).order_by(EvaluationRun.created_at.desc()).limit(10)
    )
    return {
        **stats,
        "recent_traces": [trace_row(db, t) for t in recent],
        "failed_evaluation_runs": [run_row(r) for r in failed],
    }


def run_row(r: EvaluationRun) -> Dict[str, Any]:
    return {
        "evaluation_id": r.evaluation_id,
        "trace_id": r.trace_id,
        "status": r.status,
        "passed": r.passed,
        "created_at": _iso(r.created_at),
        "case_id": r.case_id,
        "suite_run_id": r.suite_run_id,
        "scores": {m: getattr(r, SCORE_COLUMNS[m]) for m in ALL_METRICS},
        "error": r.error,
    }


def trace_row(db: Session, t: AgentTrace) -> Dict[str, Any]:
    latest = db.scalars(select(EvaluationRun).where(EvaluationRun.trace_id == t.trace_id).order_by(EvaluationRun.created_at.desc()).limit(1)).first()
    return {
        "trace_id": t.trace_id,
        "created_at": _iso(t.created_at),
        "question": t.question,
        "agent_name": t.agent_name,
        "model_name": t.model_name,
        "success": t.success,
        "duration_ms": t.duration_ms,
        "total_tokens": t.total_tokens,
        "estimated_cost": t.estimated_cost,
        "evaluation": run_row(latest) if latest else None,
    }


def traces(db: Session, f: Filters, limit: int, offset: int) -> Dict[str, Any]:
    conds = _conditions(f)
    q = select(EvaluationRun).where(*conds)
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    rows = db.scalars(q.order_by(EvaluationRun.created_at.desc()).limit(limit).offset(offset))
    items = []
    for r in rows:
        t = r.trace
        items.append({**trace_row(db, t), "evaluation": run_row(r)})
    return {"total": total, "items": items}


def trace_detail(db: Session, trace_id: str) -> Optional[Dict[str, Any]]:
    t = db.get(AgentTrace, trace_id)
    if t is None:
        return None
    evaluations = []
    for r in t.evaluations:
        evaluations.append(
            {
                **run_row(r),
                "expected": r.expected,
                "metrics": [
                    {"metric": m.metric, "status": m.status, "score": m.score, "passed": m.passed, "reason": m.reason, "confidence": m.confidence, "details": m.details}
                    for m in r.metrics
                ],
            }
        )
    return {
        "trace_id": t.trace_id,
        "session_id": t.session_id,
        "user_id": t.user_id,
        "created_at": _iso(t.created_at),
        "question": t.question,
        "answer": t.answer,
        "agent_name": t.agent_name,
        "agents_executed": t.agents_executed,
        "model_name": t.model_name,
        "models_used": t.models_used,
        "prompt_version": t.prompt_version,
        "context": t.context,
        "citations": t.citations,
        "duration_ms": t.duration_ms,
        "input_tokens": t.input_tokens,
        "output_tokens": t.output_tokens,
        "total_tokens": t.total_tokens,
        "estimated_cost": t.estimated_cost,
        "tool_call_count": t.tool_call_count,
        "success": t.success,
        "error": t.error,
        "steps": [
            {
                "seq": s.seq, "agent_name": s.agent_name, "node": s.node, "model_name": s.model_name, "call_id": s.call_id,
                "started_at": _iso(s.started_at), "latency_ms": s.latency_ms, "input_tokens": s.input_tokens,
                "output_tokens": s.output_tokens, "estimated_cost": s.estimated_cost, "success": s.success, "error": s.error,
                "input_preview": s.input_preview, "output_preview": s.output_preview,
            }
            for s in t.steps
        ],
        "tool_calls": [
            {"seq": c.seq, "agent_name": c.agent_name, "tool_name": c.tool_name, "tool_input": c.tool_input, "tool_output": c.tool_output, "latency_ms": c.latency_ms, "success": c.success, "error": c.error}
            for c in t.tool_calls
        ],
        "evaluations": evaluations,
    }


def filter_options(db: Session) -> Dict[str, Any]:
    return {
        "agents": list(db.scalars(select(AgentStep.agent_name).distinct().order_by(AgentStep.agent_name))),
        "models": list(db.scalars(select(AgentStep.model_name).where(AgentStep.model_name.is_not(None)).distinct().order_by(AgentStep.model_name))),
        "statuses": list(STATUSES),
        "metrics": list(ALL_METRICS),
        "suite_runs": [s["suite_run_id"] for s in suite_runs(db)],
    }


# ---- evaluation suites (dataset runs) --------------------------------------------------------------------

class CaseResult(BaseModel):
    case_id: str
    passed: Optional[bool]
    scores: Dict[str, Optional[float]]
    latency_ms: Optional[float] = None
    estimated_cost: Optional[float] = None
    trace_id: Optional[str] = None
    question: Optional[str] = None


class SuiteReport(BaseModel):
    suite_run_id: str
    created_at: Optional[str] = None
    prompt_versions: List[str] = []
    cases: List[CaseResult]
    metrics: Dict[str, Dict[str, Any]]
    reliability_score: Optional[float] = None
    passed: int = 0
    failed: int = 0
    not_evaluated: int = 0
    avg_latency_ms: Optional[float] = None
    avg_cost: Optional[float] = None


def suite_runs(db: Session) -> List[Dict[str, Any]]:
    rows = db.execute(
        select(EvaluationRun.suite_run_id, func.count(), func.min(EvaluationRun.created_at))
        .where(EvaluationRun.suite_run_id.is_not(None))
        .group_by(EvaluationRun.suite_run_id)
        .order_by(func.min(EvaluationRun.created_at).desc())
    ).all()
    return [{"suite_run_id": r[0], "cases": r[1], "created_at": _iso(r[2])} for r in rows]


def build_suite_report(db: Session, suite_run_id: str) -> Optional[SuiteReport]:
    runs = _runs(db, Filters(suite_run_id=suite_run_id))
    if not runs:
        return None
    done = [r for r in runs if r.status == "completed"]
    avgs = metric_averages(done)
    prompt_versions = sorted({r.trace.prompt_version for r in runs if r.trace.prompt_version})
    return SuiteReport(
        suite_run_id=suite_run_id,
        created_at=_iso(runs[0].created_at),
        prompt_versions=prompt_versions,
        cases=[
            CaseResult(
                case_id=r.case_id or r.evaluation_id, passed=r.passed,
                scores={m: getattr(r, SCORE_COLUMNS[m]) for m in ALL_METRICS},
                latency_ms=r.latency_ms, estimated_cost=r.estimated_cost, trace_id=r.trace_id, question=r.question,
            )
            for r in runs
        ],
        metrics=avgs,
        reliability_score=reliability_score(avgs),
        passed=sum(1 for r in runs if r.passed is True),
        failed=sum(1 for r in runs if r.passed is False),
        not_evaluated=sum(1 for r in runs if r.passed is None),
        avg_latency_ms=_avg(r.latency_ms for r in runs),
        avg_cost=_avg(r.estimated_cost for r in runs),
    )
