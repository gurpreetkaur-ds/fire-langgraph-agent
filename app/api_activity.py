"""Agent Activity API: the signed-in user's own workflow executions (observable events only).

Unlike /api/eval/* (admin, all traffic), this is scoped to traces whose user_id is the current user, and it never
returns prompts or LLM input/output previews: just nodes, timings, tool calls, token usage, errors and scores.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy import func, select

from evaluation.db import session_scope
from evaluation.models import AgentTrace, EvaluationRun, utcnow
from evaluation.redaction import redact_obj

from .auth import CurrentUser, current_user

router = APIRouter(prefix="/api/activity", tags=["activity"])
NO_STORE = {"Cache-Control": "no-store"}
SCORE_FIELDS = ("faithfulness_score", "relevance_score", "groundedness_score", "citation_accuracy_score", "hallucination_score")


def _iso(dt):
    return dt.isoformat() if dt else None


def _llm_ms(t: AgentTrace) -> float:
    return round(sum(s.latency_ms or 0 for s in t.steps), 1)


def _tool_ms(t: AgentTrace, *names: str) -> Optional[float]:
    vals = [c.latency_ms for c in t.tool_calls if c.tool_name in names]
    return round(sum(vals), 1) if vals else None


RETRIEVAL_TOOLS = ("embed_query", "vector_search")


def _row(t: AgentTrace) -> dict:
    return {
        "trace_id": t.trace_id, "conversation_id": t.session_id, "created_at": _iso(t.created_at),
        "question": (t.question or "")[:200], "route": t.route, "success": t.success, "error": t.error,
        "duration_ms": round(t.duration_ms or 0, 1), "llm_ms": _llm_ms(t),
        "retrieval_ms": _tool_ms(t, *RETRIEVAL_TOOLS), "embed_ms": _tool_ms(t, "embed_query"),
        "search_ms": _tool_ms(t, "vector_search"), "stopped": bool(t.error and t.error.startswith("Stopped by the user")),
        "llm_calls": len(t.steps), "tool_calls": t.tool_call_count, "total_tokens": t.total_tokens,
        "models": t.models_used or [],
    }


@router.get("")
def list_activity(limit: int = Query(30, ge=1, le=100), offset: int = Query(0, ge=0),
                  route: Optional[str] = Query(None, pattern="^(rag|research|direct)$"),
                  user: CurrentUser = Depends(current_user)):
    with session_scope() as db:
        stmt = select(AgentTrace).where(AgentTrace.user_id == user.id)
        if route:
            stmt = stmt.where(AgentTrace.route == route)
        total = db.scalar(select(func.count()).select_from(stmt.subquery()))
        traces = db.scalars(stmt.order_by(AgentTrace.created_at.desc()).offset(offset).limit(limit)).all()
        items = [_row(t) for t in traces]

        since = utcnow() - timedelta(days=7)
        recent = db.scalars(select(AgentTrace).where(AgentTrace.user_id == user.id, AgentTrace.created_at >= since)).all()
        by_route: dict = {}
        for t in recent:
            r = by_route.setdefault(t.route or "unknown", {"runs": 0, "duration_ms": 0.0, "errors": 0})
            r["runs"] += 1
            r["duration_ms"] += t.duration_ms or 0
            r["errors"] += 0 if t.success else 1
        for r in by_route.values():
            r["avg_duration_ms"] = round(r.pop("duration_ms") / r["runs"], 1) if r["runs"] else None
        retrieval = [x for x in (_tool_ms(t, *RETRIEVAL_TOOLS) for t in recent) if x is not None]
        llm = [_llm_ms(t) for t in recent if t.steps]
        summary = {
            "window_days": 7,
            "runs": len(recent),
            "errors": sum(1 for t in recent if not t.success),
            "avg_duration_ms": round(sum(t.duration_ms or 0 for t in recent) / len(recent), 1) if recent else None,
            "avg_retrieval_ms": round(sum(retrieval) / len(retrieval), 1) if retrieval else None,
            "avg_llm_ms": round(sum(llm) / len(llm), 1) if llm else None,
            "total_tokens": sum(t.total_tokens or 0 for t in recent),
            "by_route": by_route,
        }
    return JSONResponse(redact_obj({"total": total, "items": items, "summary": summary}, pii=False), headers=NO_STORE)


@router.get("/{trace_id}")
def activity_detail(trace_id: str, user: CurrentUser = Depends(current_user)):
    with session_scope() as db:
        t = db.scalar(select(AgentTrace).where(AgentTrace.trace_id == trace_id, AgentTrace.user_id == user.id))
        if t is None:
            raise HTTPException(404, "Execution not found.")
        ev = db.scalar(select(EvaluationRun).where(EvaluationRun.trace_id == trace_id).order_by(EvaluationRun.created_at.desc()))
        out = {
            **_row(t),
            "question": t.question,
            "timeline": t.timeline or [],
            "llm_steps": [
                {"seq": s.seq, "agent": s.agent_name, "node": s.node, "model": s.model_name, "latency_ms": round(s.latency_ms, 1),
                 "input_tokens": s.input_tokens, "output_tokens": s.output_tokens, "success": s.success, "error": s.error,
                 "started_at": _iso(s.started_at)}
                for s in t.steps
            ],
            "tools": [
                {"seq": c.seq, "tool": c.tool_name, "agent": c.agent_name, "latency_ms": round(c.latency_ms, 1), "success": c.success,
                 "error": c.error, "input": c.tool_input, "output": c.tool_output}
                for c in t.tool_calls
            ],
            "evaluation": None if ev is None else {
                "status": ev.status, "passed": ev.passed,
                "scores": {f.replace("_score", ""): getattr(ev, f) for f in SCORE_FIELDS},
            },
        }
    return JSONResponse(redact_obj(out, pii=False), headers=NO_STORE)
