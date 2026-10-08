"""Read-only, authenticated API for the AI Evaluation Center.

Every route requires `Authorization: Bearer <EVAL_ADMIN_TOKEN>`. If the token is not configured the whole API
answers 503 (closed by default). There are no write endpoints: scores can only be produced by the evaluation
pipeline, never submitted by a client. Responses are passed through secret redaction as defense in depth.
"""
from __future__ import annotations

import hmac
from datetime import date
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from evaluation import aggregate
from evaluation.config import ALL_METRICS, get_settings
from evaluation.db import session_scope
from evaluation.redaction import redact_obj
from evaluation.regression import compare

NO_STORE = {"Cache-Control": "no-store"}


def require_eval_access(request: Request) -> None:
    token = get_settings().admin_token
    if not token:
        raise HTTPException(503, "Evaluation API is disabled: EVAL_ADMIN_TOKEN is not configured on the server.")
    header = request.headers.get("authorization", "")
    scheme, _, supplied = header.partition(" ")
    if scheme.lower() != "bearer" or not supplied or not hmac.compare_digest(supplied.strip().encode(), token.encode()):
        raise HTTPException(401, "Invalid or missing evaluation access token.", headers={"WWW-Authenticate": "Bearer"})


router = APIRouter(prefix="/api/eval", dependencies=[Depends(require_eval_access)], tags=["evaluation"])


def _out(data: Any) -> JSONResponse:
    return JSONResponse(redact_obj(data, pii=False), headers=NO_STORE)


def _filters(
    agent: Optional[str] = None,
    model: Optional[str] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    status: Optional[str] = Query(None, pattern="^(pending|running|completed|failed)$"),
    trace_id: Optional[str] = Query(None, max_length=40),
    failed_metric: Optional[str] = None,
    suite_run_id: Optional[str] = None,
) -> aggregate.Filters:
    if failed_metric and failed_metric not in ALL_METRICS:
        raise HTTPException(422, f"failed_metric must be one of {list(ALL_METRICS)}")
    return aggregate.Filters(agent, model, date_from, date_to, status, trace_id, failed_metric, suite_run_id)


@router.get("/auth-check")
def auth_check():
    return _out({"ok": True})


@router.get("/filters")
def filters():
    with session_scope() as db:
        return _out(aggregate.filter_options(db))


@router.get("/summary")
def summary(f: aggregate.Filters = Depends(_filters)):
    with session_scope() as db:
        return _out({**aggregate.summary(db, f), "thresholds": get_settings().thresholds})


@router.get("/timeseries")
def timeseries(f: aggregate.Filters = Depends(_filters)):
    with session_scope() as db:
        return _out(aggregate.timeseries(db, f))


@router.get("/agents")
def agents(f: aggregate.Filters = Depends(_filters)):
    with session_scope() as db:
        return _out(aggregate.agents(db, f))


@router.get("/agents/{name}")
def agent_detail(name: str, f: aggregate.Filters = Depends(_filters)):
    with session_scope() as db:
        data = aggregate.agent_detail(db, f, name)
    if data is None:
        raise HTTPException(404, "Unknown agent")
    return _out(data)


@router.get("/traces")
def traces(f: aggregate.Filters = Depends(_filters), limit: int = Query(25, ge=1, le=200), offset: int = Query(0, ge=0)):
    with session_scope() as db:
        return _out(aggregate.traces(db, f, limit, offset))


@router.get("/traces/{trace_id}")
def trace_detail(trace_id: str):
    with session_scope() as db:
        data = aggregate.trace_detail(db, trace_id)
    if data is None:
        raise HTTPException(404, "Trace not found")
    return _out(data)


@router.get("/suites")
def suites():
    with session_scope() as db:
        return _out(aggregate.suite_runs(db))


@router.get("/suites/{suite_run_id}")
def suite(suite_run_id: str):
    with session_scope() as db:
        report = aggregate.build_suite_report(db, suite_run_id)
    if report is None:
        raise HTTPException(404, "Suite run not found")
    return _out(report.model_dump())


@router.get("/regression")
def regression(baseline: str, candidate: str):
    with session_scope() as db:
        b, c = aggregate.build_suite_report(db, baseline), aggregate.build_suite_report(db, candidate)
    if b is None or c is None:
        raise HTTPException(404, "Suite run not found")
    report = compare(b, c)
    return _out({**report.model_dump(), "has_regression": report.has_regression})
