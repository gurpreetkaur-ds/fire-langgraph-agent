from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

from evaluation.tracing import setup_logging  # noqa: E402

setup_logging()  # same INFO logging as before, plus trace_id on every line
logger = logging.getLogger("fire_agent.main")

from evaluation.config import get_settings as get_eval_settings  # noqa: E402
from evaluation.db import init_engine, run_migrations  # noqa: E402
from evaluation.worker import EvaluationWorker, get_worker, set_worker  # noqa: E402

from .eval_api import router as eval_router  # noqa: E402
from .runner import WorkflowRun  # noqa: E402  (import after dotenv/logging setup)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Fire LangGraph Agent")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


app.include_router(eval_router)
_background_tasks: set[asyncio.Task] = set()  # keeps references so finalize tasks aren't garbage-collected


class AnalyzeRequest(BaseModel):
    question: str = Field(max_length=4000)
    session_id: str | None = Field(default=None, max_length=100)


@app.on_event("startup")
async def check_config() -> None:
    # Only ever check PRESENCE of the key, never its value.
    if not os.environ.get("FIRE_API_KEY"):
        logger.warning("FIRE_API_KEY is not set -- requests to Fire will fail until it is configured in .env")
    else:
        logger.info("FIRE_API_KEY is configured (value not logged)")

    eval_settings = get_eval_settings()
    if eval_settings.auto_migrate:
        await asyncio.to_thread(run_migrations)
    init_engine()
    if eval_settings.enabled and eval_settings.worker_mode == "inline":
        worker = EvaluationWorker()
        set_worker(worker)
        worker.start()
    if not eval_settings.admin_token:
        logger.warning("EVAL_ADMIN_TOKEN is not set -- /api/eval/* is disabled until it is configured")


@app.on_event("shutdown")
async def stop_worker() -> None:
    worker = get_worker()
    if worker is not None:
        await worker.stop()
        set_worker(None)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "fire_key_configured": bool(os.environ.get("FIRE_API_KEY")),
        "access_required": bool(os.environ.get("AGENT_ACCESS_TOKEN")),
    }


def _check_agent_access(supplied: str | None) -> None:
    """When AGENT_ACCESS_TOKEN is set (public deploys), asking questions needs it; each question spends Fire credit."""
    expected = os.environ.get("AGENT_ACCESS_TOKEN")
    if not expected:
        return
    if not supplied or not hmac.compare_digest(supplied.strip().encode(), expected.encode()):
        raise HTTPException(401, "Wrong or missing access password.")


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/api/analyze/stream")
async def analyze_stream(req: AnalyzeRequest, x_access_token: str | None = Header(default=None)):
    _check_agent_access(x_access_token)
    question = (req.question or "").strip()

    async def event_gen():
        if not question:
            yield _sse({"node": "error", "message": "Question cannot be empty."})
            return
        run = WorkflowRun(question, session_id=req.session_id)
        try:
            async for event in run.events():
                yield _sse(event)
        finally:
            # Persist + queue evaluation after the answer has been streamed; never blocks the response.
            task = asyncio.create_task(run.finalize())
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)

    return StreamingResponse(event_gen(), media_type="text/event-stream")
