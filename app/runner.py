"""Runs the LangGraph workflow for one request with tracing, then hands the trace to the evaluation pipeline.

Shared by the HTTP endpoint and the evaluation CLI so both exercise exactly the same code path.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional

from evaluation import store
from evaluation.schemas import Expected
from evaluation.tracing import TraceCollector, log_event, set_current_trace
from evaluation.worker import get_worker

from .graph import PROMPT_VERSION, graph

logger = logging.getLogger("fire_agent.runner")


class WorkflowRun:
    def __init__(
        self,
        question: str,
        *,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        expected: Optional[Expected] = None,
        suite_run_id: Optional[str] = None,
        case_id: Optional[str] = None,
        claim_evaluation: bool = False,
    ):
        self.collector = TraceCollector(question=question, session_id=session_id, user_id=user_id, prompt_version=PROMPT_VERSION)
        self.expected = expected
        self.suite_run_id = suite_run_id
        self.case_id = case_id
        self.claim_evaluation = claim_evaluation  # True: caller evaluates the run itself (dataset runner)
        self.state: dict = {}
        self.error: Optional[str] = None
        self.evaluation_id: Optional[str] = None
        self._finalized = False

    @property
    def trace_id(self) -> str:
        return self.collector.trace_id

    async def events(self) -> AsyncIterator[dict]:
        """Yield the same event payloads the SSE endpoint has always streamed (plus trace_id on `done`)."""
        set_current_trace(self.collector)
        initial_state = {"question": self.collector.question, "sub_questions": [], "qa_pairs": [], "final_answer": "", "error": None}
        log_event(logger, "request_start")
        try:
            async for update in graph.astream(initial_state, stream_mode="updates"):
                for node_name, node_update in update.items():
                    if node_update.get("error"):
                        self.error = node_update["error"]
                        yield {"node": "error", "message": self.error}
                        return
                    self.state.update(node_update)
                    yield {"node": node_name, "data": node_update}
            yield {"node": "done", "trace_id": self.trace_id}
        except Exception as exc:
            logger.exception("analyze_stream_unhandled_error")
            self.error = f"{type(exc).__name__}"
            yield {"node": "error", "message": "An internal error occurred while running the workflow."}

    async def finalize(self) -> Optional[str]:
        """Persist the trace and queue its evaluation. Idempotent; failures are logged, never raised."""
        if self._finalized:
            return self.evaluation_id
        self._finalized = True
        answer = self.state.get("final_answer") or None
        error = self.error
        if answer is None and error is None:
            error = "Workflow ended without a final answer."
        context = [
            {"id": f"note_{i + 1}", "source": "research_agent", "text": f"Q: {p['question']}\nA: {p['answer']}"}
            for i, p in enumerate(self.state.get("qa_pairs", []))
        ]
        try:
            self.evaluation_id = await asyncio.to_thread(
                store.persist_trace, self.collector, answer=answer, error=error, context=context,
                expected=self.expected, suite_run_id=self.suite_run_id, case_id=self.case_id,
                claim=self.claim_evaluation,
            )
            log_event(logger, "request_done", success=error is None, duration_ms=round(self.collector.elapsed_ms()), evaluation_id=self.evaluation_id)
            worker = get_worker()
            if worker is not None:
                worker.notify()
        except Exception:
            logger.exception("event=trace_persist_failed trace_id=%s", self.trace_id)
        return self.evaluation_id
