"""Runs the LangGraph workflow for one request with tracing, then hands the trace to the evaluation pipeline.

Shared by the HTTP endpoints and the evaluation CLI so all of them exercise exactly the same code path.

- `events()`  : the original protocol of /api/analyze/stream (node updates of the research pipeline).
- `stream()`  : the chat protocol: observable node events, answer tokens, and a final payload with sources.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, List, Optional

from evaluation import store
from evaluation.schemas import Expected
from evaluation.tracing import TraceCollector, log_event, set_current_trace
from evaluation.worker import get_worker

from .graph import PROMPT_VERSION, graph
from .rag import retrieval
from .rag import store as kb_store

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
        force_route: Optional[str] = "research",
        history: Optional[List[dict]] = None,
        kb_mode: str = "auto",
        document_ids: Optional[List[str]] = None,
    ):
        self.collector = TraceCollector(question=question, session_id=session_id, user_id=user_id, prompt_version=PROMPT_VERSION)
        self.expected = expected
        self.suite_run_id = suite_run_id
        self.case_id = case_id
        self.claim_evaluation = claim_evaluation  # True: caller evaluates the run itself (dataset runner)
        self.force_route = force_route  # "research" keeps the original pipeline; None lets the analyzer route
        self.history = history or []
        self.kb_mode = kb_mode
        self.document_ids = document_ids or []
        self.state: dict = {}
        self.partial: List[str] = []  # streamed answer text so far (kept if the user stops generation)
        self.error: Optional[str] = None
        self.stopped = False
        self.evaluation_id: Optional[str] = None
        self._finalized = False

    @property
    def trace_id(self) -> str:
        return self.collector.trace_id

    def _initial_state(self, kb_documents: Optional[list] = None) -> dict:
        state = {"question": self.collector.question, "sub_questions": [], "qa_pairs": [], "final_answer": "", "error": None}
        if self.force_route is not None:
            state["force_route"] = self.force_route
        if self.collector.user_id is not None or self.force_route is None:
            state.update(user_id=self.collector.user_id, history=self.history, kb_mode=self.kb_mode,
                         document_ids=self.document_ids, kb_documents=kb_documents or [])
        return state

    async def events(self) -> AsyncIterator[dict]:
        """Yield the same event payloads the SSE endpoint has always streamed (plus trace_id on `done`)."""
        set_current_trace(self.collector)
        initial_state = self._initial_state()
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
            self.collector.route = self.state.get("route") or self.force_route
            yield {"node": "done", "trace_id": self.trace_id}
        except Exception as exc:
            logger.exception("analyze_stream_unhandled_error")
            self.error = f"{type(exc).__name__}"
            yield {"node": "error", "message": "An internal error occurred while running the workflow."}

    async def stream(self) -> AsyncIterator[dict]:
        """Chat protocol. Events: node | token | answer | final | error (see docs/architecture.md)."""
        set_current_trace(self.collector)
        log_event(logger, "chat_request_start", route=self.force_route or "auto")
        kb_documents: list = []
        if self.collector.user_id and self.kb_mode != "off":
            kb_documents = await asyncio.to_thread(kb_store.ready_documents, self.collector.user_id)
        try:
            async for mode, chunk in graph.astream(self._initial_state(kb_documents), stream_mode=["updates", "custom"]):
                if mode == "custom":
                    if chunk.get("type") == "token":
                        self.partial.append(chunk.get("text", ""))
                    yield chunk
                    continue
                for node_name, node_update in chunk.items():
                    if not node_update:
                        continue
                    if node_update.get("error"):
                        self.error = node_update["error"]
                        self.collector.route = self.state.get("route")
                        yield {"type": "error", "node": node_name, "message": _user_safe(self.error)}
                        return
                    self.state.update(node_update)
                    if node_name == "validate_citations":
                        # citations may have been cleaned (invalid markers removed): replace the streamed text
                        yield {"type": "answer", "text": self.state.get("final_answer", "")}
            self.collector.route = self.state.get("route")
            yield {"type": "final", **self.result()}
        except Exception as exc:
            logger.exception("chat_stream_unhandled_error")
            self.error = type(exc).__name__
            yield {"type": "error", "message": "An internal error occurred while running the workflow."}

    def result(self) -> dict:
        """Observable outcome of the run (safe to send to the browser and to store with the message)."""
        st = self.state
        sources = st.get("sources") or []
        report = st.get("citation_report") or {}
        steps = self.collector.steps
        in_tok = sum(s.input_tokens or 0 for s in steps)
        out_tok = sum(s.output_tokens or 0 for s in steps)
        return {
            "answer": st.get("final_answer") or "".join(self.partial),
            "route": st.get("route") or self.force_route,
            "route_reason": st.get("route_reason"),
            "standalone_question": st.get("standalone_question"),
            "search_queries": st.get("search_queries") or [],
            "sources": retrieval.pick_sources_for_ui(sources, report.get("used_sources")),
            "citations": st.get("citations") or [],
            "grounded": st.get("grounded"),
            "retrieval": st.get("retrieval_stats"),
            "sub_questions": st.get("sub_questions") or [],
            "trace_id": self.trace_id,
            "duration_ms": round(self.collector.elapsed_ms()),
            "usage": {"input_tokens": in_tok, "output_tokens": out_tok, "llm_calls": len(steps)},
            "models": list(dict.fromkeys(s.model_name for s in steps if s.model_name)),
        }

    def _eval_context(self) -> list:
        sources = self.state.get("sources")
        if sources:
            return [{"id": s["id"], "source": s["filename"] + (f" ({retrieval.location(s)})" if retrieval.location(s) else ""),
                     "text": s["text"]} for s in sources]
        return [
            {"id": f"note_{i + 1}", "source": "research_agent", "text": f"Q: {p['question']}\nA: {p['answer']}"}
            for i, p in enumerate(self.state.get("qa_pairs", []))
        ]

    async def finalize(self) -> Optional[str]:
        """Persist the trace and queue its evaluation. Idempotent; failures are logged, never raised."""
        if self._finalized:
            return self.evaluation_id
        self._finalized = True
        answer = self.state.get("final_answer") or ("".join(self.partial) if self.stopped else None) or None
        error = self.error
        if self.stopped:
            error = "Stopped by the user before the answer was complete."
        if answer is None and error is None:
            error = "Workflow ended without a final answer."
        if self.collector.route is None:
            self.collector.route = self.state.get("route") or self.force_route
        try:
            self.evaluation_id = await asyncio.to_thread(
                store.persist_trace, self.collector, answer=answer, error=error, context=self._eval_context(),
                citations=self.state.get("citations") or [], expected=self.expected, suite_run_id=self.suite_run_id,
                case_id=self.case_id, claim=self.claim_evaluation, enqueue_evaluation=not self.stopped,
            )
            log_event(logger, "request_done", success=error is None, duration_ms=round(self.collector.elapsed_ms()), evaluation_id=self.evaluation_id)
            worker = get_worker()
            if worker is not None:
                worker.notify()
        except Exception:
            logger.exception("event=trace_persist_failed trace_id=%s", self.trace_id)
        return self.evaluation_id


def _user_safe(error: str) -> str:
    """Errors shown in the chat: Fire status messages are safe (they never include the key); keep them short."""
    if not error:
        return "The workflow hit an error."
    if "FIRE_API_KEY" in error:
        return "The server is missing its Fire API key."
    if "HTTP 429" in error:
        return "The AI provider is rate-limiting requests. Please try again in a moment."
    if "Could not reach" in error:
        return "Could not reach the AI provider. Please try again."
    return "The AI provider returned an error. Please try again."
