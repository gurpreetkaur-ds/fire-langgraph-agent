"""Per-request trace collection, structured logging with trace_id, and the LLM-call tracing helper."""
from __future__ import annotations

import logging
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Optional

from langchain_core.messages import BaseMessage

from .pricing import estimate_cost
from .redaction import redact_obj, redact_text, truncate

logger = logging.getLogger("fire_agent.trace")

_current_trace: ContextVar[Optional["TraceCollector"]] = ContextVar("current_trace", default=None)
PREVIEW_CHARS = 2000


def new_trace_id() -> str:
    return "req_" + uuid.uuid4().hex[:12]


def get_current_trace() -> Optional["TraceCollector"]:
    return _current_trace.get()


def set_current_trace(trace: Optional["TraceCollector"]) -> None:
    _current_trace.set(trace)


# ---- logging ---------------------------------------------------------------------------------------------

class TraceIdFilter(logging.Filter):
    """Adds `trace_id` to every log record so any line can be tied back to a request."""

    def filter(self, record: logging.LogRecord) -> bool:
        trace = _current_trace.get()
        record.trace_id = trace.trace_id if trace else "-"
        return True


def setup_logging(level: int = logging.INFO) -> None:
    fmt = "%(asctime)s %(levelname)s %(name)s trace_id=%(trace_id)s %(message)s"
    logging.basicConfig(level=level, format=fmt)
    for handler in logging.getLogger().handlers:
        if not any(isinstance(f, TraceIdFilter) for f in handler.filters):
            handler.addFilter(TraceIdFilter())
        handler.setFormatter(logging.Formatter(fmt))


def log_event(log: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Structured key=value event line, e.g. `event=llm_call agent=answer_agent latency_ms=182`."""
    parts = [f"event={event}"] + [f"{k}={v}" for k, v in fields.items() if v is not None]
    log.log(level, " ".join(parts))


# ---- collector -------------------------------------------------------------------------------------------

@dataclass
class StepRecord:
    seq: int
    agent_name: str
    node: str
    model_name: Optional[str]
    call_id: Optional[str]
    started_at: datetime
    latency_ms: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    estimated_cost: Optional[float]
    success: bool
    error: Optional[str]
    input_preview: Optional[str]
    output_preview: Optional[str]


@dataclass
class ToolCallData:
    seq: int
    agent_name: Optional[str]
    tool_name: str
    tool_input: Optional[dict]
    tool_output: Any
    latency_ms: float
    success: bool
    error: Optional[str]


@dataclass
class TraceCollector:
    question: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    prompt_version: Optional[str] = None
    trace_id: str = field(default_factory=new_trace_id)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    steps: List[StepRecord] = field(default_factory=list)
    tool_calls: List[ToolCallData] = field(default_factory=list)
    _t0: float = field(default_factory=time.perf_counter)

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def record_step(self, **kw: Any) -> StepRecord:
        step = StepRecord(seq=len(self.steps) + 1, **kw)
        self.steps.append(step)
        return step

    def record_tool_call(
        self,
        tool_name: str,
        *,
        agent_name: Optional[str] = None,
        tool_input: Optional[dict] = None,
        tool_output: Any = None,
        latency_ms: float = 0.0,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Tools call this (or use `traced_tool`) so inputs/outputs are persisted, redacted."""
        self.tool_calls.append(
            ToolCallData(
                seq=len(self.tool_calls) + 1,
                agent_name=agent_name,
                tool_name=tool_name,
                tool_input=redact_obj(tool_input),
                tool_output=redact_obj(tool_output),
                latency_ms=latency_ms,
                success=success,
                error=redact_text(error) if error else None,
            )
        )
        log_event(logger, "tool_call", agent=agent_name, tool=tool_name, latency_ms=round(latency_ms), success=success)


def _fire_meta(message: BaseMessage) -> dict:
    meta = getattr(message, "response_metadata", None) or {}
    return meta.get("fire") or {}


async def traced_invoke(agent_name: str, node: str, llm: Any, messages: List[BaseMessage]) -> BaseMessage:
    """Invoke an LLM and record the call (latency, tokens, cost, previews) on the active trace.

    Works without an active trace (the call just isn't recorded) so graph code stays usable standalone.
    Exceptions propagate unchanged after the failed step is recorded.
    """
    trace = get_current_trace()
    model = getattr(llm, "species_name", None) or "fire-default"
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    prompt_preview = truncate(redact_text("\n".join(str(m.content) for m in messages)), PREVIEW_CHARS)
    try:
        response = await llm.ainvoke(messages)
    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000.0
        if trace:
            trace.record_step(
                agent_name=agent_name, node=node, model_name=model, call_id=None, started_at=started,
                latency_ms=latency, input_tokens=None, output_tokens=None, estimated_cost=None,
                success=False, error=redact_text(f"{type(exc).__name__}: {exc}"),
                input_preview=prompt_preview, output_preview=None,
            )
        log_event(logger, "llm_call", logging.ERROR, agent=agent_name, model=model, latency_ms=round(latency), success=False)
        raise
    latency = (time.perf_counter() - t0) * 1000.0
    meta = _fire_meta(response)
    usage = meta.get("usage") or {}
    in_tok, out_tok = usage.get("input"), usage.get("output")
    cost = estimate_cost(model, in_tok, out_tok)
    if trace:
        trace.record_step(
            agent_name=agent_name, node=node, model_name=model, call_id=meta.get("call_id"), started_at=started,
            latency_ms=latency, input_tokens=in_tok, output_tokens=out_tok, estimated_cost=cost,
            success=True, error=None, input_preview=prompt_preview,
            output_preview=truncate(redact_text(str(response.content)), PREVIEW_CHARS),
        )
    log_event(
        logger, "llm_call", agent=agent_name, model=model, latency_ms=round(latency),
        input_tokens=in_tok, output_tokens=out_tok, call_id=meta.get("call_id"),
    )
    return response
