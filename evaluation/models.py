"""ORM models: agent_traces, agent_steps, tool_calls, evaluation_runs, evaluation_metrics."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class AgentTrace(Base):
    """One row per AI request: what was asked, what ran, what came back."""

    __tablename__ = "agent_traces"

    trace_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[Optional[str]] = mapped_column(Text)
    agent_name: Mapped[str] = mapped_column(String(100), index=True)  # agent responsible for the final answer
    agents_executed: Mapped[list[Any]] = mapped_column(JSON, default=list)
    model_name: Mapped[Optional[str]] = mapped_column(String(100), index=True)  # model of the responsible agent
    models_used: Mapped[list[Any]] = mapped_column(JSON, default=list)
    model_version: Mapped[Optional[str]] = mapped_column(String(100))
    prompt_version: Mapped[Optional[str]] = mapped_column(String(40))
    context: Mapped[list[Any]] = mapped_column(JSON, default=list)  # [{"id","source","text"}] shown to the answering agent
    citations: Mapped[list[Any]] = mapped_column(JSON, default=list)  # [{"claim","source_id"}]
    duration_ms: Mapped[Optional[float]] = mapped_column(Float)
    input_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    output_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    estimated_cost: Mapped[Optional[float]] = mapped_column(Float)  # NULL = unknown (no price configured)
    tool_call_count: Mapped[int] = mapped_column(Integer, default=0)
    success: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    error: Mapped[Optional[str]] = mapped_column(Text)

    steps: Mapped[list["AgentStep"]] = relationship(
        back_populates="trace", cascade="all, delete-orphan", order_by="AgentStep.seq"
    )
    tool_calls: Mapped[list["ToolCall"]] = relationship(
        back_populates="trace", cascade="all, delete-orphan", order_by="ToolCall.seq"
    )
    evaluations: Mapped[list["EvaluationRun"]] = relationship(
        back_populates="trace", cascade="all, delete-orphan", order_by="EvaluationRun.created_at"
    )


class AgentStep(Base):
    """One LLM call made by one agent inside a trace (the unit used for per-agent statistics)."""

    __tablename__ = "agent_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(ForeignKey("agent_traces.trace_id", ondelete="CASCADE"), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    agent_name: Mapped[str] = mapped_column(String(100), index=True)
    node: Mapped[str] = mapped_column(String(100))
    model_name: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    call_id: Mapped[Optional[str]] = mapped_column(String(100))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    input_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    output_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    estimated_cost: Mapped[Optional[float]] = mapped_column(Float)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error: Mapped[Optional[str]] = mapped_column(Text)
    input_preview: Mapped[Optional[str]] = mapped_column(Text)
    output_preview: Mapped[Optional[str]] = mapped_column(Text)

    trace: Mapped[AgentTrace] = relationship(back_populates="steps")


class ToolCall(Base):
    """A tool invocation inside a trace. The current workflow has no tools; the table and recorder exist so
    adding one automatically feeds the tool-accuracy evaluator."""

    __tablename__ = "tool_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(ForeignKey("agent_traces.trace_id", ondelete="CASCADE"), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    agent_name: Mapped[Optional[str]] = mapped_column(String(100))
    tool_name: Mapped[str] = mapped_column(String(100), index=True)
    tool_input: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    tool_output: Mapped[Optional[Any]] = mapped_column(JSON)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error: Mapped[Optional[str]] = mapped_column(Text)

    trace: Mapped[AgentTrace] = relationship(back_populates="tool_calls")


class EvaluationRun(Base):
    """One evaluation of one trace. Per-metric score columns are denormalised from evaluation_metrics
    so dashboards can aggregate without joins; evaluation_metrics keeps the reasoning and status."""

    __tablename__ = "evaluation_runs"

    evaluation_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    trace_id: Mapped[str] = mapped_column(ForeignKey("agent_traces.trace_id", ondelete="CASCADE"), index=True)
    suite_run_id: Mapped[Optional[str]] = mapped_column(String(60), index=True)
    case_id: Mapped[Optional[str]] = mapped_column(String(60))
    expected: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)  # dataset expectations, if any
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)  # pending|running|completed|failed
    passed: Mapped[Optional[bool]] = mapped_column(Boolean, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    agent_name: Mapped[str] = mapped_column(String(100), index=True)
    model_name: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[Optional[str]] = mapped_column(Text)
    faithfulness_score: Mapped[Optional[float]] = mapped_column(Float)
    relevance_score: Mapped[Optional[float]] = mapped_column(Float)
    groundedness_score: Mapped[Optional[float]] = mapped_column(Float)
    citation_accuracy_score: Mapped[Optional[float]] = mapped_column(Float)
    tool_accuracy_score: Mapped[Optional[float]] = mapped_column(Float)
    hallucination_score: Mapped[Optional[float]] = mapped_column(Float)  # fraction of hallucinated claims: lower is better
    safety_score: Mapped[Optional[float]] = mapped_column(Float)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float)
    input_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    output_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    estimated_cost: Mapped[Optional[float]] = mapped_column(Float)
    error: Mapped[Optional[str]] = mapped_column(Text)

    trace: Mapped[AgentTrace] = relationship(back_populates="evaluations")
    metrics: Mapped[list["EvaluationMetric"]] = relationship(
        back_populates="evaluation", cascade="all, delete-orphan", order_by="EvaluationMetric.metric"
    )

    __table_args__ = (Index("ix_eval_runs_status_created", "status", "created_at"),)


class EvaluationMetric(Base):
    __tablename__ = "evaluation_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evaluation_id: Mapped[str] = mapped_column(ForeignKey("evaluation_runs.evaluation_id", ondelete="CASCADE"), index=True)
    metric: Mapped[str] = mapped_column(String(40), index=True)
    status: Mapped[str] = mapped_column(String(20))  # ok | not_available | error
    score: Mapped[Optional[float]] = mapped_column(Float)
    passed: Mapped[Optional[bool]] = mapped_column(Boolean)
    reason: Mapped[Optional[str]] = mapped_column(Text)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    details: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)

    evaluation: Mapped[EvaluationRun] = relationship(back_populates="metrics")
