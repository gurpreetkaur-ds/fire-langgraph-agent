"""Pydantic models shared by evaluators, the engine and the API."""
from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


class ContextItem(BaseModel):
    id: str
    source: str
    text: str


class Citation(BaseModel):
    claim: str
    source_id: str


class ToolCallRecord(BaseModel):
    tool_name: str
    tool_input: Optional[dict[str, Any]] = None
    tool_output: Optional[Any] = None
    success: bool = True
    error: Optional[str] = None


class Expected(BaseModel):
    """Optional expectations from an evaluation dataset case."""

    answer: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    refusal: bool = False  # the correct behaviour is to decline / not comply


class EvalInput(BaseModel):
    trace_id: str
    question: str
    answer: Optional[str]
    success: bool
    context: List[ContextItem] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    expected: Expected = Field(default_factory=Expected)


class EvaluatorResult(BaseModel):
    """Structured outcome of one evaluator.

    status "not_available" means the data needed to compute the metric was missing; score is then None.
    status "error" means the evaluator itself failed (e.g. judge unreachable); score is then None.
    """

    metric: str
    status: Literal["ok", "not_available", "error"]
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    passed: Optional[bool] = None
    reason: str = ""
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    details: dict[str, Any] = Field(default_factory=dict)
