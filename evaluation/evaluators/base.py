from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from ..config import LOWER_IS_BETTER, get_settings
from ..judge import Judge
from ..schemas import EvalInput, EvaluatorResult


class EvalContext:
    """Per-evaluation shared state: the judge plus memoised intermediate results (e.g. claim analysis),
    so evaluators that need the same expensive step share one computation."""

    def __init__(self, judge: Judge):
        self.judge = judge
        self._memo: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def memo(self, key: str, factory: Callable[[], Any]) -> Any:
        async with self._lock:
            if key not in self._memo:
                self._memo[key] = await factory()
            return self._memo[key]


class Evaluator(ABC):
    """One metric. Subclasses implement `evaluate` and may raise `JudgeError`; the engine converts that
    into an `error` result rather than inventing a score."""

    metric: str

    @abstractmethod
    async def evaluate(self, data: EvalInput, ctx: EvalContext) -> EvaluatorResult: ...

    # helpers -----------------------------------------------------------------------------------------
    def not_available(self, reason: str) -> EvaluatorResult:
        return EvaluatorResult(metric=self.metric, status="not_available", reason=reason)

    def scored(self, score: float, reason: str, confidence: Optional[float] = 1.0, **details: Any) -> EvaluatorResult:
        score = max(0.0, min(1.0, score))
        threshold = get_settings().thresholds[self.metric]
        passed = score <= threshold if self.metric in LOWER_IS_BETTER else score >= threshold
        return EvaluatorResult(
            metric=self.metric, status="ok", score=round(score, 4), passed=passed,
            reason=reason, confidence=confidence, details=details,
        )
