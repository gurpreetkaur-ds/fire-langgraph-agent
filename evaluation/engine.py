"""EvaluationEngine: runs the modular evaluators against one trace and returns structured results."""
from __future__ import annotations

import asyncio
import logging
from typing import Iterable, List, Optional, Sequence, Type

from .evaluators import DEFAULT_EVALUATORS, EvalContext, Evaluator
from .judge import Judge, JudgeError
from .redaction import redact_text
from .schemas import EvalInput, EvaluatorResult

logger = logging.getLogger("fire_agent.eval.engine")


class EvaluationEngine:
    def __init__(self, judge: Optional[Judge] = None, evaluators: Optional[Iterable[Evaluator | Type[Evaluator]]] = None):
        if judge is None:
            from .judge import FireJudge

            judge = FireJudge()
        self.judge = judge
        items = evaluators if evaluators is not None else DEFAULT_EVALUATORS
        self.evaluators: List[Evaluator] = [e() if isinstance(e, type) else e for e in items]

    async def evaluate(self, data: EvalInput) -> List[EvaluatorResult]:
        ctx = EvalContext(self.judge)
        results = await asyncio.gather(*(self._run_one(ev, data, ctx) for ev in self.evaluators))
        return list(results)

    async def _run_one(self, ev: Evaluator, data: EvalInput, ctx: EvalContext) -> EvaluatorResult:
        try:
            return await ev.evaluate(data, ctx)
        except JudgeError as exc:
            logger.warning("evaluator_failed metric=%s trace_id=%s error=%s", ev.metric, data.trace_id, exc)
            return EvaluatorResult(metric=ev.metric, status="error", reason=redact_text(f"Judge failure: {exc}"))
        except Exception as exc:  # an evaluator bug must not lose the other metrics
            logger.exception("evaluator_crashed metric=%s trace_id=%s", ev.metric, data.trace_id)
            return EvaluatorResult(metric=ev.metric, status="error", reason=redact_text(f"{type(exc).__name__}: {exc}"))


def overall_passed(results: Sequence[EvaluatorResult], trace_success: bool) -> Optional[bool]:
    """False if the request failed or any computed metric failed. None (unknown) if nothing could be computed or
    any evaluator errored without a computed failure -- an evaluator error must never read as a pass."""
    if not trace_success:
        return False
    scored = [r for r in results if r.status == "ok"]
    if any(not r.passed for r in scored):
        return False
    if not scored or any(r.status == "error" for r in results):
        return None
    return True
