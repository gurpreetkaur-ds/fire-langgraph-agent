from __future__ import annotations

from ..schemas import EvalInput, EvaluatorResult
from .base import EvalContext, Evaluator


class ToolAccuracyEvaluator(Evaluator):
    """Deterministic tool-use check (no LLM).

    With expected tools: F1 of the set of tools called vs expected (calling nothing when tools were expected
    scores 0; calling tools when none were expected scores 0). Without expectations: the share of tool calls
    that executed successfully. Not available when no tool was expected and none was called.
    """

    metric = "tool_accuracy"

    async def evaluate(self, data: EvalInput, ctx: EvalContext) -> EvaluatorResult:
        called = [t.tool_name for t in data.tool_calls]
        expected = list(data.expected.tools)
        if not called and not expected:
            return self.not_available("No tools were expected or called in this request.")
        if expected:
            exp, got = set(expected), set(called)
            tp = len(exp & got)
            precision = tp / len(got) if got else 0.0
            recall = tp / len(exp)
            f1 = 0.0 if tp == 0 else 2 * precision * recall / (precision + recall)
            failed = [t.tool_name for t in data.tool_calls if not t.success]
            score = f1 * (1 - len(failed) / len(called)) if called else 0.0
            return self.scored(
                score,
                f"Expected {sorted(exp)}, called {sorted(got)}; precision={precision:.2f} recall={recall:.2f}"
                + (f"; failed calls: {failed}" if failed else ""),
                1.0, expected=sorted(exp), called=called, failed=failed,
            )
        ok = sum(1 for t in data.tool_calls if t.success)
        return self.scored(
            ok / len(data.tool_calls),
            f"{ok} of {len(data.tool_calls)} tool calls succeeded (no expected tools were specified).",
            1.0, called=called,
        )
