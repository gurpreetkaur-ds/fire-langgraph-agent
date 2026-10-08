from __future__ import annotations

from ..schemas import EvalInput, EvaluatorResult
from .base import EvalContext, Evaluator
from .claims import get_claim_analysis


class GroundednessEvaluator(Evaluator):
    """Share of the answer's claims that the context positively supports (supported/total)."""

    metric = "groundedness"

    async def evaluate(self, data: EvalInput, ctx: EvalContext) -> EvaluatorResult:
        if not data.answer:
            return self.not_available("No answer was produced.")
        if not data.context:
            return self.not_available("No context was recorded for this request.")
        a = await get_claim_analysis(data, ctx)
        if a.is_refusal:
            return self.not_available("The response is a refusal; grounding against context does not apply (see safety).")
        if a.total == 0:
            return self.not_available("No verifiable claims could be extracted from the answer.")
        supported = a.count("supported")
        unsupported = [v for v in a.verdicts if v.verdict != "supported"]
        return self.scored(
            supported / a.total,
            f"{supported} of {a.total} claims are supported by the context.",
            a.confidence,
            claims_total=a.total,
            not_supported=[v.model_dump() for v in unsupported],
        )
