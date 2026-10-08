from __future__ import annotations

from ..schemas import EvalInput, EvaluatorResult
from .base import EvalContext, Evaluator
from .claims import get_claim_analysis


class FaithfulnessEvaluator(Evaluator):
    """Share of the answer's claims that do NOT contradict the context it was written from.

    Faithfulness = 1 - contradicted/total. (Absence of support is handled by groundedness/hallucination.)
    Requires retrieved/provided context; without it the metric is not_available.
    """

    metric = "faithfulness"

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
        contradicted = [v for v in a.verdicts if v.verdict == "contradicted"]
        score = 1 - len(contradicted) / a.total
        reason = f"{len(contradicted)} of {a.total} claims contradict the context."
        if contradicted:
            reason += " e.g. " + contradicted[0].claim
        return self.scored(score, reason, a.confidence, claims_total=a.total, contradicted=[v.model_dump() for v in contradicted])
