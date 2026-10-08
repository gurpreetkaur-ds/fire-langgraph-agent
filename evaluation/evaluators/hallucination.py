from __future__ import annotations

from ..schemas import EvalInput, EvaluatorResult
from .base import EvalContext, Evaluator
from .claims import get_claim_analysis


class HallucinationEvaluator(Evaluator):
    """Hallucination rate (LOWER is better): fraction of claims that are contradicted by the context, or that
    are unsupported *specific facts* (names, numbers, dates, ...). Unsupported general-knowledge statements are
    not counted as hallucinations (they lower groundedness instead).
    """

    metric = "hallucination"

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
        bad = [
            v for v in a.verdicts
            if v.verdict == "contradicted" or (v.verdict == "unsupported" and v.kind == "specific_fact")
        ]
        rate = len(bad) / a.total
        return self.scored(
            rate,
            f"{len(bad)} of {a.total} claims are contradicted or are unsupported specific facts.",
            a.confidence,
            claims_total=a.total,
            hallucinated=[v.model_dump() for v in bad],
        )
