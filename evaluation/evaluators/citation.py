from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from ..schemas import EvalInput, EvaluatorResult
from .base import EvalContext, Evaluator


class _CitationVerdict(BaseModel):
    index: int
    supports: bool
    note: str = ""


class _Citations(BaseModel):
    verdicts: List[_CitationVerdict]
    confidence: float = Field(ge=0.0, le=1.0)


SYSTEM = (
    "For each numbered CITATION (a claim plus the source it cites) decide whether the cited SOURCE TEXT actually "
    "supports the claim. supports=false if the source is unrelated, only loosely related, or says something "
    "different. Use only the source text provided. Return one verdict per citation using its index."
)


class CitationEvaluator(Evaluator):
    """Citation accuracy = share of citations whose cited source text really supports the cited claim.
    Also fails citations that point to a source id that was never provided. Not available when the answer
    carries no citations (the current workflow does not produce any)."""

    metric = "citation_accuracy"

    async def evaluate(self, data: EvalInput, ctx: EvalContext) -> EvaluatorResult:
        if not data.citations:
            return self.not_available("The response contains no citations.")
        by_id = {c.id: c for c in data.context}
        dangling = [c for c in data.citations if c.source_id not in by_id]
        checkable = [c for c in data.citations if c.source_id in by_id]
        correct = 0
        confidence = 1.0
        verdicts: list = []
        if checkable:
            blocks = "\n\n".join(
                f"CITATION {i}: claim={c.claim!r}\nSOURCE TEXT [{c.source_id}]:\n{by_id[c.source_id].text}"
                for i, c in enumerate(checkable)
            )
            out = await ctx.judge.complete(SYSTEM, blocks, _Citations)
            supports = {v.index: v.supports for v in out.verdicts}
            correct = sum(1 for i in range(len(checkable)) if supports.get(i, False))
            confidence = out.confidence
            verdicts = [v.model_dump() for v in out.verdicts]
        total = len(data.citations)
        return self.scored(
            correct / total,
            f"{correct} of {total} citations are supported by their cited source"
            + (f"; {len(dangling)} cite a source that was not retrieved." if dangling else "."),
            confidence,
            dangling=[c.source_id for c in dangling],
            verdicts=verdicts,
        )
