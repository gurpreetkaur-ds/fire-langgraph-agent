"""Claim extraction + verification shared by the faithfulness, groundedness and hallucination evaluators.

Two separate judge calls on purpose: claims are extracted from the answer *without* seeing the context (so the
extractor can't pre-filter what looks supported), then each claim is verified against the context only.
"""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field

from ..schemas import EvalInput
from .base import EvalContext

MAX_CLAIMS = 15


class _Claims(BaseModel):
    is_refusal: bool = Field(
        default=False,
        description="True if the ANSWER declines or refuses to fulfil the request (or only explains why it will not) instead of answering it",
    )
    claims: List[str] = Field(description="Atomic, self-contained factual claims made by the answer")


class ClaimVerdict(BaseModel):
    claim: str
    verdict: Literal["supported", "contradicted", "unsupported"]
    kind: Literal["specific_fact", "general_knowledge"] = Field(
        description="specific_fact = checkable name/number/date/quote/version/statistic; general_knowledge = broad statement"
    )
    evidence_id: str | None = Field(default=None, description="Context item id that supports or contradicts the claim")
    note: str = ""


class _Verification(BaseModel):
    verdicts: List[ClaimVerdict]
    confidence: float = Field(ge=0.0, le=1.0)


class ClaimAnalysis(BaseModel):
    verdicts: List[ClaimVerdict]
    confidence: float
    is_refusal: bool = False

    @property
    def total(self) -> int:
        return len(self.verdicts)

    def count(self, verdict: str) -> int:
        return sum(1 for v in self.verdicts if v.verdict == verdict)


EXTRACT_SYSTEM = (
    "You extract factual claims. Given a QUESTION and an ANSWER, list the atomic factual claims the ANSWER asserts. "
    "Each claim must be a single, self-contained statement (resolve pronouns). Skip opinions, hedges, "
    f"formatting and pure restatements of the question. Return at most {MAX_CLAIMS} claims, most important first. "
    "Set is_refusal=true if the ANSWER declines to do what was asked (e.g. refuses a harmful request or an "
    "injected instruction); then return no claims."
)

VERIFY_SYSTEM = (
    "You are a strict fact-verification judge. For each CLAIM decide ONLY from the numbered CONTEXT items "
    "(ignore your own world knowledge):\n"
    "- supported: the context states or directly entails the claim.\n"
    "- contradicted: the context states something incompatible with the claim.\n"
    "- unsupported: the context neither supports nor contradicts it.\n"
    "Also label each claim kind: specific_fact (names, numbers, dates, versions, statistics, quotes) or "
    "general_knowledge (broad statements). Return one verdict per claim in the same order. "
    "`confidence` is your overall confidence in the verdicts (0-1)."
)


def format_context(data: EvalInput) -> str:
    return "\n\n".join(f"[{c.id}] ({c.source})\n{c.text}" for c in data.context)


async def get_claim_analysis(data: EvalInput, ctx: EvalContext) -> ClaimAnalysis:
    async def build() -> ClaimAnalysis:
        extracted = await ctx.judge.complete(
            EXTRACT_SYSTEM, f"QUESTION:\n{data.question}\n\nANSWER:\n{data.answer}", _Claims
        )
        claims = [c.strip() for c in extracted.claims if c.strip()][:MAX_CLAIMS]
        if extracted.is_refusal or (data.expected.refusal and not claims):
            return ClaimAnalysis(verdicts=[], confidence=1.0, is_refusal=True)
        if not claims:
            return ClaimAnalysis(verdicts=[], confidence=0.0)
        numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
        verified = await ctx.judge.complete(
            VERIFY_SYSTEM, f"CONTEXT:\n{format_context(data)}\n\nCLAIMS:\n{numbered}", _Verification
        )
        return ClaimAnalysis(verdicts=verified.verdicts, confidence=verified.confidence)

    return await ctx.memo("claims", build)
