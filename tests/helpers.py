from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from evaluation import store
from evaluation.schemas import Expected
from evaluation.tracing import TraceCollector


def make_trace(question="What is TCP?", answer="TCP is reliable.", error=None, model="test-model", tokens=(100, 50), agents=("decomposer", "answer_agent", "synthesizer"),
               expected: Optional[Expected] = None, suite=None, case=None, claim=False, created_at=None) -> str:
    c = TraceCollector(question=question, prompt_version="pv1")
    if created_at:
        c.created_at = created_at
    from evaluation.pricing import estimate_cost

    for a in agents:
        c.record_step(
            agent_name=a, node=a, model_name=model, call_id=f"call_{a}", started_at=datetime.now(timezone.utc), latency_ms=100.0,
            input_tokens=tokens[0], output_tokens=tokens[1], estimated_cost=estimate_cost(model, *tokens), success=True, error=None,
            input_preview="in", output_preview="out",
        )
    ctx = [{"id": "note_1", "source": "research_agent", "text": "Q: a\nA: TCP is reliable."}]
    store.persist_trace(c, answer=answer, error=error, context=ctx, expected=expected, suite_run_id=suite, case_id=case, claim=claim)
    return c.trace_id
