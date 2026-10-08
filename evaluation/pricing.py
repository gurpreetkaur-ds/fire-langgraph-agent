"""Cost estimation from a user-supplied price table (EVAL_PRICING_JSON). Unknown model => None, never a guess."""
from __future__ import annotations

from typing import Optional

from .config import get_settings


def estimate_cost(model: Optional[str], input_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[float]:
    if not model or input_tokens is None or output_tokens is None:
        return None
    price = get_settings().pricing.get(model)
    if price is None:
        return None
    return (input_tokens * price["input_per_mtok"] + output_tokens * price["output_per_mtok"]) / 1_000_000
