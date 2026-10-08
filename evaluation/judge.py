"""Replaceable LLM-judge provider. Evaluators depend only on the `Judge` protocol."""
from __future__ import annotations

import asyncio
import json
import re
from typing import Optional, Protocol, Type, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from .config import get_settings

T = TypeVar("T", bound=BaseModel)


class JudgeError(RuntimeError):
    """The judge was unreachable or returned output that does not match the requested schema."""


class Judge(Protocol):
    async def complete(self, system: str, user: str, schema: Type[T]) -> T: ...


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    candidates = [fence.group(1)] if fence else []
    candidates.append(text)
    brace = re.search(r"\{.*\}", text, re.S)
    if brace:
        candidates.append(brace.group(0))
    for cand in candidates:
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    raise JudgeError("Judge did not return a JSON object")


class FireJudge:
    """Judge backed by the project's own Fire gateway (the only outbound LLM path in this app).

    Model comes from EVAL_JUDGE_MODEL; unset means Fire's default model. Temperature is 0 for repeatability.
    Output is validated against the evaluator's Pydantic schema, with one repair retry.
    """

    def __init__(self, model: Optional[str] = None, max_tokens: int = 3500):
        from app.fire_llm import FireChatModel

        settings = get_settings()
        self._llm = FireChatModel(
            species_name=model or settings.judge_model,
            temperature=0.0,
            max_tokens=max_tokens,
            request_timeout=settings.judge_timeout,
        )

    async def complete(self, system: str, user: str, schema: Type[T]) -> T:
        from app.fire_llm import FireAPIError

        schema_hint = json.dumps(schema.model_json_schema())
        sys_msg = (
            f"{system}\n\nRespond with ONLY one JSON object (no prose, no markdown fences) "
            f"matching this JSON Schema:\n{schema_hint}"
        )
        messages = [SystemMessage(content=sys_msg), HumanMessage(content=user)]
        last_err: Optional[Exception] = None
        for attempt in range(2):
            resp = None
            for transport_attempt in range(2):  # one retry for transient network/API failures
                try:
                    resp = await self._llm.ainvoke(messages)
                    break
                except FireAPIError as exc:
                    if transport_attempt == 1:
                        raise JudgeError(f"Judge call failed: {exc}") from exc
                    await asyncio.sleep(2.0)
            try:
                return schema.model_validate(_extract_json_object(str(resp.content)))
            except (JudgeError, ValidationError) as exc:
                last_err = exc
                messages = messages + [
                    HumanMessage(content="Your previous reply was not valid for the schema. Reply with ONLY the corrected JSON object.")
                ]
        raise JudgeError(f"Judge output failed schema validation: {last_err}")
