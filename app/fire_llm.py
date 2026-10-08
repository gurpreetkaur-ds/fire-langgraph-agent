"""Custom LangChain-compatible chat model that routes every call through Fire's
POST /v1/chat endpoint. This is the ONLY place in the project that makes an
outbound LLM call -- no other provider (OpenAI, Anthropic, Gemini, OpenRouter, ...)
is ever contacted directly.
"""
from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

logger = logging.getLogger("fire_agent.fire_llm")


class FireAPIError(RuntimeError):
    """Raised for any Fire API failure: missing key, network error, or non-200 response.

    Messages here are built from status codes and response bodies only -- never
    from the key itself -- so they are always safe to log or show in the UI.
    """


def _build_messages(messages: List[BaseMessage]) -> tuple[list[dict], Optional[str]]:
    system_prompt: Optional[str] = None
    fire_messages: list[dict] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            system_prompt = str(m.content)
        elif isinstance(m, AIMessage):
            fire_messages.append({"role": "assistant", "content": str(m.content)})
        else:
            fire_messages.append({"role": "user", "content": str(m.content)})
    return fire_messages, system_prompt


class FireChatModel(BaseChatModel):
    """A BaseChatModel implementation backed by Fire's /v1/chat endpoint.

    species_name selects the Fire model (see GET /v1/capabilities for the catalog);
    omit it to use Fire's default (claude-sonnet-4-5 at the time this was built).
    """

    species_name: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 1024
    request_timeout: float = 60.0
    fire_base_url: str = Field(
        default_factory=lambda: os.environ.get("FIRE_BASE_URL", "https://fire.prosaga.net/v1")
    )

    @property
    def _llm_type(self) -> str:
        return "fire-chat"

    @staticmethod
    def _api_key() -> str:
        key = os.environ.get("FIRE_API_KEY")
        if not key:
            raise FireAPIError(
                "FIRE_API_KEY is not set. Add it to a local .env file (see .env.example)."
            )
        return key

    def _payload(self, messages: List[BaseMessage]) -> dict:
        fire_messages, system_prompt = _build_messages(messages)
        payload: dict[str, Any] = {
            "messages": fire_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if system_prompt:
            payload["system_prompt"] = system_prompt
        if self.species_name:
            payload["species_name"] = self.species_name
        return payload

    def _parse_response(self, resp: httpx.Response) -> tuple[str, dict]:
        if resp.status_code != 200:
            logger.error("fire_chat_http_error status=%d", resp.status_code)
            raise FireAPIError(f"Fire API returned HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        content = data.get("content")
        if content is None:
            raise FireAPIError("Fire API response was missing the 'content' field.")
        logger.info(
            "fire_chat_ok species=%s call_id=%s usage=%s",
            self.species_name or "default",
            data.get("call_id"),
            data.get("usage"),
        )
        # Surface usage/call_id on the message so callers (e.g. evaluation tracing) can record them.
        return content, {"call_id": data.get("call_id"), "usage": data.get("usage"), "species_name": self.species_name}

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        headers = {"Authorization": f"Bearer {self._api_key()}", "Content-Type": "application/json"}
        payload = self._payload(messages)
        logger.info("fire_chat_request species=%s n_messages=%d", self.species_name or "default", len(payload["messages"]))
        try:
            with httpx.Client(timeout=self.request_timeout) as client:
                resp = client.post(f"{self.fire_base_url}/chat", json=payload, headers=headers)
        except httpx.HTTPError as exc:
            logger.error("fire_chat_network_error error=%s", type(exc).__name__)
            raise FireAPIError(f"Could not reach Fire API: {type(exc).__name__}") from exc
        content, fire_meta = self._parse_response(resp)
        message = AIMessage(content=content, response_metadata={"fire": fire_meta})
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        headers = {"Authorization": f"Bearer {self._api_key()}", "Content-Type": "application/json"}
        payload = self._payload(messages)
        logger.info("fire_chat_request species=%s n_messages=%d", self.species_name or "default", len(payload["messages"]))
        try:
            async with httpx.AsyncClient(timeout=self.request_timeout) as client:
                resp = await client.post(f"{self.fire_base_url}/chat", json=payload, headers=headers)
        except httpx.HTTPError as exc:
            logger.error("fire_chat_network_error error=%s", type(exc).__name__)
            raise FireAPIError(f"Could not reach Fire API: {type(exc).__name__}") from exc
        content, fire_meta = self._parse_response(resp)
        message = AIMessage(content=content, response_metadata={"fire": fire_meta})
        return ChatResult(generations=[ChatGeneration(message=message)])
