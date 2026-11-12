"""Embeddings through Fire's POST /v1/embed (the same provider and key as every chat call).

Implements LangChain's `Embeddings` interface so it plugs into any LangChain component, plus async batch
helpers that also report token usage. Transient failures (network errors, HTTP 429/5xx) are retried with
exponential backoff; each retry is recorded on the active trace so it shows up in Agent Activity.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import httpx
import numpy as np
from langchain_core.embeddings import Embeddings
from tenacity import AsyncRetrying, RetryCallState, retry_if_exception, stop_after_attempt, wait_exponential

from evaluation.tracing import get_current_trace, log_event

from ..fire_llm import FireAPIError
from ..settings import get_app_settings

logger = logging.getLogger("fire_agent.embeddings")
MAX_INPUT_CHARS = 30000


class FireTransientError(FireAPIError):
    """Retryable Fire failure (network, 429, 5xx)."""


@dataclass
class EmbeddingResult:
    vectors: List[np.ndarray]
    input_tokens: int
    model: str
    latency_ms: float


def _normalise(vec: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0 else arr


def _on_retry(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    log_event(logger, "embed_retry", logging.WARNING, attempt=state.attempt_number, error=type(exc).__name__ if exc else None)
    trace = get_current_trace()
    if trace is not None and hasattr(trace, "add_event"):
        trace.add_event("retry", "embeddings", f"Embedding call failed ({type(exc).__name__}); retry {state.attempt_number}")


class FireEmbeddings(Embeddings):
    def __init__(self, model: Optional[str] = None, dimensions: Optional[int] = None, timeout: float = 60.0, attempts: int = 4):
        settings = get_app_settings()
        self.model = model or settings.embedding_model
        self.dimensions = dimensions or settings.embedding_dimensions
        self.batch_size = settings.embedding_batch_size
        self.timeout = timeout
        self.attempts = attempts

    @property
    def base_url(self) -> str:
        return os.environ.get("FIRE_BASE_URL", "https://fire.prosaga.net/v1")

    @staticmethod
    def _key() -> str:
        key = os.environ.get("FIRE_API_KEY")
        if not key:
            raise FireAPIError("FIRE_API_KEY is not set. Add it to a local .env file (see .env.example).")
        return key

    async def _call(self, client: httpx.AsyncClient, texts: List[str]) -> tuple[list, int]:
        payload = {"input": [t[:MAX_INPUT_CHARS] or " " for t in texts], "species_name": self.model, "dimensions": self.dimensions}
        try:
            resp = await client.post(f"{self.base_url}/embed", json=payload, headers={"Authorization": f"Bearer {self._key()}"})
        except httpx.HTTPError as exc:
            raise FireTransientError(f"Could not reach Fire API: {type(exc).__name__}") from exc
        if resp.status_code == 429 or resp.status_code >= 500:
            raise FireTransientError(f"Fire embed returned HTTP {resp.status_code}")
        if resp.status_code != 200:
            raise FireAPIError(f"Fire embed returned HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        rows = sorted(data.get("data") or [], key=lambda d: d.get("index", 0))
        if len(rows) != len(texts):
            raise FireAPIError("Fire embed returned the wrong number of vectors.")
        vectors = [r["embedding"] for r in rows]
        if any(len(v) != self.dimensions for v in vectors):
            raise FireAPIError(f"Fire embed returned vectors that are not {self.dimensions}-dimensional.")
        return vectors, int((data.get("usage") or {}).get("input_tokens") or 0)

    async def embed_many(self, texts: Sequence[str]) -> EmbeddingResult:
        texts = list(texts)
        t0 = time.perf_counter()
        vectors: List[np.ndarray] = []
        tokens = 0
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(self.attempts),
                    wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
                    retry=retry_if_exception(lambda e: isinstance(e, FireTransientError)),
                    before_sleep=_on_retry,
                    reraise=True,
                ):
                    with attempt:
                        raw, used = await self._call(client, batch)
                vectors.extend(_normalise(v) for v in raw)
                tokens += used
        latency = (time.perf_counter() - t0) * 1000
        log_event(logger, "embed_ok", n=len(texts), tokens=tokens, latency_ms=round(latency), model=self.model)
        return EmbeddingResult(vectors=vectors, input_tokens=tokens, model=self.model, latency_ms=latency)

    # ---- LangChain Embeddings interface ----
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return [v.tolist() for v in (await self.embed_many(texts)).vectors]

    async def aembed_query(self, text: str) -> List[float]:
        return (await self.embed_many([text])).vectors[0].tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return asyncio.run(self.aembed_documents(texts))

    def embed_query(self, text: str) -> List[float]:
        return asyncio.run(self.aembed_query(text))


_embedder: Optional[FireEmbeddings] = None


def get_embedder() -> FireEmbeddings:
    """Module-level accessor so tests can swap in a deterministic embedder."""
    global _embedder
    if _embedder is None:
        _embedder = FireEmbeddings()
    return _embedder


def set_embedder(embedder) -> None:
    global _embedder
    _embedder = embedder
