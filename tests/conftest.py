from __future__ import annotations

import os
from typing import Any, Callable, Dict, List

import pytest
from langchain_core.messages import AIMessage

TEST_TOKEN = "test-admin-token-1234567890"

os.environ.setdefault("FIRE_API_KEY", "fire_sk_test_key_not_real_123456")
# Import the app now: it loads .env (python-dotenv), and that must happen before the per-test fixture below
# removes production-only settings, not in the middle of a test.
import app.main  # noqa: E402,F401


# Settings that a production .env (loaded by app.main via python-dotenv) must never leak into tests.
_PROD_ONLY = ("AGENT_ACCESS_TOKEN", "DATABASE_URL", "UPLOAD_DIR", "COOKIE_SECURE", "ALLOW_SIGNUP", "EMBEDDING_MODEL",
              "RAG_SIMILARITY_THRESHOLD", "RAG_TOP_K", "RAG_CHUNK_SIZE", "RAG_CHUNK_OVERLAP")


@pytest.fixture(autouse=True)
def eval_env(monkeypatch, tmp_path):
    for name in _PROD_ONLY:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'eval.db'}")
    monkeypatch.setenv("EVAL_DATABASE_URL", f"sqlite:///{tmp_path / 'eval.db'}")
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("INGEST_WORKER", "off")  # tests drive indexing explicitly
    monkeypatch.setenv("EVAL_ADMIN_TOKEN", TEST_TOKEN)
    monkeypatch.setenv("EVAL_WORKER_MODE", "external")  # tests drive the worker explicitly
    monkeypatch.setenv("EVAL_PRICING_JSON", '{"test-model": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}}')
    monkeypatch.setenv("EVAL_ENABLED", "true")
    monkeypatch.setenv("EVAL_AUTO_MIGRATE", "true")
    from app.ratelimit import limiter
    from app.rag.embeddings import set_embedder
    from app.settings import reload_app_settings

    reload_app_settings()
    limiter.reset()
    set_embedder(HashEmbedder())
    yield
    set_embedder(None)
    reload_app_settings()


@pytest.fixture
def db(eval_env):
    """A fresh SQLite database created through the real Alembic migrations."""
    from evaluation.config import get_settings
    from evaluation.db import init_engine, run_migrations

    run_migrations(get_settings().database_url)
    engine = init_engine()
    yield engine
    engine.dispose()


class HashEmbedder:
    """Deterministic stand-in for Fire /v1/embed: hashed bag of words (+ bigrams), L2-normalised.

    Texts sharing words get high cosine similarity, unrelated texts ~0, so retrieval behaves realistically.
    """

    model = "test-hash-embedding"

    def __init__(self, dims: int = 1536):
        self.dims = dims
        self.calls: List[List[str]] = []

    def vector(self, text: str):
        import hashlib
        import re

        import numpy as np

        words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2]
        feats = words + [a + "_" + b for a, b in zip(words, words[1:])]
        v = np.zeros(self.dims, dtype=np.float32)
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % self.dims] += 1.0 if (h >> 64) & 1 else -1.0
        n = float(np.linalg.norm(v))
        return v / n if n else v

    async def embed_many(self, texts):
        from app.rag.embeddings import EmbeddingResult

        texts = list(texts)
        self.calls.append(texts)
        return EmbeddingResult(vectors=[self.vector(t) for t in texts], input_tokens=sum(len(t.split()) for t in texts),
                               model=self.model, latency_ms=1.0)


class ScriptedJudge:
    """Judge returning canned structured answers keyed by schema class name (records the calls)."""

    def __init__(self, responses: Dict[str, Any]):
        self.responses = responses
        self.calls: List[str] = []

    async def complete(self, system: str, user: str, schema):
        self.calls.append(schema.__name__)
        r = self.responses[schema.__name__]
        r = r(user) if callable(r) else r
        return schema.model_validate(r)


class FakeLLM:
    def __init__(self, species_name: str, reply: Callable[[list], str]):
        self.species_name = species_name
        self._reply = reply

    async def ainvoke(self, messages):
        return AIMessage(
            content=self._reply(messages),
            response_metadata={"fire": {"call_id": "call_" + self.species_name, "usage": {"input": 100, "output": 50}}},
        )


@pytest.fixture
def fake_llms(monkeypatch):
    import app.graph as g

    monkeypatch.setattr(g, "decomposer_llm", FakeLLM("test-model", lambda m: '["What is TCP?", "What is UDP?"]'))
    monkeypatch.setattr(g, "answer_llm", FakeLLM("test-model", lambda m: "It is a transport protocol."))
    monkeypatch.setattr(g, "synth_llm", FakeLLM("test-model", lambda m: "TCP is reliable; UDP is fast."))


GOOD_JUDGE = {
    "_Claims": {"claims": ["TCP is reliable", "UDP is fast"]},
    "_Verification": {
        "verdicts": [
            {"claim": "TCP is reliable", "verdict": "supported", "kind": "general_knowledge", "evidence_id": "note_1"},
            {"claim": "UDP is fast", "verdict": "supported", "kind": "general_knowledge", "evidence_id": "note_2"},
        ],
        "confidence": 0.9,
    },
    "_Relevance": {"aspects": [{"aspect": "TCP", "coverage": "full"}, {"aspect": "UDP", "coverage": "full"}], "off_topic": False, "reason": "covers both", "confidence": 0.9},
    "_Safety": {"violations": [], "refused_or_declined": False, "reason": "benign", "confidence": 0.95},
    "_Citations": {"verdicts": [], "confidence": 1.0},
}


@pytest.fixture
def good_judge():
    return ScriptedJudge(dict(GOOD_JUDGE))
