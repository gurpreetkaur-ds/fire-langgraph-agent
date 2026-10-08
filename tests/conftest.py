from __future__ import annotations

import os
from typing import Any, Callable, Dict, List

import pytest
from langchain_core.messages import AIMessage

TEST_TOKEN = "test-admin-token-1234567890"

os.environ.setdefault("FIRE_API_KEY", "fire_sk_test_key_not_real_123456")


@pytest.fixture(autouse=True)
def eval_env(monkeypatch, tmp_path):
    monkeypatch.setenv("EVAL_DATABASE_URL", f"sqlite:///{tmp_path / 'eval.db'}")
    monkeypatch.setenv("EVAL_ADMIN_TOKEN", TEST_TOKEN)
    monkeypatch.setenv("EVAL_WORKER_MODE", "external")  # tests drive the worker explicitly
    monkeypatch.setenv("EVAL_PRICING_JSON", '{"test-model": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}}')
    monkeypatch.setenv("EVAL_ENABLED", "true")
    monkeypatch.setenv("EVAL_AUTO_MIGRATE", "true")
    yield


@pytest.fixture
def db(eval_env):
    """A fresh SQLite database created through the real Alembic migrations."""
    from evaluation.config import get_settings
    from evaluation.db import init_engine, run_migrations

    run_migrations(get_settings().database_url)
    engine = init_engine()
    yield engine
    engine.dispose()


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
