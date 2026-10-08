import json

import pytest
from fastapi.testclient import TestClient

from evaluation.engine import EvaluationEngine
from evaluation.worker import EvaluationWorker
from tests.conftest import GOOD_JUDGE, TEST_TOKEN, ScriptedJudge
from tests.helpers import make_trace

AUTH = {"Authorization": f"Bearer {TEST_TOKEN}"}
ROUTES = ["/api/eval/summary", "/api/eval/timeseries", "/api/eval/agents", "/api/eval/agents/synthesizer", "/api/eval/traces",
          "/api/eval/traces/req_x", "/api/eval/filters", "/api/eval/suites", "/api/eval/auth-check"]


@pytest.fixture
def client(db):
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("path", ROUTES)
def test_every_eval_route_requires_the_token(client, path):
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get(path, headers={"Authorization": f"Basic {TEST_TOKEN}"}).status_code == 401


def test_api_is_closed_when_no_token_is_configured(client, monkeypatch):
    monkeypatch.delenv("EVAL_ADMIN_TOKEN")
    r = client.get("/api/eval/summary", headers=AUTH)
    assert r.status_code == 503


def test_eval_api_is_read_only(client):
    for method in ("post", "put", "patch", "delete"):
        assert getattr(client, method)("/api/eval/summary", headers=AUTH).status_code == 405
    assert client.post("/api/eval/traces/req_x", headers=AUTH, json={"faithfulness": 1.0}).status_code == 405


async def test_dashboard_endpoints_return_real_data(client):
    tid = make_trace()
    await EvaluationWorker(EvaluationEngine(judge=ScriptedJudge(GOOD_JUDGE))).drain()
    s = client.get("/api/eval/summary", headers=AUTH).json()
    assert s["evaluations_total"] == 1 and s["metrics"]["faithfulness"]["avg"] == 1.0 and s["thresholds"]["safety"] == 0.95
    assert client.get("/api/eval/timeseries", headers=AUTH).json()[0]["evaluations"] == 1
    assert [a["agent"] for a in client.get("/api/eval/agents", headers=AUTH).json()] == ["answer_agent", "decomposer", "synthesizer"]
    assert client.get("/api/eval/agents/synthesizer", headers=AUTH).json()["executions"] == 1
    assert client.get("/api/eval/agents/ghost", headers=AUTH).status_code == 404
    listing = client.get("/api/eval/traces", headers=AUTH, params={"agent": "synthesizer", "status": "completed"}).json()
    assert listing["total"] == 1 and listing["items"][0]["trace_id"] == tid
    detail = client.get(f"/api/eval/traces/{tid}", headers=AUTH).json()
    assert detail["evaluations"][0]["metrics"] and len(detail["steps"]) == 3
    assert client.get("/api/eval/traces/nope", headers=AUTH).status_code == 404
    assert "no-store" in client.get("/api/eval/summary", headers=AUTH).headers["cache-control"]


def test_filter_validation(client):
    assert client.get("/api/eval/summary", headers=AUTH, params={"status": "bogus"}).status_code == 422
    assert client.get("/api/eval/summary", headers=AUTH, params={"failed_metric": "bogus"}).status_code == 422
    assert client.get("/api/eval/summary", headers=AUTH, params={"date_from": "not-a-date"}).status_code == 422


def test_responses_never_contain_secrets(client, monkeypatch):
    secret = "sk-abcdefghijklmnop1234"
    monkeypatch.setenv("MY_SERVICE_API_KEY", "env-secret-value-98765")
    tid = make_trace(question=f"use {secret} and env-secret-value-98765", answer=f"Bearer abcdefghijkl1234 {secret}")
    body = client.get(f"/api/eval/traces/{tid}", headers=AUTH).text + client.get("/api/eval/traces", headers=AUTH).text
    for leaked in (secret, "env-secret-value-98765", "abcdefghijkl1234"):
        assert leaked not in body


def test_regression_endpoint(client):
    assert client.get("/api/eval/regression", headers=AUTH, params={"baseline": "a", "candidate": "b"}).status_code == 404


# ---- full flow: HTTP request -> LangGraph -> agents -> final answer -> trace -> evaluation -> database -> API ----

async def test_end_to_end_request_to_evaluated_trace(client, fake_llms):
    from evaluation.db import session_scope
    from evaluation.models import EvaluationRun

    r = client.post("/api/analyze/stream", json={"question": "Compare TCP and UDP", "session_id": "sess-1"})
    events = [json.loads(line[6:]) for line in r.text.split("\n\n") if line.startswith("data: ")]
    assert [e["node"] for e in events] == ["decompose", "answer", "synthesize", "done"]  # existing SSE protocol unchanged
    trace_id = events[-1]["trace_id"]
    assert events[2]["data"]["final_answer"] == "TCP is reliable; UDP is fast."

    # trace + pending evaluation are persisted after the answer was already streamed
    import time
    for _ in range(100):
        with session_scope() as s:
            if s.query(EvaluationRun).count():
                break
        time.sleep(0.05)
    with session_scope() as s:
        assert s.query(EvaluationRun).one().status == "pending"

    judge = ScriptedJudge(GOOD_JUDGE)
    assert await EvaluationWorker(EvaluationEngine(judge=judge)).drain() == 1

    d = client.get(f"/api/eval/traces/{trace_id}", headers=AUTH).json()
    assert d["question"] == "Compare TCP and UDP" and d["session_id"] == "sess-1" and d["success"]
    assert d["answer"] == "TCP is reliable; UDP is fast." and d["agent_name"] == "synthesizer"
    assert [s["agent_name"] for s in d["steps"]] == ["decomposer", "answer_agent", "answer_agent", "synthesizer"]
    assert d["total_tokens"] == 4 * 150 and d["estimated_cost"] == pytest.approx(4 * (100 + 100) / 1e6) and d["prompt_version"]
    assert [c["id"] for c in d["context"]] == ["note_1", "note_2"]
    ev = d["evaluations"][0]
    assert ev["status"] == "completed" and ev["passed"] is True
    assert {m["metric"]: m["status"] for m in ev["metrics"]}["citation_accuracy"] == "not_available"


def test_empty_question_still_rejected_without_trace(client):
    from evaluation.db import session_scope
    from evaluation.models import AgentTrace

    r = client.post("/api/analyze/stream", json={"question": "   "})
    assert "Question cannot be empty" in r.text
    with session_scope() as s:
        assert s.query(AgentTrace).count() == 0


def test_llm_failure_is_traced_as_failed_request(client, monkeypatch):
    import time

    import app.graph as g
    from app.fire_llm import FireAPIError
    from evaluation.db import session_scope
    from evaluation.models import AgentTrace

    class Down:
        species_name = "test-model"

        async def ainvoke(self, messages):
            raise FireAPIError("Fire API returned HTTP 503")

    monkeypatch.setattr(g, "decomposer_llm", Down())
    r = client.post("/api/analyze/stream", json={"question": "Why?"})
    assert '"node": "error"' in r.text
    for _ in range(100):
        with session_scope() as s:
            t = s.query(AgentTrace).first()
            if t:
                break
        time.sleep(0.05)
    assert t.success is False and "503" in t.error and t.answer is None
