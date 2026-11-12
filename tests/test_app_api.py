"""HTTP API: accounts, CSRF, conversations, streaming chat, documents, activity, isolation between users."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import app.graph as g
from tests.rag_helpers import FakeLLM, FakeStreamLLM, make_pdf
from tests.test_rag_graph import analyzer
from tests.test_rag_retrieval import HANDBOOK

H = {"X-Requested-With": "fetch"}
PASSWORD = "correct horse battery"


@pytest.fixture
def client(db):
    from app.main import app

    with TestClient(app) as c:
        yield c


def signup(c, username="alice", password=PASSWORD, **kw):
    return c.post("/api/auth/signup", json={"username": username, "password": password, **kw}, headers=H)


@pytest.fixture
def alice(client):
    assert signup(client).status_code == 200
    return client


@pytest.fixture
def bob(db):
    from app.main import app

    with TestClient(app) as c:
        assert signup(c, "bob").status_code == 200
        yield c


@pytest.fixture
def llms(monkeypatch):
    monkeypatch.setattr(g, "analyzer_llm", analyzer("knowledge_base", "How many vacation days do employees get per year?"))
    monkeypatch.setattr(g, "rewriter_llm", FakeLLM(lambda m: '["vacation days per year"]'))
    monkeypatch.setattr(g, "rag_llm", FakeStreamLLM(lambda m: "You get **25 days** [S1]."))
    monkeypatch.setattr(g, "direct_llm", FakeStreamLLM(lambda m: "Hi!"))


def sse(text):
    return [json.loads(chunk[6:]) for chunk in text.split("\n\n") if chunk.startswith("data: ")]


async def upload_and_index(c, name, data):
    from app.rag.ingest import IngestionWorker

    r = c.post("/api/documents", files=[("files", (name, data, "application/octet-stream"))], headers=H)
    assert r.status_code == 200, r.text
    await IngestionWorker().drain()
    return r.json()["results"][0]


# ---- accounts ----

def test_signup_login_logout_flow(client):
    assert client.get("/api/me").status_code == 401
    r = signup(client, display_name="Alice A.")
    assert r.status_code == 200 and r.json()["user"]["display_name"] == "Alice A."
    cookie = r.headers["set-cookie"]
    assert "HttpOnly" in cookie and "samesite=lax" in cookie.lower()
    assert client.get("/api/me").json()["user"]["username"] == "alice"
    assert client.post("/api/auth/logout", headers=H).status_code == 200
    assert client.get("/api/me").status_code == 401
    assert client.post("/api/auth/login", json={"username": "alice", "password": "wrong password!"}, headers=H).status_code == 401
    assert client.post("/api/auth/login", json={"username": "ALICE", "password": PASSWORD}, headers=H).status_code == 200
    assert client.get("/api/me").status_code == 200


def test_signup_validation(client):
    assert signup(client, password="short").status_code == 422
    assert signup(client, username="no spaces!").status_code == 422
    assert signup(client).status_code == 200
    assert signup(client).status_code == 409


def test_invite_code_required_when_access_token_set(client, monkeypatch):
    monkeypatch.setenv("AGENT_ACCESS_TOKEN", "invite-123")
    assert client.get("/api/config").json()["invite_required"] is True
    assert signup(client).status_code == 403
    assert signup(client, invite_code="nope").status_code == 403
    assert signup(client, invite_code="invite-123").status_code == 200


def test_signup_can_be_closed(client, monkeypatch):
    from app.settings import reload_app_settings

    monkeypatch.setenv("ALLOW_SIGNUP", "false")
    reload_app_settings()
    assert signup(client).status_code == 403


def test_state_changing_routes_require_csrf_header(alice):
    assert alice.post("/api/auth/login", json={"username": "alice", "password": PASSWORD}).status_code == 403
    assert alice.post("/api/chat/stream", json={"message": "hi"}).status_code == 403
    assert alice.post("/api/documents", files=[("files", ("a.txt", b"hello world", "text/plain"))]).status_code == 403
    assert alice.patch("/api/me", json={"display_name": "x"}).status_code == 403


def test_login_rate_limit(client):
    signup(client)
    codes = [client.post("/api/auth/login", json={"username": "alice", "password": "wrong password!"}, headers=H).status_code
             for _ in range(11)]
    assert codes[:10] == [401] * 10 and codes[10] == 429


def test_preferences_and_password_change(alice):
    r = alice.patch("/api/me", json={"display_name": "Al", "preferences": {"kb_mode": "always"}}, headers=H)
    assert r.json()["user"]["preferences"] == {"kb_mode": "always"} and r.json()["user"]["display_name"] == "Al"
    assert alice.patch("/api/me", json={"preferences": {"kb_mode": "bogus"}}, headers=H).status_code == 422
    assert alice.post("/api/me/password", json={"current_password": "nope nope nope", "new_password": "another long pw"}, headers=H).status_code == 401
    assert alice.post("/api/me/password", json={"current_password": PASSWORD, "new_password": "another long pw"}, headers=H).status_code == 200
    assert alice.get("/api/me").status_code == 200  # this browser stays signed in


def test_security_headers(client):
    r = client.get("/")
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"] and r.headers["x-content-type-options"] == "nosniff"
    assert client.get("/favicon.ico").headers["content-type"].startswith("image/svg")


# ---- chat ----

async def test_chat_stream_persists_conversation_with_sources(alice, llms):
    await upload_and_index(alice, "handbook.pdf", make_pdf(HANDBOOK))
    r = alice.post("/api/chat/stream", json={"message": "how many vacation days?"}, headers=H)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = sse(r.text)
    types = [e["type"] for e in events]
    assert types[0] == "meta" and types[-2:] == ["final", "done"] and "token" in types and "node" in types
    cid = events[0]["conversation_id"]
    final = next(e for e in events if e["type"] == "final")
    assert final["route"] == "rag" and final["sources"][0]["filename"] == "handbook.pdf"
    # prompts are never streamed to the browser
    assert "You are a precise assistant" not in r.text and "SOURCES:" not in r.text

    convo = alice.get(f"/api/conversations/{cid}").json()
    assert convo["title"] == "how many vacation days?"
    user_m, bot_m = convo["messages"]
    assert user_m["role"] == "user" and bot_m["role"] == "assistant" and bot_m["status"] == "complete"
    assert bot_m["content"] == "You get **25 days** [S1]." and bot_m["meta"]["sources"][0]["page"] == 1
    assert bot_m["trace_id"] == final["trace_id"]

    # follow-up in the same conversation carries history
    r2 = alice.post("/api/chat/stream", json={"message": "and sick days?", "conversation_id": cid}, headers=H)
    assert sse(r2.text)[0]["conversation_id"] == cid
    assert len(alice.get(f"/api/conversations/{cid}").json()["messages"]) == 4
    assert "how many vacation days?" in json.dumps([str(m.content) for m in g.rag_llm.calls[-1]])


async def test_regenerate_replaces_last_answer(alice, llms):
    events = sse(alice.post("/api/chat/stream", json={"message": "hello"}, headers=H).text)
    cid = events[0]["conversation_id"]
    r = alice.post("/api/chat/stream", json={"regenerate": True, "conversation_id": cid}, headers=H)
    assert sse(r.text)[0]["regenerate"] is True
    msgs = alice.get(f"/api/conversations/{cid}").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]


def test_chat_validation_and_rate_limit(alice, llms, monkeypatch):
    from app.settings import reload_app_settings

    assert alice.post("/api/chat/stream", json={"message": "  "}, headers=H).status_code == 422
    assert alice.post("/api/chat/stream", json={"message": "x" * 4001}, headers=H).status_code == 422
    assert alice.post("/api/chat/stream", json={"regenerate": True}, headers=H).status_code == 422
    from app.ratelimit import limiter

    monkeypatch.setenv("RATE_LIMIT_CHAT_PER_MINUTE", "2")
    reload_app_settings()
    limiter.reset()
    codes = [alice.post("/api/chat/stream", json={"message": "hi"}, headers=H).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


async def test_chat_error_is_saved_and_reported(alice, monkeypatch):
    from app.fire_llm import FireAPIError

    class Down:
        species_name = "m"

        async def ainvoke(self, messages):
            raise FireAPIError("Fire API returned HTTP 429: slow down")

    monkeypatch.setattr(g, "analyzer_llm", Down())
    events = sse(alice.post("/api/chat/stream", json={"message": "hi"}, headers=H).text)
    err = next(e for e in events if e["type"] == "error")
    assert "rate-limiting" in err["message"]
    msgs = alice.get(f"/api/conversations/{events[0]['conversation_id']}").json()["messages"]
    assert msgs[-1]["status"] == "error"


def test_conversation_search_rename_delete(alice, llms):
    cid = sse(alice.post("/api/chat/stream", json={"message": "Tell me about pelicans"}, headers=H).text)[0]["conversation_id"]
    sse(alice.post("/api/chat/stream", json={"message": "Kubernetes question"}, headers=H).text)
    assert [c["title"] for c in alice.get("/api/conversations").json()] == ["Kubernetes question", "Tell me about pelicans"]
    found = alice.get("/api/conversations", params={"q": "pelican"}).json()
    assert len(found) == 1 and found[0]["id"] == cid and "pelicans" in found[0]["snippet"]
    assert alice.patch(f"/api/conversations/{cid}", json={"title": "Birds"}, headers=H).json()["title"] == "Birds"
    assert alice.delete(f"/api/conversations/{cid}", headers=H).status_code == 200
    assert alice.get(f"/api/conversations/{cid}").status_code == 404


def test_conversations_are_private(alice, bob, llms):
    cid = sse(alice.post("/api/chat/stream", json={"message": "private thought"}, headers=H).text)[0]["conversation_id"]
    assert bob.get(f"/api/conversations/{cid}").status_code == 404
    assert bob.get("/api/conversations").json() == []
    assert bob.post("/api/chat/stream", json={"message": "x", "conversation_id": cid}, headers=H).status_code == 404
    assert bob.delete(f"/api/conversations/{cid}", headers=H).status_code == 404
    assert bob.post("/api/chat/stream", json={"regenerate": True, "conversation_id": cid}, headers=H).status_code == 404


# ---- documents ----

async def test_upload_validation_duplicates_and_quota(alice, monkeypatch):
    from app.settings import reload_app_settings

    files = [("files", ("notes.txt", b"Plain text notes about caching.", "text/plain")),
             ("files", ("fake.pdf", b"this is not a pdf", "application/pdf")),
             ("files", ("virus.exe", b"MZ", "application/octet-stream"))]
    res = alice.post("/api/documents", files=files, headers=H).json()["results"]
    assert [r["ok"] for r in res] == [True, False, False]
    assert "PDF" in res[1]["error"] and "Unsupported" in res[2]["error"]
    again = alice.post("/api/documents", files=[("files", ("copy.txt", b"Plain text notes about caching.", "text/plain"))], headers=H).json()
    assert again["results"][0]["duplicate"] is True and len(alice.get("/api/documents").json()) == 1

    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    reload_app_settings()
    big = alice.post("/api/documents", files=[("files", ("big.txt", b"a" * (1024 * 1024 + 1), "text/plain"))], headers=H).json()
    assert "larger than" in big["results"][0]["error"]
    monkeypatch.setenv("MAX_DOCUMENTS_PER_USER", "1")
    reload_app_settings()
    q = alice.post("/api/documents", files=[("files", ("new.txt", b"something else entirely", "text/plain"))], headers=H).json()
    assert "limit of 1 documents" in q["results"][0]["error"]


async def test_uploaded_file_is_stored_privately_under_generated_name(alice):
    import os
    import stat

    from app.settings import get_app_settings

    res = await upload_and_index(alice, "../../evil name.txt", b"Content about databases and their indexes.")
    doc = res["document"]
    assert doc["filename"] == "evil name.txt"
    files = [p for p in get_app_settings().upload_dir.rglob("*") if p.is_file()]
    assert len(files) == 1 and files[0].name == f"{doc['id']}.txt"
    assert stat.S_IMODE(os.stat(files[0]).st_mode) == 0o600


async def test_document_library_preview_search_reindex_delete(alice):
    res = await upload_and_index(alice, "handbook.pdf", make_pdf(HANDBOOK))
    doc_id = res["document"]["id"]
    doc = alice.get(f"/api/documents/{doc_id}").json()
    assert doc["status"] == "ready" and doc["chunk_count"] == 3 and doc["meta"]["title"] == "Test Handbook"
    chunks = alice.get(f"/api/documents/{doc_id}/chunks", params={"page": 2}).json()
    assert chunks["total"] == 1 and "Fridays" in chunks["items"][0]["text"]
    one = alice.get(f"/api/chunks/{chunks['items'][0]['id']}").json()
    assert one["filename"] == "handbook.pdf" and one["page"] == 2

    found = alice.get("/api/documents/search", params={"q": "meals reimbursed during business travel"}).json()
    assert found["passages"][0]["page"] == 3 and found["passages"][0]["relevant"] and found["passages"][0]["text"] is None

    stats = alice.get("/api/documents/stats").json()
    assert stats["total_documents"] == 1 and stats["total_chunks"] == 3 and stats["success_rate"] == 1.0

    dl = alice.get(f"/api/documents/{doc_id}/file")
    assert dl.status_code == 200 and dl.content.startswith(b"%PDF") and "attachment" in dl.headers["content-disposition"]

    assert alice.post(f"/api/documents/{doc_id}/reindex", headers=H).json()["status"] == "queued"
    assert alice.delete(f"/api/documents/{doc_id}", headers=H).status_code == 200
    assert alice.get(f"/api/documents/{doc_id}").status_code == 404
    assert alice.get("/api/documents/stats").json()["total_chunks"] == 0


async def test_documents_are_private(alice, bob):
    res = await upload_and_index(alice, "secret.txt", b"The vault combination is 31-41-59.")
    doc_id = res["document"]["id"]
    chunk_id = alice.get(f"/api/documents/{doc_id}/chunks").json()["items"][0]["id"]
    for path in (f"/api/documents/{doc_id}", f"/api/documents/{doc_id}/chunks", f"/api/documents/{doc_id}/file", f"/api/chunks/{chunk_id}"):
        assert bob.get(path).status_code == 404, path
    assert bob.post(f"/api/documents/{doc_id}/reindex", headers=H).status_code == 404
    assert bob.delete(f"/api/documents/{doc_id}", headers=H).status_code == 404
    assert bob.get("/api/documents/search", params={"q": "vault combination"}).json()["passages"] == []
    assert bob.get("/api/documents").json() == []


async def test_chat_cannot_target_another_users_documents(alice, bob, llms):
    res = await upload_and_index(alice, "secret.txt", b"Employees get 99 vacation days per year in the secret plan.")
    events = sse(bob.post("/api/chat/stream", json={"message": "vacation days per year", "document_ids": [res["document"]["id"]], "kb_mode": "always"}, headers=H).text)
    final = next(e for e in events if e["type"] == "final")
    assert final["sources"] == [] and final["route"] != "rag"  # bob has no documents: retrieval never runs


def test_api_requires_sign_in(client):
    for path in ("/api/conversations", "/api/documents", "/api/documents/stats", "/api/activity"):
        assert client.get(path).status_code == 401, path


# ---- activity ----

async def test_activity_shows_only_own_runs_without_prompts(alice, bob, llms):
    await upload_and_index(alice, "handbook.pdf", make_pdf(HANDBOOK))
    events = sse(alice.post("/api/chat/stream", json={"message": "vacation days?"}, headers=H).text)
    trace_id = next(e for e in events if e["type"] == "final")["trace_id"]
    import time

    for _ in range(100):
        if alice.get("/api/activity").json()["total"]:
            break
        time.sleep(0.05)
    data = alice.get("/api/activity").json()
    assert data["total"] == 1 and data["items"][0]["route"] == "rag" and data["items"][0]["retrieval_ms"] is not None
    assert data["summary"]["runs"] == 1 and data["summary"]["by_route"]["rag"]["runs"] == 1
    detail = alice.get(f"/api/activity/{trace_id}")
    body = detail.json()
    assert [t["tool"] for t in body["tools"]] == ["embed_query", "vector_search"]
    assert body["llm_steps"][-1]["agent"] == "rag_generator" and body["timeline"]
    assert "input_preview" not in detail.text and "You are a precise assistant" not in detail.text
    assert bob.get(f"/api/activity/{trace_id}").status_code == 404 and bob.get("/api/activity").json()["total"] == 0


def test_request_bodies_are_capped_in_the_app(alice, monkeypatch):
    from app.settings import reload_app_settings

    big_json = '{"message": "' + "x" * 70000 + '"}'
    r = alice.post("/api/chat/stream", content=big_json, headers={**H, "Content-Type": "application/json"})
    assert r.status_code == 413 and "too large" in r.json()["detail"]
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    reload_app_settings()
    files = [("files", (f"f{i}.txt", b"a" * (1024 * 1024), "text/plain")) for i in range(12)]  # > 10 x 1 MB + overhead
    assert alice.post("/api/documents", files=files, headers=H).status_code == 413


async def test_chunked_upload_without_content_length_is_cut_off(db, monkeypatch):
    from app.bodylimit import BodySizeLimitMiddleware

    received = []

    async def app(scope, receive, send):
        while True:
            m = await receive()
            received.append(len(m.get("body", b"")))
            if not m.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    chunks = [{"type": "http.request", "body": b"x" * 40000, "more_body": True} for _ in range(3)]
    sent = []

    async def receive():
        return chunks.pop(0)

    async def send(m):
        sent.append(m)

    await BodySizeLimitMiddleware(app)({"type": "http", "method": "POST", "path": "/api/chat/stream", "headers": []}, receive, send)
    assert sent[0]["status"] == 413 and sum(received) <= 64 * 1024 + 40000


async def test_documents_over_the_chunk_cap_are_refused(alice, monkeypatch):
    from app.settings import reload_app_settings

    monkeypatch.setenv("MAX_CHUNKS_PER_DOCUMENT", "3")
    monkeypatch.setenv("RAG_CHUNK_SIZE", "200")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "0")
    reload_app_settings()
    res = await upload_and_index(alice, "long.txt", ("A sentence about something. " * 60).encode())
    doc = alice.get(f"/api/documents/{res['document']['id']}").json()
    assert doc["status"] == "failed" and "limit is 3" in doc["error"] and doc["chunk_count"] == 0
