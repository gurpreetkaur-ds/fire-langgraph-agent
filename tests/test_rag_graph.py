"""LangGraph routing and the RAG branch (fake LLMs, real retrieval over the SQLite vector fallback)."""
from __future__ import annotations

import json

import pytest

import app.graph as g
from app.runner import WorkflowRun
from tests.rag_helpers import FakeLLM, FakeStreamLLM, last_user_text, make_pdf
from tests.test_rag_retrieval import HANDBOOK, _make_user, add_doc, index_all


def analyzer(route, standalone=None):
    return FakeLLM(lambda m: json.dumps({"route": route, "standalone_question": standalone or last_user_text(m).split("Latest message: ")[-1], "reason": "test"}))


@pytest.fixture
def llms(monkeypatch):
    """Default fakes for every LLM in the graph; tests override individual ones."""
    fakes = {
        "analyzer_llm": analyzer("direct"),
        "rewriter_llm": FakeLLM(lambda m: '["employee vacation days per year", "paid leave allowance"]'),
        "rag_llm": FakeStreamLLM(lambda m: "Employees get 25 days of paid vacation per year [S1]. Unused days carry over [S1][S4]."),
        "direct_llm": FakeStreamLLM(lambda m: "Hello! How can I help?"),
        "decomposer_llm": FakeLLM(lambda m: '["What is TCP?", "What is UDP?"]'),
        "answer_llm": FakeLLM(lambda m: "It is a transport protocol."),
        "synth_llm": FakeLLM(lambda m: "TCP is reliable; UDP is fast."),
    }
    for name, fake in fakes.items():
        monkeypatch.setattr(g, name, fake)
    return fakes


@pytest.fixture
def alice(db):
    return _make_user("alice")


async def run_chat(question, user_id, **kw):
    run = WorkflowRun(question, user_id=user_id, session_id="cnv_test", force_route=None, **kw)
    events = [e async for e in run.stream()]
    await run.finalize()
    return run, events


def nodes(events):
    return [e["node"] for e in events if e.get("type") == "node" and e["status"] == "done"]


async def test_rag_route_runs_full_pipeline_with_citations(llms, alice, monkeypatch):
    monkeypatch.setattr(g, "analyzer_llm", analyzer("knowledge_base", "How many vacation days do employees get per year?"))
    add_doc(alice, "handbook.pdf", make_pdf(HANDBOOK), "pdf")
    await index_all()
    run, events = await run_chat("how many vacation days?", alice)
    assert nodes(events) == ["analyze_query", "rewrite_query", "retrieve", "rerank", "build_context", "generate_answer", "validate_citations"]
    final = events[-1]
    assert final["type"] == "final" and final["route"] == "rag"
    # streamed tokens arrive before the validated answer
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert "[S4]" in tokens  # the model "hallucinated" a citation...
    assert "[S4]" not in final["answer"]  # ...which validation removed
    assert any(e["type"] == "answer" for e in events)
    assert final["sources"][0]["filename"] == "handbook.pdf" and final["sources"][0]["page"] == 1
    assert final["sources"][0]["cited"] is True and "text" not in final["sources"][0]
    assert {c["source_id"] for c in final["citations"]} == {"S1"} and final["grounded"] is True
    assert final["search_queries"][0] == "How many vacation days do employees get per year?"
    # the generator saw the retrieved context, labelled
    prompt = last_user_text(g.rag_llm.calls[-1])
    assert "[S1] handbook.pdf (page 1)" in prompt and "25 days" in prompt


async def test_rag_trace_feeds_evaluation_and_activity(llms, alice, monkeypatch):
    from evaluation.db import session_scope
    from evaluation.models import AgentTrace

    monkeypatch.setattr(g, "analyzer_llm", analyzer("knowledge_base", "vacation days per year"))
    add_doc(alice, "handbook.pdf", make_pdf(HANDBOOK), "pdf")
    await index_all()
    run, _ = await run_chat("vacation?", alice)
    with session_scope() as s:
        t = s.get(AgentTrace, run.trace_id)
        assert t.route == "rag" and t.user_id == alice and t.agent_name == "rag_generator"
        assert t.context[0]["id"] == "S1" and t.context[0]["source"].startswith("handbook.pdf (page")
        assert t.citations and t.citations[0]["source_id"] == "S1"
        assert [c.tool_name for c in t.tool_calls] == ["embed_query", "vector_search"]
        assert [e["node"] for e in t.timeline if e["kind"] == "node"][:3] == ["analyze_query", "rewrite_query", "retrieve"]
        assert all(e["duration_ms"] >= 0 for e in t.timeline)
        assert [st.agent_name for st in t.steps] == ["query_analyzer", "query_rewriter", "rag_generator"]
        assert t.steps[-1].input_tokens == 300  # usage from the stream's message_done


async def test_no_relevant_passages_says_documents_do_not_contain_it(llms, alice, monkeypatch):
    monkeypatch.setattr(g, "analyzer_llm", analyzer("knowledge_base", "What is the capital of Australia?"))
    monkeypatch.setattr(g, "rewriter_llm", FakeLLM(lambda m: '["capital city of Australia"]'))
    no_ctx = FakeStreamLLM(lambda m: "Your uploaded documents don't contain information about this.")
    monkeypatch.setattr(g, "rag_llm", no_ctx)
    add_doc(alice, "handbook.pdf", make_pdf(HANDBOOK), "pdf")
    await index_all()
    _, events = await run_chat("capital of Australia?", alice)
    final = events[-1]
    assert final["sources"] == [] and final["citations"] == [] and final["grounded"] is None
    assert final["retrieval"]["kept"] == 0 and final["retrieval"]["below_threshold"] >= 1
    assert str(no_ctx.calls[0][0].content).startswith("You are an assistant answering questions about the user's documents. A search")


async def test_answer_without_any_citation_is_flagged_ungrounded(llms, alice, monkeypatch):
    monkeypatch.setattr(g, "analyzer_llm", analyzer("knowledge_base", "vacation days per year employees"))
    monkeypatch.setattr(g, "rag_llm", FakeStreamLLM(lambda m: "Employees get plenty of vacation."))
    add_doc(alice, "handbook.pdf", make_pdf(HANDBOOK), "pdf")
    await index_all()
    _, events = await run_chat("vacation?", alice)
    assert events[-1]["grounded"] is False


async def test_direct_route_skips_retrieval(llms, alice):
    add_doc(alice, "handbook.pdf", make_pdf(HANDBOOK), "pdf")
    await index_all()
    _, events = await run_chat("hi there", alice)
    assert nodes(events) == ["analyze_query", "direct_answer"]
    assert events[-1]["answer"] == "Hello! How can I help?" and events[-1]["sources"] == []
    assert "".join(e["text"] for e in events if e["type"] == "token") == "Hello! How can I help?"


async def test_research_route_reuses_original_pipeline(llms, alice, monkeypatch):
    monkeypatch.setattr(g, "analyzer_llm", analyzer("research", "Compare TCP and UDP"))
    _, events = await run_chat("compare them", alice, history=[{"role": "user", "content": "Tell me about TCP and UDP"},
                                                               {"role": "assistant", "content": "Sure."}])
    assert nodes(events) == ["analyze_query", "decompose", "answer", "synthesize"]
    assert events[-1]["answer"] == "TCP is reliable; UDP is fast." and events[-1]["sub_questions"] == ["What is TCP?", "What is UDP?"]
    # the decomposer received the standalone (pronoun-resolved) question
    assert last_user_text(g.decomposer_llm.calls[0]) == "Compare TCP and UDP"


async def test_kb_off_or_empty_never_retrieves(llms, alice, monkeypatch):
    monkeypatch.setattr(g, "analyzer_llm", analyzer("knowledge_base"))
    _, events = await run_chat("what do my docs say?", alice)  # no documents at all
    assert "retrieve" not in nodes(events) and events[-1]["route"] == "direct"
    add_doc(alice, "handbook.pdf", make_pdf(HANDBOOK), "pdf")
    await index_all()
    _, events = await run_chat("what do my docs say?", alice, kb_mode="off")
    assert "retrieve" not in nodes(events)
    assert "do NOT choose knowledge_base" in last_user_text(g.analyzer_llm.calls[-1])


async def test_kb_always_forces_retrieval(llms, alice, monkeypatch):
    monkeypatch.setattr(g, "analyzer_llm", analyzer("direct"))
    add_doc(alice, "handbook.pdf", make_pdf(HANDBOOK), "pdf")
    await index_all()
    _, events = await run_chat("vacation days", alice, kb_mode="always")
    assert "retrieve" in nodes(events) and events[-1]["route"] == "rag"


async def test_analyzer_garbage_falls_back_safely(llms, alice, monkeypatch):
    monkeypatch.setattr(g, "analyzer_llm", FakeLLM(lambda m: "I think you should search!"))
    add_doc(alice, "handbook.pdf", make_pdf(HANDBOOK), "pdf")
    await index_all()
    _, events = await run_chat("vacation days per year", alice)
    assert events[-1]["route"] == "rag" and "fallback" in events[-1]["route_reason"]


async def test_document_filter_restricts_retrieval(llms, alice, monkeypatch):
    monkeypatch.setattr(g, "analyzer_llm", analyzer("knowledge_base", "vacation days per year"))
    add_doc(alice, "handbook.pdf", make_pdf(HANDBOOK), "pdf")
    other = add_doc(alice, "other.txt", b"Vacation days in the other company are 30 per year.", "txt")
    await index_all()
    _, events = await run_chat("vacation days", alice, document_ids=[other])
    assert {s["document_id"] for s in events[-1]["sources"]} == {other}


async def test_conversation_history_reaches_generator(llms, alice):
    history = [{"role": "user", "content": "My name is Sam."}, {"role": "assistant", "content": "Nice to meet you, Sam."}]
    await run_chat("what's my name?", alice, history=history)
    contents = [str(m.content) for m in g.direct_llm.calls[-1]]
    assert contents[1:3] == ["My name is Sam.", "Nice to meet you, Sam."]


async def test_rewriter_failure_still_retrieves_with_original_question(llms, alice, monkeypatch):
    from app.fire_llm import FireAPIError

    class Down:
        species_name = "test-model"

        async def ainvoke(self, messages):
            raise FireAPIError("Fire API returned HTTP 503")

    monkeypatch.setattr(g, "analyzer_llm", analyzer("knowledge_base", "vacation days per year employees"))
    monkeypatch.setattr(g, "rewriter_llm", Down())
    add_doc(alice, "handbook.pdf", make_pdf(HANDBOOK), "pdf")
    await index_all()
    _, events = await run_chat("vacation", alice)
    assert events[-1]["type"] == "final" and events[-1]["search_queries"] == ["vacation days per year employees"]
    assert events[-1]["sources"]


async def test_embedding_failure_ends_with_user_safe_error(llms, alice, monkeypatch):
    from app.fire_llm import FireAPIError
    from app.rag.embeddings import set_embedder

    class DownEmbedder:
        async def embed_many(self, texts):
            raise FireAPIError("Could not reach Fire API: ConnectError")

    monkeypatch.setattr(g, "analyzer_llm", analyzer("knowledge_base"))
    add_doc(alice, "handbook.pdf", make_pdf(HANDBOOK), "pdf")
    await index_all()
    set_embedder(DownEmbedder())
    run, events = await run_chat("vacation", alice)
    assert events[-1] == {"type": "error", "node": "retrieve", "message": "Could not reach the AI provider. Please try again."}
    assert run.error and "fire_sk" not in json.dumps(events)


async def test_legacy_protocol_unchanged(llms):
    run = WorkflowRun("Compare TCP and UDP")
    events = [e async for e in run.events()]
    assert [e["node"] for e in events] == ["decompose", "answer", "synthesize", "done"]
    assert "analyze_query" not in json.dumps(events)
