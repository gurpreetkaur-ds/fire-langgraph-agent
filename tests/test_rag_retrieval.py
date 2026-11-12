"""Embedding generation, vector storage, retrieval relevance, isolation, filters, reranking, context, citations."""
from __future__ import annotations

import hashlib

import httpx
import numpy as np
import pytest

from app.rag import retrieval
from app.rag import store as kb
from app.rag.embeddings import FireEmbeddings, get_embedder
from app.rag.ingest import IngestionWorker
from app.rag.store import Hit
from tests.rag_helpers import make_pdf

HANDBOOK = [
    "Vacation policy. Every full-time employee receives 25 days of paid vacation per year. Unused vacation days carry over up to 5 days.",
    "Remote work. Employees may work remotely on Fridays. The VPN must be used for all remote access to internal systems.",
    "Expenses. Meals during business travel are reimbursed up to 60 euros per day with receipts submitted within 30 days.",
]


def _make_user(username="alice"):
    from app.models import User
    from evaluation.db import session_scope

    with session_scope() as db:
        u = User(username=username, display_name=username, password_hash="x", preferences={})
        db.add(u)
        db.flush()
        return u.id


def add_doc(user_id: str, filename: str, data: bytes, ftype: str) -> str:
    from app.models import new_id
    from app.settings import get_app_settings

    doc_id = new_id("doc")
    rel = f"{user_id}/{doc_id}.{ftype}"
    path = get_app_settings().upload_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    kb.create_document(user_id, doc_id=doc_id, filename=filename, file_type=ftype, size_bytes=len(data),
                       sha256=hashlib.sha256(data).hexdigest(), storage_path=rel)
    return doc_id


async def index_all():
    return await IngestionWorker().drain()


async def search(user_id, query, k=5, **kw):
    vecs = (await get_embedder().embed_many([query])).vectors
    return kb.search(user_id, vecs, k, **kw)


@pytest.fixture
def users(db):
    return _make_user("alice"), _make_user("bob")


async def test_ingestion_pipeline_stores_chunks_with_metadata(users):
    alice, _ = users
    doc_id = add_doc(alice, "handbook.pdf", make_pdf(HANDBOOK), "pdf")
    assert kb.get_document(alice, doc_id)["status"] == "queued"
    assert await index_all() == 1
    d = kb.get_document(alice, doc_id)
    assert d["status"] == "ready" and d["chunk_count"] == 3 and d["page_count"] == 3
    assert d["embedding_model"] == "test-hash-embedding" and d["embedding_dimensions"] == 1536 and d["indexed_at"]
    chunks = kb.list_chunks(alice, doc_id)["items"]
    assert [c["page"] for c in chunks] == [1, 2, 3]
    one = kb.get_chunk(alice, chunks[1]["id"])
    assert one["filename"] == "handbook.pdf" and "Fridays" in one["text"] and one["position"] == 2


async def test_failed_extraction_marks_document_failed(users):
    alice, _ = users
    doc_id = add_doc(alice, "scan.pdf", make_pdf([""]), "pdf")
    await index_all()
    d = kb.get_document(alice, doc_id)
    assert d["status"] == "failed" and "scanned" in d["error"] and d["chunk_count"] == 0
    assert kb.stats(alice)["success_rate"] == 0.0


async def test_retrieval_finds_the_relevant_chunk(users):
    alice, _ = users
    add_doc(alice, "handbook.pdf", make_pdf(HANDBOOK), "pdf")
    await index_all()
    hits = await search(alice, "how many vacation days do employees get per year")
    assert hits[0].page == 1 and "25 days" in hits[0].text
    hits = await search(alice, "remote access VPN on Fridays")
    assert hits[0].page == 2


async def test_users_cannot_retrieve_each_others_documents(users):
    alice, bob = users
    doc_id = add_doc(alice, "secret.txt", b"The launch code for project falcon is 7731.", "txt")
    await index_all()
    assert await search(bob, "project falcon launch code") == []
    assert (await search(alice, "project falcon launch code"))[0].document_id == doc_id
    for fn in (kb.get_document, kb.list_chunks, kb.delete_document, kb.requeue_document, kb.storage_path_of):
        with pytest.raises(kb.NotFound):
            fn(bob, doc_id)
    chunk_id = kb.list_chunks(alice, doc_id)["items"][0]["id"]
    with pytest.raises(kb.NotFound):
        kb.get_chunk(bob, chunk_id)
    assert kb.list_documents(bob) == [] and kb.stats(bob)["total_chunks"] == 0


async def test_metadata_filters(users):
    alice, _ = users
    a = add_doc(alice, "a.txt", b"Kubernetes pods restart automatically when health checks fail.", "txt")
    b = add_doc(alice, "b.md", b"# Notes\nKubernetes deployments roll out pods gradually.", "md")
    await index_all()
    assert {h.document_id for h in await search(alice, "kubernetes pods")} == {a, b}
    assert {h.document_id for h in await search(alice, "kubernetes pods", document_ids=[b])} == {b}
    assert {h.document_id for h in await search(alice, "kubernetes pods", file_types=["txt"])} == {a}


async def test_only_ready_documents_are_searchable_and_delete_removes_vectors(users):
    alice, _ = users
    doc_id = add_doc(alice, "a.txt", b"Quarterly revenue grew by twelve percent.", "txt")
    assert await search(alice, "quarterly revenue") == []  # still queued
    await index_all()
    assert await search(alice, "quarterly revenue")
    kb.delete_document(alice, doc_id)
    assert await search(alice, "quarterly revenue") == [] and kb.stats(alice)["total_chunks"] == 0


async def test_reindex_replaces_chunks(users):
    alice, _ = users
    doc_id = add_doc(alice, "a.txt", b"Alpha beta gamma delta epsilon zeta eta theta.", "txt")
    await index_all()
    first = kb.list_chunks(alice, doc_id)["items"][0]["id"]
    kb.requeue_document(alice, doc_id)
    await index_all()
    items = kb.list_chunks(alice, doc_id)["items"]
    assert len(items) == 1 and items[0]["id"] != first


async def test_interrupted_processing_is_requeued(users):
    alice, _ = users
    doc_id = add_doc(alice, "a.txt", b"Some useful content about databases and indexes.", "txt")
    assert kb.claim_next_document()["id"] == doc_id  # worker "crashed" while processing
    assert kb.get_document(alice, doc_id)["status"] == "processing"
    assert kb.reset_interrupted() == 1
    await index_all()
    assert kb.get_document(alice, doc_id)["status"] == "ready"


async def test_stats_are_computed_from_real_rows(users):
    alice, _ = users
    add_doc(alice, "a.txt", b"First document about networking protocols.", "txt")
    add_doc(alice, "b.pdf", make_pdf([""]), "pdf")
    await index_all()
    st = kb.stats(alice)
    assert st["total_documents"] == 2 and st["by_status"]["ready"] == 1 and st["by_status"]["failed"] == 1
    assert st["success_rate"] == 0.5 and st["total_chunks"] == 1 and st["storage_bytes"] > 0
    assert st["by_type"] == {"txt": 1, "pdf": 1}


# ---- reranking / context / citations ----

def _hit(cid, text, score, doc="d1", page=None):
    return Hit(cid, doc, "f.pdf", "pdf", 0, page, None, text, score)


def test_rerank_applies_threshold_dedupes_and_limits():
    hits = [
        _hit("a", "The VPN is mandatory for remote access to internal systems.", 0.70),
        _hit("b", "The VPN is mandatory for remote access to internal systems!", 0.69),  # near duplicate
        _hit("c", "Remote employees use the VPN client from the IT portal.", 0.55),
        _hit("d", "Cafeteria opens at eight.", 0.10),  # below threshold
    ]
    kept, st = retrieval.rerank("Is the VPN required for remote access?", hits, top_k=5, threshold=0.3)
    assert [r.hit.chunk_id for r in kept] == ["a", "c"]
    assert st == {"candidates": 4, "below_threshold": 1, "below_relative": 0, "duplicates": 1}


def test_rerank_lexical_signal_breaks_close_vector_ties():
    hits = [_hit("x", "General information about the company history.", 0.52),
            _hit("y", "Parental leave lasts sixteen weeks at full pay.", 0.50)]
    kept, _ = retrieval.rerank("how long is parental leave", hits, top_k=2, threshold=0.3)
    assert kept[0].hit.chunk_id == "y"


def test_build_context_labels_sources_and_respects_budget():
    ranked = [retrieval.Ranked(_hit(f"c{i}", "word " * 300, 0.9 - i * 0.1, page=i + 1), 0.9, 0.0, 0.9 - i * 0.1) for i in range(5)]
    sources, ctx = retrieval.build_context(ranked, 3500)
    assert [s["id"] for s in sources] == ["S1", "S2", "S3"]  # the last one is truncated to fit
    assert sources[2]["text"].endswith("…") and not sources[1]["text"].endswith("…")
    assert len(ctx) <= 3500 and ctx.startswith("[S1] f.pdf (page 1)")


def test_validate_citations_removes_fabricated_markers():
    sources = [{"id": "S1", "text": "a"}, {"id": "S2", "text": "b"}]
    answer = "Vacation is 25 days [S1]. Fridays are remote [S2, S7]. The CEO likes cats [S9].\nNo citation here."
    rep = retrieval.validate_citations(answer, sources)
    assert "[S7]" not in rep["answer"] and "[S9]" not in rep["answer"]
    assert "The CEO likes cats." in rep["answer"]
    assert rep["removed_markers"] == ["S7", "S9"]
    assert rep["citations"] == [{"claim": "Vacation is 25 days.", "source_id": "S1"},
                                {"claim": "Fridays are remote.", "source_id": "S2"}]
    assert rep["used_sources"] == ["S1", "S2"] and rep["unused_sources"] == []


def test_says_not_found():
    assert retrieval.says_not_found("Your documents don't contain information about pricing.")
    assert not retrieval.says_not_found("Pricing is 10 dollars [S1].")


# ---- Fire embeddings client ----

def _transport(responses):
    calls = []

    def handler(request: httpx.Request):
        calls.append(request)
        status, body = responses[min(len(calls) - 1, len(responses) - 1)]
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler), calls


async def test_fire_embeddings_retries_transient_errors(monkeypatch):
    vec = [0.0] * 1536
    vec[0] = 3.0
    ok = {"data": [{"index": 1, "embedding": [0.0] * 1535 + [1.0]}, {"index": 0, "embedding": vec}], "usage": {"input_tokens": 7}}
    transport, calls = _transport([(503, {"error": "busy"}), (429, {"error": "slow"}), (200, ok)])
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=transport, **kw))
    monkeypatch.setattr("tenacity.asyncio._portable_async_sleep", lambda s: _noop())
    emb = FireEmbeddings(model="text-embedding-3-small", dimensions=1536)
    res = await emb.embed_many(["first", "second"])
    assert len(calls) == 3 and res.input_tokens == 7
    assert res.vectors[0][0] == pytest.approx(1.0)  # ordered by index and L2-normalised
    assert res.vectors[1][-1] == pytest.approx(1.0)
    body = calls[-1].read()
    assert b'"dimensions":1536' in body.replace(b" ", b"") and calls[-1].headers["authorization"].startswith("Bearer ")


async def test_fire_embeddings_does_not_retry_client_errors(monkeypatch):
    from app.fire_llm import FireAPIError

    transport, calls = _transport([(422, {"error": "too long"})])
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=transport, **kw))
    with pytest.raises(FireAPIError, match="422"):
        await FireEmbeddings(dimensions=1536).embed_many(["x"])
    assert len(calls) == 1


async def test_fire_embeddings_rejects_wrong_dimensions(monkeypatch):
    from app.fire_llm import FireAPIError

    transport, _ = _transport([(200, {"data": [{"index": 0, "embedding": [1.0, 0.0]}], "usage": {}})])
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=transport, **kw))
    with pytest.raises(FireAPIError, match="1536"):
        await FireEmbeddings(dimensions=1536).embed_many(["x"])


async def _noop():
    return None


def test_hash_embedder_similarity_is_meaningful():
    e = get_embedder()
    a, b, c = e.vector("vacation days policy"), e.vector("policy on vacation days"), e.vector("kubernetes cluster nodes")
    assert float(np.dot(a, b)) > 0.5 > float(np.dot(a, c))


def test_citation_on_its_own_line_cites_the_preceding_block():
    sources = [{"id": "S1", "text": "a"}, {"id": "S2", "text": "b"}]
    answer = "Q4 priorities:\n\n| Priority | Details |\n|---|---|\n| Migrate search | pgvector |\n\n[S2]"
    rep = retrieval.validate_citations(answer, sources)
    assert rep["citations"] == [{"claim": "Migrate search | pgvector", "source_id": "S2"}]
    assert rep["used_sources"] == ["S2"]


def test_relative_cutoff_drops_the_weak_tail_but_keeps_the_best():
    hits = [_hit("a", "Employees may work abroad for 20 days.", 0.77), _hit("b", "Core hours are ten to three.", 0.44),
            _hit("c", "Expenses are reimbursed monthly.", 0.62)]
    kept, st = retrieval.rerank("work abroad days", hits, top_k=5, threshold=0.3, relative_cutoff=0.75)
    assert [r.hit.chunk_id for r in kept] == ["a", "c"] and st["below_relative"] == 1
    kept, _ = retrieval.rerank("work abroad days", hits, top_k=5, threshold=0.3, relative_cutoff=0)
    assert len(kept) == 3


def test_citations_are_stripped_from_statements_of_absence():
    sources = [{"id": "S1", "text": "Remote work policy."}]
    rep = retrieval.validate_citations("There is no information about who the CEO is [S1].\n\nYour documents don't contain this.", sources)
    assert "[S1]" not in rep["answer"] and rep["citations"] == [] and rep["absence_citations_removed"] == ["S1"]
    assert retrieval.says_not_found(rep["answer"])
