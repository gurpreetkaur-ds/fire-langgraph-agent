"""The same storage/retrieval guarantees on real PostgreSQL + pgvector (skipped unless TEST_PG_URL is set).

    TEST_PG_URL="postgresql+psycopg://agent@/agent_test?host=$HOME/.local/pgsock-copilot&port=55433" pytest tests/test_pgvector.py
"""
from __future__ import annotations

import os

import pytest

PG_URL = os.environ.get("TEST_PG_URL")
pytestmark = pytest.mark.skipif(not PG_URL, reason="TEST_PG_URL not set")


@pytest.fixture
def pg(monkeypatch):
    from alembic import command
    from alembic.config import Config

    from evaluation.config import BASE_DIR
    from evaluation.db import init_engine

    monkeypatch.setenv("DATABASE_URL", PG_URL)
    monkeypatch.setenv("EVAL_DATABASE_URL", PG_URL)
    cfg = Config(str(BASE_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BASE_DIR / "migrations"))
    cfg.set_main_option("sqlalchemy.url", PG_URL.replace("%", "%%"))
    command.downgrade(cfg, "base")  # every run starts from an empty schema, through the real migrations
    command.upgrade(cfg, "head")
    engine = init_engine(PG_URL)
    yield engine
    engine.dispose()


def test_migration_creates_pgvector_column(pg):
    from sqlalchemy import text

    with pg.connect() as c:
        col = c.execute(text("SELECT format_type(atttypid, atttypmod) FROM pg_attribute WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'")).scalar()
    assert col == "vector(1536)"


async def test_pgvector_retrieval_isolation_filters_and_delete(pg):
    from app.rag import store as kb
    from app.rag.embeddings import get_embedder
    from tests.rag_helpers import make_pdf
    from tests.test_rag_retrieval import HANDBOOK, _make_user, add_doc, index_all

    alice, bob = _make_user("alice"), _make_user("bob")
    hb = add_doc(alice, "handbook.pdf", make_pdf(HANDBOOK), "pdf")
    other = add_doc(alice, "notes.txt", b"Remote employees connect through the VPN gateway every morning.", "txt")
    add_doc(bob, "bob.txt", b"Bob's vacation days: 40 per year, secret.", "txt")
    assert await index_all() == 3

    async def q(user, text, **kw):
        vecs = (await get_embedder().embed_many([text])).vectors
        return kb.search(user, vecs, 5, **kw)

    hits = await q(alice, "how many vacation days do employees get per year")
    assert hits[0].document_id == hb and hits[0].page == 1 and 0 < hits[0].score <= 1
    assert all(h.document_id in {hb, other} for h in hits)  # never Bob's chunk
    assert {h.document_id for h in await q(alice, "VPN remote", document_ids=[other])} == {other}
    assert {h.file_type for h in await q(alice, "VPN remote", file_types=["pdf"])} == {"pdf"}
    assert {h.document_id for h in await q(bob, "vacation days per year")} <= {d["id"] for d in kb.list_documents(bob)}

    # scores match a numpy cosine of the stored vectors
    import numpy as np
    from sqlalchemy import select

    from app.models import Chunk
    from evaluation.db import session_scope

    qv = (await get_embedder().embed_many(["remote work fridays"])).vectors[0]
    top = (await q(alice, "remote work fridays"))[0]
    with session_scope() as db:
        stored = db.scalar(select(Chunk.embedding).where(Chunk.id == top.chunk_id))
    assert isinstance(stored, np.ndarray) and stored.shape == (1536,)
    assert top.score == pytest.approx(float(np.dot(stored, qv) / (np.linalg.norm(stored) * np.linalg.norm(qv))), abs=1e-4)

    kb.delete_document(alice, hb)
    assert all(h.document_id != hb for h in await q(alice, "vacation days"))
    assert kb.stats(alice)["total_documents"] == 1
