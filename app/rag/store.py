"""Knowledge-base persistence: documents, chunks, vector search and statistics.

Every public function takes the owner's `user_id` and filters on it, so a user can never read, search, preview,
re-index or delete another user's documents. Functions are synchronous (SQLAlchemy); async callers use to_thread.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
from sqlalchemy import delete, func, select, text, update

from evaluation.db import get_engine, session_scope
from evaluation.models import utcnow

from ..models import Chunk, Document
from ..settings import get_app_settings
from .chunking import ChunkData

logger = logging.getLogger("fire_agent.rag.store")


class NotFound(LookupError):
    pass


class QuotaExceeded(ValueError):
    pass


@dataclass
class Hit:
    chunk_id: str
    document_id: str
    filename: str
    file_type: str
    chunk_index: int
    page: Optional[int]
    section: Optional[str]
    text: str
    score: float  # cosine similarity in [-1, 1]

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def document_dict(d: Document) -> dict:
    return {
        "id": d.id, "filename": d.filename, "file_type": d.file_type, "size_bytes": d.size_bytes,
        "status": d.status, "error": d.error, "page_count": d.page_count, "char_count": d.char_count,
        "chunk_count": d.chunk_count, "embedding_model": d.embedding_model, "embedding_dimensions": d.embedding_dimensions,
        "embedding_tokens": d.embedding_tokens, "meta": d.meta or {}, "sha256": d.sha256,
        "created_at": _iso(d.created_at), "updated_at": _iso(d.updated_at), "indexed_at": _iso(d.indexed_at),
    }


# ---- documents -------------------------------------------------------------------------------------------

def find_duplicate(user_id: str, sha256: str) -> Optional[dict]:
    with session_scope() as db:
        d = db.scalar(select(Document).where(Document.user_id == user_id, Document.sha256 == sha256))
        return document_dict(d) if d else None


def check_quota(user_id: str, incoming_bytes: int) -> None:
    s = get_app_settings()
    with session_scope() as db:
        count, used = db.execute(
            select(func.count(Document.id), func.coalesce(func.sum(Document.size_bytes), 0)).where(Document.user_id == user_id)
        ).one()
    if count >= s.max_documents_per_user:
        raise QuotaExceeded(f"You have reached the limit of {s.max_documents_per_user} documents.")
    if used + incoming_bytes > s.max_storage_bytes_per_user:
        raise QuotaExceeded(f"This upload would exceed your {s.max_storage_bytes_per_user // (1024 * 1024)} MB storage quota.")


def create_document(user_id: str, *, doc_id: str, filename: str, file_type: str, size_bytes: int, sha256: str, storage_path: str) -> dict:
    now = utcnow()
    with session_scope() as db:
        d = Document(id=doc_id, user_id=user_id, filename=filename, file_type=file_type, size_bytes=size_bytes, sha256=sha256,
                     storage_path=storage_path, status="queued", chunk_count=0, meta={}, attempts=0, created_at=now, updated_at=now)
        db.add(d)
        db.flush()
        return document_dict(d)


def list_documents(user_id: str, q: Optional[str] = None) -> List[dict]:
    with session_scope() as db:
        stmt = select(Document).where(Document.user_id == user_id)
        if q:
            stmt = stmt.where(func.lower(Document.filename).contains(q.lower()))
        return [document_dict(d) for d in db.scalars(stmt.order_by(Document.created_at.desc())).all()]


def get_document(user_id: str, doc_id: str) -> dict:
    with session_scope() as db:
        d = db.scalar(select(Document).where(Document.id == doc_id, Document.user_id == user_id))
        if d is None:
            raise NotFound(doc_id)
        return document_dict(d)


def storage_path_of(user_id: str, doc_id: str) -> Path:
    with session_scope() as db:
        rel = db.scalar(select(Document.storage_path).where(Document.id == doc_id, Document.user_id == user_id))
    if rel is None:
        raise NotFound(doc_id)
    return get_app_settings().upload_dir / rel


def delete_document(user_id: str, doc_id: str) -> None:
    with session_scope() as db:
        d = db.scalar(select(Document).where(Document.id == doc_id, Document.user_id == user_id))
        if d is None:
            raise NotFound(doc_id)
        rel = d.storage_path
        db.execute(delete(Chunk).where(Chunk.document_id == doc_id, Chunk.user_id == user_id))
        db.delete(d)
    path = get_app_settings().upload_dir / rel
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("event=upload_unlink_failed doc_id=%s", doc_id)


def requeue_document(user_id: str, doc_id: str) -> dict:
    with session_scope() as db:
        d = db.scalar(select(Document).where(Document.id == doc_id, Document.user_id == user_id))
        if d is None:
            raise NotFound(doc_id)
        if d.status == "processing":
            raise ValueError("This document is being indexed right now.")
        d.status, d.error, d.updated_at = "queued", None, utcnow()
        return document_dict(d)


# ---- worker-side (no user filter: the worker only touches documents it claimed by id) ----------------------

def reset_interrupted() -> int:
    """Documents left `processing` by a crash or restart go back to the queue."""
    with session_scope() as db:
        return db.execute(update(Document).where(Document.status == "processing").values(status="queued")).rowcount or 0


def claim_next_document() -> Optional[dict]:
    with session_scope() as db:
        for doc_id in db.scalars(select(Document.id).where(Document.status == "queued").order_by(Document.created_at).limit(5)).all():
            res = db.execute(
                update(Document).where(Document.id == doc_id, Document.status == "queued")
                .values(status="processing", attempts=Document.attempts + 1, updated_at=utcnow())
            )
            if res.rowcount == 1:
                d = db.get(Document, doc_id)
                return {"id": d.id, "user_id": d.user_id, "file_type": d.file_type, "storage_path": d.storage_path, "filename": d.filename}
    return None


def save_chunks(doc_id: str, user_id: str, chunks: Sequence[ChunkData], vectors: Sequence[np.ndarray], *, page_count, char_count,
                meta: dict, model: str, dimensions: int, tokens: int) -> bool:
    """Replace the document's chunks and mark it ready. Returns False if the document was deleted meanwhile."""
    now = utcnow()
    with session_scope() as db:
        d = db.get(Document, doc_id)
        if d is None or d.user_id != user_id:
            return False
        db.execute(delete(Chunk).where(Chunk.document_id == doc_id))
        db.add_all([
            Chunk(document_id=doc_id, user_id=user_id, chunk_index=c.index, page=c.page, section=(c.section or None) and c.section[:300],
                  text=c.text, char_start=c.char_start, char_end=c.char_end, embedding=v, created_at=now)
            for c, v in zip(chunks, vectors)
        ])
        d.status, d.error = "ready", None
        d.chunk_count, d.page_count, d.char_count = len(chunks), page_count, char_count
        d.meta, d.embedding_model, d.embedding_dimensions, d.embedding_tokens = meta, model, dimensions, tokens
        d.indexed_at = d.updated_at = now
    return True


def mark_failed(doc_id: str, message: str) -> None:
    with session_scope() as db:
        d = db.get(Document, doc_id)
        if d is not None:
            d.status, d.error, d.updated_at = "failed", message[:500], utcnow()
            db.execute(delete(Chunk).where(Chunk.document_id == doc_id))
            d.chunk_count = 0


# ---- chunks / previews -----------------------------------------------------------------------------------

def chunk_dict(c: Chunk, with_text: bool = True) -> dict:
    out = {"id": c.id, "document_id": c.document_id, "chunk_index": c.chunk_index, "page": c.page, "section": c.section,
           "char_start": c.char_start, "char_end": c.char_end, "chars": len(c.text)}
    if with_text:
        out["text"] = c.text
    return out


def list_chunks(user_id: str, doc_id: str, offset: int = 0, limit: int = 50, page: Optional[int] = None) -> dict:
    with session_scope() as db:
        if db.scalar(select(func.count()).select_from(Document).where(Document.id == doc_id, Document.user_id == user_id)) == 0:
            raise NotFound(doc_id)
        stmt = select(Chunk).where(Chunk.document_id == doc_id, Chunk.user_id == user_id)
        if page is not None:
            stmt = stmt.where(Chunk.page == page)
        total = db.scalar(select(func.count()).select_from(stmt.subquery()))
        rows = db.scalars(stmt.order_by(Chunk.chunk_index).offset(offset).limit(limit)).all()
        return {"total": total, "items": [chunk_dict(c) for c in rows]}


def get_chunk(user_id: str, chunk_id: str) -> dict:
    with session_scope() as db:
        c = db.scalar(select(Chunk).where(Chunk.id == chunk_id, Chunk.user_id == user_id))
        if c is None:
            raise NotFound(chunk_id)
        d = c.document
        return {**chunk_dict(c), "filename": d.filename, "file_type": d.file_type,
                "position": c.chunk_index + 1, "chunk_count": d.chunk_count}


# ---- vector search ---------------------------------------------------------------------------------------

def search(user_id: str, query_vectors: Iterable[np.ndarray], k: int, *, document_ids: Optional[Sequence[str]] = None,
           file_types: Optional[Sequence[str]] = None) -> List[Hit]:
    """Top-k chunks per query vector (cosine), restricted to the user's ready documents; merged by best score."""
    vecs = [np.asarray(v, dtype=np.float32) for v in query_vectors]
    if not vecs:
        return []
    is_pg = get_engine().dialect.name == "postgresql"
    best: dict[str, Hit] = {}
    for hit in (_search_pg if is_pg else _search_py)(user_id, vecs, k, document_ids, file_types):
        if hit.chunk_id not in best or hit.score > best[hit.chunk_id].score:
            best[hit.chunk_id] = hit
    return sorted(best.values(), key=lambda h: h.score, reverse=True)


def _filters_sql(document_ids, file_types) -> tuple[str, dict]:
    sql, params = "", {}
    if document_ids:
        sql += " AND c.document_id = ANY(:doc_ids)"
        params["doc_ids"] = list(document_ids)
    if file_types:
        sql += " AND d.file_type = ANY(:ftypes)"
        params["ftypes"] = list(file_types)
    return sql, params


def _search_pg(user_id, vecs, k, document_ids, file_types) -> List[Hit]:
    extra, params = _filters_sql(document_ids, file_types)
    sql = text(
        "SELECT c.id, c.document_id, d.filename, d.file_type, c.chunk_index, c.page, c.section, c.text, "
        "1 - (c.embedding <=> CAST(:q AS vector)) AS score "
        "FROM chunks c JOIN documents d ON d.id = c.document_id "
        "WHERE c.user_id = :uid AND d.user_id = :uid AND d.status = 'ready'" + extra +
        " ORDER BY c.embedding <=> CAST(:q AS vector) LIMIT :k"
    )
    hits: List[Hit] = []
    with session_scope() as db:
        for v in vecs:
            q = "[" + ",".join(f"{x:.7g}" for x in v.tolist()) + "]"
            for row in db.execute(sql, {"q": q, "uid": user_id, "k": k, **params}):
                hits.append(Hit(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], float(row[8])))
    return hits


def _search_py(user_id, vecs, k, document_ids, file_types) -> List[Hit]:
    """SQLite fallback (dev/tests): exact cosine in numpy over the user's chunks."""
    with session_scope() as db:
        stmt = (select(Chunk.id, Chunk.document_id, Document.filename, Document.file_type, Chunk.chunk_index, Chunk.page,
                       Chunk.section, Chunk.text, Chunk.embedding)
                .join(Document, Document.id == Chunk.document_id)
                .where(Chunk.user_id == user_id, Document.user_id == user_id, Document.status == "ready"))
        if document_ids:
            stmt = stmt.where(Chunk.document_id.in_(list(document_ids)))
        if file_types:
            stmt = stmt.where(Document.file_type.in_(list(file_types)))
        rows = db.execute(stmt).all()
    if not rows:
        return []
    matrix = np.vstack([r[8] for r in rows])
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0
    hits: List[Hit] = []
    for v in vecs:
        qn = float(np.linalg.norm(v)) or 1.0
        scores = (matrix @ v) / (norms * qn)
        for i in np.argsort(-scores)[:k]:
            r = rows[int(i)]
            hits.append(Hit(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], float(scores[int(i)])))
    return hits


# ---- knowledge-base summary for the agent and the dashboard -------------------------------------------------

def ready_documents(user_id: str, limit: int = 40) -> List[dict]:
    with session_scope() as db:
        rows = db.execute(
            select(Document.id, Document.filename, Document.meta).where(Document.user_id == user_id, Document.status == "ready")
            .order_by(Document.created_at.desc()).limit(limit)
        ).all()
    return [{"id": r[0], "filename": r[1], "title": (r[2] or {}).get("title")} for r in rows]


def stats(user_id: str) -> dict:
    with session_scope() as db:
        by_status = dict(db.execute(select(Document.status, func.count()).where(Document.user_id == user_id).group_by(Document.status)).all())
        total_docs, storage = db.execute(
            select(func.count(Document.id), func.coalesce(func.sum(Document.size_bytes), 0)).where(Document.user_id == user_id)
        ).one()
        chunks = db.scalar(select(func.count()).select_from(Chunk).where(Chunk.user_id == user_id)) or 0
        by_type = dict(db.execute(select(Document.file_type, func.count()).where(Document.user_id == user_id).group_by(Document.file_type)).all())
        recent = db.scalars(select(Document).where(Document.user_id == user_id).order_by(Document.created_at.desc()).limit(5)).all()
        recent_list = [document_dict(d) for d in recent]
    finished = by_status.get("ready", 0) + by_status.get("failed", 0)
    s = get_app_settings()
    return {
        "total_documents": total_docs,
        "total_chunks": chunks,
        "storage_bytes": int(storage),
        "storage_quota_bytes": s.max_storage_bytes_per_user,
        "document_quota": s.max_documents_per_user,
        "by_status": {k: by_status.get(k, 0) for k in ("queued", "processing", "ready", "failed")},
        "by_type": by_type,
        # share of finished indexing jobs that succeeded; null until something has finished
        "success_rate": (by_status.get("ready", 0) / finished) if finished else None,
        "recent": recent_list,
    }
