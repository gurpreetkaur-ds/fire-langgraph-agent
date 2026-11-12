"""Knowledge-base API: upload, list, stats, preview, semantic search, re-index, delete, download.

Uploads are validated (extension + content sniffing + size + per-user quota), stored under a generated id
(never the client's filename), de-duplicated per user by SHA-256, and indexed in the background.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from evaluation.tracing import log_event

from .auth import CurrentUser, current_user, write_user
from .fire_llm import FireAPIError
from .models import new_id
from .rag import retrieval
from .rag import store as kb_store
from .rag.embeddings import get_embedder
from .rag.extract import UploadRejected, detect_type, safe_filename
from .rag.ingest import get_ingest_worker
from .ratelimit import limiter
from .settings import get_app_settings

logger = logging.getLogger("fire_agent.documents")
router = APIRouter(prefix="/api", tags=["knowledge base"])
MAX_FILES_PER_REQUEST = 10
MIME = {"pdf": "application/pdf", "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "txt": "text/plain; charset=utf-8", "md": "text/markdown; charset=utf-8", "csv": "text/csv; charset=utf-8"}


def _nf(fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except kb_store.NotFound:
        raise HTTPException(404, "Document not found.")


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    chunks, total = [], 0
    while True:
        part = await upload.read(1024 * 1024)
        if not part:
            return b"".join(chunks)
        total += len(part)
        if total > limit:
            raise UploadRejected(f"File is larger than the {limit // (1024 * 1024)} MB limit.")
        chunks.append(part)


def _store_file(user_id: str, doc_id: str, ext: str, data: bytes) -> str:
    settings = get_app_settings()
    rel = f"{user_id}/{doc_id}.{ext}"
    path = settings.upload_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return rel


@router.post("/documents")
async def upload_documents(files: List[UploadFile] = File(...), user: CurrentUser = Depends(write_user)):
    settings = get_app_settings()
    if not files:
        raise HTTPException(422, "No files were uploaded.")
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(422, f"Upload at most {MAX_FILES_PER_REQUEST} files at a time.")
    results = []
    queued = 0
    for upload in files:
        name = safe_filename(upload.filename or "")
        try:
            limiter.check(f"upload:{user.id}", settings.uploads_per_hour, 3600, message="Upload limit reached. Try again later.")
            data = await _read_limited(upload, settings.max_upload_bytes)
            ftype = detect_type(name, data)
            sha = hashlib.sha256(data).hexdigest()
            dup = await asyncio.to_thread(kb_store.find_duplicate, user.id, sha)
            if dup:
                results.append({"filename": name, "ok": True, "duplicate": True, "document": dup})
                continue
            await asyncio.to_thread(kb_store.check_quota, user.id, len(data))
            doc_id = new_id("doc")
            ext = {"md": "md"}.get(ftype, ftype)
            rel = await asyncio.to_thread(_store_file, user.id, doc_id, ext, data)
            doc = await asyncio.to_thread(kb_store.create_document, user.id, doc_id=doc_id, filename=name, file_type=ftype,
                                          size_bytes=len(data), sha256=sha, storage_path=rel)
            results.append({"filename": name, "ok": True, "duplicate": False, "document": doc})
            queued += 1
            log_event(logger, "document_uploaded", doc_id=doc_id, type=ftype, bytes=len(data))
        except UploadRejected as exc:
            results.append({"filename": name, "ok": False, "error": str(exc)})
        except kb_store.QuotaExceeded as exc:
            results.append({"filename": name, "ok": False, "error": str(exc)})
        except HTTPException as exc:
            if exc.status_code != 429:
                raise
            results.append({"filename": name, "ok": False, "error": exc.detail})
        finally:
            await upload.close()
    worker = get_ingest_worker()
    if queued and worker is not None:
        worker.notify()
    return {"results": results}


@router.get("/documents")
def list_documents(q: Optional[str] = Query(None, max_length=100), user: CurrentUser = Depends(current_user)):
    return kb_store.list_documents(user.id, (q or "").strip() or None)


@router.get("/documents/stats")
def document_stats(user: CurrentUser = Depends(current_user)):
    return kb_store.stats(user.id)


@router.get("/documents/search")
async def search_documents(q: str = Query(..., min_length=2, max_length=500), k: int = Query(8, ge=1, le=20),
                           user: CurrentUser = Depends(current_user)):
    """Semantic search across the user's documents (also matches file names)."""
    limiter.check(f"search:{user.id}", 30, 60, message="Too many searches. Please wait a moment.")
    by_name = await asyncio.to_thread(kb_store.list_documents, user.id, q)
    try:
        emb = await get_embedder().embed_many([q])
    except FireAPIError:
        raise HTTPException(502, "Search is unavailable right now (embedding service error).")
    hits = await asyncio.to_thread(kb_store.search, user.id, emb.vectors, k)
    s = get_app_settings()
    return {
        "documents": by_name[:10],
        "passages": [{**h.as_dict(), "text": None, "preview": retrieval.preview(h.text, 360), "score": round(h.score, 4),
                      "relevant": h.score >= s.similarity_threshold} for h in hits],
    }


@router.get("/documents/{doc_id}")
def get_document(doc_id: str, user: CurrentUser = Depends(current_user)):
    return _nf(kb_store.get_document, user.id, doc_id)


@router.get("/documents/{doc_id}/chunks")
def document_chunks(doc_id: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200),
                    page: Optional[int] = Query(None, ge=1), user: CurrentUser = Depends(current_user)):
    return _nf(kb_store.list_chunks, user.id, doc_id, offset, limit, page)


@router.get("/chunks/{chunk_id}")
def get_chunk(chunk_id: str, user: CurrentUser = Depends(current_user)):
    try:
        return kb_store.get_chunk(user.id, chunk_id)
    except kb_store.NotFound:
        raise HTTPException(404, "Passage not found (the document may have been re-indexed or deleted).")


@router.post("/documents/{doc_id}/reindex")
def reindex_document(doc_id: str, user: CurrentUser = Depends(write_user)):
    try:
        doc = _nf(kb_store.requeue_document, user.id, doc_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    worker = get_ingest_worker()
    if worker is not None:
        worker.notify()
    return doc


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str, user: CurrentUser = Depends(write_user)):
    _nf(kb_store.delete_document, user.id, doc_id)
    return {"ok": True}


@router.get("/documents/{doc_id}/file")
def download_document(doc_id: str, user: CurrentUser = Depends(current_user)):
    doc = _nf(kb_store.get_document, user.id, doc_id)
    path = _nf(kb_store.storage_path_of, user.id, doc_id)
    if not path.exists():
        raise HTTPException(404, "The original file is no longer stored.")
    return FileResponse(path, media_type=MIME.get(doc["file_type"], "application/octet-stream"), filename=doc["filename"],
                        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "private, no-store",
                                 "Content-Security-Policy": "sandbox"})
