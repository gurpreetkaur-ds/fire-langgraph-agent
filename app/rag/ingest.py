"""Background document indexing: extract -> clean -> chunk -> embed -> store.

Uploads return immediately with status `queued`; this worker (an asyncio task inside the API process, like the
inline evaluation worker) processes one document at a time. Status moves queued -> processing -> ready | failed.
On startup, documents interrupted mid-processing are re-queued. Embedding calls retry transient errors;
a document that still fails is marked `failed` with a user-safe message and can be re-indexed from the UI.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from evaluation.tracing import log_event

from ..fire_llm import FireAPIError
from ..settings import get_app_settings
from . import store
from .chunking import chunk_sections
from .embeddings import get_embedder
from .extract import ExtractionError, extract

logger = logging.getLogger("fire_agent.rag.ingest")


def embedding_text(filename: str, title, chunk) -> str:
    """Contextual chunk header: the embedded text carries the document name/title and the chunk's page/section.

    Without it, a short title page ("Q3 Engineering Report") can outscore the page that actually answers a question
    that mentions the document's name. The stored and displayed chunk text stays unchanged.
    """
    parts = [filename]
    if title and title != filename:
        parts.append(title)
    if chunk.page:
        parts.append(f"page {chunk.page}")
    if chunk.section:
        parts.append(chunk.section)
    return " — ".join(parts) + "\n" + chunk.text


async def index_document(doc: dict) -> None:
    settings = get_app_settings()
    doc_id, user_id = doc["id"], doc["user_id"]
    path = settings.upload_dir / doc["storage_path"]
    log_event(logger, "index_start", doc_id=doc_id, type=doc["file_type"])
    try:
        extracted = await asyncio.to_thread(extract, path, doc["file_type"], max_pdf_pages=settings.max_pdf_pages)
        chunks = chunk_sections(extracted.sections, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise ExtractionError("No readable text was found in this file.")
        if len(chunks) > settings.max_chunks_per_document:
            raise ExtractionError(
                f"This document produces {len(chunks):,} passages; the limit is {settings.max_chunks_per_document:,}. "
                "Split it into smaller files.")
        result = await get_embedder().embed_many([embedding_text(doc["filename"], extracted.meta.get("title"), c) for c in chunks])
        meta = {**extracted.meta, "sections": len({c.section for c in chunks if c.section})}
        saved = await asyncio.to_thread(
            store.save_chunks, doc_id, user_id, chunks, result.vectors, page_count=extracted.page_count,
            char_count=extracted.char_count, meta=meta, model=result.model, dimensions=len(result.vectors[0]),
            tokens=result.input_tokens,
        )
        log_event(logger, "index_done", doc_id=doc_id, chunks=len(chunks), tokens=result.input_tokens, saved=saved)
    except ExtractionError as exc:
        await asyncio.to_thread(store.mark_failed, doc_id, str(exc))
        log_event(logger, "index_failed", logging.WARNING, doc_id=doc_id, reason="extraction")
    except FileNotFoundError:
        await asyncio.to_thread(store.mark_failed, doc_id, "The uploaded file is missing on the server. Please upload it again.")
    except FireAPIError as exc:
        await asyncio.to_thread(store.mark_failed, doc_id, "Embedding failed (Fire API). Try Re-index in a minute.")
        log_event(logger, "index_failed", logging.ERROR, doc_id=doc_id, reason="embedding", error=str(exc)[:200])
    except Exception:
        logger.exception("event=index_crashed doc_id=%s", doc_id)
        await asyncio.to_thread(store.mark_failed, doc_id, "Indexing failed because of an internal error.")


class IngestionWorker:
    def __init__(self, poll_seconds: float = 5.0):
        self.poll_seconds = poll_seconds
        self._wake = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._stopping = False

    def notify(self) -> None:
        self._wake.set()

    async def drain(self) -> int:
        """Process every queued document now (tests and CLI)."""
        n = 0
        while True:
            doc = await asyncio.to_thread(store.claim_next_document)
            if doc is None:
                return n
            await index_document(doc)
            n += 1

    async def _loop(self) -> None:
        reset = await asyncio.to_thread(store.reset_interrupted)
        if reset:
            log_event(logger, "index_requeued_after_restart", count=reset)
        while not self._stopping:
            try:
                await self.drain()
            except Exception:
                logger.exception("event=ingest_loop_error")
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self.poll_seconds)
            except asyncio.TimeoutError:
                pass

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stopping = True
        self._wake.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None


_worker: Optional[IngestionWorker] = None


def get_ingest_worker() -> Optional[IngestionWorker]:
    return _worker


def set_ingest_worker(worker: Optional[IngestionWorker]) -> None:
    global _worker
    _worker = worker
