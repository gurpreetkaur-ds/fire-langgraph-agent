"""Application settings for accounts, conversations and the RAG pipeline, read from the environment.

Nothing here is a secret except where noted; secrets (FIRE_API_KEY, AGENT_ACCESS_TOKEN) are only ever checked
for presence and never returned to clients.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from evaluation.config import BASE_DIR

# Allowed upload types: extension -> canonical type. Content is verified separately (see app/rag/extract.py).
ALLOWED_TYPES = {".pdf": "pdf", ".docx": "docx", ".txt": "txt", ".md": "md", ".markdown": "md", ".csv": "csv"}


def _int(name: str, default: int, lo: int | None = None, hi: int | None = None) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        val = int(raw) if raw else default
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc
    if lo is not None and val < lo or hi is not None and val > hi:
        raise ValueError(f"Environment variable {name} must be between {lo} and {hi}, got {val}")
    return val


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}") from exc


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    return default if not raw else raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppSettings:
    # accounts
    cookie_secure: bool
    session_days: int
    allow_signup: bool
    # uploads
    upload_dir: Path
    max_upload_bytes: int
    max_documents_per_user: int
    max_storage_bytes_per_user: int
    max_pdf_pages: int
    max_chunks_per_document: int  # cost guard: embedding a huge file is refused instead of silently spending credit
    # chunking
    chunk_size: int  # characters
    chunk_overlap: int  # characters
    # embeddings
    embedding_model: str
    embedding_dimensions: int
    embedding_batch_size: int
    # retrieval
    top_k: int
    candidate_k: int
    similarity_threshold: float
    relative_cutoff: float  # keep chunks scoring >= this fraction of the best chunk's score (0 disables)
    context_char_budget: int
    history_turns: int
    # rate limits (per user)
    chat_per_minute: int
    uploads_per_hour: int

    @classmethod
    def from_env(cls) -> "AppSettings":
        chunk_size = _int("RAG_CHUNK_SIZE", 1200, 200, 8000)
        overlap = _int("RAG_CHUNK_OVERLAP", 200, 0, 4000)
        if overlap >= chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")
        top_k = _int("RAG_TOP_K", 6, 1, 20)
        return cls(
            cookie_secure=_bool("COOKIE_SECURE", False),
            session_days=_int("SESSION_DAYS", 14, 1, 365),
            allow_signup=_bool("ALLOW_SIGNUP", True),
            upload_dir=Path(os.environ.get("UPLOAD_DIR") or BASE_DIR / "data" / "uploads"),
            max_upload_bytes=_int("MAX_UPLOAD_MB", 15, 1, 200) * 1024 * 1024,
            max_documents_per_user=_int("MAX_DOCUMENTS_PER_USER", 200, 1),
            max_storage_bytes_per_user=_int("MAX_STORAGE_MB_PER_USER", 250, 1) * 1024 * 1024,
            max_pdf_pages=_int("MAX_PDF_PAGES", 600, 1),
            max_chunks_per_document=_int("MAX_CHUNKS_PER_DOCUMENT", 2000, 1),
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small").strip(),
            # Fixed by the database schema (migration 0002); changing it needs a new migration + re-index.
            embedding_dimensions=_int("EMBEDDING_DIMENSIONS", 1536, 32, 2000),
            embedding_batch_size=_int("EMBEDDING_BATCH_SIZE", 64, 1, 128),
            top_k=top_k,
            candidate_k=max(top_k, _int("RAG_CANDIDATE_K", 24, 1, 200)),
            similarity_threshold=_float("RAG_SIMILARITY_THRESHOLD", 0.30),
            relative_cutoff=_float("RAG_RELATIVE_CUTOFF", 0.75),
            context_char_budget=_int("RAG_CONTEXT_CHARS", 12000, 1000, 120000),
            history_turns=_int("CHAT_HISTORY_TURNS", 6, 0, 30),
            chat_per_minute=_int("RATE_LIMIT_CHAT_PER_MINUTE", 12, 1),
            uploads_per_hour=_int("RATE_LIMIT_UPLOADS_PER_HOUR", 60, 1),
        )

    def public(self) -> dict:
        """Non-secret configuration shown on the Settings page."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "relative_cutoff": self.relative_cutoff,
            "context_char_budget": self.context_char_budget,
            "max_upload_mb": self.max_upload_bytes // (1024 * 1024),
            "max_documents": self.max_documents_per_user,
            "allowed_types": sorted(set(ALLOWED_TYPES)),
        }


@lru_cache(maxsize=1)
def _cached() -> AppSettings:
    return AppSettings.from_env()


def get_app_settings() -> AppSettings:
    return _cached()


def reload_app_settings() -> AppSettings:
    """Tests change the environment between cases."""
    _cached.cache_clear()
    return _cached()
