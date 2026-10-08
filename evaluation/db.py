"""Database engine/session management (SQLAlchemy; SQLite by default, any SQLAlchemy URL via EVAL_DATABASE_URL)."""
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .config import BASE_DIR, get_settings

logger = logging.getLogger("fire_agent.eval.db")

_lock = threading.Lock()
_engine: Optional[Engine] = None
_factory: Optional[sessionmaker] = None


def init_engine(url: Optional[str] = None) -> Engine:
    """(Re)initialise the global engine. Tests pass an explicit URL."""
    global _engine, _factory
    with _lock:
        if _engine is not None:
            _engine.dispose()
        url = url or get_settings().database_url
        kwargs: dict = {"future": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
            if ":memory:" not in url:
                Path(url.replace("sqlite:///", "", 1)).parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):

            @event.listens_for(_engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _record):  # pragma: no cover - trivial
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()

        _factory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
        return _engine


def get_engine() -> Engine:
    return _engine or init_engine()


@contextmanager
def session_scope() -> Iterator[Session]:
    if _factory is None:
        init_engine()
    assert _factory is not None
    session = _factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def run_migrations(url: Optional[str] = None) -> None:
    """Apply Alembic migrations up to head."""
    from alembic import command
    from alembic.config import Config

    logging.getLogger("alembic").setLevel(logging.WARNING)  # keep startup logs quiet

    cfg = Config(str(BASE_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BASE_DIR / "migrations"))
    cfg.set_main_option("sqlalchemy.url", (url or get_settings().database_url).replace("%", "%%"))
    command.upgrade(cfg, "head")
