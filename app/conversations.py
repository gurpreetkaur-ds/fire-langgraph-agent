"""Conversation and message persistence. Every function is scoped to the owning user."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import delete, func, or_, select

from evaluation.db import session_scope
from evaluation.models import utcnow

from .models import Conversation, Message


class ConversationNotFound(LookupError):
    pass


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def message_dict(m: Message) -> dict:
    return {"id": m.id, "role": m.role, "content": m.content, "status": m.status, "meta": m.meta or {},
            "trace_id": m.trace_id, "created_at": _iso(m.created_at)}


def conversation_dict(c: Conversation, message_count: Optional[int] = None, snippet: Optional[str] = None) -> dict:
    out = {"id": c.id, "title": c.title, "created_at": _iso(c.created_at), "updated_at": _iso(c.updated_at)}
    if message_count is not None:
        out["message_count"] = message_count
    if snippet:
        out["snippet"] = snippet
    return out


def make_title(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= 60 else text[:57].rsplit(" ", 1)[0] + "…"


def _owned(db, user_id: str, conversation_id: str) -> Conversation:
    c = db.scalar(select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user_id))
    if c is None:
        raise ConversationNotFound(conversation_id)
    return c


def list_conversations(user_id: str, q: Optional[str] = None, limit: int = 100) -> List[dict]:
    with session_scope() as db:
        counts = (select(Message.conversation_id, func.count(Message.id).label("n")).group_by(Message.conversation_id).subquery())
        stmt = (select(Conversation, func.coalesce(counts.c.n, 0))
                .outerjoin(counts, counts.c.conversation_id == Conversation.id)
                .where(Conversation.user_id == user_id))
        if q:
            like = f"%{q.lower()}%"
            matching = select(Message.conversation_id).where(func.lower(Message.content).like(like))
            stmt = stmt.where(or_(func.lower(Conversation.title).like(like), Conversation.id.in_(matching)))
        rows = db.execute(stmt.order_by(Conversation.updated_at.desc()).limit(limit)).all()
        out = []
        for c, n in rows:
            snippet = None
            if q:
                m = db.scalar(select(Message.content).where(Message.conversation_id == c.id, func.lower(Message.content).like(f"%{q.lower()}%")).limit(1))
                if m:
                    i = m.lower().find(q.lower())
                    snippet = ("…" if i > 40 else "") + m[max(0, i - 40): i + 80].replace("\n", " ") + "…"
            out.append(conversation_dict(c, n, snippet))
        return out


def create_conversation(user_id: str, title: str) -> dict:
    now = utcnow()
    with session_scope() as db:
        c = Conversation(user_id=user_id, title=make_title(title) or "New conversation", created_at=now, updated_at=now)
        db.add(c)
        db.flush()
        return conversation_dict(c, 0)


def get_conversation(user_id: str, conversation_id: str) -> dict:
    with session_scope() as db:
        c = _owned(db, user_id, conversation_id)
        return {**conversation_dict(c, len(c.messages)), "messages": [message_dict(m) for m in c.messages]}


def rename_conversation(user_id: str, conversation_id: str, title: str) -> dict:
    title = " ".join((title or "").split())[:200]
    if not title:
        raise ValueError("Title cannot be empty.")
    with session_scope() as db:
        c = _owned(db, user_id, conversation_id)
        c.title = title
        return conversation_dict(c)


def delete_conversation(user_id: str, conversation_id: str) -> None:
    with session_scope() as db:
        c = _owned(db, user_id, conversation_id)
        db.delete(c)


def add_message(user_id: str, conversation_id: str, role: str, content: str, *, status: str = "complete",
                meta: Optional[dict] = None, trace_id: Optional[str] = None) -> dict:
    now = utcnow()
    with session_scope() as db:
        c = _owned(db, user_id, conversation_id)
        m = Message(conversation_id=c.id, role=role, content=content, status=status, meta=meta or {}, trace_id=trace_id, created_at=now)
        db.add(m)
        c.updated_at = now
        db.flush()
        return message_dict(m)


def history_before(user_id: str, conversation_id: str, before_message_id: Optional[str], turns: int) -> List[dict]:
    """The last `turns` user/assistant pairs before a message (complete answers only, oldest first)."""
    if turns <= 0:
        return []
    with session_scope() as db:
        c = _owned(db, user_id, conversation_id)
        msgs = list(c.messages)
    if before_message_id:
        idx = next((i for i, m in enumerate(msgs) if m.id == before_message_id), len(msgs))
        msgs = msgs[:idx]
    msgs = [m for m in msgs if m.role == "user" or (m.role == "assistant" and m.status == "complete" and m.content)]
    return [{"role": m.role, "content": m.content} for m in msgs[-2 * turns:]]


def prepare_regenerate(user_id: str, conversation_id: str) -> dict:
    """Remove the assistant replies after the last user message and return that user message."""
    with session_scope() as db:
        c = _owned(db, user_id, conversation_id)
        msgs = list(c.messages)
        last_user = next((m for m in reversed(msgs) if m.role == "user"), None)
        if last_user is None:
            raise ValueError("There is no question to regenerate.")
        after = [m.id for m in msgs if m.role == "assistant" and m.created_at >= last_user.created_at and msgs.index(m) > msgs.index(last_user)]
        if after:
            db.execute(delete(Message).where(Message.id.in_(after)))
        return message_dict(last_user)
