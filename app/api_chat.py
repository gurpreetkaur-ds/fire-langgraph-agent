"""Conversations API and the streaming chat endpoint (the routed LangGraph workflow).

POST /api/chat/stream streams Server-Sent Events:
  meta    {conversation_id, user_message_id, conversation}
  node    {node, status: running|done|error, duration_ms?, summary?, data?}   observable workflow state
  token   {text}                                                              answer deltas
  answer  {text}                                                              replaces streamed text (after citation validation)
  final   {answer, route, sources, citations, grounded, retrieval, usage, trace_id, duration_ms, ...}
  done    {message_id}
  error   {message}
Stopping: the browser aborts the request; the partial answer is saved with status "stopped".
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from evaluation.tracing import log_event

from . import conversations as convs
from .auth import CurrentUser, current_user, write_user
from .rag import store as kb_store
from .ratelimit import limiter
from .runner import WorkflowRun
from .settings import get_app_settings

logger = logging.getLogger("fire_agent.chat")
router = APIRouter(prefix="/api", tags=["chat"])
_background: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)


def _not_found(fn, *args):
    try:
        return fn(*args)
    except convs.ConversationNotFound:
        raise HTTPException(404, "Conversation not found.")


class ChatIn(BaseModel):
    message: Optional[str] = Field(default=None, max_length=4000)
    conversation_id: Optional[str] = Field(default=None, max_length=40)
    regenerate: bool = False
    kb_mode: Literal["auto", "always", "off"] = "auto"
    document_ids: List[str] = Field(default_factory=list, max_length=20)


class RenameIn(BaseModel):
    title: str = Field(max_length=200)


@router.get("/conversations")
def list_conversations(q: Optional[str] = Query(None, max_length=100), user: CurrentUser = Depends(current_user)):
    return convs.list_conversations(user.id, (q or "").strip() or None)


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, user: CurrentUser = Depends(current_user)):
    return _not_found(convs.get_conversation, user.id, conversation_id)


@router.patch("/conversations/{conversation_id}")
def rename_conversation(conversation_id: str, body: RenameIn, user: CurrentUser = Depends(write_user)):
    try:
        return _not_found(convs.rename_conversation, user.id, conversation_id, body.title)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, user: CurrentUser = Depends(write_user)):
    _not_found(convs.delete_conversation, user.id, conversation_id)
    return {"ok": True}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


@router.post("/chat/stream")
async def chat_stream(body: ChatIn, user: CurrentUser = Depends(write_user)):
    settings = get_app_settings()
    limiter.check(f"chat:{user.id}", settings.chat_per_minute, 60, message="You're sending messages too quickly. Please wait a moment.")

    # --- resolve the question and conversation (all ownership-checked) ---
    if body.regenerate:
        if not body.conversation_id:
            raise HTTPException(422, "conversation_id is required to regenerate.")
        try:
            user_msg = await asyncio.to_thread(_not_found, convs.prepare_regenerate, user.id, body.conversation_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        conversation_id, question = body.conversation_id, user_msg["content"]
        conversation = None
    else:
        question = (body.message or "").strip()
        if not question:
            raise HTTPException(422, "Message cannot be empty.")
        if body.conversation_id:
            await asyncio.to_thread(_not_found, convs.get_conversation, user.id, body.conversation_id)
            conversation_id, conversation = body.conversation_id, None
        else:
            conversation = await asyncio.to_thread(convs.create_conversation, user.id, question)
            conversation_id = conversation["id"]
        user_msg = await asyncio.to_thread(convs.add_message, user.id, conversation_id, "user", question)

    history = await asyncio.to_thread(convs.history_before, user.id, conversation_id, user_msg["id"], settings.history_turns)
    # metadata filter: only the user's own documents can be selected
    doc_ids: List[str] = []
    if body.document_ids:
        owned = {d["id"] for d in await asyncio.to_thread(kb_store.ready_documents, user.id, 1000)}
        doc_ids = [d for d in dict.fromkeys(body.document_ids) if d in owned]

    run = WorkflowRun(question, session_id=conversation_id, user_id=user.id, force_route=None, history=history,
                      kb_mode=body.kb_mode, document_ids=doc_ids)

    async def persist_assistant(status: str, final: Optional[dict]) -> None:
        result = final or run.result()
        meta = {k: result.get(k) for k in ("route", "route_reason", "sources", "citations", "grounded", "retrieval",
                                           "usage", "duration_ms", "search_queries", "sub_questions", "models")}
        if status == "error":
            meta["error"] = final.get("error_message") if final else None
        content = result.get("answer") or ""
        try:
            msg = await asyncio.to_thread(convs.add_message, user.id, conversation_id, "assistant", content,
                                          status=status, meta=meta, trace_id=run.trace_id)
            return msg
        except convs.ConversationNotFound:
            return None  # deleted while answering
        finally:
            await run.finalize()

    async def event_gen():
        final: Optional[dict] = None
        status = "complete"
        yield _sse({"type": "meta", "conversation_id": conversation_id, "user_message_id": user_msg["id"],
                    "conversation": conversation, "trace_id": run.trace_id, "regenerate": body.regenerate})
        try:
            async for event in run.stream():
                if event.get("type") == "final":
                    final = event
                    yield _sse(event)
                elif event.get("type") == "error":
                    status = "error"
                    final = {**run.result(), "error_message": event.get("message")}
                    yield _sse(event)
                else:
                    yield _sse(event)
            msg = await persist_assistant(status, final)
            yield _sse({"type": "done", "message_id": msg["id"] if msg else None, "status": status})
        except (asyncio.CancelledError, GeneratorExit):
            # client pressed Stop / disconnected: keep what was generated so far
            run.stopped = True
            log_event(logger, "chat_stopped_by_client", partial_chars=len("".join(run.partial)))
            _spawn(persist_assistant("stopped", None))
            raise

    headers = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}
    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)
