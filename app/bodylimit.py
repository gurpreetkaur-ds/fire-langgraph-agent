"""Request body size limits enforced inside the app (pure ASGI, streaming).

The reverse proxy's LimitRequestBody turned out not to apply to these proxied requests, and Starlette's multipart parser
spools a whole upload to disk before a route runs, so the cap must be enforced while the body streams in. Requests
whose Content-Length is too large are rejected immediately; chunked bodies are counted and cut off at the limit.
"""
from __future__ import annotations

import json

from .settings import get_app_settings

DEFAULT_LIMIT = 64 * 1024  # JSON API calls (a 4000-character question is ~16 KB even fully escaped)
UPLOAD_PATH = "/api/documents"
MAX_FILES = 10


class BodyTooLarge(Exception):
    pass


def limit_for(path: str, method: str) -> int:
    if method == "POST" and path == UPLOAD_PATH:
        return get_app_settings().max_upload_bytes * MAX_FILES + 1024 * 1024  # files + multipart overhead
    return DEFAULT_LIMIT


class BodySizeLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] in ("GET", "HEAD", "OPTIONS"):
            return await self.app(scope, receive, send)
        limit = limit_for(scope["path"], scope["method"])
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    if int(value) > limit:
                        return await _reject(send, limit)
                except ValueError:
                    return await _reject(send, limit, status=400, message="Invalid Content-Length.")
        seen = 0
        started = False

        async def limited_receive():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > limit:
                    raise BodyTooLarge()
            return message

        async def tracking_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except BodyTooLarge:
            if not started:
                await _reject(send, limit)


async def _reject(send, limit: int, status: int = 413, message: str = "") -> None:
    mb = limit / (1024 * 1024)
    detail = message or (f"Request body too large (limit {mb:.0f} MB)." if mb >= 1 else f"Request body too large (limit {limit // 1024} KB).")
    body = json.dumps({"detail": detail}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()), (b"connection", b"close")]})
    await send({"type": "http.response.body", "body": body})
