"""Small in-process sliding-window rate limiter.

The service runs as a single uvicorn worker (deploy/agent-api.service), so process memory is the right scope.
With several workers or hosts this would need a shared store (e.g. Redis).
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException


class RateLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window_seconds: float, *, message: str = "Too many requests. Please slow down.") -> None:
        """Record a hit for `key`; raise HTTP 429 if it exceeds `limit` hits within the window."""
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and q[0] <= now - window_seconds:
                q.popleft()
            if len(q) >= limit:
                retry = max(1, int(q[0] + window_seconds - now) + 1)
                raise HTTPException(429, message, headers={"Retry-After": str(retry)})
            q.append(now)
            if len(self._hits) > 50_000:  # bound memory: forget idle keys
                for k in [k for k, v in self._hits.items() if not v]:
                    del self._hits[k]

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


limiter = RateLimiter()
