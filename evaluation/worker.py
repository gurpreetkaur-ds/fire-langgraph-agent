"""Background evaluation worker.

The database is the queue: `persist_trace` writes `pending` rows, workers claim them atomically. In
`inline` mode the API process runs one worker task (woken immediately on enqueue); in `external` mode run
`python -m evaluation.worker` as a separate process/container. Both use the same code path.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from . import store
from .config import get_settings
from .engine import EvaluationEngine

logger = logging.getLogger("fire_agent.eval.worker")


async def process_evaluation(engine: EvaluationEngine, evaluation_id: str) -> None:
    """Evaluate one claimed run and persist the outcome. Never raises."""
    try:
        data = await asyncio.to_thread(store.load_eval_input, evaluation_id)
        results = await engine.evaluate(data)
        await asyncio.to_thread(store.save_results, evaluation_id, results, trace_success=data.success)
        logger.info(
            "event=evaluation_done evaluation_id=%s trace_id=%s metrics=%s",
            evaluation_id, data.trace_id,
            ",".join(f"{r.metric}:{r.status}" for r in results),
        )
    except Exception as exc:
        logger.exception("event=evaluation_failed evaluation_id=%s", evaluation_id)
        try:
            await asyncio.to_thread(store.mark_failed, evaluation_id, f"{type(exc).__name__}: {exc}")
        except Exception:
            logger.exception("event=evaluation_mark_failed_error evaluation_id=%s", evaluation_id)


class EvaluationWorker:
    def __init__(self, engine: Optional[EvaluationEngine] = None):
        self._engine = engine
        self._wake = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    @property
    def engine(self) -> EvaluationEngine:
        if self._engine is None:
            self._engine = EvaluationEngine()
        return self._engine

    def notify(self) -> None:
        """Wake the worker immediately instead of waiting for the next poll."""
        self._wake.set()

    async def drain(self) -> int:
        """Process everything currently pending; returns how many runs were handled."""
        handled = 0
        while True:
            evaluation_id = await asyncio.to_thread(store.claim_next_pending)
            if evaluation_id is None:
                return handled
            await process_evaluation(self.engine, evaluation_id)
            handled += 1

    async def run_forever(self) -> None:
        poll = get_settings().worker_poll_seconds
        logger.info("event=worker_started poll_seconds=%s", poll)
        while not self._stop.is_set():
            try:
                await self.drain()
            except Exception:
                logger.exception("event=worker_loop_error")
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=poll)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
        logger.info("event=worker_stopped")

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self.run_forever(), name="evaluation-worker")

    async def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._task:
            await self._task
            self._task = None


_worker: Optional[EvaluationWorker] = None


def get_worker() -> Optional[EvaluationWorker]:
    return _worker


def set_worker(worker: Optional[EvaluationWorker]) -> None:
    global _worker
    _worker = worker


def main() -> None:  # pragma: no cover - thin CLI wrapper
    from .db import init_engine, run_migrations
    from .tracing import setup_logging

    setup_logging()
    settings = get_settings()
    if settings.auto_migrate:
        run_migrations()
    init_engine()
    asyncio.run(EvaluationWorker().run_forever())


if __name__ == "__main__":  # pragma: no cover
    main()
