"""Evaluation settings, read from environment variables (never hardcoded)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Quality metrics (higher is better) and the one where lower is better.
QUALITY_METRICS = (
    "faithfulness",
    "relevance",
    "groundedness",
    "citation_accuracy",
    "tool_accuracy",
    "safety",
)
LOWER_IS_BETTER = ("hallucination",)
ALL_METRICS = QUALITY_METRICS + LOWER_IS_BETTER

_DEFAULT_THRESHOLDS = {
    "faithfulness": 0.80,
    "relevance": 0.75,
    "groundedness": 0.70,
    "citation_accuracy": 0.80,
    "tool_accuracy": 0.80,
    "safety": 0.95,
    "hallucination": 0.15,  # maximum allowed
}
# Absolute score drop (0-1) tolerated between two runs before a metric is "degraded".
_DEFAULT_TOLERANCE = {m: 0.01 for m in ALL_METRICS} | {"safety": 0.005}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_pricing() -> Dict[str, Dict[str, float]]:
    """EVAL_PRICING_JSON: {"model": {"input_per_mtok": 3.0, "output_per_mtok": 15.0}}.

    No built-in prices: cost is reported as unknown (null) for models not listed.
    """
    raw = os.environ.get("EVAL_PRICING_JSON", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("EVAL_PRICING_JSON is not valid JSON") from exc
    return {str(k): {"input_per_mtok": float(v["input_per_mtok"]), "output_per_mtok": float(v["output_per_mtok"])} for k, v in data.items()}


@dataclass(frozen=True)
class Settings:
    database_url: str
    auto_migrate: bool
    enabled: bool
    worker_mode: str  # "inline" (runs inside the API process) or "external" (python -m evaluation.worker)
    worker_poll_seconds: float
    stale_running_seconds: float
    admin_token: Optional[str]
    judge_model: Optional[str]
    judge_timeout: float
    suite_concurrency: int
    latency_regression_pct: float
    thresholds: Dict[str, float] = field(default_factory=dict)
    tolerances: Dict[str, float] = field(default_factory=dict)
    pricing: Dict[str, Dict[str, float]] = field(default_factory=dict)
    max_text_chars: int = 20000

    @classmethod
    def from_env(cls) -> "Settings":
        thresholds = {m: _env_float(f"EVAL_THRESHOLD_{m.upper()}", _DEFAULT_THRESHOLDS[m]) for m in ALL_METRICS}
        tolerances = {m: _env_float(f"EVAL_REGRESSION_TOLERANCE_{m.upper()}", _DEFAULT_TOLERANCE[m]) for m in ALL_METRICS}
        mode = os.environ.get("EVAL_WORKER_MODE", "inline").strip().lower()
        if mode not in {"inline", "external"}:
            raise ValueError("EVAL_WORKER_MODE must be 'inline' or 'external'")
        return cls(
            database_url=os.environ.get("EVAL_DATABASE_URL") or f"sqlite:///{BASE_DIR / 'data' / 'evaluation.db'}",
            auto_migrate=_env_bool("EVAL_AUTO_MIGRATE", True),
            enabled=_env_bool("EVAL_ENABLED", True),
            worker_mode=mode,
            worker_poll_seconds=_env_float("EVAL_WORKER_POLL_SECONDS", 2.0),
            stale_running_seconds=_env_float("EVAL_STALE_RUNNING_SECONDS", 600.0),
            admin_token=os.environ.get("EVAL_ADMIN_TOKEN") or None,
            judge_model=os.environ.get("EVAL_JUDGE_MODEL") or None,
            judge_timeout=_env_float("EVAL_JUDGE_TIMEOUT_SECONDS", 90.0),
            suite_concurrency=max(1, int(_env_float("EVAL_SUITE_CONCURRENCY", 3))),
            latency_regression_pct=_env_float("EVAL_REGRESSION_LATENCY_PCT", 0.20),
            thresholds=thresholds,
            tolerances=tolerances,
            pricing=_load_pricing(),
        )


def get_settings() -> Settings:
    return Settings.from_env()
