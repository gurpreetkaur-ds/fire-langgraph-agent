"""Run the evaluation dataset through the real workflow and evaluate every answer.

    python -m evaluation.run [--dataset PATH] [--limit N] [--category C] [--baseline REF] [--set-baseline]
                             [--fail-on-regression] [--concurrency N]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from . import store
from .aggregate import SuiteReport, build_suite_report
from .config import BASE_DIR, get_settings
from .dataset import EvalCase, load_dataset
from .db import init_engine, run_migrations, session_scope
from .engine import EvaluationEngine
from .regression import LABELS, compare, load_report, render_markdown, render_text
from .tracing import setup_logging
from .worker import process_evaluation

logger = logging.getLogger("fire_agent.eval.run")
RESULTS_DIR = BASE_DIR / "evaluations" / "results"
REPORTS_DIR = BASE_DIR / "evaluations" / "reports"
BASELINE_PATH = BASE_DIR / "evaluations" / "regression" / "baseline.json"


async def run_case(case: EvalCase, suite_id: str, engine: EvaluationEngine, sem: asyncio.Semaphore) -> None:
    from app.runner import WorkflowRun  # imported lazily: pulls in the graph / Fire client

    async with sem:
        run = WorkflowRun(case.question, expected=case.expected(), suite_run_id=suite_id, case_id=case.id, claim_evaluation=True)
        async for _ in run.events():
            pass
        evaluation_id = await run.finalize()
        if evaluation_id is None:
            logger.error("case=%s evaluation disabled or trace persistence failed", case.id)
            return
        await process_evaluation(engine, evaluation_id)
        print(f"  finished {case.id}", flush=True)


def _pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v * 100:5.1f}%"


def render_summary(report: SuiteReport) -> str:
    L = ["Evaluation Run", "─" * 28, f"{'Suite:':<22}{report.suite_run_id}", f"{'Cases:':<22}{len(report.cases)}", ""]
    for m in ("faithfulness", "relevance", "groundedness", "citation_accuracy", "tool_accuracy", "safety", "hallucination"):
        d = report.metrics[m]
        label = "Hallucination Rate" if m == "hallucination" else LABELS[m]
        note = f"   (n={d['n']})" if d["n"] != len(report.cases) else ""
        L.append(f"{label + ':':<22}{_pct(d['avg']):>7}{note}")
    L += [""]
    L.append(f"{'Average Latency:':<22}{'n/a' if report.avg_latency_ms is None else f'{report.avg_latency_ms / 1000:.2f} sec'}")
    L.append(f"{'Average Cost:':<22}{'n/a (no pricing configured)' if report.avg_cost is None else f'${report.avg_cost:.4f}'}")
    L += ["", f"{'Passed:':<22}{report.passed}/{len(report.cases)}", f"{'Failed:':<22}{report.failed}/{len(report.cases)}"]
    if report.not_evaluated:
        L.append(f"{'Not evaluated:':<22}{report.not_evaluated}/{len(report.cases)}")
    L.append("\nMetrics marked n<cases were 'not_available' for some cases (e.g. no tools / citations exist in this workflow).")
    failed = [c for c in report.cases if c.passed is not True]
    if failed:
        L.append("\nCases not passed: " + ", ".join(c.case_id for c in failed))
    return "\n".join(L)


async def run_suite(cases: List[EvalCase], concurrency: int) -> SuiteReport:
    suite_id = "suite_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    engine = EvaluationEngine()
    sem = asyncio.Semaphore(concurrency)
    print(f"Running {len(cases)} cases (concurrency={concurrency}) as {suite_id}")
    await asyncio.gather(*(run_case(c, suite_id, engine, sem) for c in cases))
    with session_scope() as db:
        report = build_suite_report(db, suite_id)
    if report is None:
        raise SystemExit("No results were stored; check the logs above.")
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run the evaluation dataset")
    ap.add_argument("--dataset", type=Path)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--category")
    ap.add_argument("--concurrency", type=int)
    ap.add_argument("--baseline", help="suite_run_id or JSON path to compare against (default: evaluations/regression/baseline.json)")
    ap.add_argument("--set-baseline", action="store_true", help="store this run as the new regression baseline")
    ap.add_argument("--fail-on-regression", action="store_true")
    args = ap.parse_args(argv)

    setup_logging(logging.WARNING)
    settings = get_settings()
    if settings.auto_migrate:
        run_migrations()
    init_engine()
    if not settings.enabled:
        raise SystemExit("EVAL_ENABLED is false; enable evaluation to run the suite.")

    dataset = load_dataset(args.dataset)
    cases = [c for c in dataset.cases if not args.category or c.category == args.category]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No cases selected.")

    report = asyncio.run(run_suite(cases, args.concurrency or settings.suite_concurrency))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump_json(indent=2)
    (RESULTS_DIR / f"{report.suite_run_id}.json").write_text(payload)
    (RESULTS_DIR / "latest.json").write_text(payload)
    summary = render_summary(report)
    (REPORTS_DIR / f"{report.suite_run_id}.md").write_text("```\n" + summary + "\n```\n")
    print("\n" + summary)

    exit_code = 0
    baseline_ref = args.baseline or (str(BASELINE_PATH) if BASELINE_PATH.exists() else None)
    if baseline_ref:
        reg = compare(load_report(baseline_ref), report, settings)
        print("\n" + render_text(reg))
        (REPORTS_DIR / f"{report.suite_run_id}_regression.md").write_text(render_markdown(reg))
        if args.fail_on_regression and reg.has_regression:
            exit_code = 1
    else:
        print("\nNo baseline found; use --set-baseline to record this run as the baseline.")
    if args.set_baseline:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(payload)
        print(f"Baseline updated: {BASELINE_PATH.relative_to(BASE_DIR)}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
