"""Compare two evaluation suite runs and flag regressions (thresholds from configuration)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

from .aggregate import SuiteReport, build_suite_report
from .config import ALL_METRICS, LOWER_IS_BETTER, Settings, get_settings
from .db import init_engine, session_scope

LABELS = {
    "faithfulness": "Faithfulness", "relevance": "Relevance", "groundedness": "Groundedness",
    "citation_accuracy": "Citation Accuracy", "tool_accuracy": "Tool Accuracy",
    "hallucination": "Hallucination Rate", "safety": "Safety",
}


class MetricDelta(BaseModel):
    metric: str
    baseline: Optional[float]
    candidate: Optional[float]
    delta: Optional[float]  # candidate - baseline (raw, sign not flipped for hallucination)
    status: str  # improved | degraded | unchanged | not_comparable
    tolerance: float


class RegressionReport(BaseModel):
    baseline_id: str
    candidate_id: str
    metrics: List[MetricDelta]
    latency_change_pct: Optional[float] = None
    latency_degraded: bool = False
    cost_change_pct: Optional[float] = None
    newly_failing: List[str]
    newly_passing: List[str]
    missing_cases: List[str]
    warnings: List[str]

    @property
    def has_regression(self) -> bool:
        return any(m.status == "degraded" for m in self.metrics) or bool(self.newly_failing) or self.latency_degraded

    def by_status(self, status: str) -> List[MetricDelta]:
        return [m for m in self.metrics if m.status == status]


def _pct_change(old: Optional[float], new: Optional[float]) -> Optional[float]:
    if old is None or new is None or old == 0:
        return None
    return (new - old) / old


def compare(baseline: SuiteReport, candidate: SuiteReport, settings: Optional[Settings] = None) -> RegressionReport:
    settings = settings or get_settings()
    deltas: List[MetricDelta] = []
    warnings: List[str] = []
    for m in ALL_METRICS:
        b, c = baseline.metrics[m]["avg"], candidate.metrics[m]["avg"]
        tol = settings.tolerances[m]
        if b is None or c is None:
            deltas.append(MetricDelta(metric=m, baseline=b, candidate=c, delta=None, status="not_comparable", tolerance=tol))
            continue
        delta = c - b
        # for "lower is better" metrics an increase is the bad direction
        worse = delta > tol if m in LOWER_IS_BETTER else delta < -tol
        better = delta < -tol if m in LOWER_IS_BETTER else delta > tol
        status = "degraded" if worse else "improved" if better else "unchanged"
        deltas.append(MetricDelta(metric=m, baseline=b, candidate=c, delta=delta, status=status, tolerance=tol))
        if worse:
            verb = "increased" if m in LOWER_IS_BETTER else "decreased"
            warnings.append(f"WARNING: {LABELS[m]} {'rate' if m in LOWER_IS_BETTER and 'Rate' not in LABELS[m] else 'score'} {verb} by {abs(delta) * 100:.1f}%.")

    base_cases = {c.case_id: c for c in baseline.cases}
    cand_cases = {c.case_id: c for c in candidate.cases}
    newly_failing = sorted(i for i, c in cand_cases.items() if c.passed is False and i in base_cases and base_cases[i].passed is True)
    newly_passing = sorted(i for i, c in cand_cases.items() if c.passed is True and i in base_cases and base_cases[i].passed is False)
    missing = sorted(set(base_cases) - set(cand_cases))

    lat = _pct_change(baseline.avg_latency_ms, candidate.avg_latency_ms)
    lat_bad = lat is not None and lat > settings.latency_regression_pct
    if lat_bad:
        warnings.append(f"WARNING: Average latency increased by {lat * 100:.0f}% (limit {settings.latency_regression_pct * 100:.0f}%).")
    for cid in newly_failing:
        warnings.append(f"WARNING: Case {cid} passed in the baseline but fails now.")
    return RegressionReport(
        baseline_id=baseline.suite_run_id, candidate_id=candidate.suite_run_id, metrics=deltas,
        latency_change_pct=lat, latency_degraded=lat_bad, cost_change_pct=_pct_change(baseline.avg_cost, candidate.avg_cost),
        newly_failing=newly_failing, newly_passing=newly_passing, missing_cases=missing, warnings=warnings,
    )


def _fmt(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def render_text(r: RegressionReport) -> str:
    lines = [f"Regression: {r.baseline_id}  ->  {r.candidate_id}", "─" * 60]
    for label, status in (("Improved", "improved"), ("Degraded", "degraded"), ("Unchanged", "unchanged"), ("Not comparable", "not_comparable")):
        items = r.by_status(status)
        if not items:
            continue
        lines.append(f"{label}:")
        for d in items:
            delta = "" if d.delta is None else f"  ({d.delta * 100:+.1f} pts)"
            lines.append(f"  {LABELS[d.metric]:<20} {_fmt(d.baseline):>7} -> {_fmt(d.candidate):>7}{delta}")
    if r.latency_change_pct is not None:
        lines.append(f"Latency change: {r.latency_change_pct * 100:+.0f}%")
    if r.cost_change_pct is not None:
        lines.append(f"Cost change:    {r.cost_change_pct * 100:+.0f}%")
    lines.append(f"Newly failing cases: {', '.join(r.newly_failing) or 'none'}")
    lines.append(f"Newly passing cases: {', '.join(r.newly_passing) or 'none'}")
    if r.missing_cases:
        lines.append(f"Cases missing from candidate: {', '.join(r.missing_cases)}")
    lines += [""] + r.warnings if r.warnings else ["", "No regressions detected."]
    return "\n".join(lines)


def render_markdown(r: RegressionReport) -> str:
    rows = "\n".join(
        f"| {LABELS[d.metric]} | {_fmt(d.baseline)} | {_fmt(d.candidate)} | {'' if d.delta is None else f'{d.delta * 100:+.1f}'} | {d.status} |"
        for d in r.metrics
    )
    warn = "\n".join(f"- {w}" for w in r.warnings) or "- none"
    return (
        f"# Regression report\n\nBaseline: `{r.baseline_id}`  \nCandidate: `{r.candidate_id}`\n\n"
        f"| Metric | Baseline | Candidate | Δ (pts) | Status |\n|---|---|---|---|---|\n{rows}\n\n"
        f"Newly failing cases: {', '.join(r.newly_failing) or 'none'}  \nNewly passing cases: {', '.join(r.newly_passing) or 'none'}\n\n"
        f"## Warnings\n{warn}\n"
    )


def load_report(ref: str) -> SuiteReport:
    """`ref` is a path to a saved suite JSON, or a suite_run_id stored in the database."""
    path = Path(ref)
    if path.suffix == ".json" and path.exists():
        return SuiteReport.model_validate_json(path.read_text())
    init_engine()
    with session_scope() as db:
        report = build_suite_report(db, ref)
    if report is None:
        raise SystemExit(f"No suite run or file found for {ref!r}")
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compare two evaluation suite runs")
    ap.add_argument("--baseline", required=True, help="suite_run_id or path to a suite JSON")
    ap.add_argument("--candidate", required=True, help="suite_run_id or path to a suite JSON")
    ap.add_argument("--fail-on-regression", action="store_true", help="exit 1 if any regression is found")
    args = ap.parse_args(argv)
    report = compare(load_report(args.baseline), load_report(args.candidate))
    print(render_text(report))
    return 1 if args.fail_on_regression and report.has_regression else 0


if __name__ == "__main__":
    sys.exit(main())
