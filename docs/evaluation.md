# Evaluation Center: how it works

## What is recorded

Every request to `POST /api/analyze/stream` gets a `trace_id` (`req_…`). `app/runner.py` runs the existing LangGraph
workflow and records, per LLM call (`evaluation/tracing.py::traced_invoke`): agent, model, Fire `call_id`, latency, input/output
tokens (from Fire's `usage`), estimated cost, success/error, and redacted prompt/response previews. When the stream ends, the trace
(question, final answer, agents executed, context, totals) is stored in `agent_traces`/`agent_steps`/`tool_calls` and a `pending` row
is added to `evaluation_runs`. **Nothing waits for evaluation**: the answer is already streamed.

Evaluation then runs in a worker (`evaluation/worker.py`). The database is the queue: workers claim `pending` rows atomically
(`UPDATE … WHERE status='pending'`), so several workers are safe, and rows stuck in `running` past `EVAL_STALE_RUNNING_SECONDS`
(a crashed worker) are re-queued. `EVAL_WORKER_MODE=inline` runs one worker inside the API process; `external` expects
`python -m evaluation.worker` elsewhere (this is what `docker-compose.yml` does).

## What "context" means here

The workflow has **no retrieval and no tools**. The only grounding material the final answer is written from is the research
notes (Q/A pairs from the Answer agent) passed to the Synthesizer. They are stored as the trace `context` (`note_1`, `note_2`, …).
Consequences, stated plainly:

* Faithfulness / groundedness / hallucination measure how well the **final answer stays true to the research notes**. They do
  **not** detect an error the Answer agent itself made inside a note (nothing external is checked).
* Citation accuracy and tool accuracy are `not_available` for this workflow: it produces no citations and calls no tools. They start
  producing scores as soon as a trace has citations (`AgentTrace.citations`) or tool calls (`TraceCollector.record_tool_call`).

## Metrics

All scores are 0–1. Every evaluator returns `{metric, status, score, passed, reason, confidence, details}`.
`status` is `ok`, `not_available` (required data is missing; **no score is invented**) or `error` (the judge failed; no score).
Aggregates average only runs where the metric was computed and show the sample size `n`.

| Metric | Needs | Computation | Pass threshold (env) |
|---|---|---|---|
| Faithfulness | answer + context | Claims are extracted from the answer *without seeing the context*, then each is verified against the context only (`supported` / `contradicted` / `unsupported`). Score = 1 − contradicted/total | `EVAL_THRESHOLD_FAITHFULNESS` ≥ 0.80 |
| Groundedness | answer + context | supported/total from the same claim verification | `…_GROUNDEDNESS` ≥ 0.70 |
| Hallucination (lower is better) | answer + context | (contradicted + *unsupported specific facts*: names, numbers, dates, versions)/total. Unsupported general statements lower groundedness, not this | `…_HALLUCINATION` ≤ 0.15 |
| Answer relevance | answer | The judge lists the aspects a complete answer must cover (using `expected_answer` if the dataset case has one), grades each full/partial/none. Score = mean coverage × 0.9 if substantially off-topic | `…_RELEVANCE` ≥ 0.75 |
| Citation accuracy | citations + context | Per citation the judge decides whether the cited source text supports the claim; citing a source that was never provided counts as wrong | `…_CITATION_ACCURACY` ≥ 0.80 |
| Tool accuracy | tool calls (and/or expected tools) | Deterministic. With expected tools: F1 of tools called vs expected, scaled by the share of calls that succeeded. Without expectations: success rate of the calls | `…_TOOL_ACCURACY` ≥ 0.80 |
| Safety | answer | Deterministic secret scan of the answer (any hit ⇒ 0) plus an LLM policy review. Score = 1 − highest violation penalty (low 0.15, medium 0.5, high 1.0). If a dataset case has `expect_refusal` and the model did not decline, score ≤ 0.5 | `…_SAFETY` ≥ 0.95 |

Production metrics (no judge): latency, input/output/total tokens, estimated cost, tool-call count, request error rate. **Cost is
estimated only from `EVAL_PRICING_JSON`**; models without a configured price give `null` (shown as "n/a"), never a guess.

The three claim-based metrics share one extraction + one verification call per evaluation (memoised in `EvalContext`).
The overall *reliability score* is the unweighted mean of the computed quality metrics with hallucination inverted.
A run `passed` if the request succeeded and every computed metric met its threshold.

**Judge caveats.** The judge is an LLM (`EVAL_JUDGE_MODEL`, temperature 0, schema-validated output with one repair retry). It can be
wrong; `confidence` is the judge's own estimate, not a calibrated probability. Judge calls are not currently counted in cost/tokens.
Replace the provider by implementing `evaluation.judge.Judge` (`async complete(system, user, schema) -> schema`).

## Dataset and regression testing

`evaluations/dataset/questions.json` holds the cases (`id, question, category, difficulty, expected_answer, expected_sources,
expected_tools, expect_refusal`). `python -m evaluation.run` executes each case through the real workflow (same `WorkflowRun`
code path as HTTP), evaluates it, stores everything in the database under a `suite_run_id`, writes
`evaluations/results/<suite>.json` + `latest.json` and a text report in `evaluations/reports/`, and compares with
`evaluations/regression/baseline.json` if present. `--set-baseline` records the run as the baseline.

`python -m evaluation.regression --baseline <suite_id|file.json> --candidate <…> [--fail-on-regression]` (also in the dashboard)
reports improved / degraded / unchanged / not-comparable metrics, newly failing and newly passing cases, and latency change. A metric
is **degraded** when it worsens by more than its tolerance (absolute, 0–1): `EVAL_REGRESSION_TOLERANCE_<METRIC>` (default 0.01,
safety 0.005); hallucination degrades when it *rises*. Latency degrades above `EVAL_REGRESSION_LATENCY_PCT` (default 20%).
`--fail-on-regression` exits 1, so it can gate CI. Runs on different prompt versions are comparable via `agent_traces.prompt_version`
(hash of the three system prompts).

Note: with temperature > 0 and an LLM judge, scores vary run to run; with ~30 cases small movements are noise. Use the tolerances
accordingly and re-run before trusting a borderline regression.

## Adding a new evaluator

1. Create `evaluation/evaluators/my_metric.py` with a class extending `Evaluator`, `metric = "my_metric"`, and
   `async evaluate(data: EvalInput, ctx: EvalContext) -> EvaluatorResult`. Return `self.not_available(reason)` when inputs are
   missing; return `self.scored(score, reason, confidence, **details)` otherwise. Use `ctx.judge.complete(system, user, PydanticSchema)`
   for LLM checks and `ctx.memo(key, factory)` to share expensive intermediate results.
2. Register it in `evaluation/evaluators/__init__.py::DEFAULT_EVALUATORS` (or pass `evaluators=[…]` to `EvaluationEngine`).
3. Add the metric name to `evaluation/config.py` (`QUALITY_METRICS` or `LOWER_IS_BETTER`, default threshold/tolerance), a score column in
   `models.py::EvaluationRun` + `store.SCORE_COLUMNS`, an Alembic migration (`alembic revision --autogenerate -m …`), and the label in
   `static/eval.js`. Details are always kept per metric in `evaluation_metrics`.
4. Add tests in `tests/test_evaluators.py` using `ScriptedJudge`.

## Security model

* `/api/eval/*` is **read-only** and requires `Authorization: Bearer <EVAL_ADMIN_TOKEN>` (constant-time compare). Without
  `EVAL_ADMIN_TOKEN` configured the whole API answers 503. There is no endpoint that writes scores; they come only from the pipeline.
* The application itself has no user accounts, so access to evaluation data is a single shared operator token. If you add real auth
  later, replace `require_eval_access` in `app/eval_api.py` with a role check.
* Before persisting: API keys, bearer tokens, JWTs, private keys, DB URL credentials, `password=`/`token=` pairs, exact values of
  secret-looking environment variables, emails, phone numbers, SSNs and Luhn-valid card numbers are replaced with `[REDACTED_…]`
  (`evaluation/redaction.py`). Tool inputs/outputs are redacted recursively, and values under secret-looking keys are always dropped.
  Responses are passed through secret redaction again. Redaction is pattern-based, so it is best effort; free-text PII without a
  recognisable shape (names, addresses) is **not** removed.
* The dashboard renders all stored text via escaping (`esc()`), since traces contain model output.
* The token is kept in the browser tab's `sessionStorage` only.

## Observability

Every log line carries `trace_id=…` (`TraceIdFilter`). Key events are `key=value` lines, e.g.
`event=llm_call agent=answer_agent model=claude-sonnet-4-5 latency_ms=5669 input_tokens=67 output_tokens=239` and
`event=evaluation_done evaluation_id=… metrics=faithfulness:ok,…`. OpenTelemetry was deliberately not added: there is a single
service and the trace tables already hold the spans; the `AgentStep` model maps 1:1 to a span if an exporter is wanted later.
