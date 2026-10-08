# Fire LangGraph Agent + AI Evaluation Center

A multi-agent LangGraph workflow (every LLM call routed through Fire's `/v1/chat`) with a production-style LLM evaluation
platform: request tracing, modular evaluators, a background evaluation worker, a regression-testing suite, and an in-app
**Evaluation Center** dashboard.

## Architecture

```
Browser ──► FastAPI (app/main.py) ──► WorkflowRun (app/runner.py) ──► LangGraph (app/graph.py)
                │                          │  trace_id, per-call tracing         decompose ► answer ► synthesize
                │                          ▼                                         │ every LLM call
                │                 persist trace (redacted)                           ▼
                │                          │                              FireChatModel ► Fire /v1/chat
                │                          ▼
                │                evaluation_runs (status=pending)  ◄── the database is the queue
                │                          │
                │                          ▼
                │              Evaluation worker (inline or separate process)
                │                EvaluationEngine ─ Faithfulness · Relevance · Groundedness · Citation
                │                                    Tool accuracy · Hallucination · Safety  (LLM judge via Fire)
                │                          ▼
                └── /api/eval/* (token-protected, read-only) ◄── SQLite/any SQLAlchemy DB ──► Evaluation Center UI
```

**Multi-agent workflow** (unchanged): *Decomposer* (splits the question into 2–4 sub-questions) → *Answer agent* (answers each) →
*Synthesizer* (writes the final answer). The answer is streamed to the user first; tracing/evaluation never add user-facing latency.

**Evaluation architecture.** After the stream ends the trace is stored and an evaluation is queued. A worker claims it, runs all
evaluators concurrently, and stores per-metric results (`evaluation_metrics`) plus denormalised scores for fast dashboards
(`evaluation_runs`). Tables: `agent_traces`, `agent_steps`, `tool_calls`, `evaluation_runs`, `evaluation_metrics` (Alembic migration `0001`).

**Metrics:** faithfulness, answer relevance, groundedness, citation accuracy, tool accuracy, hallucination rate, safety; plus latency,
tokens, estimated cost, tool-call count and error rate. Metrics that cannot be computed are reported as `not_available`, never
invented. For this workflow citation and tool accuracy are `not_available` (it has no citations or tools); faithfulness,
groundedness and hallucination measure the final answer against the **research notes**, not against external truth. Full method:
[docs/evaluation.md](docs/evaluation.md).

## Local setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
cp .env.example .env        # set FIRE_API_KEY and EVAL_ADMIN_TOKEN (python -c "import secrets;print(secrets.token_urlsafe(32))")
.venv/bin/uvicorn app.main:app --port 8000     # migrations run automatically on startup
```

Open http://127.0.0.1:8000 → **Agent** tab to ask questions, **Evaluation Center** tab (enter `EVAL_ADMIN_TOKEN`) for the dashboard.

## Tests

```bash
.venv/bin/python -m pytest            # unit + API + authorization + redaction + migrations + end-to-end flow (LLM calls are faked)
```

## Running the evaluation suite and regression tests

```bash
.venv/bin/python -m evaluation.run                       # all cases in evaluations/dataset/questions.json (calls Fire!)
.venv/bin/python -m evaluation.run --limit 5 --category security
.venv/bin/python -m evaluation.run --set-baseline        # save this run as evaluations/regression/baseline.json
.venv/bin/python -m evaluation.run --fail-on-regression  # compare with the baseline; exit 1 on regression (CI)
.venv/bin/python -m evaluation.regression --baseline suite_A --candidate suite_B
```

Outputs: `evaluations/results/<suite>.json`, `latest.json`, `evaluations/reports/<suite>.md`. The suite makes real Fire calls
(roughly 4–8 workflow calls and ~4 judge calls per case); check your quota first.

## Dashboard

Overall reliability, per-metric scores vs. targets, counts, latency/cost, trend charts (scores, latency, tokens, cost, failure and
hallucination rate), per-agent statistics with drill-down, a filterable trace list (agent, model, date range, status, failed metric,
trace ID), a trace detail view (execution flow, steps, context, final answer, per-metric reasons), and regression comparison between suite runs.

![Evaluation Center](docs/img/evaluation-center.png)
![Trace detail](docs/img/trace-detail.png)

## Observability & security (summary)

Every log line has `trace_id=`; key events are `event=… key=value`. Secrets and common PII are redacted before storage; the
evaluation API is read-only, closed unless `EVAL_ADMIN_TOKEN` is set, and requires a bearer token. See [docs/evaluation.md](docs/evaluation.md#security-model).

## Production deployment

```bash
docker compose up --build        # api (EVAL_WORKER_MODE=external) + eval-worker sharing a volume
```

* One image, two processes: `uvicorn app.main:app` and `python -m evaluation.worker`. Migrations run at startup
  (`EVAL_AUTO_MIGRATE`) or explicitly with `alembic upgrade head`.
* SQLite is fine for a single host. For multiple replicas use a server database: set `EVAL_DATABASE_URL` and install its driver.
* Put the app behind TLS; the evaluation token and Fire key travel in headers. Configure `EVAL_PRICING_JSON` to get cost figures.
* Set `EVAL_ENABLED=false` to turn evaluation off entirely (traces are still stored).

Environment variables are documented in `.env.example`.
