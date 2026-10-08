# Instructor Report — LangGraph Agent Routed Through Fire

## What I Asked For

A small LangGraph agent with at least two meaningful steps (Question Decomposer →
Research/Answer Agent → combined Final Answer), with every LLM call routed through
Fire's `/v1/chat` endpoint (no direct OpenAI/Anthropic/Gemini calls), a custom Fire
wrapper compatible with LangChain/LangGraph, strict API key hygiene (`.env` only,
never logged/displayed/committed), a modern/polished web UI showing the workflow
live, basic testing, phone-accessible deployment, and this report plus a short
LangGraph-vs-Claude-Code comparison.

## What I Built

**Architecture:** `Browser → FastAPI → LangGraph → FireChatModel wrapper → Fire /v1/chat → LLM`

- **`app/fire_llm.py`** — `FireChatModel`, a `langchain_core.BaseChatModel` subclass.
  It is the only code path in the project that makes an outbound LLM call. It reads
  `FIRE_API_KEY`/`FIRE_BASE_URL` from the environment, builds the Fire `/v1/chat`
  request (`messages`, `system_prompt`, `species_name`, `temperature`, `max_tokens`),
  and raises a clean `FireAPIError` (status code + truncated body, never the key) on
  any failure.
- **`app/graph.py`** — A 3-node LangGraph `StateGraph`:
  `START → decompose → answer → synthesize → END`.
  - `decompose`: `claude-haiku-4-5` (via Fire) breaks the question into 2–4
    sub-questions (JSON output, parsed defensively with a regex fallback; degrades
    to the original question if parsing fails).
  - `answer`: `claude-sonnet-4-5` (via Fire) answers each sub-question in turn.
  - `synthesize`: `claude-sonnet-4-5` (via Fire) combines all Q/A pairs into one
    final answer.
  - Any `FireAPIError` at any node short-circuits the remaining nodes and surfaces
    a plain-language error instead of crashing.
- **`app/main.py`** — FastAPI app. `POST /api/analyze/stream` runs
  `graph.astream(..., stream_mode="updates")` and forwards each node's output to the
  browser as Server-Sent Events, so the UI lights up stage-by-stage in real time
  rather than waiting for the whole pipeline. `GET /api/health` reports only whether
  `FIRE_API_KEY` is *present* (boolean), never its value.
- **`static/`** — A dark, glassmorphism dashboard (no build step, plain HTML/CSS/JS):
  question input with example-question chips, a live 3-stage workflow visualization
  with two distinct SVG bot avatars (cyan "Decomposer Bot", magenta "Answer Bot"),
  progressive reveal of sub-questions / sub-answers / final answer, an error banner,
  a Fire-connectivity status pill, and a "How It Works" section with an architecture
  diagram.

**Two distinct Fire models** are used deliberately (`claude-haiku-4-5` for fast
decomposition, `claude-sonnet-4-5` for research/synthesis) so the two agent steps are
visibly and functionally different, not just two calls to the same model.

## What Works (actually tested, not assumed)

- `GET /v1/capabilities` fetched and used to pick real `species_name` values and the
  correct `/v1/chat` request/response shape — nothing guessed.
- Fire auth confirmed two ways: a placeholder key correctly got `HTTP 401
  REVOKED_CREDENTIAL` from Fire (proves the wrapper sends the header correctly and
  surfaces errors cleanly); the real key then returned `HTTP 200` on every call.
- Full end-to-end run with the real key and the suggested example question ("What
  are the main differences between RAG and fine-tuning...") — 6 real Fire calls (1
  decompose + 4 answer + 1 synthesize), all `200 OK`, producing 4 sensible
  sub-questions, 4 solid answers, and a well-organized final synthesis.
- LangGraph executed all 3 nodes in order; `stream_mode="updates"` correctly emitted
  one SSE event per node.
- UI loads (`GET /` → 200), static assets load, health pill reflects key presence,
  example chips populate the textarea, and submitting streams and renders each stage
  live in the browser.
- Empty-question input is rejected with a clean in-UI error, no crash.
- Security audit (see below) passed: no key in any HTTP response, log line, source
  file, or git history.

## What Broke During Development

- `python3 -m venv` failed — `ensurepip` wasn't installed on the system Python and
  there was no passwordless `sudo` to install `python3.12-venv`. Fixed by creating
  the venv with `--without-pip` and bootstrapping pip manually via
  `bootstrap.pypa.io/get-pip.py` run inside that venv — no system packages touched.
- First background server start got killed mid-boot by an unrelated `pkill`+`sleep`
  command chain timing out; `ps aux` showed no uvicorn process. Fixed by restarting
  it as a clean detached (`disown`) background process and re-verifying with `ps`.
- A key-leak grep during the pre-commit check matched `fire_sk_your_key_here` in
  `.env.example` and looked like a positive hit at first glance. Re-checked with the
  exact real-key substring to confirm it was the placeholder, not the secret, before
  committing.

## What Is Left / Known Limitations

- **Phone access requires one manual step.** This machine is WSL2 behind Windows'
  NAT (WSL IP `172.20.x.x` vs. the Windows Wi-Fi IP `192.168.12.125`), and no
  reverse proxy/tunnel tool (ngrok, cloudflared) is installed. Setting up a Windows
  firewall rule and `netsh interface portproxy` requires an elevated PowerShell
  command on the Windows side — a host-level network/firewall change — which I did
  not run unilaterally. See **Demo** below for the exact command and the no-setup
  alternative.
- The decomposer's JSON output is parsed defensively (regex fallback, graceful
  degradation to the original question), but it is still a text-based contract with
  the model rather than a guaranteed structured output — in rare cases a malformed
  response could still yield an unhelpful sub-question split.
- No automated test suite (pytest) was written; verification was done via live
  `curl`/browser testing against the real Fire API, which is more meaningful for a
  demo but doesn't give regression protection.
- Bot avatars are hand-built inline SVG, not Fire-generated images — a deliberate
  choice to avoid extra image-generation cost/latency for a 3-minute demo; they are
  still two distinct, professional-looking illustrations.

## Architecture (recap)

```
Browser → Web App (FastAPI) → LangGraph → Fire LLM Wrapper → Fire /v1/chat → LLM
                                   ↑___________________________________|
                            (3 nodes: decompose → answer → synthesize)
```

## Demo

- **Code:** `/home/gurpr/fire-langgraph-agent` (git repo, `main`/`master` branch has
  one commit; `.env` is untracked and gitignored)
- **Run it:** `cd /home/gurpr/fire-langgraph-agent && .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8787`
  (already running during this session)
- **On this machine:** http://localhost:8787
- **From the phone (same Wi-Fi), one-time setup:** run this once in an **elevated**
  PowerShell on Windows (not from WSL — this changes the host firewall/NAT, so it's
  left for you to run intentionally):
  ```powershell
  netsh interface portproxy add v4tov4 listenport=8787 listenaddress=0.0.0.0 connectport=8787 connectaddress=172.20.145.172
  New-NetFirewallRule -DisplayName "Fire LangGraph Demo" -Direction Inbound -LocalPort 8787 -Protocol TCP -Action Allow
  ```
  Then open `http://192.168.12.125:8787` on the phone (must be on the same Wi-Fi).
  If the WSL IP changes after a reboot, re-run the first command with the new IP
  (check with `hostname -I` inside WSL).
- **No-setup fallback:** present directly from the laptop screen at
  `http://localhost:8787` — fully functional, just not on the phone.

## Fire Security

`FIRE_API_KEY` is stored only in a local `.env` file (permissions `600`), is listed
in `.gitignore`, is never committed, and was verified absent from: every HTTP
response the server sends, every log line, every source file, and the full git
history. The UI and API only ever expose a boolean "is a key configured" — never the
key itself. The key is not reproduced anywhere in this report.
