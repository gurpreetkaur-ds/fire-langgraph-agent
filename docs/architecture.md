# Architecture

Fire Agent is a document-aware AI assistant: a LangGraph agent that routes each question to retrieval over the
user's own documents (RAG), to the original multi-agent research pipeline, or to a direct answer, streams the answer
with verifiable citations, and records every run for observability and automatic evaluation.

```
Browser (static/, no build step)
  │  cookie session + X-Requested-With (CSRF)            SSE: node · token · answer · final · done
  ▼
FastAPI  app/main.py ─ api_auth · api_chat · api_documents · api_activity · eval_api (admin)
  │                         │                    │
  │                         ▼                    ▼
  │                 WorkflowRun (runner.py)   uploads ─► data/uploads/<user>/<doc>.<ext>
  │                         │                    │
  │                         ▼                    ▼
  │                 LangGraph (graph.py)     IngestionWorker (rag/ingest.py)
  │                         │                 extract ► clean ► chunk ► embed ► store
  │   FireChatModel ◄───────┤                    │
  │   (/v1/chat, /v1/chat/stream)                │
  │   FireEmbeddings (/v1/embed) ◄───────────────┤
  ▼                                              ▼
PostgreSQL + pgvector  (SQLite + numpy in development)
  users · user_sessions · conversations · messages · documents · chunks(vector 1536)
  agent_traces(+route, timeline) · agent_steps · tool_calls · evaluation_runs · evaluation_metrics
                         ▲
Evaluation worker ───────┘  LLM-judge metrics for every run (unchanged from the original project)
```

## How the existing agent and the RAG system work together

The original agent (Decomposer → Research agent → Synthesizer) was **kept, not replaced**. It is one branch of a larger
graph. A new *Query Analyzer* node looks at the question, the recent conversation and the list of the user's indexed
documents, and picks a route:

| Route | When | Path |
|---|---|---|
| `rag` | the question is about the user's documents | rewrite_query → retrieve → rerank → build_context → generate_answer → validate_citations |
| `research` | a substantial general question with several parts | decompose → answer → synthesize (the original pipeline) |
| `direct` | greetings, follow-ups, formatting, simple questions | direct_answer |

The analyzer also rewrites follow-ups into a standalone question ("what about outside the EU?" → "What do I need to
work from outside the EU?"), which both the retriever and the research pipeline use. Retrieval is never forced on
questions that don't need it, and it is impossible when the user has no documents or has set the knowledge-base mode
to *Off*. Mode *Always* (or attaching files to a message) forces the RAG route.

The original `/api/analyze/stream` endpoint and the evaluation dataset runner enter the graph directly at `decompose`
(`force_route="research"`), so their event stream and behaviour are exactly what they were before.

## LangGraph workflow

```
                 ┌──────────────► rewrite_query ► retrieve ─┬─► rerank ► build_context ► generate_answer ─┬─► validate_citations ► END
START ─► analyze_query ──(route)──┤                          └─(error)► END                            └─(error)► END
   │             ├──────────────► decompose ► answer ► synthesize ► END
   │             └──────────────► direct_answer ► END
   └─(force_route="research")───► decompose
```

State (`AgentState` in `app/graph.py`) carries the question, history, user id, KB mode, document filter, the route,
search queries, retrieved / reranked chunks, the labelled sources and context, citations and a grounded flag. Every node
is wrapped by `observed(...)`, which records start/finish on the trace timeline and emits `node` events to the browser
with an observable one-line summary (counts, routes, timings), never prompts or model reasoning.

| Node | Kind | What it does |
|---|---|---|
| analyze_query | LLM (Haiku) | route + standalone question + short reason (JSON); heuristic fallback if unparseable |
| rewrite_query | LLM (Haiku) | 1–3 search queries (paraphrase + keyword variants); falls back to the original question |
| retrieve | tools | `embed_query` (one Fire `/v1/embed` batch) + `vector_search` (top `RAG_CANDIDATE_K` per query, user-scoped) |
| rerank | deterministic | absolute threshold, relative cutoff, hybrid score (0.8 cosine + 0.2 IDF term overlap), near-duplicate removal, top-k |
| build_context | deterministic | labels passages S1…Sn, fits them into `RAG_CONTEXT_CHARS` |
| generate_answer | LLM (Sonnet, streamed) | answers only from sources with `[S#]` markers, or states the documents don't contain it |
| validate_citations | deterministic | removes markers for sources that weren't provided, strips citations from "not in your documents" sentences, extracts `{claim, source_id}` pairs, flags answers that cite nothing |
| decompose / answer / synthesize | LLM | the original research pipeline (sub-questions are now answered concurrently; synthesis streams) |
| direct_answer | LLM (Sonnet, streamed) | conversational answer with history |

Errors: a failed rewriter degrades to the original query; a failed embedding call or LLM call ends the run with a
user-safe message (never the raw provider error or any key); a database error during vector search degrades to "no
context". Embedding calls retry 429/5xx/network errors with exponential backoff, and each retry is recorded on the timeline.

## RAG architecture

**Ingestion** (`app/rag/`): upload → validation → background indexing.

1. *Validation* (`extract.detect_type`): extension allow-list **and** content sniffing (PDF magic; DOCX must be a ZIP
   containing `word/document.xml` and expanding to < 120 MB; text types must decode and contain no NUL bytes), size limit,
   per-user document/storage quota, SHA-256 de-duplication per user. Files are stored as `<user_id>/<doc_id>.<ext>`
   (mode 0600), never under the client's filename.
2. *Extraction*: PDF per page (pypdf; page numbers kept, encrypted/scanned PDFs fail with a clear message); DOCX in
   document order with headings and tables; Markdown split at headings (not inside code fences); CSV rows rendered as
   `Row n: column: value; …` in groups of 25 rows, with header detection that survives mixed-type columns; TXT with
   encoding fallback.
3. *Cleaning*: Unicode NFKC, control characters removed, hyphenated line breaks re-joined, whitespace normalised.
4. *Chunking* (`chunking.py`): `RAG_CHUNK_SIZE` characters with `RAG_CHUNK_OVERLAP` overlap, splitting at paragraph →
   line → sentence → word boundaries; chunks never cross a page/section, so each has exactly one page or heading.
5. *Metadata*: document id/name/type/size/hash/upload date/page count/title/author/CSV columns; per chunk the index,
   page, section, character offsets, owner.
6. *Embedding*: Fire `/v1/embed`, `text-embedding-3-small` at 1536 dimensions, batches of 64, L2-normalised; the model and
   dimension are stored on the document.
7. *Storage*: `chunks.embedding vector(1536)` in PostgreSQL (pgvector); status `queued → processing → ready | failed`,
   interrupted jobs are re-queued on startup, failed documents can be re-indexed.

**Retrieval**: exact cosine search in SQL (`embedding <=> query`), always filtered by `user_id` and `status='ready'`,
optionally by document ids (attachments, "Ask about this document") or file type. There is deliberately no ANN index:
with pgvector < 0.8 a filtered HNSW scan can return fewer than *k* rows, which would silently hide a user's documents;
the user_id b-tree index plus an exact scan is correct and fast at this scale. Add HNSW with iterative scans (pgvector ≥ 0.8)
when one user has hundreds of thousands of chunks.

**Grounding & citations**: sources are labelled S1…Sn in the prompt, which tells the model that text inside sources is
data, not instructions (prompt-injection hygiene). The validator guarantees every citation the user sees points to a
passage that was actually retrieved; the UI renders each `[S#]` as a chip that opens that exact passage (with its
neighbours) in a preview drawer. If nothing passes the thresholds, the generator gets a "no context" prompt and must say
the documents don't contain the answer; anything from general knowledge goes in a separate "Not from your documents"
paragraph.

## Database schema (migration `0002`)

| Table | Key columns |
|---|---|
| `users` | id, username (unique), display_name, password_hash (scrypt), preferences JSON, created_at, last_login_at |
| `user_sessions` | token_hash (SHA-256 of the cookie, PK), user_id → users, expires_at |
| `conversations` | id, user_id → users, title, created_at, updated_at |
| `messages` | id, conversation_id → conversations, role, content, status (complete/stopped/error), meta JSON (route, sources, citations, retrieval stats, usage, timings), trace_id |
| `documents` | id, user_id → users, filename, file_type, size_bytes, sha256, storage_path, status, error, page_count, char_count, chunk_count, embedding_model, embedding_dimensions, embedding_tokens, meta JSON, attempts, created/updated/indexed_at |
| `chunks` | id, document_id → documents (cascade), user_id, chunk_index, page, section, text, char_start, char_end, embedding vector(1536) |
| `agent_traces` (+2 columns) | route, timeline JSON `[{kind, node, status, started_ms, duration_ms, summary}]` |

The evaluation tables from `0001` are unchanged. All user-owned rows cascade-delete with their parent.

## Chat streaming protocol (`POST /api/chat/stream`)

`meta {conversation_id, user_message_id, trace_id}` → `node {node, status: running|done|error, duration_ms, summary, data}`
(repeated) → `token {text}` (repeated) → `answer {text}` (RAG only: the validated answer replaces the streamed text) →
`final {answer, route, sources, citations, grounded, retrieval, usage, duration_ms, …}` → `done {message_id}`; or `error {message}`.
**Stop** aborts the request; the server keeps the partial answer as a `stopped` message. **Regenerate** deletes the last
answer and re-runs the last question.

## Security model

- Accounts with scrypt-hashed passwords; HttpOnly + SameSite=Lax session cookie (`Secure` in production); session tokens are
  stored hashed; password change revokes other sessions. Sign-up requires the invite code (`AGENT_ACCESS_TOKEN`) when set.
- CSRF: every state-changing route also requires `X-Requested-With: fetch` (forces a CORS preflight that is never granted).
- Isolation: every query on conversations, messages, documents, chunks, files and activity filters on the owner; a foreign
  id returns 404. Tests cover each route with a second user.
- Rate limits (per user / per IP): chat, uploads, semantic search, sign-in, sign-up.
- Body-size limits enforced while the request streams in (`app/bodylimit.py`): 64 KB for API calls, 10 × `MAX_UPLOAD_MB`
  for uploads; per-file size, per-user document/storage quotas, and `MAX_CHUNKS_PER_DOCUMENT` (a cost guard that refuses
  documents that would need thousands of embeddings).
- Strict CSP (`script-src 'self'`, all libraries vendored), `frame-ancestors 'none'`, `nosniff`; Markdown is rendered through
  DOMPurify with an allow-list; downloads are served as attachments with `Content-Security-Policy: sandbox`.
- The browser never receives prompts, LLM input/output previews, API keys or provider error bodies. The admin Evaluation
  Center remains behind `EVAL_ADMIN_TOKEN`.

## Observability

Structured `event=… key=value` logs with `trace_id` on every line; per run: node timeline, LLM calls (model, latency,
tokens), tool calls (embedding and vector-search latency, hit counts), retries and errors. The **Agent activity** page
shows the signed-in user's own runs; the **Evaluation Center** (admin) adds LLM-judge scores across all users.
