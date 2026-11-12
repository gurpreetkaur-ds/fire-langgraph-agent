// Right-hand intelligence panel: observable workflow state of the live (or selected) run.
// Shows routes, node status/timings, tools, retrieval stats and sources: never prompts or model reasoning.
import { api } from "./api.js";
import { openCitation } from "./preview.js";
import { state } from "./state.js";
import { NODE_LABEL, ROUTE_LABEL, clear, fmt, h, icon } from "./ui.js";

const PLAN = {
  rag: ["analyze_query", "rewrite_query", "retrieve", "rerank", "build_context", "generate_answer", "validate_citations"],
  research: ["analyze_query", "decompose", "answer", "synthesize"],
  direct: ["analyze_query", "direct_answer"],
};
const TOOL_NODES = { retrieve: ["embed_query", "vector_search"] };

let timer = null;
const body = () => document.getElementById("panelBody");

export function newRun() {
  return { status: "running", startedAt: performance.now(), durationMs: null, route: null, routeReason: null, nodes: {}, order: [],
           sources: [], retrieval: null, usage: null, traceId: null, queries: [], grounded: undefined, tools: [] };
}

/** Apply one SSE event from /api/chat/stream to a run object. */
export function applyEvent(run, ev) {
  if (ev.type === "meta") run.traceId = ev.trace_id;
  if (ev.type === "node") {
    if (!run.nodes[ev.node]) run.order.push(ev.node);
    run.nodes[ev.node] = { ...(run.nodes[ev.node] || {}), status: ev.status, duration_ms: ev.duration_ms, summary: ev.summary, data: ev.data };
    if (ev.node === "analyze_query" && ev.status === "done" && ev.data) run.route = ev.data.route;
    if (ev.node === "decompose" && !run.route) run.route = "research";
    if (ev.node === "rewrite_query" && ev.status === "done") run.queries = ev.data?.queries || [];
    if (ev.node === "retrieve" && ev.status === "done") {
      run.tools.push({ name: "embed_query", ms: ev.data?.embed_ms, ok: true }, { name: "vector_search", ms: ev.data?.search_ms, ok: true, detail: `${ev.data?.candidates ?? 0} candidates` });
      run.retrieval = { ...(run.retrieval || {}), candidates: ev.data?.candidates, documents: ev.data?.documents };
    }
    if (ev.node === "rerank" && ev.status === "done") run.retrieval = { ...(run.retrieval || {}), ...ev.data };
    if (ev.node === "build_context" && ev.status === "done") run.sources = ev.data?.sources || [];
  }
  if (ev.type === "final") {
    run.status = "done";
    run.route = ev.route; run.routeReason = ev.route_reason; run.sources = ev.sources || run.sources;
    run.retrieval = ev.retrieval || run.retrieval; run.usage = ev.usage; run.durationMs = ev.duration_ms; run.grounded = ev.grounded;
    run.queries = ev.search_queries || run.queries; run.models = ev.models;
  }
  if (ev.type === "error") { run.status = "error"; run.error = ev.message; }
}

/** Build a run view from a stored assistant message (+ its trace timeline, if still available). */
export async function runFromMessage(msg) {
  const m = msg.meta || {};
  const run = { ...newRun(), status: msg.status === "complete" ? "done" : msg.status, route: m.route, routeReason: m.route_reason,
    sources: m.sources || [], retrieval: m.retrieval, usage: m.usage, durationMs: m.duration_ms, traceId: msg.trace_id,
    queries: m.search_queries || [], grounded: m.grounded, models: m.models, error: m.error };
  if (msg.trace_id) {
    try {
      const t = await api(`/api/activity/${encodeURIComponent(msg.trace_id)}`);
      (t.timeline || []).filter((e) => e.kind === "node").forEach((e) => {
        if (!run.nodes[e.node]) run.order.push(e.node);
        run.nodes[e.node] = { status: e.status === "ok" ? "done" : "error", duration_ms: e.duration_ms, summary: e.summary };
      });
      run.tools = (t.tools || []).map((x) => ({ name: x.tool, ms: x.latency_ms, ok: x.success }));
      run.evaluation = t.evaluation;
    } catch { /* trace not persisted yet or deleted: show what the message has */ }
  }
  return run;
}

export function showRun(run) {
  state.run = run;
  render();
  clearInterval(timer);
  if (run && run.status === "running") timer = setInterval(render, 100);
}

export function render() {
  const root = body();
  if (!root) return;
  const run = state.run;
  clear(root);
  if (!run) {
    root.append(h("div.empty-state", icon("sparkle"), h("h3", "Nothing running"),
      h("p", "Ask a question to watch the agent work: routing, retrieval, sources and timings appear here. Select an answer’s details to inspect it later.")));
    return;
  }
  if (run.status !== "running") clearInterval(timer);
  const elapsed = run.durationMs ?? (performance.now() - run.startedAt);
  const statusPill = {
    running: h("span.pill.accent", h("span.dot.pulse"), "Running"), done: h("span.pill.ok", h("span.dot"), "Completed"),
    error: h("span.pill.err", h("span.dot"), "Error"), stopped: h("span.pill.warn", h("span.dot"), "Stopped"),
  }[run.status] || h("span.pill", run.status);

  const stat = (k, v) => h("div.p-stat", h("div.k", k), h("div.v", v));
  root.append(h("div.p-section",
    h("div.p-title", h("span", "Agent status"), statusPill),
    h("div.p-stats",
      stat("Route", run.route ? ROUTE_LABEL[run.route] || run.route : "Analyzing…"),
      stat("Execution", fmt.ms(elapsed)),
      stat("LLM calls", run.usage ? run.usage.llm_calls : "—"),
      stat("Tokens", run.usage ? fmt.num(run.usage.input_tokens + run.usage.output_tokens) : "—")),
    run.routeReason ? h("div.faint", { style: { fontSize: "12px", marginTop: "8px" } }, "Routing: ", run.routeReason) : null,
    run.error ? h("div.banner.err", { style: { marginTop: "10px" } }, icon("alert", "sm"), h("span", run.error)) : null));

  // LangGraph nodes: the planned path for the route, with live status
  const plan = run.route ? PLAN[run.route] : ["analyze_query"];
  const names = [...new Set([...(plan || []), ...run.order])];
  const steps = h("div.steps", names.map((n) => {
    const s = run.nodes[n];
    const st = s ? s.status : run.status === "running" ? "pending" : "skipped";
    const mark = st === "done" ? icon("check", "sm") : st === "running" ? h("span.spinner") : st === "error" ? icon("x", "sm") : null;
    return h("div.step." + st, h("span.st", mark),
      h("div", h("div.sn", NODE_LABEL[n] || n), s?.summary ? h("div.ss", s.summary) : null),
      h("span.sd", s?.duration_ms != null && st !== "running" ? fmt.ms(s.duration_ms) : ""));
  }));
  root.append(h("div.p-section", h("div.p-title", h("span", "LangGraph workflow"), h("span", run.order.length ? `${Object.values(run.nodes).filter((x) => x.status === "done").length} nodes` : "")), steps));

  // tools
  root.append(h("div.p-section", h("div.p-title", h("span", "Tools")),
    run.tools.length ? h("div", run.tools.map((t) => h("div.tool-row", icon(t.name === "vector_search" ? "db" : "layers", "sm"), h("span.grow", t.name), t.detail ? h("span.faint", t.detail) : null,
      h("span.faint.num", fmt.ms(t.ms)), t.ok ? null : h("span.pill.err", "failed"))))
      : h("div.panel-empty", run.route && run.route !== "rag" ? "No tools needed for this route." : run.status === "running" ? "Waiting…" : "No tools were called.")));

  // retrieval
  if (run.route === "rag" || run.retrieval) {
    const r = run.retrieval || {};
    root.append(h("div.p-section", h("div.p-title", h("span", "Retrieval")),
      h("div.p-stats",
        stat("Candidates", r.candidates ?? "—"), stat("Kept chunks", r.kept ?? "—"),
        stat("Too weak", r.below_threshold == null ? "—" : r.below_threshold + (r.below_relative || 0)), stat("Duplicates removed", r.duplicates ?? "—")),
      run.queries?.length ? h("div", { style: { marginTop: "10px" } }, h("div.faint", { style: { fontSize: "12px", marginBottom: "4px" } }, "Search queries"),
        run.queries.map((q) => h("div", { style: { fontSize: "12.5px", color: "var(--text-2)", padding: "2px 0" } }, "· ", q))) : null,
      run.grounded === false ? h("div.banner.err", { style: { marginTop: "10px" } }, icon("alert", "sm"), h("span", "The answer did not cite any retrieved passage.")) : null));
  }

  // sources
  if (run.sources?.length || run.route === "rag") {
    root.append(h("div.p-section", h("div.p-title", h("span", "Sources"), h("span", run.sources.length ? `${run.sources.filter((s) => s.cited !== false).length} cited` : "")),
      run.sources.length ? h("div.source-list", { style: { padding: 0 } }, run.sources.map((s) => sourceButton(s)))
        : h("div.panel-empty", run.status === "running" ? "Waiting for retrieval…" : "No passage was relevant enough to use.")));
  }

  if (run.evaluation) {
    const sc = run.evaluation.scores || {};
    root.append(h("div.p-section", h("div.p-title", h("span", "Evaluation"), h("span", run.evaluation.status)),
      h("div.p-stats", ["faithfulness", "groundedness", "relevance", "citation_accuracy"].map((k) => stat(k.replace("_", " "), sc[k] == null ? "—" : fmt.pct(sc[k]))))));
  }
  if (run.traceId) root.append(h("div.faint.mono", { style: { marginTop: "4px" } }, run.traceId));
}

export function sourceButton(s) {
  const loc = [s.page ? "p. " + s.page : null, s.section].filter(Boolean).join(" · ");
  return h("button.source" + (s.cited === false ? ".uncited" : ""), { type: "button", title: "Open the cited passage", on: { click: () => openCitation(s) } },
    h("span.sid", s.id),
    h("div", { style: { minWidth: 0 } }, h("div.sname", h("span", s.filename), loc ? h("span.sloc", loc) : null), h("div.sprev", s.preview || "")),
    h("span.sscore", { title: "Similarity" }, (s.similarity ?? s.score)?.toFixed?.(2) ?? ""));
}
