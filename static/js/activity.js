// Agent Activity: the user's own executions, with a node timeline, tool calls, LLM calls and token usage.
import { api } from "./api.js";
import { NODE_LABEL, ROUTE_LABEL, clear, fmt, h, icon } from "./ui.js";

let root, statsEl, routesEl, listEl, detailEl, routeFilter = "", selected = null, mounted = false;

export function mountActivity(container) {
  root = container;
  statsEl = h("div.act-grid");
  routesEl = h("div.card.pad-sm");
  listEl = h("div.run-list");
  detailEl = h("div");
  const seg = h("div.seg", [["", "All"], ["rag", "Knowledge base"], ["research", "Research"], ["direct", "Direct"]].map(([v, l]) =>
    h("button" + (v === routeFilter ? ".on" : ""), { type: "button", dataset: { v }, on: { click: () => {
      routeFilter = v; seg.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.v === v)); load();
    } } }, l)));
  root.append(h("div.page",
    h("div.page-head", h("div", h("h2", "Agent activity"), h("p", "Every workflow you ran: which LangGraph nodes executed, tools, retrieval and model latency, tokens and errors.")),
      h("button.btn.sm", { type: "button", on: { click: () => load() } }, icon("refresh", "sm"), "Refresh")),
    statsEl,
    h("div.act-layout",
      h("div", h("div.toolbar", seg), routesEl, h("div", { style: { height: "12px" } }), listEl),
      detailEl)));
  mounted = true;
}

export async function showActivity() { if (mounted) await load(); }

async function load() {
  try {
    const data = await api(`/api/activity?limit=40${routeFilter ? "&route=" + routeFilter : ""}`);
    renderSummary(data.summary);
    renderList(data.items);
    if (!selected && data.items.length) openRun(data.items[0].trace_id);
    if (!data.items.length) clear(detailEl);
  } catch (e) {
    clear(listEl).append(h("div.banner.err", icon("alert", "sm"), h("span", e.message)));
  }
}

function renderSummary(s) {
  const tile = (k, ic, v, n) => h("div.stat", h("div.k", icon(ic, "sm"), k), h("div.v", v), n ? h("div.n", n) : null);
  clear(statsEl).append(
    tile("Runs (7 days)", "activity", fmt.num(s.runs)),
    tile("Avg total time", "clock", fmt.ms(s.avg_duration_ms)),
    tile("Avg retrieval", "db", fmt.ms(s.avg_retrieval_ms), "embedding + vector search"),
    tile("Avg LLM time", "zap", fmt.ms(s.avg_llm_ms), "sum of model calls per run"),
    tile("Tokens (7 days)", "layers", fmt.num(s.total_tokens)),
    tile("Errors", "alert", fmt.num(s.errors), s.runs ? fmt.pct(s.errors / s.runs) + " of runs" : null));
  const routes = Object.entries(s.by_route || {});
  const max = Math.max(1, ...routes.map(([, r]) => r.runs));
  clear(routesEl).append(h("div.p-title", h("span", "Routes taken (7 days)")),
    routes.length ? h("div.route-bars", routes.sort((a, b) => b[1].runs - a[1].runs).map(([k, r]) => h("div.route-bar",
      h("span.ellipsis", ROUTE_LABEL[k] || k), h("div.track", h("i", { style: { width: (r.runs / max) * 100 + "%" } })),
      h("span.faint.num", { style: { textAlign: "right" } }, `${r.runs} · ${fmt.ms(r.avg_duration_ms)}`))))
      : h("div.panel-empty", "No runs yet. Ask the agent something."));
}

function renderList(items) {
  clear(listEl);
  if (!items.length) { listEl.append(h("div.card", h("div.empty-state", icon("activity"), h("h3", "No activity yet"), h("p", "Runs appear here as soon as you chat with the agent.")))); return; }
  items.forEach((t) => listEl.append(h("button.run" + (t.trace_id === selected ? ".active" : ""), { type: "button", dataset: { id: t.trace_id }, on: { click: () => openRun(t.trace_id) } },
    h("div.q", t.question),
    h("div.d", fmt.ms(t.duration_ms)),
    h("div.m", t.route ? h("span.pill", ROUTE_LABEL[t.route] || t.route) : null, t.stopped ? h("span.pill.warn", "stopped") : t.success ? null : h("span.pill.err", "error"),
      h("span", fmt.rel(t.created_at)), h("span", `${t.llm_calls} LLM · ${t.tool_calls} tools`)),
    h("div.d.faint", fmt.num(t.total_tokens) + " tok"))));
}

async function openRun(id) {
  selected = id;
  listEl.querySelectorAll(".run").forEach((b) => b.classList.toggle("active", b.dataset.id === id));
  clear(detailEl).append(h("div.card", h("div.skeleton", { style: { height: "220px" } })));
  let t;
  try { t = await api(`/api/activity/${encodeURIComponent(id)}`); }
  catch (e) { clear(detailEl).append(h("div.banner.err", icon("alert", "sm"), h("span", e.message))); return; }

  const total = Math.max(t.duration_ms || 1, ...(t.timeline || []).map((e) => e.started_ms + e.duration_ms));
  const nodes = (t.timeline || []).filter((e) => e.kind === "node");
  const events = (t.timeline || []).filter((e) => e.kind !== "node");
  const bar = (start, dur, cls) => h("div.g-track", h("div.g-bar" + (cls ? "." + cls : ""), { style: { left: (start / total) * 100 + "%", width: Math.max(0.6, (dur / total) * 100) + "%" } }));
  const stat = (k, v) => h("div.p-stat", h("div.k", k), h("div.v", v));

  clear(detailEl).append(h("div.card",
    h("div.row", { style: { marginBottom: "6px", alignItems: "flex-start" } }, h("div.grow", h("div.faint", { style: { fontSize: "12px" } }, fmt.date(t.created_at)),
      h("h3", { style: { margin: "2px 0 0", fontSize: "15px", lineHeight: 1.4 } }, t.question)),
      t.route ? h("span.pill.accent", ROUTE_LABEL[t.route] || t.route) : null),
    t.error ? h("div.banner.err", { style: { margin: "10px 0" } }, icon("alert", "sm"), h("span", t.error)) : null,
    h("div.p-stats", { style: { gridTemplateColumns: "repeat(4, minmax(0,1fr))", margin: "14px 0 18px" } },
      stat("Total", fmt.ms(t.duration_ms)), stat("LLM time", fmt.ms(t.llm_ms)), stat("Retrieval", fmt.ms(t.retrieval_ms)), stat("Tokens", fmt.num(t.total_tokens))),
    h("div.p-title", h("span", "Execution timeline"), h("span", fmt.ms(total))),
    nodes.length ? h("div.gantt",
      nodes.map((e) => h("div.g-row", h("span.gl", { title: e.summary }, NODE_LABEL[e.node] || e.node), bar(e.started_ms, e.duration_ms, e.status !== "ok" ? "err" : ""), h("span.gd", fmt.ms(e.duration_ms)))),
      t.tools.map((x) => h("div.g-row", h("span.gl", "↳ " + x.tool), h("div.g-track"), h("span.gd", fmt.ms(x.latency_ms)))),
      events.map((e) => h("div.g-row", h("span.gl", { title: e.summary }, (e.kind === "retry" ? "↻ retry: " : "⚠ ") + (NODE_LABEL[e.node] || e.node)), bar(e.started_ms, 0, "evt"), h("span.gd", e.kind))))
      : h("div.panel-empty", "No node timeline was recorded for this run (it ran before observability was added)."),
    nodes.length ? h("div", { style: { margin: "10px 0 18px" } }, nodes.filter((e) => e.summary).map((e) =>
      h("div", { style: { fontSize: "12.5px", color: "var(--text-2)", padding: "2px 0" } }, h("b", { style: { color: "var(--text)", fontWeight: 550 } }, (NODE_LABEL[e.node] || e.node) + ": "), e.summary))) : null,
    h("div.p-title", h("span", "Model calls")),
    t.llm_steps.length ? h("div", { style: { overflowX: "auto" } }, h("table.plain",
      h("thead", h("tr", h("th", "Agent"), h("th", "Model"), h("th", "Latency"), h("th", "In"), h("th", "Out"), h("th", ""))),
      h("tbody", t.llm_steps.map((s) => h("tr", h("td", s.agent), h("td.faint", s.model), h("td.num", fmt.ms(s.latency_ms)), h("td.num", fmt.num(s.input_tokens)), h("td.num", fmt.num(s.output_tokens)),
        h("td", s.success ? h("span.pill.ok", "ok") : h("span.pill.err", { title: s.error || "" }, "failed"))))))) : h("div.panel-empty", "No model calls."),
    h("div.p-title", { style: { marginTop: "18px" } }, h("span", "Tool invocations")),
    t.tools.length ? h("div", { style: { overflowX: "auto" } }, h("table.plain",
      h("thead", h("tr", h("th", "Tool"), h("th", "Input"), h("th", "Output"), h("th", "Latency"))),
      h("tbody", t.tools.map((x) => h("tr", h("td", x.tool, x.success ? null : h("span.pill.err", { style: { marginLeft: "6px" } }, "failed")),
        h("td.mono", JSON.stringify(x.input || {})), h("td.mono", JSON.stringify(x.output || {})), h("td.num", fmt.ms(x.latency_ms))))))) : h("div.panel-empty", "No tools were called."),
    t.evaluation ? h("div", h("div.p-title", { style: { marginTop: "18px" } }, h("span", "Automatic evaluation"), h("span", t.evaluation.status)),
      h("div.p-stats", { style: { gridTemplateColumns: "repeat(5, minmax(0,1fr))" } }, Object.entries(t.evaluation.scores).map(([k, v]) => stat(k.replace("_", " "), v == null ? "—" : fmt.pct(v)))))
      : null,
    h("div.faint.mono", { style: { marginTop: "14px" } }, t.trace_id)));
}
