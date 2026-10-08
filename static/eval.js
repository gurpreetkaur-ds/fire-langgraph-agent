/* AI Evaluation Center. All dynamic text goes through esc() / textContent: traces contain model output. */
(() => {
  const METRICS = [
    ["faithfulness", "Faithfulness"], ["relevance", "Answer Relevance"], ["groundedness", "Groundedness"],
    ["citation_accuracy", "Citation Accuracy"], ["tool_accuracy", "Tool Accuracy"], ["safety", "Safety"],
    ["hallucination", "Hallucination"],
  ];
  const LABEL = Object.fromEntries(METRICS);
  const LOWER_BETTER = new Set(["hallucination"]);
  const SERIES_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500"]; // fixed order, validated on the dark surface
  const PAGE = 15;
  const $ = (id) => document.getElementById(id);
  const state = { token: null, offset: 0, total: 0, thresholds: {}, series: null, table: false };

  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (v, d = 1) => (v == null ? "n/a" : (v * 100).toFixed(d) + "%");
  const ms = (v) => (v == null ? "n/a" : v >= 1000 ? (v / 1000).toFixed(2) + "s" : Math.round(v) + "ms");
  const money = (v) => (v == null ? "n/a" : "$" + (v < 0.01 ? v.toFixed(5) : v.toFixed(4)));
  const num = (v) => (v == null ? "n/a" : Math.round(v).toLocaleString());

  try { state.token = sessionStorage.getItem("evalToken"); } catch { /* storage may be blocked */ }

  // ---------- API ----------
  async function api(path, params) {
    const qs = new URLSearchParams();
    Object.entries(params || {}).forEach(([k, v]) => { if (v !== "" && v != null) qs.set(k, v); });
    const res = await fetch("/api/eval/" + path + (qs.toString() ? "?" + qs : ""), { headers: { Authorization: "Bearer " + state.token } });
    if (res.status === 401 || res.status === 503) {
      const body = await res.json().catch(() => ({}));
      const err = new Error(body.detail || "Not authorized");
      err.auth = true;
      throw err;
    }
    if (!res.ok) throw new Error("Request failed (HTTP " + res.status + ")");
    return res.json();
  }

  function filters() {
    return {
      agent: $("fAgent").value, model: $("fModel").value, status: $("fStatus").value, failed_metric: $("fMetric").value,
      date_from: $("fFrom").value, date_to: $("fTo").value, trace_id: $("fTrace").value.trim(),
    };
  }

  // ---------- gate / tabs ----------
  function show(view) {
    const evalOn = view === "eval";
    $("agentView").hidden = evalOn;
    $("evalView").hidden = !evalOn;
    $("tabAgent").classList.toggle("active", !evalOn);
    $("tabEval").classList.toggle("active", evalOn);
    if (evalOn) enterEval();
  }

  function gate(message) {
    $("evalGate").hidden = false;
    $("evalBody").hidden = true;
    const b = $("evalGateError");
    b.hidden = !message;
    b.textContent = message || "";
  }

  async function enterEval() {
    if (!state.token) return gate();
    try {
      const opts = await api("filters");
      fillSelect($("fAgent"), opts.agents, "All");
      fillSelect($("fModel"), opts.models, "All");
      fillSelect($("fStatus"), opts.statuses, "All");
      fillSelect($("fMetric"), opts.metrics, "Any", (m) => LABEL[m] || m);
      fillSelect($("regBase"), opts.suite_runs, "Select run");
      fillSelect($("regCand"), opts.suite_runs, "Select run");
      $("evalGate").hidden = true;
      $("evalBody").hidden = false;
      await refreshAll();
    } catch (e) {
      if (e.auth) { state.token = null; try { sessionStorage.removeItem("evalToken"); } catch {} gate(e.message); }
      else gate(e.message);
    }
  }

  function fillSelect(sel, values, placeholder, labeller) {
    const keep = sel.value;
    sel.innerHTML = "";
    const first = document.createElement("option");
    first.value = ""; first.textContent = placeholder;
    sel.appendChild(first);
    values.forEach((v) => { const o = document.createElement("option"); o.value = v; o.textContent = labeller ? labeller(v) : v; sel.appendChild(o); });
    if (values.includes(keep)) sel.value = keep;
  }

  async function refreshAll() {
    state.offset = 0;
    $("traceDetail").hidden = true;
    $("agentDetail").innerHTML = "";
    await Promise.all([loadSummary(), loadCharts(), loadAgents(), loadTraces()].map((p) => p.catch(handleError)));
  }

  function handleError(e) {
    if (e.auth) { state.token = null; gate(e.message); return; }
    $("overview").innerHTML = '<div class="error-banner">' + esc(e.message) + "</div>";
  }

  // ---------- overview ----------
  function badge(metric, v) {
    if (v == null) return '<span class="badge na">not available</span>';
    const th = state.thresholds[metric];
    const ok = LOWER_BETTER.has(metric) ? v <= th : v >= th;
    return ok ? '<span class="badge pass">&#10003; meets target</span>' : '<span class="badge fail">&#10007; below target</span>';
  }

  async function loadSummary() {
    const s = await api("summary", filters());
    state.thresholds = s.thresholds || {};
    const tiles = METRICS.map(([key, label]) => {
      const m = s.metrics[key];
      const tgt = state.thresholds[key];
      return `<div class="tile"><div class="k">${esc(label)}</div><div class="v">${pct(m.avg)}</div>
        <div class="n">${m.n ? "over " + m.n + " evaluations" : "no data (not applicable)"}${tgt != null ? " &middot; target " + (LOWER_BETTER.has(key) ? "&le;" : "&ge;") + pct(tgt, 0) : ""}</div>
        <div style="margin-top:6px">${badge(key, m.avg)}</div></div>`;
    }).join("");
    const reliable = s.reliability_score;
    $("overview").innerHTML = `
      <h2>AI Evaluation Center</h2>
      <div class="hero">
        <div><div class="hero-score">${pct(reliable)}</div><div class="hero-label">Overall reliability score<br><span class="muted">mean of computed quality metrics, hallucination inverted</span></div></div>
        <div class="tiles" style="flex:1;min-width:280px">
          <div class="tile"><div class="k">Evaluations</div><div class="v">${num(s.evaluations_total)}</div><div class="n">${num(s.evaluations_pending)} pending</div></div>
          <div class="tile"><div class="k">Failed evaluations</div><div class="v">${num(s.failed_evaluations)}</div><div class="n">${num(s.evaluation_errors)} crashed evaluation runs</div></div>
          <div class="tile"><div class="k">Request error rate</div><div class="v">${pct(s.request_error_rate)}</div></div>
          <div class="tile"><div class="k">Average latency</div><div class="v">${ms(s.avg_latency_ms)}</div></div>
          <div class="tile"><div class="k">Average cost</div><div class="v">${money(s.avg_cost)}</div><div class="n">${s.cost_known_for ? "priced for " + s.cost_known_for + " requests" : "no pricing configured"}</div></div>
          <div class="tile"><div class="k">Total tokens</div><div class="v">${num(s.total_tokens)}</div></div>
        </div>
      </div>
      <div class="tiles">${tiles}</div>`;
  }

  // ---------- charts (inline SVG, no external libraries) ----------
  const W = 520, H = 190, PL = 44, PR = 14, PT = 10, PB = 24;

  function niceMax(v) { if (!v || v <= 0) return 1; const p = Math.pow(10, Math.floor(Math.log10(v))); return Math.ceil(v / p) * p; }

  function chart({ title, sub, dates, series, fmt, yMax, kind = "line", sparseNote }) {
    const box = document.createElement("div");
    box.className = "chart";
    const has = series.some((s) => s.values.some((v) => v != null));
    box.innerHTML = `<h4>${esc(title)}</h4><div class="sub">${esc(sub || "")}</div>`;
    if (!has) { box.insertAdjacentHTML("beforeend", `<div class="empty">${esc(sparseNote || "No data for the current filters")}</div>`); return box; }
    const max = yMax || niceMax(Math.max(...series.flatMap((s) => s.values.filter((v) => v != null))));
    const n = dates.length;
    const x = (i) => PL + (n === 1 ? (W - PL - PR) / 2 : (i * (W - PL - PR)) / (n - 1));
    const y = (v) => PT + (1 - v / max) * (H - PT - PB);
    let g = "";
    [0, 0.5, 1].forEach((t) => {
      const yy = PT + (1 - t) * (H - PT - PB);
      g += `<line x1="${PL}" x2="${W - PR}" y1="${yy}" y2="${yy}" stroke="var(--grid)" stroke-width="1"/><text x="${PL - 6}" y="${yy + 4}" text-anchor="end" font-size="10" fill="var(--muted)">${esc(fmt(max * t))}</text>`;
    });
    g += `<text x="${PL}" y="${H - 6}" font-size="10" fill="var(--muted)">${esc(dates[0])}</text>`;
    if (n > 1) g += `<text x="${W - PR}" y="${H - 6}" text-anchor="end" font-size="10" fill="var(--muted)">${esc(dates[n - 1])}</text>`;
    if (kind === "bar") {
      const bw = Math.max(4, Math.min(28, ((W - PL - PR) / n) * 0.6));
      series.forEach((s) => s.values.forEach((v, i) => {
        if (v == null) return;
        const top = y(v), base = y(0), r = Math.min(4, bw / 2, base - top);
        g += `<path d="M${x(i) - bw / 2},${base} V${top + r} Q${x(i) - bw / 2},${top} ${x(i) - bw / 2 + r},${top} H${x(i) + bw / 2 - r} Q${x(i) + bw / 2},${top} ${x(i) + bw / 2},${top + r} V${base} Z" fill="${s.color}"/>`;
      }));
    } else {
      series.forEach((s) => {
        let d = "", pen = false;
        s.values.forEach((v, i) => { if (v == null) { pen = false; return; } d += (pen ? "L" : "M") + x(i) + "," + y(v); pen = true; });
        g += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
        s.values.forEach((v, i) => { if (v != null) g += `<circle cx="${x(i)}" cy="${y(v)}" r="${n > 20 ? 2 : 3.5}" fill="${s.color}" stroke="#151a2b" stroke-width="2"/>`; });
      });
    }
    // hover layer: one full-height hit column per x position
    const colW = n === 1 ? W - PL - PR : (W - PL - PR) / (n - 1);
    dates.forEach((_, i) => { g += `<rect class="hit" data-i="${i}" x="${x(i) - colW / 2}" y="${PT}" width="${colW}" height="${H - PT - PB}" fill="transparent"/>`; });
    box.insertAdjacentHTML("beforeend", `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}">${g}</svg>`);
    if (series.length > 1) box.insertAdjacentHTML("beforeend", `<div class="legend">${series.map((s) => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join("")}</div>`);
    const tip = document.createElement("div");
    tip.className = "tooltip"; tip.hidden = true;
    box.appendChild(tip);
    box.querySelectorAll(".hit").forEach((r) => {
      r.addEventListener("mousemove", (ev) => {
        const i = +r.dataset.i;
        tip.innerHTML = `<div class="muted" style="margin-bottom:4px">${esc(dates[i])}</div>` + series.map((s) => `<div class="row"><span><i style="background:${s.color}"></i> ${esc(s.name)}</span><b>${esc(s.values[i] == null ? "n/a" : fmt(s.values[i]))}</b></div>`).join("");
        const b = box.getBoundingClientRect();
        tip.hidden = false;
        tip.style.left = Math.min(ev.clientX - b.left + 12, b.width - 170) + "px";
        tip.style.top = ev.clientY - b.top + 12 + "px";
      });
      r.addEventListener("mouseleave", () => { tip.hidden = true; });
    });
    return box;
  }

  async function loadCharts() {
    const pts = await api("timeseries", filters());
    state.series = pts;
    const dates = pts.map((p) => p.date);
    const sc = (m) => pts.map((p) => p.scores[m]);
    const host = $("charts");
    host.innerHTML = "";
    const trend = ["faithfulness", "relevance", "groundedness", "safety"];
    host.append(
      chart({ title: "Evaluation score trends", sub: "Daily average, higher is better", dates, fmt: (v) => pct(v, 0), yMax: 1, series: trend.map((m, i) => ({ name: LABEL[m], color: SERIES_COLORS[i], values: sc(m) })) }),
      chart({ title: "Latency", sub: "Average end-to-end request time", dates, fmt: ms, series: [{ name: "Latency", color: SERIES_COLORS[0], values: pts.map((p) => p.avg_latency_ms) }] }),
      chart({ title: "Token usage", sub: "Tokens per day", dates, fmt: num, series: [{ name: "Input", color: SERIES_COLORS[0], values: pts.map((p) => p.input_tokens) }, { name: "Output", color: SERIES_COLORS[1], values: pts.map((p) => p.output_tokens) }] }),
      chart({ title: "Cost", sub: "Estimated spend per day", dates, fmt: money, kind: "bar", sparseNote: "No cost data. Set EVAL_PRICING_JSON to price models.", series: [{ name: "Cost", color: SERIES_COLORS[0], values: pts.map((p) => p.total_cost) }] }),
      chart({ title: "Failure rate", sub: "Share of evaluations that failed", dates, fmt: (v) => pct(v, 0), yMax: 1, series: [{ name: "Failure rate", color: SERIES_COLORS[0], values: pts.map((p) => p.failure_rate) }] }),
      chart({ title: "Hallucination rate", sub: "Lower is better", dates, fmt: (v) => pct(v, 1), yMax: 1, series: [{ name: "Hallucination", color: SERIES_COLORS[0], values: pts.map((p) => p.hallucination_rate) }] }),
    );
    renderDataTable();
  }

  function renderDataTable() {
    const box = $("dataTable");
    box.hidden = !state.table;
    $("toggleTable").textContent = state.table ? "Hide data table" : "Show data table";
    if (!state.table) return;
    const rows = (state.series || []).map((p) => `<tr><td>${esc(p.date)}</td><td>${p.evaluations}</td>${["faithfulness", "relevance", "groundedness", "safety", "hallucination"].map((m) => `<td>${pct(p.scores[m])}</td>`).join("")}<td>${ms(p.avg_latency_ms)}</td><td>${num(p.total_tokens)}</td><td>${money(p.total_cost)}</td><td>${pct(p.failure_rate, 0)}</td></tr>`).join("");
    box.innerHTML = `<div class="scroll-x"><table class="data"><thead><tr><th>Date</th><th>Evals</th><th>Faithfulness</th><th>Relevance</th><th>Groundedness</th><th>Safety</th><th>Hallucination</th><th>Latency</th><th>Tokens</th><th>Cost</th><th>Failure</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  }

  // ---------- agents ----------
  async function loadAgents() {
    const list = await api("agents", filters());
    if (!list.length) { $("agentsTable").innerHTML = '<p class="muted">No agent executions recorded yet.</p>'; return; }
    $("agentsTable").innerHTML = `<div class="scroll-x"><table class="data"><thead><tr><th>Agent</th><th>Reliability*</th><th>Executions</th><th>Success</th><th>Avg latency</th><th>Avg tokens</th><th>Avg cost</th></tr></thead><tbody>` +
      list.map((a) => `<tr class="click" tabindex="0" data-agent="${esc(a.agent)}"><td><b>${esc(a.agent)}</b></td><td>${pct(a.reliability_score)}</td><td>${num(a.executions)}</td><td>${pct(a.success_rate, 0)}</td><td>${ms(a.avg_latency_ms)}</td><td>${num(a.avg_tokens)}</td><td>${money(a.avg_cost)}</td></tr>`).join("") +
      `</tbody></table></div><p class="muted" style="font-size:12px">*Evaluation is per request. An agent&rsquo;s reliability is the score of the requests it took part in.</p>`;
    $("agentsTable").querySelectorAll("tr.click").forEach((tr) => {
      const open = () => openAgent(tr.dataset.agent);
      tr.addEventListener("click", open);
      tr.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); });
    });
  }

  async function openAgent(name) {
    const d = await api("agents/" + encodeURIComponent(name), filters());
    const kv = (k, v) => `<div class="tile"><div class="k">${esc(k)}</div><div class="v" style="font-size:20px">${v}</div></div>`;
    $("agentDetail").innerHTML = `<h3>${esc(d.agent)}</h3><div class="kv">
      ${kv("Total executions", num(d.executions))}${kv("Success rate", pct(d.success_rate))}${kv("Avg latency", ms(d.avg_latency_ms))}${kv("Avg tokens", num(d.avg_tokens))}${kv("Avg cost", money(d.avg_cost))}${kv("Failed evaluations", num(d.failed_evaluations))}${kv("Models", esc(d.models.join(", ") || "n/a"))}</div>
      <div class="kv">${METRICS.map(([m, l]) => kv(l, pct(d.scores[m]))).join("")}</div>
      <h3>Recent traces</h3>${tracesTable(d.recent_traces)}`;
    bindTraceRows($("agentDetail"));
    $("agentDetail").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // ---------- traces ----------
  function passBadge(ev) {
    if (!ev) return '<span class="badge na">not evaluated</span>';
    if (ev.status === "pending" || ev.status === "running") return '<span class="badge warn">&#8987; ' + esc(ev.status) + "</span>";
    if (ev.status === "failed") return '<span class="badge fail">&#10007; evaluator error</span>';
    if (ev.passed === true) return '<span class="badge pass">&#10003; passed</span>';
    if (ev.passed === false) return '<span class="badge fail">&#10007; failed</span>';
    return '<span class="badge warn" title="An evaluator errored or no metric could be computed">&#63; not verified</span>';
  }

  function tracesTable(items) {
    if (!items.length) return '<p class="muted">No traces match the current filters.</p>';
    return `<div class="scroll-x"><table class="data"><thead><tr><th>Time</th><th>Trace</th><th>Question</th><th>Result</th><th>Latency</th><th>Tokens</th><th>Cost</th></tr></thead><tbody>` +
      items.map((t) => `<tr class="click" tabindex="0" data-trace="${esc(t.trace_id)}"><td>${esc((t.created_at || "").replace("T", " ").slice(0, 19))}</td><td class="mono">${esc(t.trace_id)}</td><td>${esc((t.question || "").slice(0, 90))}</td><td>${passBadge(t.evaluation)}${t.success ? "" : ' <span class="badge fail">request failed</span>'}</td><td>${ms(t.duration_ms)}</td><td>${num(t.total_tokens)}</td><td>${money(t.estimated_cost)}</td></tr>`).join("") +
      "</tbody></table></div>";
  }

  function bindTraceRows(root) {
    root.querySelectorAll("tr[data-trace]").forEach((tr) => {
      const open = () => openTrace(tr.dataset.trace);
      tr.addEventListener("click", open);
      tr.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); });
    });
  }

  async function loadTraces() {
    const r = await api("traces", { ...filters(), limit: PAGE, offset: state.offset });
    state.total = r.total;
    $("tracesTable").innerHTML = tracesTable(r.items);
    bindTraceRows($("tracesTable"));
    const from = r.total ? state.offset + 1 : 0;
    $("pageInfo").textContent = `${from}-${Math.min(state.offset + PAGE, r.total)} of ${r.total}`;
    $("prevPage").disabled = state.offset === 0;
    $("nextPage").disabled = state.offset + PAGE >= r.total;
  }

  async function openTrace(id) {
    const d = await api("traces/" + encodeURIComponent(id));
    const flow = d.steps.map((s) => `<span class="step ${s.success ? "" : "bad"}"><b>${esc(s.agent_name)}</b><br><span class="muted">${esc(s.model_name || "")} &middot; ${ms(s.latency_ms)}</span></span>`).join('<span class="sep">&darr;</span>');
    const ev = d.evaluations[0];
    const metrics = ev ? ev.metrics.map((m) => `<tr><td>${esc(LABEL[m.metric] || m.metric)}</td><td>${m.status === "ok" ? pct(m.score) : '<span class="badge na">' + esc(m.status.replace("_", " ")) + "</span>"}</td><td>${m.status === "ok" ? (m.passed ? '<span class="badge pass">&#10003; pass</span>' : '<span class="badge fail">&#10007; fail</span>') : ""}</td><td>${m.confidence == null ? "" : pct(m.confidence, 0)}</td><td><div class="reason">${esc(m.reason)}</div></td></tr>`).join("") : "";
    const kv = (k, v) => `<div class="tile"><div class="k">${esc(k)}</div><div class="v" style="font-size:18px">${v}</div></div>`;
    $("traceDetail").hidden = false;
    $("traceDetail").innerHTML = `
      <div class="section-head"><h2>Trace <span class="mono">${esc(d.trace_id)}</span></h2><button class="btn-ghost" type="button" id="closeTrace">Close</button></div>
      <div class="kv">${kv("Status", d.success ? '<span class="badge pass">&#10003; success</span>' : '<span class="badge fail">&#10007; failed</span>')}${kv("Latency", ms(d.duration_ms))}${kv("Input tokens", num(d.input_tokens))}${kv("Output tokens", num(d.output_tokens))}${kv("Cost", money(d.estimated_cost))}${kv("Tool calls", d.tool_call_count)}${kv("Prompt version", esc(d.prompt_version || "n/a"))}${kv("Session", esc(d.session_id || "n/a"))}</div>
      ${d.error ? `<div class="error-banner">${esc(d.error)}</div>` : ""}
      <h3>User question</h3><div class="block">${esc(d.question)}</div>
      <h3>Execution</h3><div class="flow">${flow || '<span class="muted">No steps recorded</span>'}${d.tool_calls.length ? "" : ""}</div>
      <div class="scroll-x"><table class="data"><thead><tr><th>#</th><th>Agent</th><th>Model</th><th>Latency</th><th>In</th><th>Out</th><th>Cost</th><th>Result</th></tr></thead><tbody>${d.steps.map((s) => `<tr><td>${s.seq}</td><td>${esc(s.agent_name)}</td><td>${esc(s.model_name || "")}</td><td>${ms(s.latency_ms)}</td><td>${num(s.input_tokens)}</td><td>${num(s.output_tokens)}</td><td>${money(s.estimated_cost)}</td><td>${s.success ? '<span class="badge pass">&#10003; ok</span>' : '<span class="badge fail">&#10007; ' + esc(s.error || "error") + "</span>"}</td></tr>`).join("")}</tbody></table></div>
      <h3>Tool calls</h3>${d.tool_calls.length ? `<div class="scroll-x"><table class="data"><thead><tr><th>Tool</th><th>Agent</th><th>Input</th><th>Output</th><th>Latency</th></tr></thead><tbody>${d.tool_calls.map((t) => `<tr><td>${esc(t.tool_name)}</td><td>${esc(t.agent_name || "")}</td><td class="mono">${esc(JSON.stringify(t.tool_input))}</td><td class="mono">${esc(JSON.stringify(t.tool_output)).slice(0, 300)}</td><td>${ms(t.latency_ms)}</td></tr>`).join("")}</tbody></table></div>` : '<p class="muted">This workflow made no tool calls.</p>'}
      <h3>Context / sources shown to the answering agent</h3>${d.context.length ? d.context.map((c) => `<div class="block" style="margin-bottom:8px"><b>[${esc(c.id)}]</b> ${esc(c.source)}\n${esc(c.text)}</div>`).join("") : '<p class="muted">None recorded.</p>'}
      <h3>Final response</h3><div class="block">${esc(d.answer || "(no answer produced)")}</div>
      <h3>Evaluation ${ev ? passBadge(ev) : ""}</h3>${ev ? `<div class="scroll-x"><table class="data"><thead><tr><th>Metric</th><th>Score</th><th>Result</th><th>Confidence</th><th>Reason</th></tr></thead><tbody>${metrics || '<tr><td colspan="5" class="muted">Evaluation has not finished yet.</td></tr>'}</tbody></table></div>${ev.error ? `<div class="error-banner">${esc(ev.error)}</div>` : ""}` : '<p class="muted">Not evaluated.</p>'}`;
    $("closeTrace").addEventListener("click", () => { $("traceDetail").hidden = true; });
    $("traceDetail").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ---------- regression ----------
  async function runRegression() {
    const base = $("regBase").value, cand = $("regCand").value;
    const out = $("regOut");
    if (!base || !cand) { out.innerHTML = '<p class="muted">Select a baseline and a candidate run.</p>'; return; }
    try {
      const r = await api("regression", { baseline: base, candidate: cand });
      const row = (m) => `<tr><td>${esc(LABEL[m.metric])}</td><td>${pct(m.baseline)}</td><td>${pct(m.candidate)}</td><td>${m.delta == null ? "n/a" : (m.delta * 100 >= 0 ? "+" : "") + (m.delta * 100).toFixed(1) + " pts"}</td><td>${{ improved: '<span class="badge pass">&#9650; improved</span>', degraded: '<span class="badge fail">&#9660; degraded</span>', unchanged: '<span class="badge na">= unchanged</span>', not_comparable: '<span class="badge na">not comparable</span>' }[m.status]}</td></tr>`;
      out.innerHTML = `<div class="scroll-x"><table class="data"><thead><tr><th>Metric</th><th>Baseline</th><th>Candidate</th><th>Change</th><th>Status</th></tr></thead><tbody>${r.metrics.map(row).join("")}</tbody></table></div>
        <p>Newly failing cases: <b>${esc(r.newly_failing.join(", ") || "none")}</b> &middot; Newly passing: <b>${esc(r.newly_passing.join(", ") || "none")}</b></p>
        ${r.warnings.map((w) => `<div class="warnline">&#9888; ${esc(w)}</div>`).join("") || '<p><span class="badge pass">&#10003; no regressions detected</span></p>'}`;
    } catch (e) { out.innerHTML = '<div class="error-banner">' + esc(e.message) + "</div>"; }
  }

  // ---------- wiring ----------
  $("tabAgent").addEventListener("click", () => show("agent"));
  $("tabEval").addEventListener("click", () => show("eval"));
  $("evalTokenForm").addEventListener("submit", (e) => {
    e.preventDefault();
    state.token = $("evalToken").value.trim();
    $("evalToken").value = "";
    try { sessionStorage.setItem("evalToken", state.token); } catch {}
    enterEval();
  });
  $("evalLock").addEventListener("click", () => { state.token = null; try { sessionStorage.removeItem("evalToken"); } catch {} gate(); });
  $("evalFilters").addEventListener("submit", (e) => { e.preventDefault(); refreshAll(); });
  $("fReset").addEventListener("click", () => { $("evalFilters").reset(); refreshAll(); });
  $("toggleTable").addEventListener("click", () => { state.table = !state.table; renderDataTable(); });
  $("prevPage").addEventListener("click", () => { state.offset = Math.max(0, state.offset - PAGE); loadTraces().catch(handleError); });
  $("nextPage").addEventListener("click", () => { state.offset += PAGE; loadTraces().catch(handleError); });
  $("regRun").addEventListener("click", runRegression);
})();
