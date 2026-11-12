// Knowledge base dashboard: stats, drag-and-drop upload, library, semantic search, preview, re-index, delete.
import { api } from "./api.js";
import { openCitation, openDocument } from "./preview.js";
import { emit, state } from "./state.js";
import { clear, confirmDialog, fileTypeBadge, fmt, h, icon, statusPill, toast } from "./ui.js";

let root, statsEl, tableEl, searchInput, resultsEl, uploadsEl, recentEl, filter = "all";
let pollTimer = null, searchTimer = null, mounted = false;

export function mountKnowledge(container) {
  root = container;
  statsEl = h("div.stat-grid");
  uploadsEl = h("div.upload-list");
  recentEl = h("div.recent-list");
  resultsEl = h("div.search-results");
  tableEl = h("div");
  searchInput = h("input.input", { placeholder: "Search inside your documents (semantic) or by file name…", "aria-label": "Search documents", maxlength: "200" });
  searchInput.addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(runSearch, 450); renderTable(); });

  const dz = h("div.dropzone", { tabindex: "0", role: "button", "aria-label": "Upload documents" },
    h("div.dz-icon", icon("upload", "lg")),
    h("div", h("b", "Drop files here or click to upload"),
      h("span", `PDF, DOCX, TXT, Markdown or CSV · up to ${state.config?.rag?.max_upload_mb ?? 15} MB each · indexed in the background`)));
  const pick = () => {
    const picker = document.getElementById("filePicker");
    picker.onchange = () => { if (picker.files.length) upload(picker.files); picker.value = ""; };
    picker.click();
  };
  dz.addEventListener("click", pick);
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("over"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("over"));
  dz.addEventListener("drop", (e) => { e.preventDefault(); dz.classList.remove("over"); if (e.dataTransfer.files.length) upload(e.dataTransfer.files); });

  const seg = h("div.seg", ["all", "ready", "processing", "failed"].map((f) => h("button" + (f === filter ? ".on" : ""), { type: "button", dataset: { f },
    on: { click: (e) => { filter = f; seg.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.f === f)); renderTable(); } } },
    f[0].toUpperCase() + f.slice(1))));

  root.append(h("div.page",
    h("div.page-head", h("div", h("h2", "Knowledge base"), h("p", "Documents the agent can search. Only you can see or search your documents."))),
    statsEl,
    h("div.kb-top", h("div", dz, uploadsEl), h("div.card.pad-sm", h("div.p-title", h("span", "Recent uploads")), recentEl)),
    h("div.toolbar", h("div.search", icon("search", "sm"), searchInput), seg),
    resultsEl, tableEl));
  mounted = true;
}

export async function showKnowledge() {
  if (!mounted) return;
  await refresh();
}

export async function refresh() {
  try {
    const [docs, stats] = await Promise.all([api("/api/documents"), api("/api/documents/stats")]);
    state.docs = docs; state.docStats = stats;
    emit("docs-loaded");
    if (!mounted) return;
    renderStats(stats); renderRecent(stats.recent); renderTable();
    schedulePoll();
  } catch (e) {
    if (mounted) clear(tableEl).append(h("div.banner.err", icon("alert", "sm"), h("span", e.message)));
  }
}

function schedulePoll() {
  clearTimeout(pollTimer);
  if ((state.docs || []).some((d) => d.status === "queued" || d.status === "processing")) pollTimer = setTimeout(refresh, 1500);
}

function renderStats(s) {
  const tile = (k, ic, v, n, meter) => h("div.stat", h("div.k", icon(ic, "sm"), k), h("div.v", v), n ? h("div.n", n) : null,
    meter != null ? h("div.meter", h("i", { style: { width: Math.min(100, meter * 100).toFixed(1) + "%" } })) : null);
  const finished = s.by_status.ready + s.by_status.failed;
  clear(statsEl).append(
    tile("Documents", "file", fmt.num(s.total_documents), `${s.document_quota} max`),
    tile("Indexed chunks", "layers", fmt.num(s.total_chunks), s.total_documents ? `≈ ${Math.round(s.total_chunks / Math.max(1, s.by_status.ready))} per document` : "—"),
    tile("Storage used", "db", fmt.bytes(s.storage_bytes), `of ${fmt.bytes(s.storage_quota_bytes)}`, s.storage_bytes / s.storage_quota_bytes),
    tile("Indexing success", "check", s.success_rate == null ? "—" : fmt.pct(s.success_rate), finished ? `${s.by_status.ready} of ${finished} finished` : "Nothing indexed yet"),
    tile("In progress", "clock", fmt.num(s.by_status.queued + s.by_status.processing), s.by_status.failed ? `${s.by_status.failed} failed` : "No failures"));
}

function renderRecent(list) {
  clear(recentEl);
  if (!list.length) { recentEl.append(h("div.panel-empty", "No uploads yet.")); return; }
  list.forEach((d) => recentEl.append(h("div.recent-item", fileTypeBadge(d.file_type),
    h("div.grow", h("div.ellipsis", { style: { fontWeight: 550 } }, d.filename), h("div.faint", { style: { fontSize: "12px" } }, fmt.rel(d.created_at))), statusPill(d.status, d.error))));
}

function renderTable() {
  if (!state.docs) return;
  const q = searchInput.value.trim().toLowerCase();
  let docs = state.docs.filter((d) => filter === "all" || (filter === "processing" ? d.status === "queued" || d.status === "processing" : d.status === filter));
  if (q) docs = docs.filter((d) => d.filename.toLowerCase().includes(q) || resultsEl.dataset.docs?.includes(d.id));
  clear(tableEl);
  if (!state.docs.length) {
    tableEl.append(h("div.card", h("div.empty-state", icon("library"), h("h3", "Your knowledge base is empty"),
      h("p", "Upload documents above. The agent splits them into passages, embeds them, and cites them when it answers."))));
    return;
  }
  if (!docs.length) { tableEl.append(h("div.card", h("div.empty-state", icon("filter"), h("h3", "No documents match"), h("p", "Try another filter or search.")))); return; }
  const rows = docs.map((d) => h("tr",
    h("td", h("div.doc-name", fileTypeBadge(d.file_type), h("div", { style: { minWidth: 0 } }, h("b", { title: d.filename }, d.filename),
      h("small", d.error ? d.error : [d.page_count ? `${d.page_count} pages` : null, d.meta?.title && d.meta.title !== d.filename ? d.meta.title : null].filter(Boolean).join(" · ") || d.file_type.toUpperCase())))),
    h("td", statusPill(d.status, d.error)),
    h("td.num.hide-sm", d.status === "ready" ? fmt.num(d.chunk_count) : "—"),
    h("td.num.hide-sm", fmt.bytes(d.size_bytes)),
    h("td.hide-sm.faint", { title: fmt.date(d.created_at) }, fmt.rel(d.created_at)),
    h("td", h("div.row-actions",
      h("button.btn.ghost.sm.icon-only", { type: "button", title: "Preview", "aria-label": "Preview " + d.filename, on: { click: () => openDocument(d.id) } }, icon("eye", "sm")),
      h("button.btn.ghost.sm.icon-only", { type: "button", title: "Ask about this document", "aria-label": "Ask about " + d.filename, disabled: d.status !== "ready",
        on: { click: () => emit("ask-about", d) } }, icon("chat", "sm")),
      h("button.btn.ghost.sm.icon-only", { type: "button", title: "Re-index", "aria-label": "Re-index " + d.filename, disabled: d.status === "processing" || d.status === "queued",
        on: { click: () => reindex(d) } }, icon("refresh", "sm")),
      h("button.btn.ghost.sm.icon-only.danger", { type: "button", title: "Delete", "aria-label": "Delete " + d.filename, on: { click: () => remove(d) } }, icon("trash", "sm"))))));
  tableEl.append(h("div.card", { style: { padding: "8px 8px 0" } }, h("div", { style: { overflowX: "auto" } }, h("table.doc-table",
    h("thead", h("tr", h("th", "Name"), h("th", "Status"), h("th.hide-sm", "Chunks"), h("th.hide-sm", "Size"), h("th.hide-sm", "Uploaded"), h("th"))),
    h("tbody", rows)))));
}

async function runSearch() {
  const q = searchInput.value.trim();
  clear(resultsEl); delete resultsEl.dataset.docs;
  if (q.length < 3 || !state.docs.some((d) => d.status === "ready")) { renderTable(); return; }
  resultsEl.append(h("div.skeleton", { style: { height: "64px" } }));
  try {
    const r = await api(`/api/documents/search?q=${encodeURIComponent(q)}&k=6`);
    if (searchInput.value.trim() !== q) return;
    clear(resultsEl);
    const relevant = r.passages.filter((p) => p.relevant);
    resultsEl.dataset.docs = relevant.map((p) => p.document_id).join(" ");
    resultsEl.append(h("div.p-title", h("span", "Matching passages"), h("span", relevant.length ? `${relevant.length} relevant` : "none above the similarity threshold")));
    relevant.forEach((p) => {
      const src = { id: "", chunk_id: p.chunk_id, filename: p.filename, page: p.page, section: p.section };
      resultsEl.append(h("button.source", { type: "button", on: { click: () => openCitation({ ...src, id: "Search" }) } },
        fileTypeBadge(p.file_type),
        h("div", { style: { minWidth: 0 } }, h("div.sname", h("span", p.filename), h("span.sloc", [p.page ? "p. " + p.page : null, p.section].filter(Boolean).join(" · "))), h("div.sprev", p.preview)),
        h("span.sscore", { title: "Similarity" }, p.score.toFixed(2))));
    });
    renderTable();
  } catch (e) {
    clear(resultsEl).append(h("div.banner.err", icon("alert", "sm"), h("span", e.message)));
  }
}

async function upload(files) {
  const list = [...files];
  if (list.length > 10) toast("Uploading the first 10 files (10 per batch).", "err");
  const batch = list.slice(0, 10);
  const items = batch.map((f) => {
    const row = h("div.upload-item", h("span.spinner"), h("span.grow.ellipsis", f.name), h("span.faint", fmt.bytes(f.size)));
    uploadsEl.append(row);
    return row;
  });
  const form = new FormData();
  batch.forEach((f) => form.append("files", f));
  try {
    const res = await api("/api/documents", { method: "POST", form });
    res.results.forEach((r, i) => {
      const row = items[i];
      clear(row).append(icon(r.ok ? "check" : "alert", "sm"), h("span.grow.ellipsis", r.filename),
        h("span." + (r.ok ? "faint" : "pill.err"), r.ok ? (r.duplicate ? "Already in your library" : "Queued for indexing") : r.error));
      setTimeout(() => row.remove(), r.ok ? 4000 : 12000);
    });
    const ok = res.results.filter((r) => r.ok && !r.duplicate).length;
    if (ok) toast(`${ok} document${ok > 1 ? "s" : ""} uploaded — indexing in the background.`);
  } catch (e) {
    items.forEach((row) => row.remove());
    toast(e.message, "err");
  }
  await refresh();
  emit("docs-changed");
}

async function reindex(d) {
  try { await api(`/api/documents/${encodeURIComponent(d.id)}/reindex`, { method: "POST" }); toast(`Re-indexing ${d.filename}`); await refresh(); }
  catch (e) { toast(e.message, "err"); }
}

async function remove(d) {
  const ok = await confirmDialog({ title: "Delete document?", body: `“${d.filename}” and its ${d.chunk_count} indexed passages will be permanently deleted. Past answers that cited it will no longer open the passage.`, confirm: "Delete", danger: true });
  if (!ok) return;
  try { await api(`/api/documents/${encodeURIComponent(d.id)}`, { method: "DELETE" }); toast("Document deleted"); await refresh(); emit("docs-changed"); }
  catch (e) { toast(e.message, "err"); }
}
