// Chat view: welcome screen, conversation thread, streaming answers, citations, attachments, stop/regenerate.
import { api, streamSSE } from "./api.js";
import { copyText, plainAnswer, renderMarkdown } from "./markdown.js";
import { applyEvent, newRun, runFromMessage, showRun, sourceButton } from "./panel.js";
import { openCitation } from "./preview.js";
import { emit, state } from "./state.js";
import { ROUTE_LABEL, NODE_LABEL, clear, fmt, h, icon, showMenu, statusPill, toast } from "./ui.js";

const MODE_INFO = {
  auto: { label: "Auto", desc: "The agent decides when to search your documents" },
  always: { label: "Documents", desc: "Always answer from your knowledge base" },
  off: { label: "No documents", desc: "Never search your documents" },
};

let root, thread, threadInner, textarea, sendBtn, modeBtn, attachRow, composer;
let conversation = null; // {id, title, messages}
let controller = null; // AbortController of the in-flight answer
let attachments = []; // [{id, filename, status, error}]
let pollTimer = null;

const kbMode = () => state.user?.preferences?.kb_mode || "auto";

export function mountChat(container) {
  root = container;
  thread = h("div.thread");
  threadInner = h("div.thread-inner");
  thread.append(threadInner);

  textarea = h("textarea", { rows: 1, placeholder: "Ask anything, or ask about your documents…", "aria-label": "Message", maxlength: "4000" });
  textarea.addEventListener("input", autosize);
  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); if (!controller) send(); }
  });
  sendBtn = h("button.send", { type: "button", "aria-label": "Send", title: "Send (Enter)" }, icon("send"));
  sendBtn.addEventListener("click", () => (controller ? stop() : send()));
  modeBtn = h("button.mode-btn", { type: "button", title: "Knowledge base mode" });
  modeBtn.addEventListener("click", () => showMenu(modeBtn, Object.entries(MODE_INFO).map(([k, v]) => ({
    label: v.label, desc: v.desc, selected: kbMode() === k, icon: k === "off" ? "x" : k === "always" ? "library" : "compass",
    action: () => setMode(k),
  }))));
  const attachBtn = h("button.btn.ghost.sm.icon-only", { type: "button", title: "Attach documents (PDF, DOCX, TXT, MD, CSV)", "aria-label": "Attach documents" }, icon("paperclip"));
  attachBtn.addEventListener("click", () => pickFiles());
  attachRow = h("div.grow");
  composer = h("div.composer", textarea, h("div.composer-bar", attachBtn, modeBtn, attachRow, sendBtn));
  composer.addEventListener("dragover", (e) => { e.preventDefault(); composer.classList.add("drag"); });
  composer.addEventListener("dragleave", () => composer.classList.remove("drag"));
  composer.addEventListener("drop", (e) => { e.preventDefault(); composer.classList.remove("drag"); if (e.dataTransfer.files.length) uploadAttachments(e.dataTransfer.files); });

  root.append(h("div.chat", thread, h("div.composer-wrap", composer,
    h("div.composer-foot", "Answers from your documents cite their sources. Always verify important information."))));
  renderMode();
  updateSend();
}

function autosize() {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 240) + "px";
  updateSend();
}

function updateSend() {
  const pending = attachments.some((a) => a.status === "queued" || a.status === "processing");
  sendBtn.classList.toggle("stop", !!controller);
  clear(sendBtn).append(icon(controller ? "stop" : "send"));
  sendBtn.setAttribute("aria-label", controller ? "Stop generating" : "Send");
  sendBtn.title = controller ? "Stop generating" : pending ? "Waiting for attachments to finish indexing" : "Send (Enter)";
  sendBtn.disabled = !controller && (!textarea.value.trim() || pending);
}

function renderMode() {
  const m = attachments.length ? "always" : kbMode();
  clear(modeBtn).append(icon(m === "off" ? "x" : m === "always" ? "library" : "compass", "sm"), attachments.length ? "Attached files" : MODE_INFO[m].label);
  modeBtn.classList.toggle("on", m !== "auto");
  modeBtn.disabled = !!attachments.length;
}

async function setMode(mode) {
  state.user.preferences = { ...state.user.preferences, kb_mode: mode };
  renderMode();
  try { await api("/api/me", { method: "PATCH", body: { preferences: { kb_mode: mode } } }); } catch (e) { toast(e.message, "err"); }
}

// ---- attachments: upload to the knowledge base and scope the next question to them ----
function pickFiles() {
  const picker = document.getElementById("filePicker");
  picker.onchange = () => { if (picker.files.length) uploadAttachments(picker.files); picker.value = ""; };
  picker.click();
}

async function uploadAttachments(files) {
  const form = new FormData();
  [...files].slice(0, 10).forEach((f) => form.append("files", f));
  try {
    const res = await api("/api/documents", { method: "POST", form });
    res.results.forEach((r) => {
      if (!r.ok) { toast(`${r.filename}: ${r.error}`, "err", 6000); return; }
      if (!attachments.some((a) => a.id === r.document.id)) attachments.push({ id: r.document.id, filename: r.document.filename, status: r.document.status, error: r.document.error });
    });
    renderAttachments();
    pollAttachments();
    emit("docs-changed");
  } catch (e) { toast(e.message, "err"); }
}

function renderAttachments() {
  clear(attachRow).append(...attachments.map((a) => h("span.attach-chip", { title: a.error || a.filename },
    a.status === "ready" ? icon("file", "sm") : a.status === "failed" ? icon("alert", "sm") : h("span.spinner"),
    h("span", a.filename),
    h("button", { type: "button", "aria-label": "Remove " + a.filename, on: { click: () => { attachments = attachments.filter((x) => x !== a); renderAttachments(); } } }, icon("x", "sm")))));
  renderMode();
  updateSend();
}

function pollAttachments() {
  clearTimeout(pollTimer);
  if (!attachments.some((a) => a.status === "queued" || a.status === "processing")) return;
  pollTimer = setTimeout(async () => {
    await Promise.all(attachments.map(async (a) => {
      if (a.status === "ready" || a.status === "failed") return;
      try {
        const d = await api(`/api/documents/${encodeURIComponent(a.id)}`);
        a.status = d.status; a.error = d.error;
        if (d.status === "failed") toast(`${a.filename}: ${d.error}`, "err", 6000);
      } catch { a.status = "failed"; }
    }));
    renderAttachments();
    emit("docs-changed");
    pollAttachments();
  }, 1200);
}

// ---- thread rendering ----
function greeting() {
  const hr = new Date().getHours();
  return hr < 5 ? "Working late" : hr < 12 ? "Good morning" : hr < 18 ? "Good afternoon" : "Good evening";
}

function renderWelcome() {
  clear(threadInner);
  const ready = (state.docs || []).filter((d) => d.status === "ready");
  const cards = [];
  if (ready.length) {
    cards.push({ icon: "book", title: `Summarize “${ready[0].filename}”`, sub: "Key points, with citations", q: `Summarize the key points of ${ready[0].filename}.` });
    cards.push({ icon: "search", title: "What do my documents say about…", sub: "Search across your knowledge base", q: "What are the most important facts, numbers and decisions in my documents?" });
  } else {
    cards.push({ icon: "upload", title: "Add your first document", sub: "PDF, DOCX, TXT, Markdown or CSV", action: () => pickFiles() });
    cards.push({ icon: "library", title: "How does document Q&A work here?", sub: "Retrieval, reranking and citations", q: "Explain how retrieval-augmented generation answers questions from documents, and why citations matter." });
  }
  cards.push({ icon: "layers", title: "Compare RAG and fine-tuning", sub: "Research agent: decompose → answer → synthesize", q: "What are the main differences between RAG and fine-tuning, and when should each be used?" });
  cards.push({ icon: "code", title: "Write a Python retry helper", sub: "With exponential backoff and jitter", q: "Write a small Python function that retries an async HTTP call with exponential backoff and jitter. Include a usage example." });
  threadInner.append(h("div.welcome",
    h("span.welcome-mark", h("img.mark", { src: "/static/favicon.svg?v=20261112", alt: "" }), h("i.spark.s1"), h("i.spark.s2"), h("i.spark.s3")),
    h("h2", `${greeting()}, `, h("span.grad", state.user.display_name.split(" ")[0])),
    h("p", ready.length ? `Ask anything. ${ready.length} document${ready.length > 1 ? "s are" : " is"} ready to search.` : "Ask anything, or attach documents and get answers with citations."),
    h("div.prompt-grid", cards.map((c) => h("button.prompt-card", { type: "button", on: { click: () => { if (c.action) c.action(); else { textarea.value = c.q; autosize(); send(); } } } },
      h("span.pc-icon", icon(c.icon, "sm")), h("div", h("b", c.title), h("span", c.sub)))))));
}

function userMessageEl(m) {
  const initials = (state.user.display_name || "?").trim().slice(0, 1).toUpperCase();
  return h("div.msg.user", { dataset: { id: m.id || "" } }, h("div.who", initials),
    h("div.body", h("div.meta", h("b", "You"), h("span", fmt.time(m.created_at))), h("div.bubble", m.content)));
}

function assistantMessageEl(m, { streaming = false } = {}) {
  const md = h("div.md");
  const extras = h("div");
  const actions = h("div.msg-actions");
  const metaRow = h("div.meta", h("b", "Fire Agent"), h("span.when", fmt.time(m.created_at)), h("span.route"));
  const el = h("div.msg.assistant", { dataset: { id: m.id || "" } }, h("div.who", h("img", { src: "/static/favicon.svg?v=20261112", alt: "" })),
    h("div.body", metaRow, md, extras, actions));
  el._parts = { md, extras, actions, metaRow };
  el._msg = m;
  fillAssistant(el, m, { streaming });
  return el;
}

function fillAssistant(el, m, { streaming = false } = {}) {
  const { md, extras, actions, metaRow } = el._parts;
  const meta = m.meta || {};
  const sources = meta.sources || [];
  const validIds = new Set(sources.map((s) => s.id));
  renderMarkdown(md, m.content, { streaming, validIds, onCite: (sid) => { const s = sources.find((x) => x.id === sid); if (s) openCitation(s); } });
  md.classList.toggle("cursor", streaming);
  const routeEl = metaRow.querySelector(".route");
  clear(routeEl);
  if (meta.route) routeEl.append(h("span.pill", { title: meta.route_reason || "" }, icon(meta.route === "rag" ? "library" : meta.route === "research" ? "layers" : "zap", "sm"), ROUTE_LABEL[meta.route] || meta.route));
  if (meta.duration_ms != null && !streaming) routeEl.append(h("span.faint", { style: { marginLeft: "6px" } }, fmt.ms(meta.duration_ms)));

  clear(extras);
  if (m.status === "stopped") extras.append(h("div.status-note", icon("stop", "sm"), "Stopped — this answer is incomplete."));
  if (m.status === "error") extras.append(h("div.status-note.err", icon("alert", "sm"), meta.error || "The agent hit an error before finishing."));
  if (!streaming && meta.route === "rag" && meta.grounded === false) extras.append(h("div.status-note", icon("alert", "sm"), "This answer does not cite your documents — treat it as unverified."));
  if (!streaming && sources.length) {
    const cited = sources.filter((s) => s.cited !== false).length;
    extras.append(h("details.sources", h("summary", icon("book", "sm"), `${sources.length} source${sources.length > 1 ? "s" : ""}`, h("span.faint", `· ${cited} cited`), h("span.chev", icon("chevron", "sm"))),
      h("div.source-list", sources.map((s) => sourceButton(s)))));
  }

  clear(actions);
  if (streaming) return;
  const copyBtn = h("button.btn.ghost", { type: "button", title: "Copy response" }, icon("copy", "sm"), "Copy");
  copyBtn.addEventListener("click", async () => { await copyText(plainAnswer(m.content)); copyBtn.lastChild.textContent = "Copied"; setTimeout(() => (copyBtn.lastChild.textContent = "Copy"), 1400); });
  actions.append(copyBtn);
  if (el.classList.contains("last")) actions.append(h("button.btn.ghost", { type: "button", title: "Regenerate response", on: { click: regenerate } }, icon("refresh", "sm"), "Regenerate"));
  if (m.trace_id || m.meta?.route) actions.append(h("button.btn.ghost", { type: "button", title: "Show execution details", on: { click: async () => { emit("panel", true); showRun(await runFromMessage(m)); } } }, icon("activity", "sm"), "Details"));
}

function markLast() {
  const msgs = threadInner.querySelectorAll(".msg.assistant");
  msgs.forEach((x) => x.classList.remove("last"));
  const last = msgs[msgs.length - 1];
  if (last && last === threadInner.lastElementChild) { last.classList.add("last"); fillAssistant(last, last._msg); }
}

function scrollToBottom(force) {
  const near = thread.scrollHeight - thread.scrollTop - thread.clientHeight < 160;
  if (force || near) thread.scrollTop = thread.scrollHeight;
}

export async function openConversation(id) {
  if (controller) stop();
  state.activeConversationId = id;
  emit("active-conversation", id);
  if (!id) {
    conversation = null;
    emit("title", "New conversation");
    renderWelcome();
    showRun(null);
    textarea?.focus();
    return;
  }
  clear(threadInner).append(h("div.skeleton", { style: { height: "48px", width: "60%", marginLeft: "44px", marginBottom: "24px" } }),
    h("div.skeleton", { style: { height: "120px", marginLeft: "44px" } }));
  try {
    conversation = await api(`/api/conversations/${encodeURIComponent(id)}`);
  } catch (e) {
    if (e.status === 404) { toast("That conversation no longer exists.", "err"); location.hash = "#/chat"; return; }
    clear(threadInner).append(h("div.banner.err", icon("alert", "sm"), h("span", e.message)));
    return;
  }
  emit("title", conversation.title);
  clear(threadInner);
  conversation.messages.forEach((m) => threadInner.append(m.role === "user" ? userMessageEl(m) : assistantMessageEl(m)));
  markLast();
  scrollToBottom(true);
  const lastBot = [...conversation.messages].reverse().find((m) => m.role === "assistant");
  showRun(lastBot ? await runFromMessage(lastBot) : null);
}

// ---- sending ----
async function send({ regenerate: isRegen = false } = {}) {
  const text = textarea.value.trim();
  if (!isRegen && !text) return;
  if (controller) return;
  const ready = attachments.filter((a) => a.status === "ready");
  const body = isRegen ? { regenerate: true, conversation_id: conversation.id, kb_mode: kbMode() }
    : { message: text, conversation_id: conversation?.id || null, kb_mode: ready.length ? "always" : kbMode(), document_ids: ready.map((a) => a.id) };

  if (!conversation) clear(threadInner);
  if (!isRegen) {
    threadInner.append(userMessageEl({ content: text, created_at: new Date().toISOString() }));
    textarea.value = ""; autosize();
    attachments = []; renderAttachments();
  }
  threadInner.querySelectorAll(".msg.assistant").forEach((x) => { x.classList.remove("last"); fillAssistant(x, x._msg); });
  const msg = { role: "assistant", content: "", status: "complete", meta: {}, created_at: new Date().toISOString() };
  const el = assistantMessageEl(msg, { streaming: true });
  el.classList.add("last");
  const live = h("div.live-status", h("span.typing", h("i"), h("i"), h("i")), h("span.shine", "Analyzing your question…"));
  el._parts.extras.before(live);
  threadInner.append(el);
  scrollToBottom(true);

  const run = newRun();
  emit("run-started");
  showRun(run);
  controller = new AbortController();
  updateSend();
  let raf = 0;
  const paint = () => { raf = 0; fillAssistant(el, msg, { streaming: true }); scrollToBottom(); };

  try {
    await streamSSE("/api/chat/stream", body, {
      signal: controller.signal,
      onEvent: (ev) => {
        applyEvent(run, ev);
        if (ev.type === "meta") {
          if (!conversation) {
            conversation = { id: ev.conversation_id, title: ev.conversation?.title || text, messages: [] };
            state.activeConversationId = ev.conversation_id;
            history.replaceState(null, "", "#/chat/" + ev.conversation_id);
            emit("active-conversation", ev.conversation_id);
            emit("title", conversation.title);
          }
          emit("conversations-changed");
        } else if (ev.type === "node" && ev.status === "running") {
          live.lastChild.textContent = (NODE_LABEL[ev.node] || ev.node) + "…";
        } else if (ev.type === "token") {
          live.hidden = true;
          msg.content += ev.text;
          if (!raf) raf = requestAnimationFrame(paint);
        } else if (ev.type === "answer") {
          msg.content = ev.text;
        } else if (ev.type === "final") {
          msg.content = ev.answer || msg.content;
          msg.trace_id = ev.trace_id;
          msg.meta = { route: ev.route, route_reason: ev.route_reason, sources: ev.sources, citations: ev.citations, grounded: ev.grounded,
                       retrieval: ev.retrieval, usage: ev.usage, duration_ms: ev.duration_ms, search_queries: ev.search_queries, models: ev.models };
        } else if (ev.type === "error") {
          msg.status = "error";
          msg.meta = { ...msg.meta, error: ev.message };
        } else if (ev.type === "done") {
          msg.id = ev.message_id;
        }
      },
    });
  } catch (e) {
    if (e.name === "AbortError") {
      msg.status = "stopped";
      run.status = "stopped";
    } else {
      msg.status = "error";
      msg.meta = { ...msg.meta, error: e.status === 429 ? e.message : e.message || "Connection lost." };
      run.status = "error"; run.error = msg.meta.error;
      if (e.status === 429 || e.status === 422) toast(e.message, "err");
    }
  } finally {
    if (raf) cancelAnimationFrame(raf);
    controller = null;
    live.remove();
    el._msg = msg;
    fillAssistant(el, msg);
    updateSend();
    showRun(run);
    emit("conversations-changed");
    emit("run-finished");
    if (msg.status === "stopped") setTimeout(() => refreshIds(), 800);
  }
}

async function refreshIds() {
  // the server saved the partial answer; reload so regenerate and details use the stored message
  if (conversation?.id && state.activeConversationId === conversation.id && !controller) {
    try { conversation = await api(`/api/conversations/${encodeURIComponent(conversation.id)}`); } catch { /* ignore */ }
  }
}

function stop() { if (controller) controller.abort(); }

async function regenerate() {
  if (!conversation || controller) return;
  const lastBot = threadInner.querySelector(".msg.assistant.last");
  if (lastBot) lastBot.remove();
  await send({ regenerate: true });
}

export function focusComposer() { textarea?.focus(); }
export function isStreaming() { return !!controller; }

/** Start a new conversation scoped to one document (from the Knowledge base "Ask about" action). */
export function askAbout(doc) {
  attachments = [{ id: doc.id, filename: doc.filename, status: "ready" }];
  renderAttachments();
  textarea.value = "";
  autosize();
  textarea.placeholder = `Ask about ${doc.filename}…`;
  textarea.focus();
}
