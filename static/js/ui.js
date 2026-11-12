// Small UI toolkit: DOM builder, icons, toasts, menus, confirm dialog, formatters.
// All dynamic text is set through textContent (never innerHTML) unless it went through DOMPurify (markdown.js).

const ICONS = {
  plus: '<path d="M12 5v14M5 12h14"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
  chat: '<path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12Z"/>',
  library: '<path d="M4 5.5A1.5 1.5 0 0 1 5.5 4H9v16H5.5A1.5 1.5 0 0 1 4 18.5Z"/><path d="M9 4h3.5A1.5 1.5 0 0 1 14 5.5V20H9"/><path d="m14.5 6.2 3.4-.9a1.5 1.5 0 0 1 1.8 1l3 11.3a1.5 1.5 0 0 1-1 1.8l-3.4.9"/>',
  activity: '<path d="M3 12h4l3-8 4 16 3-8h4"/>',
  settings: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1Z"/>',
  sidebar: '<rect x="3" y="4" width="18" height="16" rx="3"/><path d="M9 4v16"/>',
  panel: '<rect x="3" y="4" width="18" height="16" rx="3"/><path d="M15 4v16"/>',
  menu: '<path d="M4 7h16M4 12h16M4 17h16"/>',
  x: '<path d="M18 6 6 18M6 6l12 12"/>',
  logout: '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><path d="m16 17 5-5-5-5M21 12H9"/>',
  send: '<path d="M12 19V5M5 12l7-7 7 7"/>',
  stop: '<rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor" stroke="none"/>',
  paperclip: '<path d="m21.4 11.6-9.2 9.2a6 6 0 0 1-8.5-8.5l9.2-9.2a4 4 0 0 1 5.7 5.7l-9.2 9.2a2 2 0 0 1-2.8-2.8l8.5-8.5"/>',
  copy: '<rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
  refresh: '<path d="M21 12a9 9 0 0 1-15.5 6.2L3 16"/><path d="M3 21v-5h5"/><path d="M3 12a9 9 0 0 1 15.5-6.2L21 8"/><path d="M21 3v5h-5"/>',
  trash: '<path d="M3 6h18M8 6V4h8v2M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>',
  more: '<circle cx="5" cy="12" r="1.2" fill="currentColor"/><circle cx="12" cy="12" r="1.2" fill="currentColor"/><circle cx="19" cy="12" r="1.2" fill="currentColor"/>',
  edit: '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 1 1 3 3L7 19l-4 1 1-4Z"/>',
  upload: '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M20 16v2a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-2"/>',
  file: '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z"/><path d="M14 3v5h5"/>',
  eye: '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/>',
  download: '<path d="M12 4v12M7 11l5 5 5-5"/><path d="M4 20h16"/>',
  sparkle: '<path d="M12 3v4M12 17v4M3 12h4M17 12h4"/><path d="m6.3 6.3 2.1 2.1M15.6 15.6l2.1 2.1M6.3 17.7l2.1-2.1M15.6 8.4l2.1-2.1"/>',
  db: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
  layers: '<path d="m12 2 10 5-10 5L2 7Z"/><path d="m2 17 10 5 10-5M2 12l10 5 10-5"/>',
  route: '<circle cx="6" cy="19" r="3"/><path d="M9 19h8.5a3.5 3.5 0 0 0 0-7h-11a3.5 3.5 0 0 1 0-7H15"/><circle cx="18" cy="5" r="3"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  zap: '<path d="M13 2 3 14h9l-1 8 10-12h-9Z"/>',
  book: '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20V3H6.5A2.5 2.5 0 0 0 4 5.5Z"/><path d="M4 19.5A2.5 2.5 0 0 0 6.5 22H20v-5"/>',
  shield: '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/>',
  alert: '<circle cx="12" cy="12" r="9"/><path d="M12 8v4M12 16h.01"/>',
  chevron: '<path d="m6 9 6 6 6-6"/>',
  compass: '<circle cx="12" cy="12" r="9"/><path d="m15.5 8.5-2 5-5 2 2-5Z"/>',
  code: '<path d="m16 18 6-6-6-6M8 6l-6 6 6 6"/>',
  gauge: '<path d="M12 14 15 9"/><path d="M3.3 16a9 9 0 1 1 17.4 0"/>',
  user: '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
  key: '<circle cx="8" cy="15" r="4"/><path d="m10.8 12.2 9.2-9.2M17 6l3 3"/>',
  filter: '<path d="M3 5h18l-7 8v6l-4 2v-8Z"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M20.5 14.5A8.5 8.5 0 0 1 9.5 3.5a8.5 8.5 0 1 0 11 11Z"/>',
};

export function icon(name, size = "") {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("class", "icon" + (size ? " " + size : ""));
  svg.setAttribute("aria-hidden", "true");
  svg.innerHTML = ICONS[name] || ICONS.sparkle; // static, trusted markup
  return svg;
}

export function hydrateIcons(root = document) {
  root.querySelectorAll("[data-icon]").forEach((el) => {
    if (el.dataset.hydrated) return;
    el.dataset.hydrated = "1";
    el.prepend(icon(el.dataset.icon, el.dataset.size || ""));
  });
}

/** h("div.cls#id", {attrs, on:{click}}, ...children) */
export function h(tag, attrs, ...children) {
  const m = tag.match(/^([a-z0-9-]+)?((?:[.#][\w-]+)*)$/i);
  const el = document.createElement(m[1] || "div");
  (m[2].match(/[.#][\w-]+/g) || []).forEach((p) => (p[0] === "." ? el.classList.add(p.slice(1)) : (el.id = p.slice(1))));
  if (attrs && (typeof attrs !== "object" || attrs instanceof Node || Array.isArray(attrs))) {
    children.unshift(attrs);
    attrs = null;
  }
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "on") Object.entries(v).forEach(([ev, fn]) => el.addEventListener(ev, fn));
    else if (k === "class") el.className += " " + v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k in el && typeof v !== "string") el[k] = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

export function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }

// ---- formatters ----
export const fmt = {
  bytes(n) {
    if (n == null) return "—";
    const u = ["B", "KB", "MB", "GB"]; let i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return (i ? n.toFixed(n < 10 ? 1 : 0) : n) + " " + u[i];
  },
  ms(v) { if (v == null) return "—"; return v >= 1000 ? (v / 1000).toFixed(v >= 10000 ? 1 : 2) + " s" : Math.round(v) + " ms"; },
  num(v) { return v == null ? "—" : Math.round(v).toLocaleString(); },
  pct(v, d = 0) { return v == null ? "—" : (v * 100).toFixed(d) + "%"; },
  time(iso) { if (!iso) return ""; const d = new Date(iso); return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); },
  date(iso) { if (!iso) return ""; return new Date(iso).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }); },
  rel(iso) {
    if (!iso) return "";
    const s = (Date.now() - new Date(iso).getTime()) / 1000;
    if (s < 45) return "just now";
    if (s < 3600) return Math.round(s / 60) + " min ago";
    if (s < 86400) return Math.round(s / 3600) + " h ago";
    if (s < 86400 * 7) return Math.round(s / 86400) + " d ago";
    return new Date(iso).toLocaleDateString([], { month: "short", day: "numeric" });
  },
};

export const ROUTE_LABEL = { rag: "Knowledge base", research: "Research agent", direct: "Direct answer" };
export const NODE_LABEL = {
  analyze_query: "Query analyzer", rewrite_query: "Query rewriter", retrieve: "Document retriever", rerank: "Reranker",
  build_context: "Context builder", generate_answer: "Response generator", validate_citations: "Citation validator",
  direct_answer: "Response generator", decompose: "Question decomposer", answer: "Research agent", synthesize: "Synthesizer",
};

// ---- toasts ----
export function toast(message, kind = "ok", ms = 3800) {
  const t = h("div.toast." + kind, icon(kind === "err" ? "alert" : "check", "sm"), h("span", message));
  document.getElementById("toasts").append(t);
  setTimeout(() => { t.style.opacity = "0"; t.style.transition = "opacity .2s"; setTimeout(() => t.remove(), 220); }, ms);
}

// ---- modal confirm ----
export function confirmDialog({ title, body, confirm = "Confirm", danger = false }) {
  return new Promise((resolve) => {
    const scrim = h("div.scrim");
    const done = (v) => { scrim.remove(); modal.remove(); document.removeEventListener("keydown", onKey); resolve(v); };
    const onKey = (e) => { if (e.key === "Escape") done(false); };
    const ok = h("button.btn" + (danger ? ".danger" : ".primary"), { type: "button", on: { click: () => done(true) } }, confirm);
    const modal = h("div.modal", { role: "dialog", "aria-modal": "true" }, h("h3", title), h("p", body),
      h("div.actions", h("button.btn.ghost", { type: "button", on: { click: () => done(false) } }, "Cancel"), ok));
    scrim.addEventListener("click", () => done(false));
    document.addEventListener("keydown", onKey);
    document.body.append(scrim, modal);
    ok.focus();
  });
}

export function promptDialog({ title, value = "", confirm = "Save" }) {
  return new Promise((resolve) => {
    const input = h("input.input", { value, maxlength: "200" });
    const scrim = h("div.scrim");
    const done = (v) => { scrim.remove(); modal.remove(); resolve(v); };
    const form = h("form", { on: { submit: (e) => { e.preventDefault(); done(input.value.trim() || null); } } }, input,
      h("div.actions", h("button.btn.ghost", { type: "button", on: { click: () => done(null) } }, "Cancel"), h("button.btn.primary", { type: "submit" }, confirm)));
    const modal = h("div.modal", { role: "dialog", "aria-modal": "true" }, h("h3", { style: { marginBottom: "12px" } }, title), form);
    scrim.addEventListener("click", () => done(null));
    input.addEventListener("keydown", (e) => { if (e.key === "Escape") done(null); });
    document.body.append(scrim, modal);
    input.focus(); input.select();
  });
}

// ---- popover menu ----
let openMenu = null;
export function showMenu(anchor, items) {
  closeMenu();
  const menu = h("div.menu", { role: "menu" }, items.map((it) => h("button" + (it.danger ? ".danger" : "") + (it.selected ? ".sel" : ""),
    { type: "button", role: "menuitem", on: { click: () => { closeMenu(); it.action(); } } },
    it.icon ? icon(it.icon, "sm") : null, h("span", it.label, it.desc ? h("span.d", it.desc) : null))));
  document.body.append(menu);
  const r = anchor.getBoundingClientRect();
  const mw = menu.offsetWidth, mh = menu.offsetHeight;
  let top = r.bottom + 6, left = r.left;
  if (top + mh > window.innerHeight - 8) top = r.top - mh - 6;
  if (left + mw > window.innerWidth - 8) left = window.innerWidth - mw - 8;
  menu.style.top = Math.max(8, top) + "px";
  menu.style.left = Math.max(8, left) + "px";
  openMenu = menu;
  setTimeout(() => document.addEventListener("click", closeMenu, { once: true }), 0);
  menu.querySelector("button")?.focus();
}
export function closeMenu() { if (openMenu) { openMenu.remove(); openMenu = null; } }
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeMenu(); });

export function fileTypeBadge(type) {
  return h("span.ftype." + (type || "txt"), (type || "?").toUpperCase());
}

export function statusPill(status, error) {
  const map = {
    queued: ["warn", "Queued"], processing: ["accent", "Indexing"], ready: ["ok", "Ready"], failed: ["err", "Failed"],
  };
  const [cls, label] = map[status] || ["", status];
  return h("span.pill." + cls, { title: error || "" }, h("span.dot" + (status === "processing" || status === "queued" ? ".pulse" : "")), label);
}
