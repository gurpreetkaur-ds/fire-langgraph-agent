// App shell: authentication, routing, sidebar (conversation history), intelligence panel, health.
import { mountActivity, showActivity } from "./activity.js";
import { api, setUnauthorizedHandler } from "./api.js";
import { askAbout, isStreaming, mountChat, openConversation } from "./chat.js";
import { mountKnowledge, refresh as refreshDocs, showKnowledge } from "./knowledge.js";
import { render as renderPanel } from "./panel.js";
import { mountSettings, showSettings } from "./settings.js";
import { emit, on, state, storageGet, storageSet } from "./state.js";
import { initTheme } from "./theme.js";
import { clear, confirmDialog, h, hydrateIcons, icon, promptDialog, showMenu, toast } from "./ui.js";

const $ = (id) => document.getElementById(id);
const app = $("app");
const TITLES = { knowledge: "Knowledge base", activity: "Agent activity", settings: "Settings", eval: "Evaluation Center" };

// ---------------- auth ----------------
let signupMode = false;

function showAuth() {
  $("app").hidden = true;
  $("authScreen").hidden = false;
  const inviteRequired = state.config?.invite_required;
  const setMode = (signup) => {
    signupMode = signup;
    $("authTabLogin").classList.toggle("on", !signup); $("authTabLogin").setAttribute("aria-selected", String(!signup));
    $("authTabSignup").classList.toggle("on", signup); $("authTabSignup").setAttribute("aria-selected", String(signup));
    $("authNameField").hidden = !signup;
    $("authInviteField").hidden = !(signup && inviteRequired);
    $("authPassHint").hidden = !signup;
    $("authPass").autocomplete = signup ? "new-password" : "current-password";
    $("authSubmit").textContent = signup ? "Create account" : "Sign in";
    $("authError").hidden = true;
  };
  $("authTabLogin").onclick = () => setMode(false);
  $("authTabSignup").onclick = () => setMode(true);
  $("authTabSignup").hidden = state.config && !state.config.signup_open;
  setMode(false);
  $("authUser").focus();
}

$("authForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = $("authSubmit");
  btn.disabled = true;
  $("authError").hidden = true;
  try {
    const body = { username: $("authUser").value.trim(), password: $("authPass").value };
    if (signupMode) Object.assign(body, { display_name: $("authName").value.trim(), invite_code: $("authInvite").value.trim() || null });
    const r = await api(signupMode ? "/api/auth/signup" : "/api/auth/login", { method: "POST", body });
    $("authPass").value = ""; $("authInvite").value = "";
    state.user = r.user;
    await startApp();
  } catch (err) {
    $("authError").hidden = false;
    $("authError").textContent = err.message;
  } finally { btn.disabled = false; }
});

setUnauthorizedHandler(() => {
  if (!state.user) return;
  state.user = null;
  toast("Your session ended. Please sign in again.", "err");
  showAuth();
});

$("logoutBtn").addEventListener("click", async () => {
  try { await api("/api/auth/logout", { method: "POST" }); } catch { /* ignore */ }
  location.hash = "";
  location.reload();
});

// ---------------- sidebar ----------------
function renderUser() {
  const u = state.user;
  $("userName").textContent = u.display_name;
  $("userHandle").textContent = "@" + u.username;
  $("userAvatar").textContent = (u.display_name || "?").trim().slice(0, 1).toUpperCase();
}

function groupLabel(iso) {
  const d = new Date(iso), now = new Date();
  const day = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const diff = (day(now) - day(d)) / 86400000;
  if (diff < 1) return "Today";
  if (diff < 2) return "Yesterday";
  if (diff < 7) return "Previous 7 days";
  if (diff < 30) return "Previous 30 days";
  return d.toLocaleDateString([], { month: "long", year: "numeric" });
}

async function loadConversations() {
  try {
    state.conversations = await api("/api/conversations" + (state.convQuery ? "?q=" + encodeURIComponent(state.convQuery) : ""));
  } catch { return; }
  renderConversations();
}

function renderConversations() {
  const list = $("convList");
  clear(list);
  if (!state.conversations.length) {
    list.append(h("div.panel-empty", { style: { padding: "8px 10px" } }, state.convQuery ? "No conversations match." : "No conversations yet."));
    return;
  }
  let group = null;
  state.conversations.forEach((c) => {
    const g = state.convQuery ? "Results" : groupLabel(c.updated_at);
    if (g !== group) { group = g; list.append(h("div.conv-group", g)); }
    const more = h("span.more", { role: "button", tabindex: "0", "aria-label": "Conversation options" }, icon("more", "sm"));
    const openMenu = (e) => {
      e.preventDefault(); e.stopPropagation();
      showMenu(more, [
        { label: "Rename", icon: "edit", action: () => rename(c) },
        { label: "Delete", icon: "trash", danger: true, action: () => remove(c) },
      ]);
    };
    more.addEventListener("click", openMenu);
    more.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") openMenu(e); });
    list.append(h("a.conv" + (c.id === state.activeConversationId ? ".active" : ""), { href: "#/chat/" + c.id, title: c.title },
      h("span.t", c.title, c.snippet ? h("span.snip", c.snippet) : null), more));
  });
}

async function rename(c) {
  const title = await promptDialog({ title: "Rename conversation", value: c.title });
  if (!title) return;
  try { await api(`/api/conversations/${encodeURIComponent(c.id)}`, { method: "PATCH", body: { title } }); await loadConversations(); if (c.id === state.activeConversationId) setTitle(title); }
  catch (e) { toast(e.message, "err"); }
}

async function remove(c) {
  const ok = await confirmDialog({ title: "Delete conversation?", body: `“${c.title}” and all its messages will be permanently deleted.`, confirm: "Delete", danger: true });
  if (!ok) return;
  try {
    await api(`/api/conversations/${encodeURIComponent(c.id)}`, { method: "DELETE" });
    if (c.id === state.activeConversationId) location.hash = "#/chat";
    await loadConversations();
    toast("Conversation deleted");
  } catch (e) { toast(e.message, "err"); }
}

let searchTimer;
$("convSearch").addEventListener("input", (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => { state.convQuery = e.target.value.trim(); loadConversations(); }, 250);
});
$("newChatBtn").addEventListener("click", () => {
  if (location.hash === "#/chat" || location.hash === "" ) openConversation(null); else location.hash = "#/chat";
  closeNav();
});

function updateDocCount() {
  const n = (state.docs || []).filter((d) => d.status === "ready").length;
  $("navDocCount").textContent = n ? String(n) : "";
}

// ---------------- layout toggles ----------------
function setPanel(open, persist = true) {
  state.panelOpen = open;
  app.classList.toggle("panel-open", open);
  $("panelBtn").setAttribute("aria-pressed", String(open));
  if (persist) storageSet("panelOpen", open);
  if (open) renderPanel();
}
$("panelBtn").addEventListener("click", () => setPanel(!state.panelOpen));
$("panelClose").addEventListener("click", () => setPanel(false));
on("panel", (open) => setPanel(open));
// Below this width the panel overlays the chat: close it when the viewport shrinks (rotation, window resize).
const overlayMq = window.matchMedia("(max-width: 1100px)");
overlayMq.addEventListener("change", (e) => { if (e.matches && state.panelOpen) setPanel(false, false); });

function setCollapsed(c) { app.classList.toggle("collapsed", c); storageSet("sidebarCollapsed", c); $("collapseBtn").title = c ? "Expand sidebar" : "Collapse sidebar"; }
$("collapseBtn").addEventListener("click", () => setCollapsed(!app.classList.contains("collapsed")));
const closeNav = () => app.classList.remove("nav-open");
$("openNavBtn").addEventListener("click", () => app.classList.add("nav-open"));
$("closeNavBtn").addEventListener("click", closeNav);
// on phones the open drawer dims the page behind it (a pseudo-element of .app); tapping that dimmed area closes it
app.addEventListener("click", (e) => { if (e.target === app && app.classList.contains("nav-open")) closeNav(); });

function setTitle(t) { $("viewTitle").textContent = t; document.title = t && t !== "Chat" ? `${t} · Fire Agent` : "Fire Agent"; }
on("title", setTitle);
on("active-conversation", () => renderConversations());
on("conversations-changed", () => loadConversations());
on("docs-changed", () => refreshDocs());
on("docs-loaded", updateDocCount);
on("user-changed", renderUser);
on("ask-about", (doc) => { location.hash = "#/chat"; setTimeout(() => askAbout(doc), 50); });

// ---------------- routing ----------------
async function route() {
  if (!state.user) return;
  const [, view = "chat", id] = (location.hash || "#/chat").replace(/^#/, "").split("/");
  const name = ["chat", "knowledge", "activity", "settings", "eval"].includes(view) ? view : "chat";
  document.querySelectorAll(".view").forEach((v) => { v.hidden = v.dataset.view !== name; });
  document.querySelectorAll("[data-nav]").forEach((a) => a.classList.toggle("active", a.dataset.nav === name));
  closeNav();
  if (name === "chat") {
    if ((id || null) !== state.activeConversationId || !id) {
      if (!(id && isStreaming() && id === state.activeConversationId)) await openConversation(id || null);
    }
    setTitle(id ? (state.conversations.find((c) => c.id === id)?.title || "Chat") : "New conversation");
  } else {
    setTitle(TITLES[name]);
    if (name === "knowledge") showKnowledge();
    if (name === "activity") showActivity();
    if (name === "settings") showSettings();
    if (name === "eval") window.EvalCenter?.open();
  }
}
window.addEventListener("hashchange", route);

// ---------------- health ----------------
async function checkHealth() {
  const pill = $("healthPill");
  try {
    const d = await api("/api/health");
    pill.className = "pill " + (d.fire_key_configured ? "ok" : "err");
    $("healthText").textContent = d.fire_key_configured ? "Online" : "Model key missing";
    pill.title = d.fire_key_configured ? "Server is up; the Fire API key is configured" : "The server has no Fire API key";
  } catch {
    pill.className = "pill err";
    $("healthText").textContent = "Offline";
  }
}

// ---------------- boot ----------------
let mounted = false;
async function startApp() {
  $("authScreen").hidden = true;
  app.hidden = false;
  renderUser();
  if (!mounted) {
    mounted = true;
    mountChat($("view-chat"));
    mountKnowledge($("view-knowledge"));
    mountActivity($("view-activity"));
    mountSettings($("view-settings"));
    hydrateIcons();
    setCollapsed(storageGet("sidebarCollapsed", false));
    const prefPanel = state.user.preferences?.show_panel;
    // the panel overlays the chat on narrow screens, so it always starts closed there
    setPanel(window.innerWidth > 1100 && storageGet("panelOpen", prefPanel !== false), false);
    checkHealth();
    setInterval(checkHealth, 60000);
  }
  await Promise.all([loadConversations(), refreshDocs()]);
  await route();
}

async function boot() {
  hydrateIcons();
  initTheme();
  try { state.config = await api("/api/config"); } catch { state.config = { signup_open: true, invite_required: false, rag: {} }; }
  try {
    const r = await api("/api/me");
    state.user = r.user;
    await startApp();
  } catch {
    showAuth();
  }
}

boot();
export { emit };
