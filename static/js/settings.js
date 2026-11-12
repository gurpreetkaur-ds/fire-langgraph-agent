// Settings: profile, preferences, password, RAG configuration (read-only), admin evaluation link.
import { api } from "./api.js";
import { emit, state } from "./state.js";
import { clear, fmt, h, icon, toast } from "./ui.js";

let root;

export function mountSettings(container) { root = container; }

export function showSettings() {
  const u = state.user;
  const rag = state.config?.rag || {};
  const name = h("input.input", { value: u.display_name, maxlength: "80" });
  const saveProfile = async (e) => {
    e.preventDefault();
    try {
      const r = await api("/api/me", { method: "PATCH", body: { display_name: name.value } });
      state.user = { ...state.user, display_name: r.user.display_name };
      emit("user-changed");
      toast("Profile saved");
    } catch (err) { toast(err.message, "err"); }
  };

  const modeSel = h("select.select", ["auto", "always", "off"].map((m) => h("option", { value: m, selected: (u.preferences.kb_mode || "auto") === m },
    { auto: "Auto — the agent decides", always: "Always search my documents", off: "Never search my documents" }[m])));
  modeSel.addEventListener("change", () => savePrefs({ kb_mode: modeSel.value }));
  const panelToggle = h("input", { type: "checkbox", checked: u.preferences.show_panel !== false });
  panelToggle.addEventListener("change", () => { savePrefs({ show_panel: panelToggle.checked }); emit("panel", panelToggle.checked); });

  const cur = h("input.input", { type: "password", autocomplete: "current-password" });
  const nw = h("input.input", { type: "password", autocomplete: "new-password", minlength: "10" });
  const pwErr = h("div.banner.err", { hidden: true });
  const changePw = async (e) => {
    e.preventDefault();
    pwErr.hidden = true;
    try {
      await api("/api/me/password", { method: "POST", body: { current_password: cur.value, new_password: nw.value } });
      cur.value = nw.value = "";
      toast("Password changed. Other devices were signed out.");
    } catch (err) { pwErr.hidden = false; clear(pwErr).append(icon("alert", "sm"), h("span", err.message)); }
  };

  const kv = (k, v) => [h("dt", k), h("dd", v)];
  clear(root).append(h("div.page",
    h("div.page-head", h("div", h("h2", "Settings"), h("p", "Your account, preferences and how retrieval is configured on this server."))),
    h("div.settings-grid",
      h("div.card", h("h3", "Profile"), h("p", "@" + u.username),
        h("form.form-stack", { on: { submit: saveProfile } }, h("label.field", h("span", "Display name"), name), h("div", h("button.btn.primary", { type: "submit" }, "Save")))),
      h("div.card", h("h3", "Preferences"), h("p", "Saved to your account."),
        h("div.form-stack",
          h("label.field", h("span", "Knowledge base mode"), modeSel, h("span.hint", "Auto routes document questions to retrieval and answers everything else directly.")),
          h("label.row", { style: { justifyContent: "space-between" } }, h("span", h("div", "Show intelligence panel"), h("div.faint", { style: { fontSize: "12px" } }, "Live workflow, tools and sources beside the chat")),
            h("span.switch", panelToggle, h("i"))))),
      h("div.card", h("h3", "Password"), h("p", "Changing it signs you out everywhere else."),
        h("form.form-stack", { on: { submit: changePw } },
          h("label.field", h("span", "Current password"), cur), h("label.field", h("span", "New password"), nw, h("span.hint", "At least 10 characters.")),
          pwErr, h("div", h("button.btn", { type: "submit" }, "Change password")))),
      h("div.card", h("h3", "Retrieval configuration"), h("p", "Set by the server administrator (environment variables)."),
        h("dl.kv-list",
          kv("Embedding model", `${rag.embedding_model} (${rag.embedding_dimensions} dims)`), kv("Chunk size", `${rag.chunk_size} chars, ${rag.chunk_overlap} overlap`),
          kv("Top-K", rag.top_k), kv("Similarity threshold", rag.similarity_threshold), kv("Context budget", fmt.num(rag.context_char_budget) + " chars"),
          kv("Upload limit", `${rag.max_upload_mb} MB per file, ${rag.max_documents} documents`), kv("File types", (rag.allowed_types || []).join(" ")))),
      h("div.card", h("h3", "Evaluation Center"), h("p", "Admin dashboard with LLM-judge scores for every request (needs the evaluation token)."),
        h("a.btn", { href: "#/eval" }, icon("gauge", "sm"), "Open Evaluation Center")))));
}

async function savePrefs(prefs) {
  try {
    const r = await api("/api/me", { method: "PATCH", body: { preferences: prefs } });
    state.user.preferences = r.user.preferences;
    toast("Preference saved");
  } catch (e) { toast(e.message, "err"); }
}
