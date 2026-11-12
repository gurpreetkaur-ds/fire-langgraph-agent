// Document / citation preview drawer.
import { api } from "./api.js";
import { clear, fileTypeBadge, fmt, h, icon, statusPill, toast } from "./ui.js";

let current = null;

function close() {
  if (!current) return;
  current.scrim.remove();
  current.drawer.remove();
  document.removeEventListener("keydown", current.onKey);
  current = null;
}

function open(title, subtitle) {
  close();
  const body = h("div.drawer-body", h("div.skeleton", { style: { height: "90px", marginBottom: "10px" } }), h("div.skeleton", { style: { height: "160px" } }));
  const drawer = h("aside.drawer", { role: "dialog", "aria-modal": "true", "aria-label": title },
    h("div.drawer-head", h("div.grow", h("h3.ellipsis", title), subtitle ? h("div.faint", { style: { fontSize: "12px" } }, subtitle) : null),
      h("button.btn.ghost.sm.icon-only", { type: "button", "aria-label": "Close", on: { click: close } }, icon("x"))),
    body);
  const scrim = h("div.scrim", { on: { click: close } });
  const onKey = (e) => { if (e.key === "Escape") close(); };
  document.addEventListener("keydown", onKey);
  document.body.append(scrim, drawer);
  current = { drawer, scrim, onKey };
  return body;
}

function chunkCard(c, highlight) {
  const loc = [c.page ? "Page " + c.page : null, c.section].filter(Boolean).join(" · ");
  return h("div.chunk" + (highlight ? ".hl" : ""), { id: "chunk-" + c.id },
    h("div.chunk-head", h("span.pill" + (highlight ? ".accent" : ""), "Chunk " + (c.chunk_index + 1)), loc ? h("span", loc) : null,
      h("span.grow"), h("span", fmt.num(c.chars) + " chars")),
    h("div.chunk-text", c.text));
}

function metaGrid(doc) {
  const stat = (k, v) => h("div.p-stat", h("div.k", k), h("div.v", { title: String(v) }, v));
  const meta = doc.meta || {};
  return h("div.meta-grid",
    stat("Status", statusPill(doc.status, doc.error)), stat("Type", doc.file_type.toUpperCase()), stat("Size", fmt.bytes(doc.size_bytes)),
    stat("Chunks", fmt.num(doc.chunk_count)), stat("Pages", doc.page_count ?? "—"), stat("Characters", fmt.num(doc.char_count)),
    stat("Uploaded", fmt.date(doc.created_at)), stat("Indexed", doc.indexed_at ? fmt.date(doc.indexed_at) : "—"),
    stat("Embedding", doc.embedding_model ? `${doc.embedding_model} · ${doc.embedding_dimensions}d` : "—"),
    meta.title ? stat("Title", meta.title) : null, meta.author ? stat("Author", meta.author) : null,
    meta.row_count != null ? stat("Rows", fmt.num(meta.row_count)) : null);
}

/** Open the passage a citation points to, with its neighbours for context. */
export async function openCitation(source) {
  const body = open(source.filename, [source.page ? "Page " + source.page : null, source.section].filter(Boolean).join(" · ") || "Cited passage");
  try {
    const chunk = await api(`/api/chunks/${encodeURIComponent(source.chunk_id)}`);
    const around = await api(`/api/documents/${encodeURIComponent(chunk.document_id)}/chunks?offset=${Math.max(0, chunk.chunk_index - 1)}&limit=3`);
    clear(body).append(
      h("div.banner.info", { style: { marginBottom: "14px" } }, icon("book", "sm"),
        h("span", `Source ${source.id} — passage ${chunk.position} of ${chunk.chunk_count} in this document.`)),
      ...around.items.map((c) => chunkCard(c, c.id === chunk.id)),
      h("div.row", { style: { marginTop: "12px" } },
        h("button.btn.sm", { type: "button", on: { click: () => openDocument(chunk.document_id, chunk.id) } }, icon("file", "sm"), "Open document"),
        h("a.btn.sm.ghost", { href: `/api/documents/${encodeURIComponent(chunk.document_id)}/file` }, icon("download", "sm"), "Download original")));
    body.querySelector(".chunk.hl")?.scrollIntoView({ block: "center" });
  } catch (e) {
    clear(body).append(h("div.banner.err", icon("alert", "sm"), h("span", e.status === 404
      ? "This passage no longer exists: the document was re-indexed or deleted after this answer."
      : e.message)));
  }
}

/** Document preview: metadata + extracted chunks (paged), optionally scrolled to one chunk. */
export async function openDocument(docId, focusChunkId) {
  const body = open("Document preview");
  try {
    const doc = await api(`/api/documents/${encodeURIComponent(docId)}`);
    current.drawer.querySelector("h3").textContent = doc.filename;
    const list = h("div");
    const more = h("button.btn.sm", { type: "button" }, "Load more");
    let offset = 0, total = 0;
    const load = async () => {
      more.disabled = true;
      const page = await api(`/api/documents/${encodeURIComponent(docId)}/chunks?offset=${offset}&limit=40`);
      total = page.total;
      page.items.forEach((c) => list.append(chunkCard(c, c.id === focusChunkId)));
      offset += page.items.length;
      more.hidden = offset >= total;
      more.disabled = false;
    };
    more.addEventListener("click", () => load().catch((e) => toast(e.message, "err")));
    clear(body).append(
      h("div.row", { style: { marginBottom: "14px" } }, fileTypeBadge(doc.file_type), h("div.grow", h("b", doc.filename), h("div.faint", { style: { fontSize: "12px" } }, doc.sha256.slice(0, 16) + "…")),
        h("a.btn.sm", { href: `/api/documents/${encodeURIComponent(docId)}/file` }, icon("download", "sm"), "Original")),
      doc.error ? h("div.banner.err", { style: { marginBottom: "12px" } }, icon("alert", "sm"), h("span", doc.error)) : null,
      metaGrid(doc),
      h("div.p-title", h("span", "Extracted content"), h("span", fmt.num(doc.chunk_count) + " chunks")),
      doc.chunk_count ? list : h("div.panel-empty", doc.status === "ready" ? "No content." : "Content appears here once indexing finishes."),
      more);
    if (doc.chunk_count) {
      await load();
      if (focusChunkId) {
        while (!list.querySelector("#chunk-" + CSS.escape(focusChunkId)) && offset < total) await load();
        list.querySelector("#chunk-" + CSS.escape(focusChunkId))?.scrollIntoView({ block: "center" });
      }
    } else more.hidden = true;
  } catch (e) {
    clear(body).append(h("div.banner.err", icon("alert", "sm"), h("span", e.message)));
  }
}

export { close as closePreview };
