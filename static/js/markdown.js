// Markdown -> sanitized HTML, with highlighted code blocks (copy button) and clickable [S#] citation chips.
import { h, icon } from "./ui.js";

const marked = window.marked;
const DOMPurify = window.DOMPurify;
const hljs = window.hljs;

marked.setOptions({ gfm: true, breaks: false });

const PURIFY = {
  ALLOWED_TAGS: ["p", "br", "strong", "em", "del", "code", "pre", "blockquote", "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "h6",
    "table", "thead", "tbody", "tr", "th", "td", "a", "hr", "span", "input"],
  ALLOWED_ATTR: ["href", "title", "class", "type", "checked", "disabled", "align"],
  ALLOWED_URI_REGEXP: /^(?:https?:|mailto:|#)/i,
};

// links open in a new tab and never leak the referrer
DOMPurify.addHook("afterSanitizeAttributes", (node) => {
  if (node.tagName === "A") { node.setAttribute("target", "_blank"); node.setAttribute("rel", "noopener noreferrer"); }
});

const CITE = /\[S(\d{1,3})\]/g;
const HAS_CITE = /\[S\d{1,3}\]/; // non-global: .test() must not carry lastIndex between nodes

/** Render markdown into `el`. onCite(sourceId) is called when a citation chip is clicked. */
export function renderMarkdown(el, text, { onCite, validIds, streaming = false } = {}) {
  const html = DOMPurify.sanitize(marked.parse(text || ""), PURIFY);
  el.innerHTML = html; // sanitized above
  enhanceCode(el, streaming);
  linkCitations(el, onCite, validIds);
}

function enhanceCode(root, streaming) {
  root.querySelectorAll("pre > code").forEach((code) => {
    const pre = code.parentElement;
    const lang = ([...code.classList].find((c) => c.startsWith("language-")) || "").slice(9);
    if (!streaming && hljs) {
      try {
        if (lang && hljs.getLanguage(lang)) code.innerHTML = hljs.highlight(code.textContent, { language: lang }).value;
        else code.innerHTML = hljs.highlightAuto(code.textContent).value;
        code.classList.add("hljs");
      } catch { /* leave as plain text */ }
    }
    const copyBtn = h("button.btn.ghost.sm", { type: "button", "aria-label": "Copy code" }, icon("copy", "sm"), "Copy");
    copyBtn.addEventListener("click", async () => {
      await copyText(code.textContent);
      copyBtn.lastChild.textContent = "Copied";
      setTimeout(() => { copyBtn.lastChild.textContent = "Copy"; }, 1400);
    });
    const block = h("div.codeblock", h("div.cb-head", h("span", lang || "code"), copyBtn));
    pre.replaceWith(block);
    block.append(pre);
  });
}

function linkCitations(root, onCite, validIds) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (n) => (n.parentElement.closest("pre, code, .cite") ? NodeFilter.FILTER_REJECT : HAS_CITE.test(n.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP),
  });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  nodes.forEach((node) => {
    const frag = document.createDocumentFragment();
    let last = 0;
    node.nodeValue.replace(CITE, (m, n, idx) => {
      frag.append(node.nodeValue.slice(last, idx));
      const sid = "S" + n;
      if (!validIds || validIds.has(sid)) {
        const b = h("button.cite", { type: "button", title: "Open source " + sid, "aria-label": "Source " + n }, n);
        b.addEventListener("click", () => onCite && onCite(sid));
        frag.append(b);
      }
      last = idx + m.length;
      return m;
    });
    frag.append(node.nodeValue.slice(last));
    node.replaceWith(frag);
  });
}

export async function copyText(text) {
  try { await navigator.clipboard.writeText(text); return true; }
  catch {
    const ta = h("textarea", { style: { position: "fixed", opacity: "0" } });
    ta.value = text; document.body.append(ta); ta.select();
    try { document.execCommand("copy"); } finally { ta.remove(); }
    return true;
  }
}

/** Plain text for "Copy response": strip citation markers. */
export function plainAnswer(text) { return (text || "").replace(/\s?\[S\d{1,3}\]/g, ""); }
