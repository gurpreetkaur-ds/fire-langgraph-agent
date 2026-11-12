// HTTP client. Every request is same-origin with the session cookie; state-changing requests carry the
// X-Requested-With header the server requires (CSRF protection).

export class ApiError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

let onUnauthorized = () => {};
export function setUnauthorizedHandler(fn) { onUnauthorized = fn; }

async function parseError(res) {
  let detail = "";
  try {
    const body = await res.json();
    detail = typeof body.detail === "string" ? body.detail
      : Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join("; ") : "";
  } catch { /* not JSON */ }
  if (res.status === 413) detail = "That file is too large for the server.";
  return new ApiError(res.status, detail || `Request failed (HTTP ${res.status})`);
}

export async function api(path, { method = "GET", body, form, signal, raw } = {}) {
  const headers = {};
  if (method !== "GET") headers["X-Requested-With"] = "fetch";
  let payload;
  if (form) payload = form;
  else if (body !== undefined) { headers["Content-Type"] = "application/json"; payload = JSON.stringify(body); }
  const res = await fetch(path, { method, headers, body: payload, signal, credentials: "same-origin" });
  if (res.status === 401 && !path.startsWith("/api/auth/")) onUnauthorized();
  if (!res.ok) throw await parseError(res);
  if (raw) return res;
  return res.status === 204 ? null : res.json();
}

/** POST and read a Server-Sent Events stream; calls onEvent(payload) per event. Abort with the signal. */
export async function streamSSE(path, body, { signal, onEvent }) {
  const res = await api(path, { method: "POST", body, signal, raw: true });
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let i;
    while ((i = buffer.indexOf("\n\n")) !== -1) {
      const chunk = buffer.slice(0, i);
      buffer = buffer.slice(i + 2);
      const line = chunk.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      try { onEvent(JSON.parse(line.slice(6))); } catch (e) { console.error("bad event", e); }
    }
  }
}
