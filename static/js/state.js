// Tiny shared store with change notifications.
const listeners = new Map();

export const state = {
  user: null,
  config: null,
  conversations: [],
  convQuery: "",
  activeConversationId: null,
  docs: [],
  docStats: null,
  panelOpen: false,
  run: null, // the live or selected run shown in the intelligence panel
};

export function on(event, fn) {
  if (!listeners.has(event)) listeners.set(event, new Set());
  listeners.get(event).add(fn);
  return () => listeners.get(event).delete(fn);
}

export function emit(event, data) {
  (listeners.get(event) || []).forEach((fn) => { try { fn(data); } catch (e) { console.error(e); } });
}

export function storageGet(key, fallback) {
  try { const v = localStorage.getItem(key); return v == null ? fallback : JSON.parse(v); } catch { return fallback; }
}
export function storageSet(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage blocked: preference just isn't remembered */ }
}
