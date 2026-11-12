// Light / dark theme. Light is the default; the choice is remembered per browser.
// The <head> runs a tiny inline script that sets data-theme before first paint (no flash); this module owns the toggle.
const KEY = "theme";
const META_COLOR = { light: "#FAF8F5", dark: "#14121C" };

export function currentTheme() {
  return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}

function apply(theme) {
  const root = document.documentElement;
  root.dataset.theme = theme;
  root.style.colorScheme = theme;
  document.querySelector('meta[name="theme-color"]')?.setAttribute("content", META_COLOR[theme]);
  document.querySelectorAll("[data-theme-toggle]").forEach((b) => {
    const next = theme === "dark" ? "light" : "dark";
    b.setAttribute("aria-label", `Switch to ${next} mode`);
    b.title = `Switch to ${next} mode`;
    b.setAttribute("aria-pressed", String(theme === "dark"));
    const label = b.querySelector(".label");
    if (label) label.textContent = theme === "dark" ? "Light mode" : "Dark mode"; // names the mode a click switches to
  });
}

export function setTheme(theme) {
  const root = document.documentElement;
  // Animate colours only while switching, so normal hovers keep their own fast timing.
  root.classList.add("theme-anim");
  apply(theme);
  try { localStorage.setItem(KEY, theme); } catch { /* storage blocked: the choice just isn't remembered */ }
  clearTimeout(setTheme._t);
  setTheme._t = setTimeout(() => root.classList.remove("theme-anim"), 450);
}

export function initTheme() {
  apply(currentTheme());
  document.querySelectorAll("[data-theme-toggle]").forEach((b) =>
    b.addEventListener("click", () => setTheme(currentTheme() === "dark" ? "light" : "dark")));
}
