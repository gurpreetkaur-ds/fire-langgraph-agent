// Runs synchronously in <head> before first paint so the saved theme applies without a flash.
// (An external file because the CSP forbids inline scripts.) Light is the default.
(function () {
  var t = "light";
  try { if (localStorage.getItem("theme") === "dark") t = "dark"; } catch (e) { /* storage blocked */ }
  document.documentElement.setAttribute("data-theme", t);
  document.documentElement.style.colorScheme = t;
})();
