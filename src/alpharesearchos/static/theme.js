"use strict";

// Resolve the theme before CSS is loaded to avoid a flash of the other palette.
(() => {
  const key = "alphaos-theme";
  const system = window.matchMedia("(prefers-color-scheme: light)");
  let preference = null;
  try {
    const stored = window.localStorage.getItem(key);
    if (stored === "light" || stored === "dark") preference = stored;
  } catch { /* Storage may be unavailable in private or restricted contexts. */ }
  function apply(theme) {
    document.documentElement.dataset.theme = theme;
    const button = document.getElementById("theme-toggle");
    if (!button) return;
    const next = theme === "dark" ? "浅色" : "深色";
    button.setAttribute("aria-label", `切换到${next}主题`);
    button.setAttribute("title", `切换到${next}主题`);
    document.getElementById("theme-label").textContent = next;
    document.getElementById("theme-icon").textContent = theme === "dark" ? "☀" : "☾";
  }
  apply(preference || (system.matches ? "light" : "dark"));
  system.addEventListener("change", () => {
    if (!preference) apply(system.matches ? "light" : "dark");
  });
  document.addEventListener("DOMContentLoaded", () => {
    apply(document.documentElement.dataset.theme);
    document.getElementById("theme-toggle").addEventListener("click", () => {
      preference = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      try { window.localStorage.setItem(key, preference); } catch { /* Keep the choice for this session. */ }
      apply(preference);
    });
  });
})();
