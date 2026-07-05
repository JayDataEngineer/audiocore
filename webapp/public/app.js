(function () {
  "use strict";

  const $ = (s) => document.querySelector(s);
  const $$ = (s) => document.querySelectorAll(s);

  function status(el, msg, ok) {
    el.textContent = msg;
    el.className = "status " + (ok ? "ok" : ok === false ? "err" : "");
  }

  function human_size(n) {
    for (const u of ["B", "KB", "MB", "GB"]) {
      if (n < 1024) return n.toFixed(u === "B" ? 0 : 1) + " " + u;
      n /= 1024;
    }
    return n.toFixed(1) + " TB";
  }

  function format_dur(s) {
    if (s == null) return "";
    return s.toFixed(2) + "s";
  }

  // ── Fetch clips ──────────────────────────────────────────────────────
  async function load_clips() {
    const list = $("#clips-list");
    try {
      const r = await fetch("/v1/clips");
      if (!r.ok) throw new Error(r.statusText);
      const data = await r.json();
      const clips = data.clips || [];
      $("#clips-header").textContent =
        "Clips (" + clips.length + ")";
      if (!clips.length) {
        list.innerHTML = '<div class="empty">No clips yet.</div>';
        return;
      }
      list.innerHTML = "";
      for (const c of clips) {
        const div = document.createElement("div");
        div.className = "clip";
        div.innerHTML =
          '<audio controls preload="metadata" src="/v1/clips/raw/' +
          enc(c.name) +
          '"></audio>' +
          '<div class="clip-info">' +
          '<div class="clip-name">' + esc(c.name) + "</div>" +
          '<div class="clip-meta">' +
          human_size(c.size) +
          (c.duration ? " &middot; " + format_dur(c.duration) : "") +
          "</div>" +
          "</div>" +
          '<form data-delete="' + enc(c.name) + '">' +
          '<button class="del" type="submit">Delete</button>' +
          "</form>";
        list.appendChild(div);
      }
      for (const f of $$("[data-delete]")) {
        f.addEventListener("submit", delete_clip);
      }
    } catch (e) {
      list.innerHTML = '<div class="empty">Failed to load clips.</div>';
      status($("#server-status"), "offline", false);
    }
  }

  function enc(s) {
    return s.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
  }
  function esc(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  // ── Upload ───────────────────────────────────────────────────────────
  $("#upload-form").addEventListener("submit", async function (e) {
    e.preventDefault();
    const st = $("#upload-status");
    const files = this.querySelector('input[type="file"]').files;
    if (!files.length) return;
    status(st, "Uploading...", null);
    const fd = new FormData();
    for (const f of files) fd.append("file", f, f.name);
    try {
      const r = await fetch("/v1/clips/upload", { method: "POST", body: fd });
      const j = await r.json();
      if (r.ok && j.ok) {
        status(st, "Uploaded " + (j.saved || []).length + " file(s).", true);
        load_clips();
      } else {
        status(st, j.error || "Upload failed.", false);
      }
    } catch (err) {
      status(st, "Upload error: " + err.message, false);
    }
    this.reset();
  });

  // ── Delete ───────────────────────────────────────────────────────────
  async function delete_clip(e) {
    e.preventDefault();
    const name = this.getAttribute("data-delete");
    if (!confirm("Delete " + name + "?")) return;
    try {
      const fd = new URLSearchParams({ name: name });
      const r = await fetch("/v1/clips/delete", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: fd,
      });
      const j = await r.json();
      if (j.ok) load_clips();
    } catch (_) {}
  }

  // ── TTS ──────────────────────────────────────────────────────────────
  async function load_models() {
    try {
      const r = await fetch("/v1/models");
      const j = await r.json();
      const sel = $("#tts-model");
      sel.innerHTML = "";
      for (const m of j.data || []) {
        const o = document.createElement("option");
        o.value = m.id;
        o.textContent = m.id + " (" + m.family + ")";
        sel.appendChild(o);
      }
    } catch (_) {}
  }

  $("#tts-form").addEventListener("submit", async function (e) {
    e.preventDefault();
    const st = $("#tts-status");
    const text = $("#tts-text").value.trim();
    if (!text) return;
    const model = $("#tts-model").value;
    const voice = $("#tts-voice").value.trim();
    const mode = $("#tts-mode").value;
    status(st, "Generating...", null);
    try {
      const body = { model: model, input: text, mode: mode };
      if (voice) body.voice = voice;
      const r = await fetch("/v1/audio/speech", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        status(st, j.error || "TTS failed (" + r.status + ")", false);
        return;
      }
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const a = new Audio(url);
      a.play();
      status(st, "Playing...", true);
      a.onended = function () {
        URL.revokeObjectURL(url);
        status(st, "", null);
      };
    } catch (err) {
      status(st, "TTS error: " + err.message, false);
    }
  });

  // ── Init ─────────────────────────────────────────────────────────────
  status($("#server-status"), "loading...", null);
  Promise.all([load_clips(), load_models()]).then(function () {
    status($("#server-status"), "", null);
  });
})();
