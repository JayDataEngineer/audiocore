(function () {
  "use strict";

  const $ = (s) => document.querySelector(s);
  const $$ = (s) => document.querySelectorAll(s);

  // ── Helpers ────────────────────────────────────────────────────────────
  function status(el, msg, type) {
    el.textContent = msg;
    el.className = "status" + (type ? " " + type : "");
  }

  function toast(msg, type) {
    const t = $("#toast");
    t.textContent = msg;
    t.className = "toast show " + type;
    clearTimeout(t._timer);
    t._timer = setTimeout(() => (t.className = "toast"), 3000);
  }

  function human_size(n) {
    for (const u of ["B", "KB", "MB", "GB"]) {
      if (n < 1024) return n.toFixed(u === "B" ? 0 : 1) + " " + u;
      n /= 1024;
    }
    return n.toFixed(1) + " TB";
  }

  function fmt_dur(s) {
    if (s == null) return "";
    const m = Math.floor(s / 60);
    const sec = (s % 60).toFixed(1);
    return m > 0 ? m + ":" + sec.padStart(4, "0") : sec + "s";
  }

  function enc(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  // ── Clips ──────────────────────────────────────────────────────────────
  async function load_clips() {
    const list = $("#clips-list");
    try {
      const r = await fetch("/v1/clips");
      if (!r.ok) throw new Error(r.statusText);
      const clips = (await r.json()).clips || [];

      $("#clips-count").textContent = clips.length ? "(" + clips.length + ")" : "";

      if (!clips.length) {
        list.innerHTML =
          '<div class="empty"><div class="empty-icon">&#9835;</div>No clips yet &mdash; upload some audio above.</div>';
        return;
      }

      list.innerHTML = "";
      for (const c of clips) {
        const el = document.createElement("div");
        el.className = "clip";
        el.innerHTML =
          '<audio controls preload="metadata" src="/v1/clips/raw/' + enc(c.name) + '"></audio>' +
          '<div class="clip-info">' +
            '<div class="clip-name" title="' + enc(c.name) + '">' + enc(c.name) + "</div>" +
            '<div class="clip-meta">' +
              human_size(c.size) +
              (c.duration ? " &middot; " + fmt_dur(c.duration) : "") +
            "</div>" +
          "</div>" +
          '<button class="btn btn-danger" data-delete="' + enc(c.name) + '">Delete</button>';
        list.appendChild(el);
      }

      for (const btn of $$("[data-delete]")) {
        btn.addEventListener("click", delete_clip);
      }
    } catch (e) {
      list.innerHTML =
        '<div class="empty"><div class="empty-icon">&#9888;</div>Could not reach server.</div>';
    }
  }

  // ── Upload ─────────────────────────────────────────────────────────────
  $("#upload-form").addEventListener("submit", async function (e) {
    e.preventDefault();
    const st = $("#upload-status");
    const btn = this.querySelector(".btn");
    const files = this.querySelector('input[type="file"]').files;
    if (!files.length) return;

    btn.disabled = true;
    status(st, "Uploading...", "loading");

    const fd = new FormData();
    for (const f of files) fd.append("file", f, f.name);

    try {
      const r = await fetch("/v1/clips/upload", { method: "POST", body: fd });
      const j = await r.json();
      if (r.ok && j.ok) {
        const n = (j.saved || []).length;
        status(st, "Uploaded " + n + " file" + (n !== 1 ? "s" : "") + ".", "ok");
        toast(n + " clip" + (n !== 1 ? "s" : "") + " uploaded", "ok");
        load_clips();
      } else {
        status(st, j.error || "Upload failed.", "err");
      }
    } catch (err) {
      status(st, "Error: " + err.message, "err");
    }

    btn.disabled = false;
    this.reset();
  });

  // ── Delete ─────────────────────────────────────────────────────────────
  async function delete_clip(e) {
    e.preventDefault();
    const name = this.getAttribute("data-delete");
    if (!confirm('Delete "' + name + '"?')) return;

    this.disabled = true;
    try {
      const r = await fetch("/v1/clips/delete", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({ name: name }),
      });
      const j = await r.json();
      if (j.ok) {
        toast("Deleted " + name, "ok");
        load_clips();
      } else {
        toast(j.error || "Delete failed", "err");
      }
    } catch (_) {
      toast("Delete failed", "err");
    }
  }

  // ── Models ─────────────────────────────────────────────────────────────
  async function load_models() {
    try {
      const r = await fetch("/v1/models");
      const j = await r.json();
      const models = j.data || [];
      const sel = $("#tts-model");
      const badge = $("#model-badge");

      sel.innerHTML = "";
      if (!models.length) {
        sel.innerHTML = '<option value="">No models loaded</option>';
        sel.disabled = true;
        badge.textContent = "no model";
        badge.style.background = "rgba(239,68,68,.12)";
        badge.style.color = "#ef4444";
        return;
      }

      for (const m of models) {
        const o = document.createElement("option");
        o.value = m.id;
        o.textContent = m.id;
        sel.appendChild(o);
      }

      badge.textContent = models[0].family.replace(/_/g, " ");
    } catch (_) {
      $("#model-badge").textContent = "offline";
    }
  }

  // ── TTS ────────────────────────────────────────────────────────────────
  let playing_audio = null;

  $("#tts-form").addEventListener("submit", async function (e) {
    e.preventDefault();
    const st = $("#tts-status");
    const btn = $("#tts-btn");
    const text = $("#tts-text").value.trim();
    if (!text) return;

    const model = $("#tts-model").value;
    if (!model) {
      status(st, "No model loaded.", "err");
      return;
    }

    const voice = $("#tts-voice").value.trim();
    const mode = $("#tts-mode").value;

    btn.disabled = true;
    btn.textContent = "Generating\u2026";
    status(st, "", "");

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
        status(st, j.error || "TTS failed (" + r.status + ")", "err");
        return;
      }

      const blob = await r.blob();
      const url = URL.createObjectURL(blob);

      if (playing_audio) {
        playing_audio.pause();
        URL.revokeObjectURL(playing_audio.src);
      }

      const a = new Audio(url);
      playing_audio = a;
      a.play();
      status(st, "Playing\u2026", "ok");
      a.onended = function () {
        URL.revokeObjectURL(url);
        if (playing_audio === a) playing_audio = null;
        status(st, "", "");
      };
    } catch (err) {
      status(st, "Error: " + err.message, "err");
    }

    btn.disabled = false;
    btn.textContent = "Synthesize";
  });

  // ── Init ───────────────────────────────────────────────────────────────
  Promise.all([load_clips(), load_models()]);
})();
