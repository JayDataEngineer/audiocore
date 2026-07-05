// ════════════════════════════════════════════════════════════════════
// audiocore voice studio — application controller                       //
// ════════════════════════════════════════════════════════════════════
(function () {
  "use strict";

  const $  = (s) => document.querySelector(s);
  const $$ = (s) => document.querySelectorAll(s);

  // ── Helpers ────────────────────────────────────────────────────────
  function status(el, msg, type) {
    if (!el) return;
    el.textContent = msg || "";
    el.className = "status" + (type ? " " + type : "");
  }

  function toast(msg, type) {
    const t = $("#toast");
    t.textContent = msg;
    t.className = "toast show " + (type || "ok");
    clearTimeout(t._timer);
    t._timer = setTimeout(() => (t.className = "toast"), 3000);
  }

  function fmt_size(n) {
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

  function esc(s) {
    const d = document.createElement("div");
    d.textContent = s == null ? "" : String(s);
    return d.innerHTML;
  }

  function set_player(el, blob_url, meta) {
    el.innerHTML = "";
    el.classList.add("has-audio");
    const a = document.createElement("audio");
    a.controls = true;
    a.autoplay = true;
    a.src = blob_url;
    el.appendChild(a);
    if (meta) {
      const m = document.createElement("div");
      m.className = "player-meta";
      m.textContent = meta;
      el.appendChild(m);
    }
    a.play().catch(() => {});
  }

  function clear_player(el) {
    el.classList.remove("has-audio");
    el.innerHTML = '<div class="player-empty">Generated audio appears here</div>';
  }

  // ── Tab switching ──────────────────────────────────────────────────
  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.tab;
      $$(".tab").forEach((t) => t.classList.toggle("active", t === tab));
      $$(".panel").forEach((p) =>
        p.classList.toggle("active", p.id === "tab-" + target)
      );
      if (target === "voices") load_voices();
      if (target === "clips")  load_clips();
    });
  });

  // ── Models ─────────────────────────────────────────────────────────
  let MODELS = [];
  let ACTIVE_TTS = null;
  let ACTIVE_VD  = null;
  let ACTIVE_MUSIC = null;

  async function load_models() {
    try {
      const r = await fetch("/v1/models");
      const j = await r.json();
      MODELS = j.data || [];
      const badge = $("#model-badge");

      if (!MODELS.length) {
        badge.textContent = "no models";
        badge.className = "badge err";
        return;
      }

      // Pick models by family/heuristic
      for (const m of MODELS) {
        const f = (m.family || "").toLowerCase();
        const id = (m.id || "").toLowerCase();
        if (f.indexOf("qwen3_tts") >= 0) {
          if (id.indexOf("voicedesign") >= 0 || id.indexOf("voice_design") >= 0)
            ACTIVE_VD = m.id;
          else if (ACTIVE_TTS == null || id.indexOf("customvoice") >= 0 ||
                   id.indexOf("cv") >= 0)
            ACTIVE_TTS = m.id;
        }
        if (f.indexOf("ace") >= 0 || id.indexOf("ace") >= 0)
          ACTIVE_MUSIC = m.id;
        if (f.indexOf("moss") >= 0 && ACTIVE_TTS == null)
          ACTIVE_TTS = m.id;
      }

      badge.textContent = MODELS.length + " model" + (MODELS.length === 1 ? "" : "s");
      badge.className = "badge";
      console.log({ ACTIVE_TTS, ACTIVE_VD, ACTIVE_MUSIC, MODELS });
    } catch (e) {
      $("#model-badge").textContent = "offline";
      $("#model-badge").className = "badge err";
    }
  }

  // ── Voice Design tab ───────────────────────────────────────────────
  $$("#vd-presets .chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      $("#vd-instruct").value = chip.dataset.desc;
      $("#vd-instruct").focus();
    });
  });

  let vd_last_blob = null;

  $("#vd-btn").addEventListener("click", async () => {
    const st = $("#vd-status");
    const btn = $("#vd-btn");
    const instruct = $("#vd-instruct").value.trim();
    const text = $("#vd-text").value.trim();
    if (!instruct) { status(st, "Describe a voice first.", "err"); return; }
    if (!text)     { status(st, "Type some sample text.", "err"); return; }
    if (!ACTIVE_VD && !ACTIVE_TTS) {
      status(st, "No TTS model loaded.", "err");
      return;
    }

    btn.disabled = true;
    btn.textContent = "Designing…";
    status(st, "Generating audio…", "loading");
    clear_player($("#vd-player"));

    try {
      const body = {
        model: ACTIVE_VD || ACTIVE_TTS,
        input: text,
        mode: "voice_design",
        instruct,
        temperature: parseFloat($("#vd-temp").value) || 0.8,
        top_p:       parseFloat($("#vd-topp").value) || 0.9,
        text_top_k:  50,
        max_new_tokens: 100,
      };
      const r = await fetch("/v1/audio/speech", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        status(st, j.error || "Voice design failed (" + r.status + ")", "err");
        return;
      }
      const blob = await r.blob();
      vd_last_blob = blob;
      const url = URL.createObjectURL(blob);
      set_player($("#vd-player"), url,
        "voice_design · " + (blob.size / 1024).toFixed(1) + " KB");
      status(st, "Done. Save the result as a .voice to reuse in Synthesize.", "ok");
      $("#vd-save").disabled = false;
    } catch (err) {
      status(st, "Error: " + err.message, "err");
    }

    btn.disabled = false;
    btn.textContent = "✨ Design voice";
  });

  $("#vd-save").addEventListener("click", async () => {
    if (!vd_last_blob) return;
    const st = $("#vd-status");
    status(st, "Encoding & saving…", "loading");
    try {
      const form = new FormData();
      form.append("file", vd_last_blob, "designed_" + Date.now() + ".wav");
      form.append("encode_voice", "true");
      const r = await fetch("/v1/clips/upload", { method: "POST", body: form });
      const j = await r.json();
      if (j.ok && j.saved && j.saved.length) {
        toast("Saved as " + j.saved[0], "ok");
        status(st, "Saved to voices. Use it from Synthesize → Voice → Uploaded.", "ok");
      } else {
        status(st, j.error || "Save failed", "err");
      }
    } catch (err) {
      status(st, "Error: " + err.message, "err");
    }
  });

  // ── Synthesize tab ─────────────────────────────────────────────────
  const PRESET_SPEAKERS = [
    "Vivian", "Serena", "Ryan", "Aiden", "Eric",
    "Dylan", "Sohee", "Ono Anna", "Uncle Fu"
  ];

  function populate_presets() {
    const sel = $("#syn-preset");
    sel.innerHTML = "";
    for (const s of PRESET_SPEAKERS) {
      const o = document.createElement("option");
      o.value = s.toLowerCase().replace(/\s+/g, "_");
      o.textContent = s;
      sel.appendChild(o);
    }
  }
  populate_presets();

  // Voice source change → toggle selects
  $("#syn-voice-source").addEventListener("change", (e) => {
    const v = e.target.value;
    $("#syn-preset").hidden   = v !== "preset";
    $("#syn-uploaded").hidden = v !== "uploaded";
    $("#syn-ref-row").hidden  = v !== "reference";
  });

  // Slider live values
  $("#syn-pitch").addEventListener("input", (e) => {
    $("#syn-pitch-val").textContent = parseFloat(e.target.value).toFixed(1);
  });
  $("#syn-speed").addEventListener("input", (e) => {
    $("#syn-speed-val").textContent = parseFloat(e.target.value).toFixed(2);
  });

  let syn_last_blob = null;
  $("#syn-btn").addEventListener("click", async () => {
    const st = $("#syn-status");
    const btn = $("#syn-btn");
    const text = $("#syn-text").value.trim();
    if (!text) { status(st, "Type something to speak.", "err"); return; }
    if (!ACTIVE_TTS) { status(st, "No TTS model loaded.", "err"); return; }

    const src = $("#syn-voice-source").value;
    const body = {
      model: ACTIVE_TTS,
      input: text,
      mode: "tts",
      instruct: $("#syn-instruct").value.trim(),
      temperature: parseFloat($("#syn-temp").value) || 0.7,
      top_p:       parseFloat($("#syn-topp").value) || 0.9,
      text_top_k:  parseInt($("#syn-topk").value, 10) || 50,
      max_new_tokens: parseInt($("#syn-max").value, 10) || 100,
      pitch_shift: parseFloat($("#syn-pitch").value) || 0,
      speed:       parseFloat($("#syn-speed").value) || 1.0,
    };

    if (src === "preset") {
      body.speaker = $("#syn-preset").value;
      body.mode = "voice_clone";
    } else if (src === "uploaded") {
      const v = $("#syn-uploaded").value;
      if (!v) { status(st, "No saved voice — generate one in Voice Design first.", "err"); return; }
      body.voice = v;
      body.mode = "voice_clone";
    } else if (src === "reference") {
      const f = $("#syn-ref-file").files[0];
      if (!f) { status(st, "Pick a reference audio file.", "err"); return; }
      // upload it first to get a server-side path
      btn.disabled = true; btn.textContent = "Uploading ref…";
      status(st, "Uploading reference audio…", "loading");
      try {
        const form = new FormData();
        form.append("file", f, f.name);
        const up = await fetch("/v1/clips/upload", { method: "POST", body: form });
        const uj = await up.json();
        if (!uj.ok || !uj.saved || !uj.saved.length) {
          status(st, "Reference upload failed", "err"); return;
        }
        body.reference_audio = uj.saved[0];
        body.reference_text  = $("#syn-ref-text").value.trim();
        body.mode = "voice_clone";
      } catch (err) {
        status(st, "Ref upload error: " + err.message, "err"); return;
      }
    }

    btn.disabled = true;
    btn.textContent = "Synthesizing…";
    status(st, "Generating audio…", "loading");
    clear_player($("#syn-player"));

    try {
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
      syn_last_blob = blob;
      const url = URL.createObjectURL(blob);
      const meta = (body.mode || "tts") + " · " + (blob.size / 1024).toFixed(1) + " KB";
      set_player($("#syn-player"), url, meta);
      $("#syn-download").disabled = false;
      status(st, "Done.", "ok");
    } catch (err) {
      status(st, "Error: " + err.message, "err");
    }

    btn.disabled = false;
    btn.textContent = "🎙️ Synthesize";
  });

  $("#syn-download").addEventListener("click", () => {
    if (!syn_last_blob) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(syn_last_blob);
    a.download = "synthesis_" + Date.now() + ".wav";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });

  // ── Music tab ──────────────────────────────────────────────────────
  let mus_last_blob = null;
  $("#mus-btn").addEventListener("click", async () => {
    const st = $("#mus-status");
    const btn = $("#mus-btn");
    if (!ACTIVE_MUSIC) {
      status(st, "No music model loaded.", "err"); return;
    }
    btn.disabled = true; btn.textContent = "Generating…";
    status(st, "Composing music (this can take a minute or more)…", "loading");
    clear_player($("#mus-player"));

    try {
      const body = {
        model: ACTIVE_MUSIC,
        caption:  $("#mus-caption").value.trim(),
        lyrics:   $("#mus-lyrics").value.trim(),
        duration: parseFloat($("#mus-dur").value) || 30,
        steps:    parseInt($("#mus-steps").value, 10) || 50,
        guidance_scale: parseFloat($("#mus-guid").value) || 7.5,
      };
      const r = await fetch("/v1/audio/music", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        status(st, j.error || "Music generation failed (" + r.status + ")", "err");
        return;
      }
      const blob = await r.blob();
      mus_last_blob = blob;
      const url = URL.createObjectURL(blob);
      set_player($("#mus-player"), url,
        "stereo · " + (blob.size / 1024 / 1024).toFixed(2) + " MB");
      $("#mus-download").disabled = false;
      status(st, "Done.", "ok");
    } catch (err) {
      status(st, "Error: " + err.message, "err");
    }
    btn.disabled = false; btn.textContent = "🎵 Generate";
  });

  $("#mus-download").addEventListener("click", () => {
    if (!mus_last_blob) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(mus_last_blob);
    a.download = "music_" + Date.now() + ".wav";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });

  // ── Voices library ─────────────────────────────────────────────────
  async function load_voices() {
    const list = $("#voices-list");
    try {
      const r = await fetch("/v1/voices");
      if (!r.ok) throw new Error(r.statusText);
      const j = await r.json();
      const voices = j.voices || [];
      // also populate the syn-uploaded select
      const sel = $("#syn-uploaded");
      sel.innerHTML = "";
      if (!voices.length) {
        list.innerHTML = '<div class="empty"><div class="empty-icon">🎭</div>No voices yet. Save one from Voice Design or upload a .voice file.</div>';
        return;
      }
      list.innerHTML = "";
      for (const v of voices) {
        const el = document.createElement("div");
        el.className = "voice-card";
        const sz = fmt_size(v.size || 0);
        const dim = v.dim ? v.dim + "-d" : "?-d";
        const l2  = v.l2_norm ? v.l2_norm.toFixed(2) : "?";
        el.innerHTML =
          '<div class="voice-name" title="' + esc(v.name) + '">' + esc(v.name) + '</div>' +
          '<div class="voice-meta">' + dim + ' · L2=' + l2 + ' · ' + sz + '</div>' +
          '<div class="voice-actions">' +
            '<button class="btn-icon use-voice" data-name="' + esc(v.name) + '">Use</button>' +
            '<button class="btn-icon play-voice" data-name="' + esc(v.name) + '">▶ Demo</button>' +
            '<button class="btn-icon danger del-voice" data-name="' + esc(v.name) + '">Delete</button>' +
          '</div>';
        list.appendChild(el);

        const o = document.createElement("option");
        o.value = v.path || v.name;
        o.textContent = v.name + " (" + dim + ")";
        sel.appendChild(o);
      }
      $$(".use-voice").forEach((b) => b.addEventListener("click", (e) => {
        const name = e.target.dataset.name;
        // switch to synthesize tab
        $('.tab[data-tab="synthesize"]').click();
        $("#syn-voice-source").value = "uploaded";
        $("#syn-voice-source").dispatchEvent(new Event("change"));
        // try to select this voice in the dropdown
        for (const o of $("#syn-uploaded").options) {
          if (o.textContent.indexOf(name) >= 0 || o.value.indexOf(name) >= 0) {
            $("#syn-uploaded").selectedIndex = o.index;
            break;
          }
        }
      }));
      $$(".play-voice").forEach((b) => b.addEventListener("click", demo_voice));
      $$(".del-voice").forEach((b) => b.addEventListener("click", async (e) => {
        const name = e.target.dataset.name;
        if (!confirm('Delete voice "' + name + '"?')) return;
        try {
          const r = await fetch("/v1/voices/delete", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name }),
          });
          const j = await r.json();
          if (j.ok) { toast("Deleted " + name, "ok"); load_voices(); }
          else      { toast(j.error || "Delete failed", "err"); }
        } catch (_) { toast("Delete failed", "err"); }
      }));
    } catch (e) {
      list.innerHTML = '<div class="empty"><div class="empty-icon">⚠</div>Could not reach server.</div>';
    }
  }

  async function demo_voice(e) {
    const name = e.target.dataset.name;
    const st = $("#voice-status");
    status(st, 'Generating demo for "' + name + '"…', "loading");
    const text = "The quick brown fox jumps over the lazy dog.";
    try {
      const r = await fetch("/v1/audio/speech", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: ACTIVE_TTS || ACTIVE_VD,
          input: text,
          voice: name,
          mode: "voice_clone",
          temperature: 0.7, top_p: 0.9, text_top_k: 50, max_new_tokens: 100,
        }),
      });
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        status(st, j.error || "Demo failed", "err"); return;
      }
      const blob = await r.blob();
      // Switch to synthesize tab to play
      const url = URL.createObjectURL(blob);
      $('.tab[data-tab="synthesize"]').click();
      set_player($("#syn-player"), url, "demo · " + name + " · " + (blob.size/1024).toFixed(1) + " KB");
      status(st, "Demo generated.", "ok");
    } catch (err) {
      status(st, "Error: " + err.message, "err");
    }
  }

  $("#voice-refresh").addEventListener("click", load_voices);

  $("#voice-upload-btn").addEventListener("click", async () => {
    const files = $("#voice-upload-file").files;
    if (!files.length) { toast("Pick a file first", "err"); return; }
    const st = $("#voice-status");
    status(st, "Uploading…", "loading");
    const fd = new FormData();
    for (const f of files) fd.append("file", f, f.name);
    try {
      const r = await fetch("/v1/voices/upload", { method: "POST", body: fd });
      const j = await r.json();
      if (j.ok) {
        toast("Uploaded " + (j.saved || []).length + " voice(s)", "ok");
        load_voices();
        status(st, "Uploaded.", "ok");
      } else status(st, j.error || "Upload failed", "err");
    } catch (err) {
      status(st, "Error: " + err.message, "err");
    }
  });

  // ── Clips library ──────────────────────────────────────────────────
  async function load_clips() {
    const list = $("#clips-list");
    try {
      const r = await fetch("/v1/clips");
      if (!r.ok) throw new Error(r.statusText);
      const j = await r.json();
      const clips = j.clips || [];

      if (!clips.length) {
        list.innerHTML = '<div class="empty"><div class="empty-icon">🎬</div>No clips yet.</div>';
        return;
      }

      list.innerHTML = "";
      for (const c of clips) {
        const el = document.createElement("div");
        el.className = "clip";
        el.innerHTML =
          '<audio controls preload="metadata" src="/v1/clips/raw/' + encodeURIComponent(c.name) + '"></audio>' +
          '<div class="clip-info">' +
            '<div class="clip-name" title="' + esc(c.name) + '">' + esc(c.name) + '</div>' +
            '<div class="clip-meta">' +
              fmt_size(c.size) +
              (c.duration ? " · " + fmt_dur(c.duration) : "") +
            '</div>' +
          '</div>' +
          '<button class="btn btn-danger" data-del-clip="' + esc(c.name) + '">Delete</button>';
        list.appendChild(el);
      }
      $$("[data-del-clip]").forEach((b) => b.addEventListener("click", async (e) => {
        const name = e.target.dataset.delClip;
        if (!confirm('Delete "' + name + '"?')) return;
        try {
          const r = await fetch("/v1/clips/delete", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name }),
          });
          const j = await r.json();
          if (j.ok) { toast("Deleted " + name, "ok"); load_clips(); }
          else      { toast(j.error || "Delete failed", "err"); }
        } catch (_) { toast("Delete failed", "err"); }
      }));
    } catch (e) {
      list.innerHTML = '<div class="empty"><div class="empty-icon">⚠</div>Could not reach server.</div>';
    }
  }

  $("#clip-refresh").addEventListener("click", load_clips);

  $("#clip-upload-btn").addEventListener("click", async () => {
    const files = $("#clip-upload-file").files;
    if (!files.length) { toast("Pick a file first", "err"); return; }
    const st = $("#clip-status");
    status(st, "Uploading…", "loading");
    const fd = new FormData();
    for (const f of files) fd.append("file", f, f.name);
    try {
      const r = await fetch("/v1/clips/upload", { method: "POST", body: fd });
      const j = await r.json();
      if (j.ok) {
        toast("Uploaded " + (j.saved || []).length + " clip(s)", "ok");
        load_clips();
        status(st, "", "");
      } else status(st, j.error || "Upload failed", "err");
    } catch (err) {
      status(st, "Error: " + err.message, "err");
    }
  });

  // ── Init ───────────────────────────────────────────────────────────
  Promise.all([load_models()]).then(() => {
    // Optional: load voices + clips lazily on tab open
  });
})();
