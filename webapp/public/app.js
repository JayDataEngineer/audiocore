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

  let toastTimer = null;
  function toast(msg, type) {
    const t = $("#toast");
    t.textContent = msg;
    t.className = "toast show " + (type || "ok");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (t.className = "toast"), 3200);
  }

  function fmt_size(n) {
    if (n == null) return "?";
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
    el.innerHTML = emptyPlayerMarkup();
  }

  function emptyPlayerMarkup(label) {
    return '<div class="player-empty">' +
      '<div class="eq-bars" aria-hidden="true">' +
        '<span></span><span></span><span></span><span></span>' +
        '<span></span><span></span><span></span>' +
      '</div>' +
      '<div>' + esc(label || "Generated audio appears here") + '</div>' +
    '</div>';
  }

  // Update slider track fill so the colored portion runs from 0 → thumb.
  function sync_slider_fill(s) {
    if (!s) return;
    const min = parseFloat(s.min || 0);
    const max = parseFloat(s.max || 100);
    const v   = parseFloat(s.value);
    const pct = ((v - min) / (max - min)) * 100;
    s.style.setProperty("--val", (isNaN(pct) ? 50 : pct) + "%");
  }
  function sync_all_sliders() {
    $$(".slider").forEach(sync_slider_fill);
  }

  // ── Tab switching ──────────────────────────────────────────────────
  const TAB_IDS = ["design", "maker", "synthesize", "music", "voices", "clips"];

  function activate_tab(target, { persist = true } = {}) {
    $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === target));
    $$(".panel").forEach((p) =>
      p.classList.toggle("active", p.id === "tab-" + target)
    );
    if (persist) {
      try { localStorage.setItem("ac:tab", target); } catch (_) {}
    }
    if (target === "voices") load_voices();
    if (target === "clips")  load_clips();
    if (target === "maker")  refresh_maker_dropdowns();
    if (target === "music")  refresh_music_sources();
    if (target === "voices" || target === "clips") {
      const search = $("#" + (target === "voices" ? "voice-search" : "clip-search"));
      if (search) setTimeout(() => search.focus(), 50);
    }
  }

  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => activate_tab(tab.dataset.tab));
  });

  // Restore last tab on load (default: design).
  try {
    const saved = localStorage.getItem("ac:tab");
    if (saved && TAB_IDS.indexOf(saved) >= 0 && saved !== "design") {
      activate_tab(saved, { persist: false });
    }
  } catch (_) {}

  // ── Models ─────────────────────────────────────────────────────────
  let MODELS = [];
  let ACTIVE_TTS = null;
  let ACTIVE_VD  = null;
  let ACTIVE_MUSIC = null;

  function set_engine_dot(id, state) {
    const el = $("#" + id);
    if (el) el.className = "engine-dot " + (state || "");
  }

  async function load_models() {
    try {
      const r = await fetch("/v1/models");
      const j = await r.json();
      MODELS = j.data || [];
      const badge = $("#model-badge");

      if (!MODELS.length) {
        badge.textContent = "no models";
        badge.className = "badge err";
        set_engine_dot("engine-tts", "");
        set_engine_dot("engine-vd", "");
        set_engine_dot("engine-music", "");
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
      set_engine_dot("engine-tts",   ACTIVE_TTS   ? "ready" : "");
      set_engine_dot("engine-vd",    ACTIVE_VD    ? "ready" : "");
      set_engine_dot("engine-music", ACTIVE_MUSIC ? "ready" : "");
      console.log({ ACTIVE_TTS, ACTIVE_VD, ACTIVE_MUSIC, MODELS });
    } catch (e) {
      $("#model-badge").textContent = "offline";
      $("#model-badge").className = "badge err";
      set_engine_dot("engine-tts", "");
      set_engine_dot("engine-vd", "");
      set_engine_dot("engine-music", "");
    }
  }

  // Mark a dot as "active" while a request is in flight, then back to ready.
  function with_engine_dot(id, promise) {
    set_engine_dot(id, "active");
    return promise.finally(() => set_engine_dot(id, ACTIVE_TTS || ACTIVE_VD || ACTIVE_MUSIC ? "ready" : ""));
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
      const p = fetch("/v1/audio/speech", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then(async (r) => {
        if (!r.ok) {
          const j = await r.json().catch(() => ({}));
          throw new Error(j.error || "Voice design failed (" + r.status + ")");
        }
        return r.blob();
      });
      const blob = await with_engine_dot("engine-vd", p);
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

  // ── Voice Maker tab ────────────────────────────────────────────────
  // Populate all the dropdowns when the user opens the tab.
  async function refresh_maker_dropdowns() {
    let voices = [];
    try {
      const r = await fetch("/v1/voices");
      const j = await r.json();
      voices = j.voices || [];
    } catch (_) {}
    // Sort: .voice files first, then .dir files (so directions cluster at the bottom).
    voices.sort((a, b) => {
      const aDir = a.name.endsWith(".dir"), bDir = b.name.endsWith(".dir");
      if (aDir !== bDir) return aDir ? 1 : -1;
      return a.name.localeCompare(b.name);
    });
    const voice_only = voices.filter(v => !v.name.endsWith(".dir"));
    const dir_only   = voices.filter(v =>  v.name.endsWith(".dir"));

    const fill = (sel, list, include_dim) => {
      if (!sel) return;
      const cur = sel.value;
      sel.innerHTML = "";
      if (!list.length) {
        const o = document.createElement("option");
        o.value = "";
        o.textContent = "(none available)";
        sel.appendChild(o);
        return;
      }
      for (const v of list) {
        const o = document.createElement("option");
        o.value = v.name;
        o.textContent = v.name + (include_dim && v.dim ? "  [" + v.dim + "-d, L2=" + (v.l2_norm||0).toFixed(2) + "]" : "");
        sel.appendChild(o);
      }
      // preserve current selection if still present
      if (cur) {
        for (const o of sel.options) if (o.value === cur) { sel.value = cur; break; }
      }
    };
    fill($("#mk-blend-a"),    voice_only, true);
    fill($("#mk-blend-b"),    voice_only, true);
    fill($("#mk-shift-base"), voice_only, true);
    fill($("#mk-shift-dir"),  dir_only,   true);
    fill($("#mk-disc-pos"),   voice_only, false);
    fill($("#mk-disc-neg"),   voice_only, false);

    // Knob 4 dropdown (kept here so a single refresh covers all four knobs).
    const shapeSel = $("#mk-shape-voice");
    if (shapeSel) {
      const cur = shapeSel.value;
      shapeSel.innerHTML = "";
      if (!voice_only.length) {
        const o = document.createElement("option");
        o.value = ""; o.textContent = "(none available)";
        shapeSel.appendChild(o);
      } else {
        for (const v of voice_only) {
          const o = document.createElement("option");
          o.value = v.name;
          const tag = v.dsp ? `  [pitch ${(v.dsp.pitch_shift||0).toFixed(1)}, sp ${(v.dsp.speed||1).toFixed(2)}]` : "";
          o.textContent = v.name + tag;
          shapeSel.appendChild(o);
        }
        if (cur) for (const o of shapeSel.options) if (o.value === cur) { shapeSel.value = cur; break; }
        if (!shapeSel.value && voice_only.length) {
          shapeSel.selectedIndex = 0;
          load_shape_for_voice();
        }
      }
    }
  }

  // Live slider value displays
  $("#mk-blend-t").addEventListener("input", (e) => {
    $("#mk-blend-t-val").textContent = parseFloat(e.target.value).toFixed(2);
    sync_slider_fill(e.target);
  });
  $("#mk-shift-scale").addEventListener("input", (e) => {
    const v = parseFloat(e.target.value);
    $("#mk-shift-scale-val").textContent = (v >= 0 ? "+" : "") + v.toFixed(2);
    sync_slider_fill(e.target);
  });

  // Knob 1: blend
  $("#mk-blend-btn").addEventListener("click", async () => {
    const st = $("#mk-blend-status");
    const a = $("#mk-blend-a").value, b = $("#mk-blend-b").value;
    if (!a || !b) { status(st, "Pick both voices.", "err"); return; }
    const body = {
      a, b,
      t: parseFloat($("#mk-blend-t").value),
      mode: $("#mk-blend-mode").value,
      out: $("#mk-blend-out").value.trim() || "blend.voice",
    };
    if (a === b) { status(st, "Pick two different voices.", "err"); return; }
    status(st, "Blending…", "loading");
    try {
      const r = await fetch("/v1/voices/blend", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const j = await r.json();
      if (j.ok) {
        const extra = j.mode === "slerp"
          ? " · Ω=" + (j.omega_deg || 0).toFixed(2) + "°"
          : "";
        status(st, "Saved " + j.name + extra + " (dim=" + j.dim + ")", "ok");
        toast("Voice blended", "ok");
        refresh_maker_dropdowns();
      } else status(st, j.error || "Blend failed", "err");
    } catch (err) { status(st, "Error: " + err.message, "err"); }
  });

  // Knob 2: shift
  $("#mk-shift-btn").addEventListener("click", async () => {
    const st = $("#mk-shift-status");
    const base = $("#mk-shift-base").value, dir = $("#mk-shift-dir").value;
    if (!base || !dir) { status(st, "Pick a base voice and a direction.", "err"); return; }
    const body = {
      base, direction: dir,
      scale: parseFloat($("#mk-shift-scale").value),
      out: $("#mk-shift-out").value.trim() || "shifted.voice",
    };
    status(st, "Shifting…", "loading");
    try {
      const r = await fetch("/v1/voices/shift", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const j = await r.json();
      if (j.ok) {
        status(st, "Saved " + j.name + " (scale=" + j.scale + ")", "ok");
        toast("Voice shifted", "ok");
        refresh_maker_dropdowns();
      } else status(st, j.error || "Shift failed", "err");
    } catch (err) { status(st, "Error: " + err.message, "err"); }
  });

  // Knob 3: discover_direction
  $("#mk-disc-btn").addEventListener("click", async () => {
    const st = $("#mk-disc-status");
    const pos_sel = $("#mk-disc-pos"), neg_sel = $("#mk-disc-neg");
    const pos = [...pos_sel.selectedOptions].map(o => o.value);
    const neg = [...neg_sel.selectedOptions].map(o => o.value);
    if (pos.length < 2 || neg.length < 2) {
      status(st, "Pick at least 2 voices in each group.", "err"); return;
    }
    const body = {
      positive: pos, negative: neg,
      out: $("#mk-disc-out").value.trim() || "custom.dir",
    };
    status(st, "Computing steering vector…", "loading");
    try {
      const r = await fetch("/v1/voices/discover_direction", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const j = await r.json();
      if (j.ok) {
        status(st, "Saved " + j.name + " (‖dir‖=" + (j.norm||0).toFixed(3) +
                    ", from " + j.positive_count + "+" + j.negative_count +
                    " voices)", "ok");
        toast("Direction discovered", "ok");
        refresh_maker_dropdowns();
      } else status(st, j.error || "Discovery failed", "err");
    } catch (err) { status(st, "Error: " + err.message, "err"); }
  });

  // ── Knob 4: Shape voice (DSP sidecar) ──────────────────────────────
  async function load_shape_for_voice() {
    const name = $("#mk-shape-voice").value;
    if (!name) return;
    try {
      const r = await fetch(`/v1/voices/${encodeURIComponent(name)}/meta`);
      const j = await r.json();
      $("#mk-shape-pitch").value = j.pitch_shift || 0;
      $("#mk-shape-speed").value = j.speed || 1.0;
      $("#mk-shape-pitch-val").textContent = (j.pitch_shift || 0).toFixed(1) + " st";
      $("#mk-shape-speed-val").textContent = (j.speed || 1.0).toFixed(2) + "×";
      sync_slider_fill($("#mk-shape-pitch"));
      sync_slider_fill($("#mk-shape-speed"));
      const st = $("#mk-shape-status");
      if (j.has_meta) {
        status(st, `Loaded saved shaping: ${(j.pitch_shift||0).toFixed(1)} st / ${(j.speed||1.0).toFixed(2)}×`, "info");
      } else {
        status(st, "No saved shaping yet — defaults loaded.", "info");
      }
    } catch (_) { /* silent */ }
  }

  $("#mk-shape-voice").addEventListener("change", load_shape_for_voice);
  $("#mk-shape-pitch").addEventListener("input", (e) => {
    const v = parseFloat(e.target.value);
    $("#mk-shape-pitch-val").textContent = (v >= 0 ? "+" : "") + v.toFixed(1) + " st";
    sync_slider_fill(e.target);
  });
  $("#mk-shape-speed").addEventListener("input", (e) => {
    const v = parseFloat(e.target.value);
    $("#mk-shape-speed-val").textContent = v.toFixed(2) + "×";
    sync_slider_fill(e.target);
  });

  // Preview: synthesize with the current slider values (one-shot, doesn't save)
  $("#mk-shape-preview").addEventListener("click", async () => {
    const st = $("#mk-shape-status");
    const player = $("#mk-shape-player");
    const name = $("#mk-shape-voice").value;
    const text = $("#mk-shape-text").value.trim();
    if (!name) { status(st, "Pick a voice first.", "err"); return; }
    if (!text) { status(st, "Type some preview text.", "err"); return; }
    const pitch = parseFloat($("#mk-shape-pitch").value);
    const speed = parseFloat($("#mk-shape-speed").value);
    status(st, `Previewing at ${pitch.toFixed(1)} st / ${speed.toFixed(2)}×…`, "loading");
    player.hidden = false;
    player.innerHTML = emptyPlayerMarkup("Generating preview…");
    try {
      const r = await fetch("/v1/audio/speech", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: window.AUDIOCORE_MODEL_ID || "default",
          input: text,
          voice: name,
          mode: "voice_clone",
          pitch_shift: pitch,
          speed: speed,
        }),
      });
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        status(st, j.error || `Preview failed (${r.status})`, "err");
        return;
      }
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      player.innerHTML = `<audio controls autoplay src="${url}"></audio>
        <div class="player-meta">preview · ${pitch.toFixed(1)} st · ${speed.toFixed(2)}× (not saved)</div>`;
      status(st, "Preview ready. Adjust sliders and Preview again, or Save shaping.", "ok");
    } catch (err) { status(st, "Error: " + err.message, "err"); }
  });

  // Save shaping: PUT the sidecar JSON
  $("#mk-shape-save").addEventListener("click", async () => {
    const st = $("#mk-shape-status");
    const name = $("#mk-shape-voice").value;
    if (!name) { status(st, "Pick a voice first.", "err"); return; }
    const pitch = parseFloat($("#mk-shape-pitch").value);
    const speed = parseFloat($("#mk-shape-speed").value);
    status(st, "Saving shaping…", "loading");
    try {
      const r = await fetch(`/v1/voices/${encodeURIComponent(name)}/meta`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pitch_shift: pitch, speed: speed }),
      });
      const j = await r.json();
      if (j.ok) {
        status(st, `Saved ${j.name} → pitch ${j.dsp.pitch_shift} st, speed ${j.dsp.speed}×. ` +
                   "Every Synthesize using this voice now applies these defaults.", "ok");
        toast("Voice shaping saved", "ok");
        refresh_maker_dropdowns();
      } else status(st, j.error || "Save failed", "err");
    } catch (err) { status(st, "Error: " + err.message, "err"); }
  });

  // Reset shaping: DELETE the sidecar
  $("#mk-shape-clear").addEventListener("click", async () => {
    const st = $("#mk-shape-status");
    const name = $("#mk-shape-voice").value;
    if (!name) return;
    status(st, "Resetting shaping…", "loading");
    try {
      const r = await fetch(`/v1/voices/${encodeURIComponent(name)}/meta`, { method: "DELETE" });
      const j = await r.json();
      $("#mk-shape-pitch").value = 0;
      $("#mk-shape-speed").value = 1.0;
      $("#mk-shape-pitch-val").textContent = "0.0 st";
      $("#mk-shape-speed-val").textContent = "1.00×";
      sync_slider_fill($("#mk-shape-pitch"));
      sync_slider_fill($("#mk-shape-speed"));
      status(st, j.ok ? "Shaping reset to defaults." : (j.error || "Nothing to reset."), "ok");
      refresh_maker_dropdowns();
    } catch (err) { status(st, "Error: " + err.message, "err"); }
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
    if (v === "uploaded") load_syn_voice_meta();
  });

  // When a saved .voice is picked in Synthesize, load its saved shaping
  // into the pitch/speed sliders so the user sees the actual values that
  // will be applied (and can fine-tune from there).
  async function load_syn_voice_meta() {
    const sel = $("#syn-uploaded");
    const name = sel && sel.value;
    if (!name) return;
    try {
      const r = await fetch(`/v1/voices/${encodeURIComponent(name)}/meta`);
      const j = await r.json();
      const pitch = j.pitch_shift || 0;
      const speed = j.speed || 1.0;
      $("#syn-pitch").value = pitch;
      $("#syn-speed").value = speed;
      $("#syn-pitch-val").textContent = pitch.toFixed(1);
      $("#syn-speed-val").textContent = speed.toFixed(2);
      sync_slider_fill($("#syn-pitch"));
      sync_slider_fill($("#syn-speed"));
    } catch (_) {}
  }
  if ($("#syn-uploaded")) {
    $("#syn-uploaded").addEventListener("change", load_syn_voice_meta);
  }

  // Slider live values
  $("#syn-pitch").addEventListener("input", (e) => {
    $("#syn-pitch-val").textContent = parseFloat(e.target.value).toFixed(1);
    sync_slider_fill(e.target);
  });
  $("#syn-speed").addEventListener("input", (e) => {
    $("#syn-speed-val").textContent = parseFloat(e.target.value).toFixed(2);
    sync_slider_fill(e.target);
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
      const p = fetch("/v1/audio/speech", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then(async (r) => {
        if (!r.ok) {
          const j = await r.json().catch(() => ({}));
          throw new Error(j.error || "TTS failed (" + r.status + ")");
        }
        return r.blob();
      });
      const blob = await with_engine_dot("engine-tts", p);
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

  // Refresh the source-clip dropdown from the clips library.
  async function refresh_music_sources() {
    const sel = $("#mus-src");
    if (!sel) return;
    try {
      const r = await fetch("/v1/clips");
      if (!r.ok) return;
      const j = await r.json();
      const cur = sel.value;
      sel.innerHTML = '<option value="">— pick a clip —</option>';
      (j.clips || []).forEach((c) => {
        const o = document.createElement("option");
        o.value = c.name;
        o.textContent = c.name + (c.duration ? ` (${c.duration.toFixed(1)}s)` : "");
        sel.appendChild(o);
      });
      if (cur) sel.value = cur;
    } catch (e) { /* ignore */ }
  }

  // Mode selector: reveal/hide source + mask rows.
  function sync_music_mode_rows() {
    const mode = $("#mus-mode").value;
    const needs_src = (mode === "cover" || mode === "repaint" || mode === "completion");
    const needs_mask = (mode === "repaint" || mode === "completion");
    $("#mus-src-row").hidden     = !needs_src;
    $("#mus-mask-row").hidden    = !needs_mask;
    $("#mus-mask-row2").hidden   = !needs_mask;
    const hint = $("#mus-mask-hint");
    if (hint) {
      hint.textContent = (mode === "repaint")
        ? "Inside [start,end] is regenerated; outside is preserved."
        : (mode === "completion")
          ? "Inside [start,end] is preserved; outside is regenerated (extension)."
          : "";
    }
    // Contextual guidance hint: in cover mode lower CFG = more faithful to source.
    const gh = $("#mus-guid-hint");
    if (gh) {
      gh.textContent = (mode === "cover")
        ? "(lower = faithful to source)"
        : "(CFG strength)";
    }
    if (needs_src) refresh_music_sources();
  }
  $("#mus-mode").addEventListener("change", sync_music_mode_rows);
  sync_music_mode_rows();
  $("#mus-m0").addEventListener("input", (e) => {
    $("#mus-m0-val").textContent = parseFloat(e.target.value).toFixed(2);
    sync_slider_fill(e.target);
  });
  $("#mus-m1").addEventListener("input", (e) => {
    $("#mus-m1-val").textContent = parseFloat(e.target.value).toFixed(2);
    sync_slider_fill(e.target);
  });
  $("#mus-pitch").addEventListener("input", (e) => {
    $("#mus-pitch-val").textContent = e.target.value;
    sync_slider_fill(e.target);
  });
  $("#mus-speed").addEventListener("input", (e) => {
    $("#mus-speed-val").textContent = parseFloat(e.target.value).toFixed(2);
    sync_slider_fill(e.target);
  });

  $("#mus-btn").addEventListener("click", async () => {
    const st = $("#mus-status");
    const btn = $("#mus-btn");
    if (!ACTIVE_MUSIC) {
      status(st, "No music model loaded.", "err"); return;
    }
    const mode = $("#mus-mode").value;
    const needs_src = (mode === "cover" || mode === "repaint" || mode === "completion");
    let src_b64 = null;
    if (needs_src) {
      const src_name = $("#mus-src").value.trim();
      if (!src_name) {
        status(st, "Pick a source clip for " + mode + " mode.", "err");
        return;
      }
      status(st, "Loading source clip '" + src_name + "'…", "loading");
      try {
        const r = await fetch("/v1/clips/raw/" + encodeURIComponent(src_name));
        if (!r.ok) throw new Error("clip fetch failed (" + r.status + ")");
        const blob = await r.blob();
        src_b64 = await new Promise((resolve, reject) => {
          const fr = new FileReader();
          fr.onload = () => resolve(fr.result.split(",")[1]);
          fr.onerror = reject;
          fr.readAsDataURL(blob);
        });
      } catch (err) {
        status(st, "Could not load source clip: " + err.message, "err");
        return;
      }
    }
    // Batch count: 1 = single render (default). >1 = N variations with
    // consecutive seeds. Seed 0 means "random" → for batches we anchor a
    // random base so all N are different but reproducible next time.
    const batch_n = Math.max(1, Math.min(8, parseInt($("#mus-batch").value, 10) || 1));
    let base_seed = parseInt($("#mus-seed").value, 10) || 0;
    if (base_seed === 0 && batch_n > 1) {
      base_seed = Math.floor(Math.random() * 2_000_000_000) + 1;
    }
    btn.disabled = true; btn.textContent = (batch_n > 1) ? "Generating batch…" : "Generating…";
    clear_player($("#mus-player"));
    const results = [];  // {seed, blob, error}
    let engine_promise = Promise.resolve();

    for (let i = 0; i < batch_n; i++) {
      const seed_i = (base_seed === 0) ? 0 : base_seed + i;
      status(st, "Composing " + mode + " variation " + (i + 1) + "/" + batch_n +
        (seed_i ? " (seed " + seed_i + ")" : "") + "…", "loading");
      try {
        const body = {
          model: ACTIVE_MUSIC,
          mode,
          caption:  $("#mus-caption").value.trim(),
          lyrics:   $("#mus-lyrics").value.trim(),
          duration: parseFloat($("#mus-dur").value) || 30,
          steps:    parseInt($("#mus-steps").value, 10) || 50,
          guidance_scale: parseFloat($("#mus-guid").value) || 7.5,
        };
        if (seed_i && seed_i > 0) body.seed = seed_i;
        if (src_b64) {
          body.input_audio = src_b64;
          body.mask_start = parseFloat($("#mus-m0").value) || 0;
          body.mask_end   = parseFloat($("#mus-m1").value) || 1;
        }
        const pitch = parseFloat($("#mus-pitch").value) || 0;
        const speed = parseFloat($("#mus-speed").value) || 1.0;
        if (Math.abs(pitch) > 0.01) body.pitch_shift = pitch;
        if (Math.abs(speed - 1.0) > 0.01) body.speed = speed;
        const req = fetch("/v1/audio/music", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }).then(async (r) => {
          if (!r.ok) {
            const j = await r.json().catch(() => ({}));
            throw new Error(j.error || ("HTTP " + r.status));
          }
          return r.blob();
        });
        // Light up the music dot during the first request.
        if (i === 0) engine_promise = with_engine_dot("engine-music", req);
        const blob = await req;
        results.push({ seed: seed_i, blob });
      } catch (err) {
        results.push({ seed: seed_i, error: err.message });
      }
    }
    await engine_promise;

    // Render results. Single success → existing player + download. Multiple
    // → numbered grid of players, each downloadable. Download button keeps
    // the LAST successful blob for the "⬇️ Download" shortcut.
    const ok = results.filter(r => r.blob);
    const failed = results.filter(r => r.error);
    const player = $("#mus-player");
    if (ok.length === 0) {
      status(st, failed[0]?.error || "All variations failed.", "err");
      btn.disabled = false; btn.textContent = "🎵 Generate";
      return;
    }
    if (ok.length === 1 && batch_n === 1) {
      mus_last_blob = ok[0].blob;
      const url = URL.createObjectURL(ok[0].blob);
      set_player(player, url,
        mode + " · stereo · " + (ok[0].blob.size / 1024 / 1024).toFixed(2) + " MB");
    } else {
      // Batch: rebuild player area as a styled grid.
      player.innerHTML = "";
      player.classList.add("has-audio");
      const grid = document.createElement("div");
      grid.className = "clips-grid";
      grid.style.gridTemplateColumns = "1fr";
      grid.style.marginTop = "0";
      for (const r of ok) {
        const url = URL.createObjectURL(r.blob);
        const card = document.createElement("div");
        card.className = "clip";
        const label = document.createElement("div");
        label.className = "clip-info";
        label.style.display = "flex";
        label.style.justifyContent = "space-between";
        label.style.alignItems = "center";
        label.style.marginBottom = "6px";
        const seedTxt = r.seed ? ("seed " + r.seed) : "random";
        label.innerHTML =
          '<div>' +
            '<div class="clip-name">#' + (results.indexOf(r) + 1) + ' · ' + esc(mode) + '</div>' +
            '<div class="clip-meta">' +
              '<span class="tag">' + esc(seedTxt) + '</span>' +
              '<span class="tag">' + (r.blob.size / 1024 / 1024).toFixed(2) + ' MB</span>' +
            '</div>' +
          '</div>';
        const dlBtn = document.createElement("button");
        dlBtn.className = "btn btn-ghost";
        dlBtn.style.padding = "6px 12px";
        dlBtn.style.fontSize = "12px";
        dlBtn.innerHTML = "⬇ Download";
        dlBtn.title = "Download this variation";
        dlBtn.onclick = () => {
          const a = document.createElement("a");
          a.href = URL.createObjectURL(r.blob);
          a.download = "music_" + mode + "_seed" + r.seed + "_" + Date.now() + ".wav";
          a.click();
          setTimeout(() => URL.revokeObjectURL(a.href), 1000);
        };
        label.appendChild(dlBtn);
        card.appendChild(label);
        const au = document.createElement("audio");
        au.controls = true;
        au.src = url;
        au.style.width = "100%";
        card.appendChild(au);
        grid.appendChild(card);
      }
      player.appendChild(grid);
      mus_last_blob = ok[ok.length - 1].blob;
    }
    $("#mus-download").disabled = !mus_last_blob;
    if (failed.length) {
      status(st, ok.length + "/" + batch_n + " OK · " + failed.length + " failed: " +
        failed[0].error, "err");
    } else if (batch_n > 1) {
      status(st, ok.length + " variations ready (seeds " +
        ok.map(r => r.seed || "?").join(", ") + ").", "ok");
    } else {
      status(st, "Done.", "ok");
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
  let VOICES_CACHE = [];
  let CLIPS_CACHE  = [];

  function apply_voice_filter() {
    const q = ($("#voice-search").value || "").trim().toLowerCase();
    const list = $("#voices-list");
    const count = $("#voice-count");
    if (!VOICES_CACHE.length) return;
    const filtered = VOICES_CACHE.filter(v =>
      !q || (v.name || "").toLowerCase().indexOf(q) >= 0
    );
    count.textContent = filtered.length + "/" + VOICES_CACHE.length + " voices";
    render_voices(filtered);
  }

  function render_voices(voices) {
    const list = $("#voices-list");
    if (!voices.length) {
      list.innerHTML = '<div class="empty">' +
        '<div class="empty-icon">🎭</div>' +
        '<div class="empty-title">No voices match</div>' +
        '<div class="empty-hint">Try a different search, or save one from Voice Design.</div>' +
      '</div>';
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
        '<div class="voice-meta">' +
          '<span class="tag">' + dim + '</span>' +
          '<span class="tag">L2 ' + l2 + '</span>' +
          '<span class="tag">' + sz + '</span>' +
        '</div>' +
        '<div class="voice-actions">' +
          '<button class="btn-icon use-voice" data-name="' + esc(v.name) + '">Use</button>' +
          '<button class="btn-icon play-voice" data-name="' + esc(v.name) + '">▶ Demo</button>' +
          '<button class="btn-icon danger del-voice" data-name="' + esc(v.name) + '">Delete</button>' +
        '</div>';
      list.appendChild(el);
    }
    $$(".use-voice").forEach((b) => b.addEventListener("click", (e) => {
      const name = e.target.dataset.name;
      // switch to synthesize tab
      activate_tab("synthesize");
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
  }

  async function load_voices() {
    const list = $("#voices-list");
    try {
      const r = await fetch("/v1/voices");
      if (!r.ok) throw new Error(r.statusText);
      const j = await r.json();
      VOICES_CACHE = j.voices || [];
      // also populate the syn-uploaded select
      const sel = $("#syn-uploaded");
      sel.innerHTML = "";
      if (!VOICES_CACHE.length) {
        $("#voice-count").textContent = "0 voices";
        list.innerHTML = '<div class="empty">' +
          '<div class="empty-icon">🎭</div>' +
          '<div class="empty-title">No voices yet</div>' +
          '<div class="empty-hint">Save one from Voice Design, blend in Voice Maker, or drag &amp; drop a .voice file here.</div>' +
        '</div>';
        return;
      }
      for (const v of VOICES_CACHE) {
        const o = document.createElement("option");
        o.value = v.path || v.name;
        const dim = v.dim ? v.dim + "-d" : "?-d";
        o.textContent = v.name + " (" + dim + ")";
        sel.appendChild(o);
      }
      apply_voice_filter();
    } catch (e) {
      $("#voice-count").textContent = "—";
      list.innerHTML = '<div class="empty">' +
        '<div class="empty-icon">⚠</div>' +
        '<div class="empty-title">Could not reach server</div>' +
        '<div class="empty-hint">Check that the audiocore process is running and healthy.</div>' +
      '</div>';
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
      activate_tab("synthesize");
      set_player($("#syn-player"), url, "demo · " + name + " · " + (blob.size/1024).toFixed(1) + " KB");
      status(st, "Demo generated.", "ok");
    } catch (err) {
      status(st, "Error: " + err.message, "err");
    }
  }

  $("#voice-refresh").addEventListener("click", load_voices);
  $("#voice-search").addEventListener("input", apply_voice_filter);

  $("#voice-upload-btn").addEventListener("click", async () => {
    const files = $("#voice-upload-file").files;
    if (!files.length) { toast("Pick a file first", "err"); return; }
    await upload_files("/v1/voices/upload", files, $("#voice-status"), () => load_voices());
    $("#voice-upload-file").value = "";
  });

  // ── Clips library ──────────────────────────────────────────────────
  function apply_clip_filter() {
    const q = ($("#clip-search").value || "").trim().toLowerCase();
    const count = $("#clip-count");
    if (!CLIPS_CACHE.length) return;
    const filtered = CLIPS_CACHE.filter(c =>
      !q || (c.name || "").toLowerCase().indexOf(q) >= 0
    );
    count.textContent = filtered.length + "/" + CLIPS_CACHE.length + " clips";
    render_clips(filtered);
  }

  function render_clips(clips) {
    const list = $("#clips-list");
    if (!clips.length) {
      list.innerHTML = '<div class="empty">' +
        '<div class="empty-icon">🎬</div>' +
        '<div class="empty-title">No clips match</div>' +
        '<div class="empty-hint">Generate audio in Voice Design, Synthesize, or Music.</div>' +
      '</div>';
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
            (c.duration ? '<span class="tag">' + fmt_dur(c.duration) + '</span>' : '') +
            '<span class="tag">' + fmt_size(c.size) + '</span>' +
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
  }

  async function load_clips() {
    const list = $("#clips-list");
    try {
      const r = await fetch("/v1/clips");
      if (!r.ok) throw new Error(r.statusText);
      const j = await r.json();
      CLIPS_CACHE = j.clips || [];

      if (!CLIPS_CACHE.length) {
        $("#clip-count").textContent = "0 clips";
        list.innerHTML = '<div class="empty">' +
          '<div class="empty-icon">🎬</div>' +
          '<div class="empty-title">No clips yet</div>' +
          '<div class="empty-hint">Generate audio in Voice Design, Synthesize, or Music.</div>' +
        '</div>';
        return;
      }
      apply_clip_filter();
    } catch (e) {
      $("#clip-count").textContent = "—";
      list.innerHTML = '<div class="empty">' +
        '<div class="empty-icon">⚠</div>' +
        '<div class="empty-title">Could not reach server</div>' +
        '<div class="empty-hint">Check that the audiocore process is running and healthy.</div>' +
      '</div>';
    }
  }

  $("#clip-refresh").addEventListener("click", load_clips);
  $("#clip-search").addEventListener("input", apply_clip_filter);

  $("#clip-upload-btn").addEventListener("click", async () => {
    const files = $("#clip-upload-file").files;
    if (!files.length) { toast("Pick a file first", "err"); return; }
    await upload_files("/v1/clips/upload", files, $("#clip-status"), () => load_clips());
    $("#clip-upload-file").value = "";
  });

  // Shared upload helper used by both buttons and drag-drop.
  async function upload_files(endpoint, files, status_el, on_done) {
    if (!files.length) return;
    if (status_el) status(status_el, "Uploading " + files.length + " file(s)…", "loading");
    const fd = new FormData();
    for (const f of files) fd.append("file", f, f.name);
    try {
      const r = await fetch(endpoint, { method: "POST", body: fd });
      const j = await r.json();
      if (j.ok) {
        const n = (j.saved || []).length;
        toast("Uploaded " + n + " file" + (n === 1 ? "" : "s"), "ok");
        if (status_el) status(status_el, "Uploaded " + n + " file" + (n === 1 ? "" : "s") + ".", "ok");
        if (on_done) on_done();
      } else {
        if (status_el) status(status_el, j.error || "Upload failed", "err");
        toast(j.error || "Upload failed", "err");
      }
    } catch (err) {
      if (status_el) status(status_el, "Error: " + err.message, "err");
      else toast("Upload failed", "err");
    }
  }

  // ── Drag & drop upload anywhere ────────────────────────────────────
  const drop_overlay = $("#drop-overlay");
  const drop_label   = $("#drop-target-label");
  let drag_counter = 0;

  function route_drop_files(files) {
    // Decide where to upload based on the currently active tab and file type.
    const active = $(".tab.active")?.dataset.tab;
    const arr = [...files];
    const voices = arr.filter(f => /\.(voice|wav|mp3|flac|ogg|m4a|opus)$/i.test(f.name) &&
                                   /\.voice$/i.test(f.name));
    const audio  = arr.filter(f => !/\.voice$/i.test(f.name) &&
                                   /\.(wav|mp3|flac|ogg|m4a|opus)$/i.test(f.name));
    // If the active tab is voices or clips, send everything there.
    if (active === "voices") {
      return upload_files("/v1/voices/upload", arr, $("#voice-status"), load_voices);
    } else if (active === "clips") {
      return upload_files("/v1/clips/upload", arr, $("#clip-status"), load_clips);
    }
    // Otherwise split: .voice → voices, audio → clips.
    const jobs = [];
    if (voices.length) jobs.push(upload_files("/v1/voices/upload", voices, $("#voice-status"), load_voices));
    if (audio.length)  jobs.push(upload_files("/v1/clips/upload",  audio,  $("#clip-status"),  load_clips));
    if (!jobs.length) {
      toast("Drop .voice or audio files", "err");
      return;
    }
    return Promise.all(jobs);
  }

  window.addEventListener("dragenter", (e) => {
    if (!e.dataTransfer || !e.dataTransfer.types || !e.dataTransfer.types.includes("Files")) return;
    e.preventDefault();
    drag_counter++;
    const active = $(".tab.active")?.dataset.tab;
    drop_label.textContent = (active === "voices") ? "Voices library"
                           : (active === "clips")  ? "Clips library"
                           : "Voices & Clips";
    drop_overlay.classList.add("show");
  });
  window.addEventListener("dragover", (e) => {
    if (!e.dataTransfer || !e.dataTransfer.types || !e.dataTransfer.types.includes("Files")) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  });
  window.addEventListener("dragleave", (e) => {
    if (!e.dataTransfer) return;
    drag_counter = Math.max(0, drag_counter - 1);
    if (drag_counter === 0) drop_overlay.classList.remove("show");
  });
  window.addEventListener("drop", (e) => {
    if (!e.dataTransfer || !e.dataTransfer.types || !e.dataTransfer.types.includes("Files")) return;
    e.preventDefault();
    drag_counter = 0;
    drop_overlay.classList.remove("show");
    const files = [...e.dataTransfer.files].filter(f => f && f.size > 0);
    if (!files.length) return;
    route_drop_files(files);
  });

  // ── Keyboard shortcuts ─────────────────────────────────────────────
  document.addEventListener("keydown", (e) => {
    // Ignore when typing in an input/textarea, unless it's a global escape.
    const tag = (e.target.tagName || "").toLowerCase();
    const typing = tag === "input" || tag === "textarea" || tag === "select" || e.target.isContentEditable;

    // Cmd/Ctrl + Enter → trigger primary action of active tab.
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      const active = $(".tab.active")?.dataset.tab;
      const map = {
        design: "#vd-btn",
        maker:  "#mk-shape-preview",
        synthesize: "#syn-btn",
        music:  "#mus-btn",
      };
      const sel = map[active];
      if (sel) { e.preventDefault(); $(sel)?.click(); }
      return;
    }

    if (typing) return;

    // 1–6: tabs
    if (/^[1-6]$/.test(e.key)) {
      const idx = parseInt(e.key, 10) - 1;
      if (TAB_IDS[idx]) { e.preventDefault(); activate_tab(TAB_IDS[idx]); }
      return;
    }

    // "/" focuses library search on voices/clips tab.
    if (e.key === "/") {
      const active = $(".tab.active")?.dataset.tab;
      if (active === "voices" || active === "clips") {
        e.preventDefault();
        $("#" + (active === "voices" ? "voice-search" : "clip-search"))?.focus();
      }
    }
  });

  // ── Footer uptime ticker ───────────────────────────────────────────
  const t0 = Date.now();
  function tick_uptime() {
    const el = $("#footer-uptime");
    if (!el) return;
    const s = Math.floor((Date.now() - t0) / 1000);
    const m = Math.floor(s / 60), sec = s % 60;
    const h = Math.floor(m / 60), mm = m % 60;
    el.textContent = (h > 0 ? h + "h " : "") + mm + "m " + sec + "s" +
      " · session uptime";
  }
  setInterval(tick_uptime, 1000);
  tick_uptime();

  // ── Init ───────────────────────────────────────────────────────────
  sync_all_sliders();
  Promise.all([load_models()]).then(() => {
    // Optional: load voices + clips lazily on tab open
  });
})();
