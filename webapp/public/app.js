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

  // ── Draggable PCA explorer shared state ──────────────────────────────
  // Holds the last /v1/voices/analyze result plus the pixel↔PC coordinate
  // transforms so we can map a dragged marker back to (PC1, PC2) weights and
  // reconstruct an embedding client-side (emb = mean + Σ wᵢ·componentsᵢ).
  const PCA = {
    data: null,         // full analyze JSON (incl. mean + components)
    weights: [],        // one signed weight per principal component
    ranges: [],         // per-PC |max| used to normalise the polygon axes
    constrain: 0,       // 0..1 manifold pull toward nearest real voice
    W: 560, H: 420, PAD: 50,
    min1: 0, max1: 1, min2: 0, max2: 1,
    // pixel → PC coordinate
    toPcX(px) { const r = (this.max1 - this.min1) + 0.01; return this.min1 + ((px - this.PAD) / (this.W - 2 * this.PAD)) * r; },
    toPcY(py) { const r = (this.max2 - this.min2) + 0.01; return this.max2 - ((py - this.PAD) / (this.H - 2 * this.PAD)) * ((this.max2 - this.min2) + 0.01); },
    // PC coordinate → pixel
    toPxX(v) { const r = (this.max1 - this.min1) + 0.01; return this.PAD + ((v - this.min1) / r) * (this.W - 2 * this.PAD); },
    toPxY(v) { const r = (this.max2 - this.min2) + 0.01; return this.H - this.PAD - ((v - this.min2) / r) * (this.H - 2 * this.PAD); },
  };

  function f32_to_b64(arr) {
    const buf = new ArrayBuffer(arr.length * 4);
    new Float32Array(buf).set(arr);
    const bytes = new Uint8Array(buf);
    let bin = "";
    for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    return btoa(bin);
  }

  // Core reconstruction: emb = mean + Σᵢ wᵢ·componentᵢ, then optional
  // manifold pull toward the nearest real voice (in PC space). Accepts an
  // explicit weight vector + constrain value so LIVE sculpting and SAVED
  // snapshots share the exact same math — an A/B comparison is truthful.
  function pca_reconstruct_from(weights, constrain) {
    const d = PCA.data;
    if (!d || !d.mean) return null;
    const mean = d.mean, comps = d.components, w = weights;
    const dim = mean.length;
    const emb = new Float32Array(dim);
    for (let i = 0; i < dim; i++) emb[i] = mean[i];
    for (let i = 0; i < w.length; i++) {
      const c = comps[i];
      if (!c) continue;
      const wi = w[i];
      if (wi === 0) continue;
      for (let j = 0; j < dim; j++) emb[j] += wi * c[j];
    }
    // Manifold constraint: pull the reconstructed embedding toward the
    // nearest real voice (in PC space) so extreme drags stay within the
    // distribution of natural speech instead of collapsing to noise.
    const c = constrain || 0;
    if (c > 0 && d.voices && d.voices.length) {
      let best = -1, bestD = Infinity;
      for (let k = 0; k < d.voices.length; k++) {
        const pc = d.voices[k].pc; let s = 0;
        for (let i = 0; i < w.length; i++) { const dd = (pc[i] || 0) - (w[i] || 0); s += dd * dd; }
        if (s < bestD) { bestD = s; best = k; }
      }
      if (best >= 0) {
        const pc = d.voices[best].pc;
        const tv = new Float32Array(dim);
        for (let j = 0; j < dim; j++) tv[j] = mean[j];
        for (let i = 0; i < pc.length; i++) {
          const cc = comps[i]; if (!cc) continue;
          const wi = pc[i] || 0; if (wi === 0) continue;
          for (let j = 0; j < dim; j++) tv[j] += wi * cc[j];
        }
        for (let j = 0; j < dim; j++) emb[j] = emb[j] * (1 - c) + tv[j] * c;
      }
    }
    return emb;
  }

  function pca_reconstruct() { return pca_reconstruct_from(PCA.weights, PCA.constrain); }

  function pca_build_voice_blob(emb) {
    // QWEN3VOICE container: 16-byte magic/pad + 4×uint32 (ver,dim,flags,rs) + f32s
    const MAGIC = "QWEN3VOICE";
    const head = new Uint8Array(16);
    for (let i = 0; i < MAGIC.length; i++) head[i] = MAGIC.charCodeAt(i);
    const meta = new Uint8Array(16); // ver(1), dim, flags(0), rs(0)
    new DataView(meta.buffer).setUint32(0, 1, true);
    new DataView(meta.buffer).setUint32(4, emb.length, true);
    new DataView(meta.buffer).setUint32(8, 0, true);
    new DataView(meta.buffer).setUint32(12, 0, true);
    const f32 = new Uint8Array(emb.buffer);
    const out = new Uint8Array(head.length + meta.length + f32.length);
    out.set(head, 0);
    out.set(meta, head.length);
    out.set(f32, head.length + meta.length);
    return new Blob([out], { type: "application/octet-stream" });
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
  const TAB_IDS = ["design", "maker", "synthesize", "music", "sfx", "voices", "clips", "models"];

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
    if (target === "models") render_model_manager();
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
  let ACTIVE_SFX = null;
  // Per-category model lists: [{id, family, loaded}, …]
  let MUSIC_MODELS = [];
  let TTS_MODELS   = [];
  let SFX_MODELS   = [];
  let VD_MODELS    = [];

  // Per-variant defaults. Turbo = 8-step flow-matching (fast), Base/SFT = 50-step.
  const MUSIC_DEFAULTS = {
    turbo: { steps: 8,  guidance: 1.0 },
    base:  { steps: 50, guidance: 1.0 },
    sft:   { steps: 50, guidance: 1.0 },
  };
  function music_variant_of(id) {
    const lid = (id || "").toLowerCase();
    if (lid.indexOf("turbo") >= 0) return "turbo";
    if (lid.indexOf("base")  >= 0) return "base";
    if (lid.indexOf("sft")   >= 0) return "sft";
    return "turbo";  // safe default — short horizon
  }
  function music_label_for(id) {
    const lid = (id || "").toLowerCase();
    const v = music_variant_of(id);
    const scrag = lid.indexOf("scrag") >= 0 ? " · ScragVAE" : "";
    const xl = lid.indexOf("xl") >= 0 ? "XL " : "";
    const map = {
      turbo: "⚡ " + xl + "Turbo",
      base:  "🎚️ " + xl + "Base",
      sft:   "🎯 " + xl + "SFT",
    };
    return (map[v] || id) + scrag + "  ·  " + MUSIC_DEFAULTS[v].steps + " steps";
  }

  // Track which models are proxied to the reference ace-server.
  // Populated from the "backend" field in /v1/models responses.
  const PROXY_MODELS = new Set();
  function is_proxy(id) { return PROXY_MODELS.has(id); }
  function apply_music_defaults(id, { force = false } = {}) {
    const v = music_variant_of(id);
    const d = MUSIC_DEFAULTS[v];
    const sInput = $("#mus-steps");
    const gInput = $("#mus-guid");
    if (sInput && (force || !sInput.dataset.userTouched)) sInput.value = d.steps;
    if (gInput && (force || !gInput.dataset.userTouched)) gInput.value = d.guidance;
    const gh = $("#mus-guid-hint");
    if (gh) gh.textContent = (v === "turbo") ? "(distilled → forced 1.0)" : "(CFG strength)";
  }

  function set_engine_dot(id, state) {
    const el = $("#" + id);
    if (el) el.className = "engine-dot " + (state || "");
  }

  // Categorise a model into one or more buckets based on family + id.
  function categorize(m) {
    const f = (m.family || "").toLowerCase();
    const id = (m.id || "").toLowerCase();
    const entry = { id: m.id, family: m.family || "", loaded: m.loaded !== false,
                    backend: m.backend || "" };
    const cats = [];
    // ACE-Step → music
    if (f.indexOf("ace") >= 0 || id.indexOf("ace") >= 0) cats.push("music");
    // MOSS-SFX or any id with "sfx" → sfx
    if (f.indexOf("moss_sfx") >= 0 || id.indexOf("sfx") >= 0) cats.push("sfx");
    // VoiceDesign variants
    if (id.indexOf("voicedesign") >= 0 || id.indexOf("voice_design") >= 0) cats.push("vd");
    // Qwen3-TTS and MOSS-TTS (non-sfx, non-vd) → tts
    if ((f.indexOf("qwen3_tts") >= 0 || f.indexOf("moss_tts") >= 0) &&
        cats.indexOf("sfx") < 0 && cats.indexOf("vd") < 0)
      cats.push("tts");
    return { entry, cats };
  }

  // Populate a <select> from a model list, preserving the current selection
  // if possible. Unloaded models are shown but greyed out.
  function populate_model_select(sel, list, labelFn) {
    if (!sel) return;
    const prev = sel.value;
    sel.innerHTML = "";
    if (!list.length) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "— no model registered —";
      opt.disabled = true;
      opt.selected = true;
      return;
    }
    for (const m of list) {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = (labelFn ? labelFn(m.id) : m.id) +
                        (m.loaded ? "" : "  (unload)");
      if (!m.loaded) opt.style.color = "var(--text-dim)";
      sel.appendChild(opt);
    }
    // Restore previous selection if still present, else pick first loaded.
    if (list.some(m => m.id === prev))
      sel.value = prev;
    else {
      const firstLoaded = list.find(m => m.loaded);
      sel.value = (firstLoaded || list[0]).id;
    }
  }

  // Auto-load a model if it's not already resident.
  async function ensure_loaded(modelId, statusEl) {
    const m = MODELS.find(x => x.id === modelId);
    if (m && m.loaded !== false) return true;
    if (statusEl) status(statusEl, "Loading " + modelId + "…", "loading");
    try {
      const r = await fetch("/v1/models/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: modelId }),
      });
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        throw new Error(j.error || "load failed (" + r.status + ")");
      }
      await load_models();   // refresh state
      return true;
    } catch (e) {
      if (statusEl) status(statusEl, "Model load failed: " + e.message, "err");
      return false;
    }
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
        set_engine_dot("engine-sfx", "");
        return;
      }

      // Categorise every model.
      MUSIC_MODELS = [];
      TTS_MODELS   = [];
      SFX_MODELS   = [];
      VD_MODELS    = [];
      PROXY_MODELS.clear();
      for (const m of MODELS) {
        const { entry, cats } = categorize(m);
        if (entry.backend === "acestep_proxy") PROXY_MODELS.add(entry.id);
        for (const c of cats) {
          if (c === "music") MUSIC_MODELS.push(entry);
          if (c === "tts")   TTS_MODELS.push(entry);
          if (c === "sfx")   SFX_MODELS.push(entry);
          if (c === "vd")    VD_MODELS.push(entry);
        }
      }

      // Resolve ACTIVE_*: keep current selection if still loaded (or if no
      // loaded alternative exists); otherwise switch to first loaded model.
      function resolve_active(list, current) {
        const cur = list.find(m => m.id === current);
        if (cur && cur.loaded) return current;          // still loaded ✓
        const firstLoaded = list.find(m => m.loaded);
        if (firstLoaded) return firstLoaded.id;          // switch to loaded
        if (cur) return current;                         // not loaded but exists
        return list.length ? list[0].id : null;          // fallback
      }
      ACTIVE_MUSIC = resolve_active(MUSIC_MODELS, ACTIVE_MUSIC);
      ACTIVE_TTS   = resolve_active(TTS_MODELS,   ACTIVE_TTS);
      ACTIVE_SFX   = resolve_active(SFX_MODELS,   ACTIVE_SFX);
      ACTIVE_VD    = resolve_active(VD_MODELS,    ACTIVE_VD);

      // Populate the four model <select> dropdowns.
      populate_model_select($("#mus-model"), MUSIC_MODELS, music_label_for);
      populate_model_select($("#syn-model"), TTS_MODELS);
      populate_model_select($("#sfx-model"), SFX_MODELS);
      populate_model_select($("#vd-model"),  VD_MODELS);

      // Apply defaults for the active music model.
      if (ACTIVE_MUSIC) apply_music_defaults(ACTIVE_MUSIC, { force: true });
      update_engine_tag();

      badge.textContent = MODELS.length + " model" + (MODELS.length === 1 ? "" : "s") +
        (PROXY_MODELS.size > 0 ? " · " + PROXY_MODELS.size + " ref⚡" : "");
      badge.className = "badge";
      set_engine_dot("engine-tts",   ACTIVE_TTS   ? "ready" : "");
      set_engine_dot("engine-vd",    ACTIVE_VD    ? "ready" : "");
      set_engine_dot("engine-music", ACTIVE_MUSIC ? "ready" : "");
      set_engine_dot("engine-sfx",   ACTIVE_SFX   ? "ready" : "");
      render_model_manager();
    } catch (e) {
      $("#model-badge").textContent = "offline";
      $("#model-badge").className = "badge err";
      set_engine_dot("engine-tts", "");
      set_engine_dot("engine-vd", "");
      set_engine_dot("engine-music", "");
      set_engine_dot("engine-sfx", "");
    }
  }

  // Mark a dot as "active" while a request is in flight, then back to ready.
  function with_engine_dot(id, promise) {
    set_engine_dot(id, "active");
    return promise.finally(() => set_engine_dot(id, ACTIVE_TTS || ACTIVE_VD || ACTIVE_MUSIC || ACTIVE_SFX ? "ready" : ""));
  }

  // ── Voice Design tab ───────────────────────────────────────────────
  $$("#vd-presets .chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      $("#vd-instruct").value = chip.dataset.desc;
      $("#vd-instruct").focus();
    });
  });

  let vd_last_blob = null;
  let vd_saved_name = null;   // auto-saved .voice name from last design

  $("#vd-btn").addEventListener("click", async () => {
    const st = $("#vd-status");
    const btn = $("#vd-btn");
    const instruct = $("#vd-instruct").value.trim();
    const text = $("#vd-text").value.trim();
    if (!instruct) { status(st, "Describe a voice first.", "err"); return; }
    if (!text)     { status(st, "Type some sample text.", "err"); return; }
    // Voice Design requires the dedicated VoiceDesign checkpoint — the
    // CustomVoice talker cannot do voice_design mode.  Show a clear error
    // instead of silently falling back to a model that will reject it.
    if (!ACTIVE_VD) {
      status(st, "Voice Design requires the qwen3-tts-voicedesign model. "
        + "Load it from the \u{1F4E6} Models tab (or unload CustomVoice first to free VRAM).", "err");
      return;
    }

    btn.disabled = true;
    btn.textContent = "Designing…";
    status(st, "Generating audio…", "loading");
    clear_player($("#vd-player"));

    try {
      const body = {
        model: ACTIVE_VD,
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
      $("#vd-save").disabled = false;

      // Auto-save the designed voice as a .voice file so it's immediately
      // available in the Synthesize tab without a manual save step.
      status(st, "Done — auto-saving voice for Synthesize…", "ok");
      try {
        const form = new FormData();
        form.append("file", blob, "designed_" + Date.now() + ".wav");
        form.append("encode_voice", "true");
        const sr = await fetch("/v1/clips/upload", { method: "POST", body: form });
        const sj = await sr.json();
        if (sj.ok && sj.saved && sj.saved.length) {
          vd_saved_name = sj.saved[0];
          $("#vd-to-syn").disabled = false;
          status(st, "Voice saved as " + vd_saved_name + ". Click “Use in Synthesize” to speak with it.", "ok");
          toast("Voice designed and saved!", "ok");
        }
      } catch (_) {
        status(st, "Done. Click “Save as .voice” to use in Synthesize.", "ok");
      }
    } catch (err) {
      status(st, "Error: " + err.message, "err");
    }

    btn.disabled = false;
    btn.textContent = "✨ Design voice";
  });

  // "Use in Synthesize" — jumps to the synthesize tab and auto-selects the
  // most recently designed voice so the user can immediately type text and
  // synthesize with it.
  $("#vd-to-syn").addEventListener("click", async () => {
    if (!vd_saved_name) return;
    // Ensure the uploaded-voice dropdown is populated.
    await load_voices();
    activate_tab("synthesize");
    $("#syn-voice-source").value = "uploaded";
    $("#syn-voice-source").dispatchEvent(new Event("change"));
    // Select the designed voice in the dropdown.
    const sel = $("#syn-uploaded");
    for (const o of sel.options) {
      if (o.value.indexOf(vd_saved_name) >= 0 || o.textContent.indexOf(vd_saved_name) >= 0) {
        sel.selectedIndex = o.index;
        break;
      }
    }
    load_syn_voice_meta();
    $("#syn-text").focus();
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

  // ── QWEN Voice Maker tab ─────────────────────────────────────────────
  // Design a voice via instruct, export as .qvoice, use with emotion/str.

  // Load saved .qvoice files.
  async function load_qvoices() {
    const sel = $("#qvm-voice-select");
    if (!sel) return;
    try {
      const r = await fetch("/v1/qvoices");
      const j = await r.json();
      const voices = j.voices || [];
      sel.innerHTML = "";
      if (!voices.length) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "— no .qvoice files —";
        opt.disabled = true;
        opt.selected = true;
        sel.appendChild(opt);
        $("#qvm-voice-controls").style.display = "none";
        return;
      }
      for (const v of voices) {
        const opt = document.createElement("option");
        opt.value = v.name;
        const type = v.size > 100 * 1024 * 1024 ? " [WDELTA]" : "";
        opt.textContent = v.name + " (" + fmt_size(v.size) + type + ")";
        sel.appendChild(opt);
      }
      sel.dispatchEvent(new Event("change"));
    } catch (err) {
      sel.innerHTML = '<option value="">— error loading —</option>';
    }
  }

  // Voice select change → show controls.
  $("#qvm-voice-select")?.addEventListener("change", () => {
    const val = $("#qvm-voice-select").value;
    $("#qvm-voice-controls").style.display = val ? "" : "none";
  });

  // Refresh button.
  $("#qvm-refresh")?.addEventListener("click", load_qvoices);

  // Strength slider display.
  $("#qvm-strength")?.addEventListener("input", () => {
    $("#qvm-strength-val").textContent = parseFloat($("#qvm-strength").value).toFixed(2);
  });

  // Preset chips.
  $$("#qvm-presets .chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      $("#qvm-instruct").value = chip.dataset.desc;
      $("#qvm-instruct").focus();
    });
  });

  // Listen button — preview the voice via CLI VoiceDesign.
  let qvm_last_blob = null;
  $("#qvm-listen")?.addEventListener("click", async () => {
    const st = $("#qvm-status");
    const btn = $("#qvm-listen");
    const instruct = $("#qvm-instruct").value.trim();
    const text = $("#qvm-text").value.trim();
    if (!instruct) { status(st, "Describe a voice first.", "err"); return; }
    if (!text)     { status(st, "Type some sample text.", "err"); return; }

    btn.disabled = true;
    btn.textContent = "🔊 Listening…";
    status(st, "Generating preview via VoiceDesign CLI…", "loading");
    clear_player($("#qvm-player"));

    try {
      const r = await fetch("/v1/voice/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ instruct, text }),
      });
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        throw new Error(j.detail || j.error || "Preview failed (" + r.status + ")");
      }
      const blob = await r.blob();
      qvm_last_blob = blob;
      const url = URL.createObjectURL(blob);
      set_player($("#qvm-player"), url,
        "preview · " + (blob.size / 1024).toFixed(1) + " KB");
      $("#qvm-export").disabled = false;
      status(st, "Preview ready. Click Export to save as .qvoice.", "ok");
    } catch (err) {
      status(st, "Error: " + err.message, "err");
    }

    btn.disabled = false;
    btn.textContent = "🔊 Listen";
  });

  // Export button — save as .qvoice via /v1/voice/export.
  $("#qvm-export")?.addEventListener("click", async () => {
    const st = $("#qvm-status");
    const instruct = $("#qvm-instruct").value.trim();
    const text = $("#qvm-text").value.trim();
    const wdelta = $("#qvm-wdelta")?.checked || false;
    if (!instruct) { status(st, "Describe a voice first.", "err"); return; }

    // Generate a name from the instruct (first 30 chars, sanitized).
    let name = instruct.replace(/[^a-zA-Z0-9 ]/g, "").trim().substring(0, 30).replace(/\s+/g, "_").toLowerCase();
    if (!name) name = "voice_" + Date.now();

    const mode = wdelta ? "WDELTA (~3GB, ~2min)" : "lite graft (~25MB, ~30s)";
    status(st, "Exporting " + name + ".qvoice… " + mode, "loading");

    try {
      const r = await fetch("/v1/voice/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, instruct, text, wdelta }),
      });
      const j = await r.json();
      if (!r.ok || !j.ok) {
        throw new Error(j.detail || j.error || "Export failed");
      }
      const type = j.wdelta ? "WDELTA" : "lite";
      status(st, "Exported " + j.name + ".qvoice (" + fmt_size(j.size) + ", " + type + "). Select it below to use.", "ok");
      toast("Voice exported!", "ok");
      await load_qvoices();
    } catch (err) {
      status(st, "Error: " + err.message, "err");
    }
  });

  // Expr weight slider display.
  $("#qvm-expr-weight")?.addEventListener("input", () => {
    $("#qvm-expr-weight-val").textContent = parseFloat($("#qvm-expr-weight").value).toFixed(2);
  });

  // Generate button — use a saved .qvoice with emotion + strength + expr.
  $("#qvm-use")?.addEventListener("click", async () => {
    const st = $("#qvm-use-status");
    const voiceName = $("#qvm-voice-select").value;
    if (!voiceName) { status(st, "Select a voice first.", "err"); return; }
    const text = $("#qvm-use-text").value.trim();
    if (!text) { status(st, "Type some text.", "err"); return; }
    const emotion = $("#qvm-emotion").value;
    const strength = parseFloat($("#qvm-strength").value) || 1.0;
    const instruct = $("#qvm-use-instruct").value.trim();
    const exprFile = $("#qvm-expr")?.value || "";
    const exprWeight = parseFloat($("#qvm-expr-weight")?.value) || 1.0;
    const temperature = parseFloat($("#qvm-temp")?.value) || 0.9;
    const topP = parseFloat($("#qvm-topp")?.value) || 1.0;
    const topK = parseInt($("#qvm-topk")?.value) || 50;
    const repPenalty = parseFloat($("#qvm-rep")?.value) || 1.05;

    // Find the CustomVoice model.
    const cvModel = MODELS.find(m => m.id.indexOf("customvoice") >= 0 && m.loaded);
    if (!cvModel) {
      status(st, "No CustomVoice model loaded. Load qwen3-tts-customvoice from Models tab.", "err");
      return;
    }

    const btn = $("#qvm-use");
    btn.disabled = true;
    btn.textContent = "🗣️ Generating…";
    status(st, "Generating with " + voiceName + "…", "loading");
    clear_player($("#qvm-use-player"));

    try {
      const body = {
        model: cvModel.id,
        input: text,
        voice: voiceName,
        emotion,
        voice_strength: strength,
        instruct,
        temperature,
        top_p: topP,
        top_k: topK,
        repetition_penalty: repPenalty,
        seed: Math.floor(Math.random() * 1000000),
      };
      if (exprFile) {
        body.expr_file = exprFile;
        body.expr_weight = exprWeight;
      }
      const p = fetch("/v1/audio/speech", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then(async (r) => {
        if (!r.ok) {
          const j = await r.json().catch(() => ({}));
          throw new Error(j.detail || j.error || "Generation failed (" + r.status + ")");
        }
        return r.blob();
      });
      const blob = await p;
      const url = URL.createObjectURL(blob);
      const desc = [emotion || "", exprFile ? "expr:" + exprFile : ""].filter(Boolean).join("+") || "neutral";
      set_player($("#qvm-use-player"), url,
        voiceName + " · " + desc + " · " + fmt_size(blob.size));
      status(st, "Done! " + voiceName + " with " + desc + ".", "ok");
      toast("Generated!", "ok");
    } catch (err) {
      status(st, "Error: " + err.message, "err");
    }

    btn.disabled = false;
    btn.textContent = "🗣️ Generate";
  });

  // Re-bake button — generate emotional audio, then bake as new voice.
  $("#qvm-rebake")?.addEventListener("click", async () => {
    const st = $("#qvm-rebake-status");
    const sourceVoice = $("#qvm-voice-select").value;
    if (!sourceVoice) { status(st, "Select a source voice first.", "err"); return; }
    const emotion = $("#qvm-emotion").value;
    const instruct = $("#qvm-use-instruct").value.trim();
    if (!emotion && !instruct) { status(st, "Pick an emotion or type an instruct.", "err"); return; }
    let newName = $("#qvm-rebake-name").value.trim();
    if (!newName) {
      newName = sourceVoice + "_" + (emotion || "custom");
      $("#qvm-rebake-name").value = newName;
    }
    const wdelta = $("#qvm-rebake-wdelta")?.checked || false;

    const mode = wdelta ? "WDELTA (~3GB, ~2min)" : "lite (~25MB, ~30s)";
    status(st, "Re-baking " + newName + " with " + (emotion || instruct) + " baked in… " + mode, "loading");

    try {
      const r = await fetch("/v1/voice/rebake", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_voice: sourceVoice,
          name: newName,
          emotion,
          instruct,
          wdelta,
        }),
      });
      const j = await r.json();
      if (!r.ok || !j.ok) throw new Error(j.detail || j.error || "Rebake failed");
      const type = j.wdelta ? "WDELTA" : "lite";
      status(st, "Re-baked " + j.name + ".qvoice (" + fmt_size(j.size) + ", " + type + "). Emotion baked in.", "ok");
      toast("Voice re-baked!", "ok");
      await load_qvoices();
    } catch (err) {
      status(st, "Error: " + err.message, "err");
    }
  });

  // Send to Synthesize tab.
  $("#qvm-use-to-syn")?.addEventListener("click", async () => {
    const voiceName = $("#qvm-voice-select").value;
    if (!voiceName) return;
    await load_voices();
    activate_tab("synthesize");
    // Set voice source to uploaded and select the voice.
    $("#syn-voice-source").value = "uploaded";
    $("#syn-voice-source").dispatchEvent(new Event("change"));
    const sel = $("#syn-uploaded");
    for (const o of sel.options) {
      if (o.value.indexOf(voiceName) >= 0 || o.textContent.indexOf(voiceName) >= 0) {
        sel.selectedIndex = o.index;
        break;
      }
    }
    load_syn_voice_meta();
    $("#syn-text").focus();
  });

  // Load qvoices on startup.
  load_qvoices();

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
    $("#syn-preset").hidden       = v !== "preset";
    $("#syn-uploaded").hidden     = !(v === "uploaded" || v === "identity_style");
    $("#syn-ref-row").hidden      = v !== "reference";
    $("#syn-style-row").hidden    = v !== "identity_style";
    // Voice strength slider: show for any voice source except "none".
    $("#syn-strength-row").hidden = v === "none";
    if (v === "uploaded" || v === "identity_style") load_syn_voice_meta();
    if (v === "designed") {
      if (vd_saved_name) {
        status($("#syn-status"), "Using last designed voice: " + vd_saved_name, "info");
      } else {
        status($("#syn-status"), "No designed voice yet — create one in the Voice Design tab.", "info");
      }
    }
  });

  // Voice strength slider visual feedback.
  $("#syn-voice-strength")?.addEventListener("input", (e) => {
    $("#syn-voice-strength-val").textContent = parseFloat(e.target.value).toFixed(1);
    sync_slider_fill(e.target);
  });
  sync_slider_fill($("#syn-voice-strength"));

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
      max_new_tokens: parseInt($("#syn-max").value, 10) || 220,
      pitch_shift: parseFloat($("#syn-pitch").value) || 0,
      speed:       parseFloat($("#syn-speed").value) || 1.0,
    };
    // Voice embed strength — scale speaker identity influence.
    // 1.0 = original, >1 = stronger voice, <1 = weaker (more from instruct).
    // Only sent when a voice source is active (not "none").
    if (src !== "none") {
      const vs = parseFloat($("#syn-voice-strength")?.value);
      if (!isNaN(vs) && vs > 0) body.embedding_strength = vs;
    }
    {
      const seed = parseInt($("#syn-seed").value, 10) || 0;
      if (seed > 0) body.seed = seed;
    }

    if (src === "preset") {
      body.speaker = $("#syn-preset").value;
      body.mode = "voice_clone";
    } else if (src === "uploaded" || src === "designed") {
      // "designed" auto-uses the most recent Voice Design output;
      // "uploaded" lets the user pick from saved .voice files.
      const v = (src === "designed") ? vd_saved_name : $("#syn-uploaded").value;
      if (!v) {
        status(st, src === "designed"
          ? "No designed voice yet — create one in the Voice Design tab."
          : "No saved voice — generate one in Voice Design first.", "err");
        return;
      }
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
    } else if (src === "identity_style") {
      // Cross-speaker style transfer: identity comes from a saved .voice
      // (your character), acting/prosody is copied from a separate expressive
      // reference WAV. The backend keeps the .voice embedding and ALSO
      // encodes the style WAV into codec ref-codes — so you hear your
      // character performing the other clip's delivery.
      const idV = $("#syn-uploaded").value;
      if (!idV) { status(st, "Pick an identity voice (.voice) first.", "err"); return; }
      const sf = $("#syn-style-file").files[0];
      if (!sf) { status(st, "Pick a style reference audio file.", "err"); return; }
      btn.disabled = true; btn.textContent = "Uploading style…";
      status(st, "Uploading style reference…", "loading");
      try {
        const form = new FormData();
        form.append("file", sf, sf.name);
        const up = await fetch("/v1/clips/upload", { method: "POST", body: form });
        const uj = await up.json();
        if (!uj.ok || !uj.saved || !uj.saved.length) {
          status(st, "Style upload failed", "err"); return;
        }
        body.voice          = idV;                       // identity (embedding)
        body.reference_audio = uj.saved[0];               // style (ref-codes)
        body.reference_text  = $("#syn-style-text").value.trim();
        body.mode = "voice_clone";
      } catch (err) {
        status(st, "Style upload error: " + err.message, "err"); return;
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
      $("#syn-save-clip").disabled = false;
      status(st, "Done.", "ok");
    } catch (err) {
      status(st, "Error: " + err.message, "err");
    }

    btn.disabled = false;
    btn.textContent = "🎙️ Synthesize";
  });

  // Save the last synthesis to the Clips library so it appears alongside
  // other generated audio. Plain TTS isn't auto-saved by the backend (to
  // avoid clip-spam), so this is the explicit user opt-in.
  $("#syn-save-clip").addEventListener("click", async () => {
    if (!syn_last_blob) return;
    const st = $("#syn-status");
    status(st, "Saving to clips…", "loading");
    try {
      const form = new FormData();
      form.append("file", syn_last_blob, "tts_" + Date.now() + ".wav");
      const r = await fetch("/v1/clips/upload", { method: "POST", body: form });
      const j = await r.json();
      if (j.ok && j.saved && j.saved.length) {
        CLIPS_CACHE = [];
        toast("Saved as " + j.saved[0], "ok");
        status(st, "Saved as " + j.saved[0] + " — find it in the Clips tab.", "ok");
      } else {
        status(st, j.error || "Save failed", "err");
      }
    } catch (err) {
      status(st, "Error: " + err.message, "err");
    }
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
    // Contextual guidance hint. Turbo models always force guidance=1.0
    // (distilled into weights), so the cover-mode "lower = faithful" hint
    // is misleading — turbo ignores guidance_scale entirely.
    const gh = $("#mus-guid-hint");
    if (gh) {
      const variant = ACTIVE_MUSIC ? music_variant_of(ACTIVE_MUSIC) : "turbo";
      if (variant === "turbo")
        gh.textContent = "(distilled → forced 1.0)";
      else
        gh.textContent = (mode === "cover")
          ? "(lower = faithful to source)"
          : "(CFG strength)";
    }
    if (needs_src) refresh_music_sources();
  }
  $("#mus-mode").addEventListener("change", sync_music_mode_rows);
  sync_music_mode_rows();

  // Show/hide the "⚡ ref" badge for proxy models.
  function update_engine_tag() {
    const tag = $("#mus-engine-tag");
    if (!tag) return;
    if (ACTIVE_MUSIC && is_proxy(ACTIVE_MUSIC)) {
      tag.classList.remove("hidden");
      tag.title = "Served by the reference acestep.cpp engine (ace-server)";
    } else {
      tag.classList.add("hidden");
    }
  }

  // ── Model picker change handlers ───────────────────────────────────
  // Each dropdown updates its ACTIVE_* and, if the selected model isn't
  // loaded yet, triggers a load via /v1/models/load before proceeding.
  $("#mus-model")?.addEventListener("change", async (e) => {
    ACTIVE_MUSIC = e.target.value;
    apply_music_defaults(ACTIVE_MUSIC, { force: true });
    sync_music_mode_rows();
    update_engine_tag();
    await ensure_loaded(ACTIVE_MUSIC, $("#mus-status"));
  });
  $("#syn-model")?.addEventListener("change", async (e) => {
    ACTIVE_TTS = e.target.value;
    await ensure_loaded(ACTIVE_TTS, $("#syn-status"));
  });
  $("#sfx-model")?.addEventListener("change", async (e) => {
    ACTIVE_SFX = e.target.value;
    await ensure_loaded(ACTIVE_SFX, $("#sfx-status"));
  });
  $("#vd-model")?.addEventListener("change", async (e) => {
    ACTIVE_VD = e.target.value;
    await ensure_loaded(ACTIVE_VD, $("#vd-status"));
  });

  // Caption preset chips: one-click genre/mood fill.
  $$("#mus-presets .chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      $("#mus-caption").value = chip.dataset.caption;
      $("#mus-caption").focus();
    });
  });

  // Track manual edits to steps/guidance so switching models doesn't clobber
  // deliberate user choices.
  ["mus-steps", "mus-guid"].forEach((id) => {
    const el = $("#" + id);
    if (el) el.addEventListener("input", () => { el.dataset.userTouched = "1"; });
  });

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
          steps:    parseInt($("#mus-steps").value, 10) ||
                    MUSIC_DEFAULTS[music_variant_of(ACTIVE_MUSIC)].steps,
          guidance_scale: parseFloat($("#mus-guid").value) || 1.0,
        };
        // Musical metadata (optional — when all are set, Phase 1 reasoning
        // is skipped and the LM goes straight to code generation with the
        // deterministic CoT YAML built from these values).
        const bpm = parseInt($("#mus-bpm")?.value, 10) || 0;
        if (bpm > 0) body.bpm = bpm;
        const ks = $("#mus-keyscale")?.value.trim();
        if (ks) body.keyscale = ks;
        const tsig = $("#mus-timesig")?.value.trim();
        if (tsig) body.timesignature = tsig;
        const lang = $("#mus-lang")?.value.trim();
        if (lang) body.vocal_language = lang;
        const lmCfg = parseFloat($("#mus-lm-cfg")?.value);
        if (lmCfg && lmCfg > 1.0) body.lm_cfg_scale = lmCfg;
        // LM sampling params — always sent so the server uses our values,
        // not its internal defaults. Matches MusicRequest defaults (0.85, 0.9).
        body.temperature = parseFloat($("#mus-temp")?.value) || 0.85;
        body.top_p       = parseFloat($("#mus-top-p")?.value) || 0.9;
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
          // The backend auto-saves each render to the clips library and
          // surfaces the filename via this header — capture it so we can
          // tell the user where to find it.
          const clip = r.headers.get("X-Audiocore-Clip");
          const blob = await r.blob();
          return { blob, clip };
        });
        // Light up the music dot during the first request.
        if (i === 0) engine_promise = with_engine_dot("engine-music", req);
        const rslt = await req;
        results.push({ seed: seed_i, blob: rslt.blob, clip: rslt.clip });
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

    // If the backend auto-saved any clips, refresh the cached clips list so
    // the Clips tab shows them without a manual refresh, and surface the
    // saved filenames in the status line.
    const savedClips = ok.map(r => r.clip).filter(Boolean);
    if (savedClips.length) {
      // Invalidate the cache so the next tab visit re-fetches.
      CLIPS_CACHE = [];
      if ($(".tab.active")?.dataset.tab === "clips") load_clips();
    }
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
    const saveNote = savedClips.length
      ? " · saved " + (savedClips.length === 1 ? savedClips[0] : savedClips.length + " clips")
      : "";
    if (failed.length) {
      status(st, ok.length + "/" + batch_n + " OK · " + failed.length + " failed: " +
        failed[0].error, "err");
    } else if (batch_n > 1) {
      status(st, ok.length + " variations ready (seeds " +
        ok.map(r => r.seed || "?").join(", ") + ")." + saveNote, "ok");
    } else {
      status(st, "Done." + saveNote, "ok");
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

  // ── SFX tab (MOSS Sound Effects) ───────────────────────────────────
  let sfx_last_blob = null;

  // SFX preset chips: one-click prompt fill.
  $$("#sfx-presets .chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      $("#sfx-text").value = chip.dataset.prompt;
      $("#sfx-text").focus();
    });
  });

  $("#sfx-btn").addEventListener("click", async () => {
    const st = $("#sfx-status");
    const btn = $("#sfx-btn");
    if (!ACTIVE_SFX) {
      status(st, "No SFX model loaded. Load moss-sfx-v2 from the \u{1F4E6} Models tab.", "err");
      return;
    }
    const text = $("#sfx-text").value.trim();
    if (!text) { status(st, "Describe the sound effect first.", "err"); return; }

    const duration = parseFloat($("#sfx-dur").value) || 5;
    const sfx_seed = parseInt($("#sfx-seed")?.value, 10) || 0;
    // moss_sfx_v2 uses duration_tokens (1 token ≈ 0.08s)
    const duration_tokens = Math.round(duration / 0.08);

    btn.disabled = true; btn.textContent = "Generating…";
    status(st, "Generating sound effect…", "loading");
    clear_player($("#sfx-player"));

    try {
      const reqBody = {
          model: ACTIVE_SFX,
          input: text,
          mode: "tts",
          duration_tokens: duration_tokens,
      };
      if (sfx_seed > 0) reqBody.seed = sfx_seed;
      const p = fetch("/v1/audio/speech", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(reqBody),
      }).then(async (r) => {
        if (!r.ok) {
          const j = await r.json().catch(() => ({}));
          throw new Error(j.error || "SFX failed (" + r.status + ")");
        }
        const clip = r.headers.get("X-Audiocore-Clip");
        const blob = await r.blob();
        return { blob, clip };
      });
      const { blob, clip } = await with_engine_dot("engine-sfx", p);
      sfx_last_blob = blob;
      const url = URL.createObjectURL(blob);
      set_player($("#sfx-player"), url,
        "sfx · " + text.substring(0, 40) + " · " + (blob.size / 1024).toFixed(1) + " KB");
      $("#sfx-download").disabled = false;
      if (clip) {
        CLIPS_CACHE = [];
        if ($(".tab.active")?.dataset.tab === "clips") load_clips();
        status(st, "Done · saved " + clip, "ok");
      } else {
        status(st, "Done.", "ok");
      }
    } catch (err) {
      status(st, "Error: " + err.message, "err");
    }
    btn.disabled = false; btn.textContent = "🔊 Generate SFX";
  });

  $("#sfx-download").addEventListener("click", () => {
    if (!sfx_last_blob) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(sfx_last_blob);
    a.download = "sfx_" + Date.now() + ".wav";
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

  // ── PCA Voice Space Analysis ───────────────────────────────────────
  const PCA_COLORS = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
                       "#1abc9c","#e67e22","#34495e","#e84393","#00cec9"];

  $("#pca-btn").addEventListener("click", async () => {
    const st = $("#pca-status");
    const btn = $("#pca-btn");
    const nComp = parseInt($("#pca-ncomp").value, 10) || 10;
    const nClu  = parseInt($("#pca-nclu").value, 10) || 5;
    btn.disabled = true; btn.textContent = "Analyzing…";
    status(st, "Running PCA + K-means on all voices…", "loading");
    try {
      const r = await fetch("/v1/voices/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          components: nComp, clusters: nClu, save_dirs: true,
        }),
      });
      const data = await r.json();
      if (data.error) { status(st, data.error, "err"); return; }
      render_pca(data);
      const cumPct = (data.cumulative_variance[data.n_components - 1] * 100).toFixed(1);
      const dirNote = data.saved_dirs && data.saved_dirs.length
        ? " · " + data.saved_dirs.length + " .dir steering vectors saved for Voice Maker"
        : "";
      status(st, data.n_voices + " voices · " + data.n_components + " PCs explain " +
             cumPct + "% variance · " + data.n_clusters + " clusters" + dirNote, "ok");
    } catch (e) {
      status(st, "Error: " + e.message, "err");
    }
    btn.disabled = false; btn.textContent = "📊 Analyze";
  });

  function render_pca(data) {
    const voices = data.voices;
    const plotEl = $("#pca-plot");
    const detEl  = $("#pca-details");

    // Persist state for the draggable explorer.
    PCA.data = data;

    // ── Scatter plot (PC1 × PC2, coloured by cluster) ──
    const pc1 = voices.map(v => v.pc[0]);
    const pc2 = voices.map(v => v.pc[1] || 0);
    const min1 = Math.min(...pc1), max1 = Math.max(...pc1);
    const min2 = Math.min(...pc2), max2 = Math.max(...pc2);
    const W = 560, H = 420, PAD = 50;
    PCA.W = W; PCA.H = H; PCA.PAD = PAD;
    PCA.min1 = min1; PCA.max1 = max1; PCA.min2 = min2; PCA.max2 = max2;
    const range1 = max1 - min1 + 0.01;
    const range2 = max2 - min2 + 0.01;
    const sx = v => PAD + ((v - min1) / range1) * (W - 2 * PAD);
    const sy = v => H - PAD - ((v - min2) / range2) * (H - 2 * PAD);

    let svg = '<svg width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + ' ' + H +
              '" style="max-width:100%;background:var(--bg-card);border-radius:8px">';
    // Axes
    svg += '<line x1="' + PAD + '" y1="' + (H - PAD) + '" x2="' + (W - PAD) +
           '" y2="' + (H - PAD) + '" stroke="var(--border)" stroke-width="1.5"/>';
    svg += '<line x1="' + PAD + '" y1="' + PAD + '" x2="' + PAD + '" y2="' + (H - PAD) +
           '" stroke="var(--border)" stroke-width="1.5"/>';
    // Axis labels
    const var1 = ((data.explained_variance_ratio[0] || 0) * 100).toFixed(1);
    const var2 = ((data.explained_variance_ratio[1] || 0) * 100).toFixed(1);
    svg += '<text x="' + (W / 2) + '" y="' + (H - 12) + '" text-anchor="middle" ' +
           'fill="var(--text-dim)" font-size="13">PC1 (' + var1 + '%)</text>';
    svg += '<text x="16" y="' + (H / 2) + '" text-anchor="middle" fill="var(--text-dim)" ' +
           'font-size="13" transform="rotate(-90 16 ' + (H / 2) + ')">PC2 (' + var2 + '%)</text>';
    // Zero lines (if in range)
    if (min1 < 0 && max1 > 0) {
      const zx = sx(0);
      svg += '<line x1="' + zx + '" y1="' + PAD + '" x2="' + zx + '" y2="' + (H - PAD) +
             '" stroke="var(--border)" stroke-width="0.5" stroke-dasharray="3,3" opacity="0.4"/>';
    }
    if (min2 < 0 && max2 > 0) {
      const zy = sy(0);
      svg += '<line x1="' + PAD + '" y1="' + zy + '" x2="' + (W - PAD) + '" y2="' + zy +
             '" stroke="var(--border)" stroke-width="0.5" stroke-dasharray="3,3" opacity="0.4"/>';
    }
    // Points
    for (const v of voices) {
      const cx = sx(v.pc[0]).toFixed(1);
      const cy = sy(v.pc[1] || 0).toFixed(1);
      const color = PCA_COLORS[v.cluster % PCA_COLORS.length];
      svg += '<circle cx="' + cx + '" cy="' + cy + '" r="5" fill="' + color +
             '" opacity="0.85" stroke="var(--bg-card)" stroke-width="1"/>';
      const label = v.name.replace(/\.voice$/, "");
      svg += '<text x="' + (parseFloat(cx) + 7) + '" y="' + (parseFloat(cy) + 4) +
             '" font-size="9" fill="var(--text-dim)">' + esc(label) + "</text>";
    }
    // ── Draggable explorer marker (initialized at the centroid = mean) ──
    // mean maps to PC (0,0); if (0,0) is outside the data range we pin it to
    // the centre of the plot so it's always grabbable.
    let mkX = (min1 <= 0 && max1 >= 0) ? PCA.toPxX(0) : (PAD + (W - 2 * PAD) / 2);
    let mkY = (min2 <= 0 && max2 >= 0) ? PCA.toPxY(0) : (PAD + (H - 2 * PAD) / 2);
    svg += '<g id="pca-marker" style="cursor:grab">';
    svg += '<line x1="' + (mkX - 9) + '" y1="' + mkY + '" x2="' + (mkX + 9) + '" y2="' + mkY +
           '" stroke="#f59e0b" stroke-width="1.5"/>';
    svg += '<line x1="' + mkX + '" y1="' + (mkY - 9) + '" x2="' + mkX + '" y2="' + (mkY + 9) +
           '" stroke="#f59e0b" stroke-width="1.5"/>';
    svg += '<circle cx="' + mkX + '" cy="' + mkY + '" r="8" fill="none" stroke="#f59e0b" ' +
           'stroke-width="2" opacity="0.9"/>';
    svg += '</g>';
    // Legend
    const clusters = [...new Set(voices.map(v => v.cluster))].sort();
    let lx = W - PAD - 60;
    for (const c of clusters) {
      const color = PCA_COLORS[c % PCA_COLORS.length];
      svg += '<rect x="' + lx + '" y="' + (PAD - 25) + '" width="10" height="10" fill="' +
             color + '" rx="2"/>';
      svg += '<text x="' + (lx + 14) + '" y="' + (PAD - 16) + '" font-size="10" fill="var(--text-dim)">C' +
             c + "</text>";
      lx += 45;
    }
    svg += "</svg>";
    plotEl.innerHTML = svg;

    // ── Draggable explorer wiring ──
    pca_init_explorer();

    // ── PC extremes table ──
    let html = '<div style="overflow-x:auto;margin-top:1em">';
    html += '<table style="width:100%;font-size:13px;border-collapse:collapse">';
    html += '<thead><tr style="text-align:left;border-bottom:2px solid var(--border)">' +
            '<th style="padding:6px">PC</th>' +
            '<th style="padding:6px">Variance</th>' +
            '<th style="padding:6px">Cumulative</th>' +
            '<th style="padding:6px">− Negative end</th>' +
            '<th style="padding:6px">+ Positive end</th>' +
            "</tr></thead><tbody>";
    for (let i = 0; i < data.n_components; i++) {
      const ext = data.pc_extremes[i];
      const v = (data.explained_variance_ratio[i] * 100).toFixed(1);
      const cv = (data.cumulative_variance[i] * 100).toFixed(1);
      html += '<tr style="border-bottom:1px solid var(--border)">' +
              '<td style="padding:6px;font-weight:600">PC' + (i + 1) + "</td>" +
              '<td style="padding:6px">' + v + "%</td>" +
              '<td style="padding:6px">' + cv + "%</td>" +
              '<td style="padding:6px">' + esc(ext.negative_end) + "</td>" +
              '<td style="padding:6px">' + esc(ext.positive_end) + "</td>" +
              "</tr>";
    }
    html += "</tbody></table></div>";

    // Cluster breakdown
    html += '<div style="margin-top:1em;display:flex;flex-wrap:wrap;gap:0.5em">';
    for (const c of clusters) {
      const members = voices.filter(v => v.cluster === c).map(v => v.name.replace(/\.voice$/, ""));
      const color = PCA_COLORS[c % PCA_COLORS.length];
      html += '<div style="border:1px solid var(--border);border-radius:8px;padding:0.5em 0.75em">' +
              '<div style="font-weight:600;margin-bottom:0.25em">' +
              '<span style="display:inline-block;width:10px;height:10px;background:' + color +
              ';border-radius:2px;margin-right:4px"></span>Cluster ' + c +
              " (" + members.length + ")</div>" +
              '<div style="font-size:11px;color:var(--text-dim)">' + esc(members.join(", ")) + "</div>" +
              "</div>";
    }
    html += "</div>";

    if (data.saved_dirs && data.saved_dirs.length) {
      html += '<p class="hint" style="margin-top:0.75em">💡 ' +
              data.saved_dirs.length + " steering directions saved (" +
              esc(data.saved_dirs.join(", ")) +
              "). Use them in Voice Maker → Apply steering vector to push any voice along these axes.</p>";
    }
    detEl.innerHTML = html;
  }

  // ── Per-axis numeric instruments ─────────────────────────────────────
  // Each principal component gets a precise numeric readout AND a slim
  // slider, both bidirectionally bound to the polygon vertex. This is the
  // cockpit: drag the polygon for feel, type here for exact values.
  function pca_render_axes() {
    const host = document.getElementById("pca-axes");
    if (!host || !PCA.data) return;
    const d = PCA.data;
    const axes = Math.min(d.n_components, 16);
    let html = "";
    for (let i = 0; i < axes; i++) {
      const rg = PCA.ranges[i] || 1;
      const varp = ((d.explained_variance_ratio[i] || 0) * 100).toFixed(0);
      const step = (rg / 200).toFixed(4);
      const al = d.axis_labels && d.axis_labels[i];
      const ext = d.pc_extremes && d.pc_extremes[i];
      const labeled = al && al.score > 0;
      const name = 'PC' + (i + 1) + (labeled ? ' ' + al.label : '');
      // Show the actual voice at each end — always unique, always descriptive
      const negName = String(ext ? ext.negative_end : '').replace(/\.voice$/, '');
      const posName = String(ext ? ext.positive_end : '').replace(/\.voice$/, '');
      const dir = negName + ' ←→ ' + posName;
      html += '<div class="pca-axis">' +
        '<span class="pca-axis-name">' + name + '</span>' +
        '<span class="pca-axis-dir">' + dir + '</span>' +
        '<span class="pca-axis-var">' + varp + '%</span>' +
        '<input type="range" class="slider pca-axis-slider" data-axis="' + i + '" ' +
          'min="' + (-rg).toFixed(3) + '" max="' + rg.toFixed(3) + '" step="' + step + '" value="0">' +
        '<input type="number" class="pca-axis-num" data-axis="' + i + '" ' +
          'min="' + (-rg).toFixed(3) + '" max="' + rg.toFixed(3) + '" step="0.01" value="0.000">' +
        '</div>';
    }
    host.innerHTML = html;
    host.querySelectorAll(".pca-axis-num, .pca-axis-slider").forEach((el) => {
      el.addEventListener("input", () => {
        const i = parseInt(el.dataset.axis, 10);
        pca_set_weight(i, parseFloat(el.value));
      });
    });
  }

  // Central single-source-of-truth setter: clamp to the observed range,
  // store, and refresh every bound view (polygon, scatter, other inputs).
  function pca_set_weight(i, val) {
    if (isNaN(val)) val = 0;
    const rg = PCA.ranges[i] || 1;
    PCA.weights[i] = Math.max(-rg, Math.min(rg, val));
    pca_refresh();
  }

  // Push the weight vector back into the axis inputs. Whatever element
  // currently has focus is skipped so typing / dragging is never clobbered.
  function pca_sync_axes() {
    const host = document.getElementById("pca-axes");
    if (!host) return;
    host.querySelectorAll(".pca-axis-num").forEach((el) => {
      const i = parseInt(el.dataset.axis, 10);
      if (document.activeElement !== el) el.value = (PCA.weights[i] || 0).toFixed(3);
    });
    host.querySelectorAll(".pca-axis-slider").forEach((el) => {
      const i = parseInt(el.dataset.axis, 10);
      if (document.activeElement !== el) el.value = PCA.weights[i] || 0;
    });
  }

  // Read the sculptor's sampling/DSP controls into a request-body fragment
  // so both the live "Play" and snapshot "A/B" speak the backend's language.
  function pca_sampling_params() {
    const o = {};
    const num = (id) => { const el = document.getElementById(id); return el ? parseFloat(el.value) : NaN; };
    const Int = (id) => { const el = document.getElementById(id); return el ? parseInt(el.value, 10) : NaN; };
    const t = num("pca-temp"), tp = num("pca-topp"), tk = Int("pca-topk"),
          mt = Int("pca-maxtok"), sd = Int("pca-seed"), pi = num("pca-pitch"), sp = num("pca-speed");
    if (!isNaN(t)) o.temperature = t;
    if (!isNaN(tp)) o.top_p = tp;
    if (!isNaN(tk)) o.text_top_k = tk;
    if (!isNaN(pi)) o.pitch_shift = pi;
    if (!isNaN(sp)) o.speed = sp;
    if (!isNaN(mt)) o.max_tokens = mt;
    if (!isNaN(sd)) o.seed = sd;
    return o;
  }

  // ── Voice-character DSP + prosody collection ───────────────────────────
  // Reads the formant/airiness/breathiness + pitch-contour/breath fields.
  // All default to neutral, so the object only carries non-zero values.
  function pca_collect_dsp() {
    const o = {};
    const f = (id) => { const el = document.getElementById(id); return el ? parseFloat(el.value) : NaN; };
    const w = f("dsp-warmth"), fo = f("dsp-formant"), br = f("dsp-brightness"),
          ai = f("dsp-airiness"), bi = f("dsp-breathiness");
    if (!isNaN(w)  && w  !== 0) o.dsp_warmth = w;
    if (!isNaN(fo) && fo !== 0) o.dsp_formant = fo;
    if (!isNaN(br) && br !== 0) o.dsp_brightness = br;
    if (!isNaN(ai) && ai !== 0) o.dsp_airiness = ai;
    if (!isNaN(bi) && bi !== 0) o.dsp_breathiness = bi;
    const sel = document.getElementById("prosody-contour");
    const shape = sel ? sel.value : "flat";
    const depth = f("prosody-contour-depth");
    if (shape && shape !== "flat" && !isNaN(depth) && depth > 0.05) {
      o.prosody_contour = shape;
      o.prosody_contour_depth = depth;
    }
    const pb = f("prosody-breath");
    if (!isNaN(pb) && pb > 0.01) o.prosody_breath = pb;
    return o;
  }

  // ── Emotion matrix ─────────────────────────────────────────────────────
  // 2D pad: X = energy (-1..1), Y = warmth (-1..1); plus intimacy (0..1).
  // Moving the pad composes a descriptive instruct string and modulates
  // temperature (energy) + embedding strength (intimacy). Manual edits to the
  // instruct field are preserved until the next pad move.
  const EMOTION = { warmth: 0.5, energy: 0.5, intimacy: 0.3 };
  let EMO_USER_INSTRUCT = false;  // user typed their own — don't clobber

  function emo_compose_instruct(w, e, i) {
    const parts = [];
    // warmth axis
    if (w > 0.7) parts.push("tender", "affectionate");
    else if (w > 0.3) parts.push("warm", "friendly");
    else if (w < -0.7) parts.push("cold", "detached");
    else if (w < -0.3) parts.push("cool", "composed");
    // energy axis
    if (e > 0.7) parts.push("energetic", "upbeat");
    else if (e > 0.3) parts.push("lively", "bright");
    else if (e < -0.7) parts.push("sleepy", "drowsy");
    else if (e < -0.3) parts.push("calm", "relaxed");
    // intimacy overlay
    if (i > 0.7) parts.push("whispered", "intimate");
    else if (i > 0.4) parts.push("soft", "close");
    if (!parts.length) return "Neutral, natural speaking voice.";
    // De-duplicate, capitalise, join.
    const seen = new Set(), out = [];
    for (const p of parts) { const lw = p.toLowerCase(); if (!seen.has(lw)) { seen.add(lw); out.push(p); } }
    out[0] = out[0][0].toUpperCase() + out[0].slice(1);
    return out.join(", ") + " tone.";
  }

  function emo_apply() {
    const dot = document.getElementById("emo-dot");
    const pad = document.getElementById("emo-pad");
    if (dot && pad) {
      // Map -1..1 → 0..100% within the pad.
      dot.style.left = ((EMOTION.energy  + 1) / 2 * 100) + "%";
      dot.style.top  = ((1 - EMOTION.warmth) / 2 * 100) + "%";
    }
    const ro = document.getElementById("emo-readout");
    if (ro) {
      const sgn = (v) => (v >= 0 ? "+" : "") + v.toFixed(2);
      ro.textContent = "warmth " + sgn(EMOTION.warmth) + " · energy " + sgn(EMOTION.energy);
    }
    const iv = document.getElementById("emo-intimacy-val");
    if (iv) iv.textContent = EMOTION.intimacy.toFixed(2);
    // Compose instruct unless the user typed their own.
    if (!EMO_USER_INSTRUCT) {
      const ins = document.getElementById("pca-instruct");
      if (ins) ins.value = emo_compose_instruct(EMOTION.warmth, EMOTION.energy, EMOTION.intimacy);
    }
    // Modulate temperature from energy, strength from intimacy (only when the
    // emotion pad is the active driver — i.e. these track the pad).
    const baseTemp = 0.55 + EMOTION.energy * 0.35;       // 0.20 .. 0.90
    const baseStr  = 0.8 + EMOTION.intimacy * 1.4;        // 0.8 .. 2.2
    const tEl = document.getElementById("pca-temp");
    const sEl = document.getElementById("pca-strength");
    if (tEl) { tEl.value = Math.max(0.1, Math.min(1.5, baseTemp)); tEl.dispatchEvent(new Event("input")); }
    if (sEl) { sEl.value = Math.max(0.1, Math.min(3.0, baseStr));  sEl.dispatchEvent(new Event("input")); }

    // ── DSP acoustic modulation ──────────────────────────────────────────
    // The emotion pad shapes the actual acoustic character via the voice-
    // enhance DSP chain, not just the instruct text. These map naturally:
    //   warmth  → low-shelf EQ (chesty ↔ thin)
    //   energy  → brightness + formant (crisp/bright ↔ dull/flat)
    //   intimacy → airiness + breathiness (close/breathy ↔ neutral)
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    const setSlider = (id, val) => {
      const el = document.getElementById(id);
      if (el) { el.value = val; el.dispatchEvent(new Event("input")); }
    };
    setSlider("dsp-warmth",     clamp(EMOTION.warmth * 0.6, -1, 1));
    setSlider("dsp-brightness", clamp(EMOTION.energy * 0.5, -1, 1));
    setSlider("dsp-formant",    clamp(EMOTION.energy * 0.3, -1, 1));
    setSlider("dsp-airiness",   clamp(EMOTION.intimacy * 0.5, -1, 1));
    setSlider("dsp-breathiness", clamp(EMOTION.intimacy * 0.7, 0, 1));
    // Intimacy also shapes the pitch contour — high intimacy = gentle dip
    // (softer, more personal), low intimacy = flat (neutral).
    const contourEl = document.getElementById("prosody-contour");
    const contourDepthEl = document.getElementById("prosody-contour-depth");
    if (contourEl && EMOTION.intimacy > 0.4) {
      contourEl.value = "dip";
      contourEl.dispatchEvent(new Event("input"));
      if (contourDepthEl) {
        contourDepthEl.value = clamp(EMOTION.intimacy * 2.5, 0, 4).toFixed(1);
        contourDepthEl.dispatchEvent(new Event("input"));
      }
    }
  }

  function emo_init() {
    const pad = document.getElementById("emo-pad");
    if (!pad || pad.dataset.bound) return;
    pad.dataset.bound = "1";
    const move = (ev) => {
      const r = pad.getBoundingClientRect();
      const px = (ev.clientX - r.left) / r.width;    // 0..1
      const py = (ev.clientY - r.top) / r.height;    // 0..1
      EMOTION.energy = Math.max(-1, Math.min(1, px * 2 - 1));
      EMOTION.warmth = Math.max(-1, Math.min(1, 1 - py * 2));
      EMO_USER_INSTRUCT = false;   // pad is driving again
      emo_apply();
    };
    pad.addEventListener("pointerdown", (e) => { pad.setPointerCapture(e.pointerId); move(e); });
    pad.addEventListener("pointermove", (e) => { if (e.buttons) move(e); });

    // Presets
    document.querySelectorAll(".emo-preset").forEach((b) => {
      b.addEventListener("click", () => {
        EMOTION.warmth = parseFloat(b.dataset.w);
        EMOTION.energy = parseFloat(b.dataset.e);
        EMO_USER_INSTRUCT = false;
        emo_apply();
      });
    });
    // Intimacy slider
    const intl = document.getElementById("emo-intimacy");
    if (intl) intl.addEventListener("input", () => {
      EMOTION.intimacy = parseFloat(intl.value);
      EMO_USER_INSTRUCT = false;
      emo_apply();
    });
    // If the user edits the instruct, stop auto-composing.
    const ins = document.getElementById("pca-instruct");
    if (ins) ins.addEventListener("input", () => { EMO_USER_INSTRUCT = true; });
    emo_apply();
  }

  // ── Profile export / import (client-side, full cockpit state) ──────────
  function pca_profile_state() {
    const num = (id) => { const el = document.getElementById(id); return el ? el.value : ""; };
    return {
      schema: "audiocore.voiceprofile.v1",
      base_voice: num("ck-base"),
      weights: PCA.weights ? PCA.weights.slice() : [],
      instruct: num("pca-instruct"),
      text: num("pca-text"),
      emotion: { warmth: EMOTION.warmth, energy: EMOTION.energy, intimacy: EMOTION.intimacy },
      sampling: {
        temp: num("pca-temp"), topp: num("pca-topp"), topk: num("pca-topk"),
        maxtok: num("pca-maxtok"), seed: num("pca-seed"),
        pitch: num("pca-pitch"), speed: num("pca-speed"),
        strength: num("pca-strength"), constrain: num("pca-constrain"),
      },
      dsp: {
        warmth: num("dsp-warmth"), formant: num("dsp-formant"),
        brightness: num("dsp-brightness"), airiness: num("dsp-airiness"),
        breathiness: num("dsp-breathiness"),
      },
      prosody: {
        contour: num("prosody-contour"), contour_depth: num("prosody-contour-depth"),
        breath: num("prosody-breath"),
      },
    };
  }

  function pca_profile_export() {
    const st = pca_profile_state();
    const name = (prompt("Profile name:", "my_voice") || "voice").trim().replace(/[^a-z0-9_-]/gi, "_");
    const blob = new Blob([JSON.stringify(st, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name + ".voiceprofile";
    document.body.appendChild(a); a.click(); a.remove();
    toast("Exported " + a.download, "ok");
  }

  function pca_profile_import(file) {
    const r = new FileReader();
    r.onload = () => {
      try {
        const st = JSON.parse(r.result);
        if (st.schema !== "audiocore.voiceprofile.v1") throw new Error("not a voiceprofile");
        const set = (id, v) => { const el = document.getElementById(id); if (el && v !== undefined && v !== null) el.value = v; };
        // Weights
        if (Array.isArray(st.weights) && PCA.data) {
          PCA.weights = st.weights.slice(0, PCA.data.n_components);
          while (PCA.weights.length < PCA.data.n_components) PCA.weights.push(0);
        }
        set("ck-base", st.base_voice);
        set("pca-text", st.text);
        if (st.emotion) {
          EMOTION.warmth = st.emotion.warmth ?? 0.5;
          EMOTION.energy = st.emotion.energy ?? 0.5;
          EMOTION.intimacy = st.emotion.intimacy ?? 0.3;
          EMO_USER_INSTRUCT = true;  // respect the imported instruct below
        }
        set("pca-instruct", st.instruct);
        if (st.sampling) {
          set("pca-temp", st.sampling.temp); set("pca-topp", st.sampling.topp);
          set("pca-topk", st.sampling.topk); set("pca-maxtok", st.sampling.maxtok);
          set("pca-seed", st.sampling.seed); set("pca-pitch", st.sampling.pitch);
          set("pca-speed", st.sampling.speed); set("pca-strength", st.sampling.strength);
          set("pca-constrain", st.sampling.constrain);
        }
        if (st.dsp) {
          set("dsp-warmth", st.dsp.warmth); set("dsp-formant", st.dsp.formant);
          set("dsp-brightness", st.dsp.brightness); set("dsp-airiness", st.dsp.airiness);
          set("dsp-breathiness", st.dsp.breathiness);
        }
        if (st.prosody) {
          set("prosody-contour", st.prosody.contour); set("prosody-contour-depth", st.prosody.contour_depth);
          set("prosody-breath", st.prosody.breath);
        }
        // Refresh every dependent surface.
        sync_all_sliders();
        emo_apply();
        pca_refresh();
        toast("Imported profile" + (st.base_voice ? " · base " + st.base_voice : ""), "ok");
      } catch (e) { toast("Import failed: " + e.message, "err"); }
    };
    r.readAsText(file);
  }

  // ── Voice-shape snapshots (localStorage) ─────────────────────────────
  // A snapshot captures the full weight vector + constrain + strength +
  // instruct, so a sculpted voice can be recalled EXACTLY, or A/B-played
  // against the current sculptor WITHOUT disturbing it.
  const PCA_SNAP_KEY = "audiocore.pca.snaps.v1";
  function pca_snap_load() {
    try { return JSON.parse(localStorage.getItem(PCA_SNAP_KEY) || "[]"); } catch (_) { return []; }
  }
  function pca_snap_store(arr) {
    try { localStorage.setItem(PCA_SNAP_KEY, JSON.stringify(arr)); } catch (_) {}
  }
  function pca_snap_add() {
    if (!PCA.data) return;
    const name = (prompt("Name this voice shape:", "shape " + (pca_snap_load().length + 1)) || "").trim();
    if (!name) return;
    const arr = pca_snap_load();
    arr.push({
      name: name,
      weights: PCA.weights.slice(),
      constrain: PCA.constrain,
      strength: parseFloat((document.getElementById("pca-strength") || {}).value) || 1.0,
      instruct: ((document.getElementById("pca-instruct") || {}).value || "").trim(),
      ts: Date.now(),
    });
    pca_snap_store(arr);
    pca_snap_render();
    toast("Saved shape: " + name, "ok");
  }
  function pca_snap_render() {
    const host = document.getElementById("pca-snapshots");
    if (!host) return;
    const arr = pca_snap_load();
    if (!arr.length) {
      host.innerHTML = '<span class="pca-snap-empty">No saved shapes yet — sculpt a voice, then 📸 Save shape to recall or A/B it.</span>';
      return;
    }
    let html = "";
    for (let i = 0; i < arr.length; i++) {
      const s = arr[i];
      html += '<div class="pca-snap">' +
        '<button class="btn btn-ghost pca-snap-recall" data-i="' + i + '" title="Recall into sculptor">↺ ' + esc(s.name) + '</button>' +
        '<button class="btn btn-ghost pca-snap-ab" data-i="' + i + '" title="Play without changing the sculptor">▶ A/B</button>' +
        '<button class="btn btn-ghost pca-snap-del" data-i="' + i + '" title="Delete this shape">✕</button>' +
        '</div>';
    }
    host.innerHTML = html;
    host.querySelectorAll(".pca-snap-recall").forEach((b) =>
      b.addEventListener("click", () => pca_snap_recall(parseInt(b.dataset.i, 10))));
    host.querySelectorAll(".pca-snap-ab").forEach((b) =>
      b.addEventListener("click", () => pca_snap_play(parseInt(b.dataset.i, 10))));
    host.querySelectorAll(".pca-snap-del").forEach((b) =>
      b.addEventListener("click", () => {
        const a = pca_snap_load();
        a.splice(parseInt(b.dataset.i, 10), 1);
        pca_snap_store(a);
        pca_snap_render();
      }));
  }
  function pca_snap_recall(i) {
    const s = pca_snap_load()[i];
    if (!s || !PCA.data) return;
    const n = PCA.data.n_components;
    PCA.weights = s.weights.slice();
    while (PCA.weights.length < n) PCA.weights.push(0);
    PCA.weights.length = n;
    PCA.constrain = s.constrain || 0;
    const cEl = document.getElementById("pca-constrain");
    if (cEl) cEl.value = PCA.constrain;
    const cVal = document.getElementById("pca-constrain-val");
    if (cVal) cVal.textContent = PCA.constrain.toFixed(2);
    const sEl = document.getElementById("pca-strength");
    if (sEl) sEl.value = s.strength;
    const sVal = document.getElementById("pca-strength-val");
    if (sVal) sVal.textContent = (s.strength || 1).toFixed(1);
    const iEl = document.getElementById("pca-instruct");
    if (iEl) iEl.value = s.instruct || "";
    pca_refresh();
    toast("Recalled shape: " + s.name, "ok");
  }
  async function pca_snap_play(i) {
    const s = pca_snap_load()[i];
    if (!s) return;
    if (!ACTIVE_TTS) { toast("Load a qwen3_tts model first", "err"); return; }
    const emb = pca_reconstruct_from(s.weights, s.constrain || 0);
    if (!emb) { toast("Reconstruct failed", "err"); return; }
    const txtEl = document.getElementById("pca-text");
    const body = Object.assign({
      model: ACTIVE_TTS,
      input: ((txtEl && txtEl.value) || "").trim() || "Hello!",
      mode: "voice_clone",
      instruct: (s.instruct || "").trim(),
      speaker_embedding: f32_to_b64(emb),
      embedding_strength: s.strength || 1.0,
    }, pca_sampling_params(), pca_collect_dsp());
    const btn = document.querySelector('.pca-snap-ab[data-i="' + i + '"]');
    if (btn) { btn.disabled = true; btn.textContent = "…"; }
    try {
      const r = await fetch("/v1/audio/speech", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        toast(j.error || ("HTTP " + r.status), "err");
      } else {
        const blob = await r.blob();
        const player = document.getElementById("pca-player");
        if (player) {
          player.hidden = false;
          set_player(player, URL.createObjectURL(blob),
            "A/B · " + s.name + " · " + (blob.size / 1024).toFixed(1) + " KB");
        }
      }
    } catch (e) { toast("A/B play failed: " + e.message, "err"); }
    if (btn) { btn.disabled = false; btn.textContent = "▶ A/B"; }
  }

  // Randomly perturb every weight within ±(amt × range). The fastest way to
  // discover an inhuman-but-beautiful voice no human larynx can produce —
  // iterate mutate → play → save-shape on the keepers.
  function pca_mutate() {
    const amtEl = document.getElementById("pca-mutate-amt");
    const amt = amtEl ? (parseFloat(amtEl.value) || 0.15) : 0.15;
    for (let i = 0; i < PCA.weights.length; i++) {
      const rg = PCA.ranges[i] || 1;
      const delta = (Math.random() * 2 - 1) * amt * rg;
      PCA.weights[i] = Math.max(-rg, Math.min(rg, (PCA.weights[i] || 0) + delta));
    }
    pca_refresh();
  }

  // ── Draggable PCA explorer: drag the marker, reconstruct the embedding,
  //    and either play it (POST to /v1/audio/speech) or save it as a .voice. ──
  // ── PCA polygon (radar) renderer ─────────────────────────────────────
  // ── Push-pull polygon ───────────────────────────────────────────────────
  // 10 vertices on a decagon, but only N_PCA_PCS underlying PCA values.
  // Each PCA gets a PAIR of opposite vertices:
  //   vertex k       = positive end of PCA k  (drag outward → weight ↑)
  //   vertex k+N     = negative end of PCA k  (drag outward → weight ↓)
  // At most one vertex of each pair is off-center at any time.
  // This gives 10 grab points but only 5 numbers — easy to understand.
  const N_PCA_PCS = 5; // show top 5 PCAs on the polygon (91% of variance)

  // ── Sweep bar: isolate one PCA at ±max to listen and label it ──
  // Each PCA gets a label + "−" / "+" buttons. Clicking zeros everything
  // else and pushes that PCA to its extreme, so you can hear what it does.
  function pca_render_sweep_bar() {
    const host = document.getElementById("pca-sweep-bar");
    if (!host || !PCA.data) return;
    const d = PCA.data;
    const np = Math.min(N_PCA_PCS, d.n_components);
    let html = "";
    for (let k = 0; k < np; k++) {
      const ext = d.pc_extremes && d.pc_extremes[k];
      const varp = ((d.explained_variance_ratio[k] || 0) * 100).toFixed(0);
      const posName = String(ext ? ext.positive_end : '?').replace(/\.voice$/, '');
      const negName = String(ext ? ext.negative_end : '?').replace(/\.voice$/, '');
      html += '<div class="pca-sweep-group">';
      html += '<span class="pca-sweep-label">PC' + (k+1) + ' · ' + varp + '%</span>';
      html += '<div class="pca-sweep-btns">';
      html += '<button class="pca-sweep-btn" data-pca="' + k + '" data-sign="-1" type="button" title="−' + negName + '">− ' + esc(negName) + '</button>';
      html += '<button class="pca-sweep-btn" data-pca="' + k + '" data-sign="1" type="button" title="+' + posName + '">+ ' + esc(posName) + '</button>';
      html += '</div></div>';
    }
    host.innerHTML = html;
    host.querySelectorAll(".pca-sweep-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const k = parseInt(btn.dataset.pca, 10);
        const sign = parseInt(btn.dataset.sign, 10);
        const rg = PCA.ranges[k] || 1;
        PCA.weights.fill(0);
        PCA.weights[k] = sign * rg;
        pca_refresh();
        // visual feedback: highlight active button briefly
        host.querySelectorAll(".pca-sweep-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        setTimeout(() => btn.classList.remove("active"), 600);
      });
    });
  }

  function pca_render_polygon() {
    const host = document.getElementById("pca-polygon");
    if (!host || !PCA.data) return;
    const d = PCA.data;
    const np = Math.min(N_PCA_PCS, d.n_components);
    const nVerts = np * 2;  // 10 vertices for 5 PCAs
    const W = 600, H = 600, cx = W / 2, cy = H / 2, R = 220;
    const dirs = [];
    for (let i = 0; i < nVerts; i++) {
      const a = -Math.PI / 2 + i * (2 * Math.PI / nVerts);
      dirs.push({ x: Math.cos(a), y: Math.sin(a) });
    }

    // Each PCA k has weight PCA.weights[k]. The positive vertex (k) sits
    // at radius = max(0, w) * R; the negative vertex (k+np) sits at
    // radius = max(0, -w) * R. So the polygon stretches toward whichever
    // end you push.
    const pcaIdx = (vi) => vi < np ? vi : vi - np;      // vertex → PCA index
    const vertexPos = (vi) => {
      const k = pcaIdx(vi);
      const w = PCA.weights[k] || 0, rg = PCA.ranges[k] || 1;
      const isPos = vi < np;
      const signed = isPos ? Math.max(0, w / rg) : Math.max(0, -w / rg);
      const rr = signed * R;
      return { x: cx + dirs[vi].x * rr, y: cy + dirs[vi].y * rr };
    };

    // Build scaffold ONCE.
    let svgEl = host.querySelector("svg");
    const needRebuild = !svgEl ||
      svgEl.dataset.np !== String(np) ||
      svgEl.dataset.W !== String(W) || svgEl.dataset.R !== String(R);
    if (needRebuild) {
      let svg = '<svg width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + ' ' + H +
                '" style="max-width:100%;background:var(--bg-card);border-radius:8px;touch-action:none">';
      // grid rings
      for (let g = 1; g <= 3; g++) {
        let pts = "";
        for (let i = 0; i < nVerts; i++) {
          const x = cx + dirs[i].x * (R * g / 3), y = cy + dirs[i].y * (R * g / 3);
          pts += x.toFixed(1) + "," + y.toFixed(1) + " ";
        }
        svg += '<polygon points="' + pts.trim() + '" fill="none" stroke="var(--border)" stroke-width="1" opacity="0.4"/>';
      }
      // diameter lines: one line per PCA, passing through center
      for (let k = 0; k < np; k++) {
        const px = cx + dirs[k].x * R, py = cy + dirs[k].y * R;
        const nx = cx + dirs[k + np].x * R, ny = cy + dirs[k + np].y * R;
        svg += '<line x1="' + px.toFixed(1) + '" y1="' + py.toFixed(1) +
               '" x2="' + nx.toFixed(1) + '" y2="' + ny.toFixed(1) +
               '" stroke="var(--border)" stroke-width="1" opacity="0.5"/>';
      }
      // labels at both ends of each diameter — show the ACTUAL voice at each
      // extreme (from pc_extremes), not the repetitive semantic tag.
      for (let vi = 0; vi < nVerts; vi++) {
        const k = pcaIdx(vi);
        const ext = d.pc_extremes && d.pc_extremes[k];
        const varp = ((d.explained_variance_ratio[k] || 0) * 100).toFixed(0);
        const isPos = vi < np;
        // Voice name at this pole (strip .voice extension for readability)
        const rawName = isPos ? (ext ? ext.positive_end : '?') : (ext ? ext.negative_end : '?');
        const voiceName = String(rawName).replace(/\.voice$/, '');
        const lx = cx + dirs[vi].x * (R + 28), ly = cy + dirs[vi].y * (R + 28);
        svg += '<text x="' + lx.toFixed(1) + '" y="' + ly.toFixed(1) + '" text-anchor="middle" ' +
               'font-size="8" fill="var(--text)">' + esc(voiceName) + '</text>';
        // PC label + variance on the positive end
        if (isPos) {
          const vx = cx + dirs[vi].x * (R + 44), vy = cy + dirs[vi].y * (R + 44);
          svg += '<text x="' + vx.toFixed(1) + '" y="' + vy.toFixed(1) + '" text-anchor="middle" ' +
                 'font-size="8" fill="var(--text-dim)" opacity="0.7">PC' + (k+1) + ' · ' + varp + '%</text>';
        }
      }
      // center dot
      svg += '<circle cx="' + cx + '" cy="' + cy + '" r="3" fill="var(--border)"/>';
      // voice-shape polygon
      svg += '<polygon class="pca-shape" points="" fill="rgba(245,158,11,0.18)" stroke="#f59e0b" stroke-width="2"/>';
      // draggable vertices — each labeled with pos/neg tag
      for (let vi = 0; vi < nVerts; vi++) {
        const k = pcaIdx(vi);
        const isPos = vi < np;
        svg += '<circle class="pca-vert" data-vi="' + vi + '" data-pca="' + k +
               '" data-sign="' + (isPos ? '1' : '-1') +
               '" cx="0" cy="0" r="7" fill="#f59e0b" stroke="var(--bg-card)" stroke-width="2" style="cursor:grab"/>';
      }
      svg += "</svg>";
      host.innerHTML = svg;
      svgEl = host.querySelector("svg");
      svgEl.dataset.np = np; svgEl.dataset.W = W; svgEl.dataset.R = R;

      // Drag listeners — project onto the PCA's axis direction.
      // Positive vertex k: weight = proj / R * range (natural sign).
      // Negative vertex k+np: weight = -proj_on_neg_dir / R * range
      //                        = proj_on_pos_dir / R * range (same formula!)
      // Both reduce to: weight(k) = (proj onto dirs[k]) / R * range
      svgEl.querySelectorAll(".pca-vert").forEach((v) => {
        v.addEventListener("pointerdown", (e) => {
          e.preventDefault();
          const k = parseInt(v.dataset.pca, 10);
          const sign = parseInt(v.dataset.sign, 10);
          v.setPointerCapture(e.pointerId);
          v.style.cursor = "grabbing";
          const move = (ev) => {
            const pt = svgEl.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
            const p = pt.matrixTransform(svgEl.getScreenCTM().inverse());
            const dx = p.x - cx, dy = p.y - cy;
            // Project onto THIS vertex's direction, then map to signed weight.
            // sign=+1: outward on positive vertex → weight increases.
            // sign=-1: outward on negative vertex → weight decreases.
            const proj = dx * dirs[parseInt(v.dataset.vi,10)].x + dy * dirs[parseInt(v.dataset.vi,10)].y;
            const rg = PCA.ranges[k] || 1;
            let w = sign * (proj / R) * rg;
            w = Math.max(-rg, Math.min(rg, w));
            PCA.weights[k] = w;
            pca_refresh();
          };
          const up = () => {
            v.style.cursor = "grab";
            svgEl.removeEventListener("pointermove", move);
            svgEl.removeEventListener("pointerup", up);
          };
          svgEl.addEventListener("pointermove", move);
          svgEl.addEventListener("pointerup", up);
        });
      });
    }

    // In-place update.
    let poly = "";
    for (let i = 0; i < nVerts; i++) { const p = vertexPos(i); poly += p.x.toFixed(1) + "," + p.y.toFixed(1) + " "; }
    const shape = svgEl.querySelector(".pca-shape");
    if (shape) shape.setAttribute("points", poly.trim());
    const verts = svgEl.querySelectorAll(".pca-vert");
    verts.forEach((v, i) => {
      if (i >= nVerts) return;
      const p = vertexPos(i);
      v.setAttribute("cx", p.x.toFixed(1));
      v.setAttribute("cy", p.y.toFixed(1));
    });
  }

  // Keep the scatter marker + coords label in sync with the weight vector.
  function pca_set_scatter() {
    const svg = document.querySelector("#pca-plot svg");
    if (!svg) return;
    const m = svg.querySelector("#pca-marker");
    if (!m) return;
    const x = PCA.toPxX(PCA.weights[0] || 0);
    const y = PCA.toPxY(PCA.weights[1] || 0);
    const c = m.querySelector("circle");
    c.setAttribute("cx", x); c.setAttribute("cy", y);
    const ls = m.querySelectorAll("line");
    ls[0].setAttribute("x1", x - 9); ls[0].setAttribute("x2", x + 9);
    ls[0].setAttribute("y1", y);      ls[0].setAttribute("y2", y);
    ls[1].setAttribute("x1", x);      ls[1].setAttribute("x2", x);
    ls[1].setAttribute("y1", y - 9);  ls[1].setAttribute("y2", y + 9);
  }

  function pca_refresh() {
    pca_render_polygon();
    pca_set_scatter();
    pca_sync_axes();
    const c = document.getElementById("pca-coords");
    if (c && PCA.data) {
      const d = PCA.data;
      // Show all 5 polygon PCA values with the voice they lean toward
      const np = Math.min(N_PCA_PCS, d.n_components);
      let parts = [];
      for (let i = 0; i < np; i++) {
        const w = PCA.weights[i] || 0;
        const ext = d.pc_extremes && d.pc_extremes[i];
        const sign = w >= 0 ? '+' : '';
        const voiceName = String(w >= 0 ? (ext?.positive_end || '?') : (ext?.negative_end || '?'))
                          .replace(/\.voice$/, '');
        parts.push('PC' + (i+1) + ':' + sign + w.toFixed(2) + '→' + voiceName);
      }
      c.textContent = parts.join(' · ');
    }
  }

  // Load a base voice into the sculptor: the mean (weights=0) or a real voice
  // (weights = that voice's own PC vector). This is how you START from a single
  // voice instead of being forced to blend two.
  function pca_load_base(name) {
    if (!PCA.data) return;
    if (!name || name === "__mean__") {
      PCA.weights = new Array(PCA.data.n_components).fill(0);
    } else {
      const v = PCA.data.voices.find(x => x.name === name);
      if (v && v.pc) {
        PCA.weights = v.pc.slice(0, PCA.data.n_components);
        while (PCA.weights.length < PCA.data.n_components) PCA.weights.push(0);
      }
    }
    pca_refresh();
  }

  // Auto-run PCA on load so the cockpit polygon is live on the landing tab,
  // without requiring the user to click Analyze or open the Voices tab.
  let PCA_AUTO_DONE = false;
  async function auto_pca_analyze() {
    if (PCA_AUTO_DONE || !VOICES_CACHE.length) return;
    PCA_AUTO_DONE = true;
    try {
      const r = await fetch("/v1/voices/analyze", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ components: 10, clusters: 5, save_dirs: true }),
      });
      const data = await r.json();
      if (data.error) return;
      render_pca(data);
    } catch (e) { /* silent — manual analyze still available */ }
  }

  function pca_init_explorer() {
    const expl = document.getElementById("pca-explorer");
    if (!expl) return;
    expl.hidden = false;
    const svg  = document.querySelector("#pca-plot svg");
    const marker = svg && svg.querySelector("#pca-marker");

    const coordsEl    = document.getElementById("pca-coords");
    const strengthEl  = document.getElementById("pca-strength");
    const strengthVal = document.getElementById("pca-strength-val");
    const txtEl       = document.getElementById("pca-text");
    const instrEl     = document.getElementById("pca-instruct");

    // Initialise the per-PC weight vector + per-axis ranges from the data.
    const d = PCA.data;
    PCA.weights = new Array(d.n_components).fill(0);
    PCA.ranges = [];
    for (let i = 0; i < d.n_components; i++) {
      let m = 0;
      for (const v of d.voices) m = Math.max(m, Math.abs(v.pc[i] || 0));
      PCA.ranges.push(m < 1e-6 ? 1 : m);
    }

    // Populate the base-voice selector (start from a default OR any real voice).
    const baseSel = document.getElementById("ck-base");
    if (baseSel && !baseSel.dataset.filled) {
      baseSel.dataset.filled = "1";
      baseSel.innerHTML = '<option value="__mean__">✨ Default (mean voice)</option>';
      for (const v of d.voices) {
        const o = document.createElement("option");
        o.value = v.name;
        o.textContent = v.name.replace(/\.voice$/, "");
        baseSel.appendChild(o);
      }
      baseSel.addEventListener("change", () => pca_load_base(baseSel.value));
    }

    if (marker) {
      const circle = marker.querySelector("circle");
      const lines  = marker.querySelectorAll("line");

      function markerXY() {
        return { x: parseFloat(circle.getAttribute("cx")),
                 y: parseFloat(circle.getAttribute("cy")) };
      }
      function clientToSvg(e) {
        const pt = svg.createSVGPoint();
        pt.x = e.clientX; pt.y = e.clientY;
        return pt.matrixTransform(svg.getScreenCTM().inverse());
      }
      function moveMarker(x, y) {
        x = Math.max(PCA.PAD, Math.min(PCA.W - PCA.PAD, x));
        y = Math.max(PCA.PAD, Math.min(PCA.H - PCA.PAD, y));
        circle.setAttribute("cx", x); circle.setAttribute("cy", y);
        lines[0].setAttribute("x1", x - 9); lines[0].setAttribute("x2", x + 9);
        lines[0].setAttribute("y1", y);      lines[0].setAttribute("y2", y);
        lines[1].setAttribute("x1", x);      lines[1].setAttribute("x2", x);
        lines[1].setAttribute("y1", y - 9);  lines[1].setAttribute("y2", y + 9);
      }

      // Dragging the 2D scatter marker adjusts PC1/PC2 and syncs the polygon.
      let dragging = false;
      marker.addEventListener("pointerdown", (e) => {
        dragging = true; marker.style.cursor = "grabbing";
        marker.setPointerCapture(e.pointerId); e.preventDefault();
      });
      marker.addEventListener("pointerup", () => { dragging = false; marker.style.cursor = "grab"; });
      svg.addEventListener("pointermove", (e) => {
        if (!dragging) return;
        const p = clientToSvg(e); moveMarker(p.x, p.y);
        PCA.weights[0] = PCA.toPcX(p.x);
        PCA.weights[1] = PCA.toPcY(p.y);
        pca_refresh();
      });
      svg.addEventListener("pointerdown", (e) => {
        if (e.target === marker || marker.contains(e.target)) return;
        const p = clientToSvg(e); moveMarker(p.x, p.y);
        PCA.weights[0] = PCA.toPcX(p.x);
        PCA.weights[1] = PCA.toPcY(p.y);
        pca_refresh();
      });
    }

    strengthEl.addEventListener("input", () => {
      strengthVal.textContent = parseFloat(strengthEl.value).toFixed(1);
    });

    const constrainEl  = document.getElementById("pca-constrain");
    const constrainVal = document.getElementById("pca-constrain-val");
    constrainEl.addEventListener("input", () => {
      PCA.constrain = parseFloat(constrainEl.value) || 0;
      constrainVal.textContent = PCA.constrain.toFixed(2);
    });

    document.getElementById("pca-reset").addEventListener("click", () => {
      PCA.weights.fill(0);
      pca_refresh();
    });

    document.getElementById("pca-play").addEventListener("click", async () => {
      if (!PCA.data) return;
      if (!ACTIVE_TTS) { toast("Load a qwen3_tts model first", "err"); return; }
      const emb = pca_reconstruct();
      if (!emb) { toast("Analyze voices first", "err"); return; }
      const body = {
        model: ACTIVE_TTS,
        input: (txtEl.value || "").trim() || "Hello!",
        mode: "voice_clone",
        instruct: (instrEl.value || "").trim(),
        speaker_embedding: f32_to_b64(emb),
        embedding_strength: parseFloat(strengthEl.value) || 1.0,
      };
      Object.assign(body, pca_sampling_params(), pca_collect_dsp());
      const btn = document.getElementById("pca-play");
      btn.disabled = true; btn.textContent = "Synthesizing…";
      try {
        const r = await fetch("/v1/audio/speech", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!r.ok) {
          const j = await r.json().catch(() => ({}));
          throw new Error(j.error || ("TTS failed " + r.status));
        }
        const blob = await r.blob();
        const player = document.getElementById("pca-player");
        player.hidden = false;
        set_player(player, URL.createObjectURL(blob),
                   "PCA voice · " + (blob.size / 1024).toFixed(1) + " KB");
      } catch (err) { toast("Play failed: " + err.message, "err"); }
      btn.disabled = false; btn.textContent = "▶️ Play";
    });

    document.getElementById("pca-save").addEventListener("click", async () => {
      if (!PCA.data) return;
      const emb = pca_reconstruct();
      if (!emb) { toast("Analyze voices first", "err"); return; }
      const blob = pca_build_voice_blob(emb);
      const name = (prompt("Save reconstructed voice as:", "pca_voice.voice") || "").trim();
      if (!name || !name.toLowerCase().endsWith(".voice")) return;
      const form = new FormData();
      form.append("file", blob, name);
      try {
        const r = await fetch("/v1/voices/upload", { method: "POST", body: form });
        const j = await r.json().catch(() => ({}));
        if (j.ok) { toast("Saved " + name, "ok"); load_voices(); }
        else toast(j.error || "Save failed", "err");
      } catch (e) { toast("Save failed: " + e.message, "err"); }
    });

    pca_render_axes();
    pca_render_sweep_bar();
    pca_snap_render();
    const snapAdd = document.getElementById("pca-snap-add");
    if (snapAdd) snapAdd.addEventListener("click", pca_snap_add);
    const mutBtn = document.getElementById("pca-mutate");
    if (mutBtn) mutBtn.addEventListener("click", pca_mutate);
    // Live value labels + track fill for the sculptor's sampling sliders.
    [["pca-temp", "pca-temp-val", 2], ["pca-topp", "pca-topp-val", 2],
     ["pca-pitch", "pca-pitch-val", 1], ["pca-speed", "pca-speed-val", 2],
     ["pca-mutate-amt", "pca-mutate-val", 0, "pct"],
     ["dsp-warmth", "dsp-warmth-val", 2], ["dsp-formant", "dsp-formant-val", 2],
     ["dsp-brightness", "dsp-brightness-val", 2], ["dsp-airiness", "dsp-airiness-val", 2],
     ["dsp-breathiness", "dsp-breathiness-val", 2],
     ["prosody-contour-depth", "prosody-contour-depth-val", 1],
     ["prosody-breath", "prosody-breath-val", 2]].forEach((spec) => {
      const id = spec[0], vid = spec[1], dp = spec[2], kind = spec[3];
      const el = document.getElementById(id), vEl = document.getElementById(vid);
      if (!el || !vEl) return;
      const upd = () => {
        vEl.textContent = kind === "pct"
          ? Math.round(parseFloat(el.value) * 100) + "%"
          : parseFloat(el.value).toFixed(dp);
        sync_slider_fill(el);
      };
      el.addEventListener("input", upd);
      upd();
    });

    // Emotion matrix + profile export/import.
    emo_init();
    const exBtn = document.getElementById("pca-profile-export");
    if (exBtn) exBtn.addEventListener("click", pca_profile_export);
    const imBtn = document.getElementById("pca-profile-import");
    const imFile = document.getElementById("pca-profile-file");
    if (imBtn && imFile) {
      imBtn.addEventListener("click", () => imFile.click());
      imFile.addEventListener("change", () => {
        if (imFile.files && imFile.files[0]) pca_profile_import(imFile.files[0]);
        imFile.value = "";
      });
    }

    pca_refresh();
  }

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
      const ts = c.mtime ? new Date(c.mtime * 1000).toLocaleString() : "";
      const enc = encodeURIComponent(c.name);
      const rating = c.rating || 0;
      let starsHtml = '<div class="clip-stars" data-clip-stars="' + esc(c.name) + '">';
      for (let s = 1; s <= 5; s++)
        starsHtml += '<span class="star' + (s <= rating ? " on" : "") + '" data-rate="' + s + '">★</span>';
      starsHtml += "</div>";
      const scoreTag = (c.naturalness != null && c.naturalness >= 0)
        ? '<span class="tag tag-score" title="VLM naturalness / intelligibility">🎯 N' + c.naturalness + ' · I' + (c.intelligibility != null ? c.intelligibility : "?") + '</span>'
        : "";
      const transcriptHtml = c.transcript
        ? '<div class="clip-transcript" title="' + esc(c.transcript) + '">“' + esc((c.transcript || "").trim().slice(0, 70)) + '”</div>'
        : "";
      const cmpOn = COMPARE.indexOf(c.name) >= 0;
      el.innerHTML =
        '<audio controls preload="metadata" src="/v1/clips/raw/' + enc + '"></audio>' +
        '<div class="clip-info">' +
          '<div class="clip-name" title="' + esc(c.name) + '">' + esc(c.name) + '</div>' +
          '<div class="clip-meta">' +
            (c.duration ? '<span class="tag">' + fmt_dur(c.duration) + '</span>' : '') +
            '<span class="tag">' + fmt_size(c.size) + '</span>' +
            (ts ? '<span class="tag">📅 ' + ts + '</span>' : '') +
            scoreTag +
          '</div>' +
          starsHtml +
          transcriptHtml +
        '</div>' +
        '<div class="clip-actions">' +
          '<button class="btn btn-ghost" data-score-clip="' + esc(c.name) + '" title="VLM score">📊</button>' +
          '<button class="btn btn-ghost' + (cmpOn ? " on" : "") + '" data-cmp-clip="' + esc(c.name) + '" title="Add to A/B compare">⇄</button>' +
          '<button class="btn btn-danger" data-del-clip="' + esc(c.name) + '">Delete</button>' +
        '</div>';
      list.appendChild(el);
    }

    // Star rating.
    $$("[data-clip-stars]").forEach((grp) => {
      grp.querySelectorAll(".star").forEach((sp) => {
        sp.addEventListener("click", async () => {
          const name = grp.dataset.clipStars;
          const rate = parseInt(sp.dataset.rate, 10);
          try {
            const r = await fetch("/v1/clips/rate", {
              method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ name, rating: rate }),
            });
            const j = await r.json();
            if (j.ok) {
              grp.querySelectorAll(".star").forEach((s, i) => s.classList.toggle("on", i < rate));
              const clip = CLIPS_CACHE.find((x) => x.name === name);
              if (clip) clip.rating = rate;
            } else toast(j.error || "Rating failed", "err");
          } catch (_) { toast("Rating failed", "err"); }
        });
      });
    });

    // VLM score.
    $$("[data-score-clip]").forEach((b) => b.addEventListener("click", async (e) => {
      const name = e.currentTarget.dataset.scoreClip;
      const btn = e.currentTarget;
      btn.disabled = true; btn.textContent = "⏳";
      try {
        const r = await fetch("/v1/clips/score", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name }),
        });
        const j = await r.json();
        if (j.ok) {
          toast("🎯 N" + j.naturalness + " · I" + j.intelligibility + " — " + name, "ok");
          load_clips();
        } else toast(j.error || (j.detail || "Score failed"), "err");
      } catch (_) { toast("Score failed", "err"); }
      btn.disabled = false; btn.textContent = "📊";
    }));

    // A/B compare toggle (max 2).
    $$("[data-cmp-clip]").forEach((b) => b.addEventListener("click", (e) => {
      const name = e.currentTarget.dataset.cmpClip;
      const idx = COMPARE.indexOf(name);
      if (idx >= 0) { COMPARE.splice(idx, 1); }
      else {
        if (COMPARE.length >= 2) COMPARE.shift();
        COMPARE.push(name);
      }
      render_clips(CLIPS_CACHE.filter((c) => clip_matches(c, $("#clip-search").value)));
      render_compare();
    }));

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

  // A/B compare tray: play two clips back-to-back to pick the winner.
  let COMPARE = [];
  function render_compare() {
    const host = $("#clips-compare");
    if (!host) return;
    if (!COMPARE.length) { host.innerHTML = ""; return; }
    let html = '<div class="compare-card"><div class="compare-head"><b>A/B compare</b>' +
               '<button class="btn btn-ghost" id="cmp-clear">Clear</button></div>';
    COMPARE.forEach((name, i) => {
      html += '<div class="compare-row"><span class="compare-label">' + esc(i === 0 ? "A" : "B") + "</span>" +
              '<span class="compare-name">' + esc(name) + "</span>" +
              '<audio controls src="/v1/clips/raw/' + encodeURIComponent(name) + '"></audio></div>';
    });
    if (COMPARE.length === 2) {
      html += '<p class="hint">Play A then B. Star-rate each above to lock in the winner.</p>';
    } else {
      html += '<p class="hint">Add one more clip (⇄) to compare.</p>';
    }
    html += "</div>";
    host.innerHTML = html;
    const clr = $("#cmp-clear");
    if (clr) clr.addEventListener("click", () => {
      COMPARE = [];
      render_clips(CLIPS_CACHE.filter((c) => clip_matches(c, $("#clip-search").value)));
      render_compare();
    });
  }
  function clip_matches(c, q) {
    q = (q || "").trim().toLowerCase();
    return !q || (c.name || "").toLowerCase().indexOf(q) >= 0;
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

  // ── Generate PCA sample clips ────────────────────────────────────────
  // For each PC: synthesise speech at +max and −max so you can HEAR what
  // each axis does. Uses a fixed seed + fixed text so the only variable is
  // the PCA weight. Clips are saved to the clips library.
  $("#clip-pca-samples")?.addEventListener("click", async () => {
    const btn = $("#clip-pca-samples");
    if (!PCA.data) { toast("Analyze voices first (Voice Design tab)", "err"); return; }
    if (!ACTIVE_TTS) { toast("Load a qwen3_tts model first", "err"); return; }
    const d = PCA.data;
    const nc = d.n_components;
    const sampleText = "Hey, I'm so happy to meet you. Tell me everything.";
    const seed = 12345;
    const statusEl = $("#clip-status");
    btn.disabled = true;
    let done = 0;
    const total = nc * 2;
    for (let k = 0; k < nc; k++) {
      const ext = d.pc_extremes && d.pc_extremes[k];
      const varp = ((d.explained_variance_ratio[k] || 0) * 100).toFixed(0);
      for (const sign of [1, -1]) {
        const w = new Array(nc).fill(0);
        w[k] = sign * (PCA.ranges[k] || 1);
        const emb = pca_reconstruct_from(w, 0);
        if (!emb) continue;
        const voiceLabel = sign > 0
          ? String(ext?.positive_end || '?').replace(/\.voice$/, '')
          : String(ext?.negative_end || '?').replace(/\.voice$/, '');
        const clipName = 'PCA_PC' + (k+1) + (sign > 0 ? '+' : '-') + '_'
                       + voiceLabel.replace(/[^a-zA-Z0-9_]/g, '_').slice(0, 20)
                       + '_v' + varp + '.wav';
        if (statusEl) status(statusEl,
          `Generating ${done+1}/${total}: PC${k+1}${sign>0?'+':'−'} ${voiceLabel}`, "loading");
        try {
          const body = {
            model: ACTIVE_TTS,
            input: sampleText,
            mode: "voice_clone",
            speaker_embedding: f32_to_b64(emb),
            embedding_strength: 1.0,
            seed: seed,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            max_tokens: 200,
          };
          const r = await fetch("/v1/audio/speech", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          });
          if (!r.ok) { console.error("TTS failed for " + clipName); continue; }
          const blob = await r.blob();
          const form = new FormData();
          form.append("file", blob, clipName);
          const ur = await fetch("/v1/clips/upload", { method: "POST", body: form });
          const uj = await ur.json().catch(() => ({}));
          if (!uj.ok) console.error("Upload failed for " + clipName);
        } catch (e) { console.error("Error generating " + clipName, e); }
        done++;
      }
    }
    btn.disabled = false;
    if (statusEl) status(statusEl, `Generated ${done} PCA sample clips`, "ok");
    load_clips();
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

  // ── Model Manager (load/unload) ───────────────────────────────────
  function render_model_manager() {
    const list = $("#model-manager-list");
    if (!list) return;
    list.innerHTML = "";
    for (const m of MODELS) {
      const isLoaded = m.loaded !== false;
      const isAce = (m.family || "").indexOf("ace") >= 0;
      const el = document.createElement("div");
      el.style.cssText = "display:flex;align-items:center;gap:1em;padding:0.75em 1em;border:1px solid var(--border);border-radius:8px";
      el.innerHTML =
        '<span style="font-size:1.3em">' + (isAce ? "🎵" : "🗣️") + "</span>" +
        '<div style="flex:1">' +
          '<div style="font-weight:600">' + esc(m.id) + "</div>" +
          '<div class="hint">' + esc(m.family || "") + " · " +
            (isLoaded
              ? '<span style="color:var(--green)">● loaded</span>'
              : '<span style="color:var(--red)">○ unloaded</span>') +
          "</div>" +
        "</div>" +
        (isLoaded
          ? '<button type="button" class="btn btn-danger" data-unload="' + esc(m.id) + '">Unload</button>'
          : '<button type="button" class="btn" data-load="' + esc(m.id) + '">Load</button>');
      list.appendChild(el);
    }
    // Wire up buttons
    $$("[data-unload]").forEach((b) => b.addEventListener("click", async () => {
      const id = b.dataset.unload;
      b.disabled = true;
      b.textContent = "Unloading…";
      try {
        await fetch("/v1/models/unload", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id }),
        });
        toast("Unloaded " + id, "ok");
        await load_models();
      } catch (e) {
        toast("Unload failed: " + e.message, "err");
        b.disabled = false;
        b.textContent = "Unload";
      }
    }));
    $$("[data-load]").forEach((b) => b.addEventListener("click", async () => {
      const id = b.dataset.load;
      b.disabled = true;
      b.textContent = "Loading…";
      try {
        const r = await fetch("/v1/models/load", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id }),
        });
        const j = await r.json();
        if (r.ok) {
          toast("Loaded " + id, "ok");
          await load_models();
        } else if (j.fetchable) {
          // Server says files are missing + auto-download is available.
          // Offer the fetch UI instead of dumping a stack trace.
          const fam = (j.fetch_payload && j.fetch_payload.family) || "?";
          b.textContent = "Fetch & Load";
          b.disabled = false;
          b.classList.add("btn-warn");
          b.dataset.fetchFamily = fam;
          b.dataset.fetchModelId = id;
          b.removeEventListener("click", arguments.callee, false);
          b.addEventListener("click", () => open_fetch_dialog(id, fam), { once: true });
          toast("Files missing for " + id + " (family: " + fam + "). Click Fetch & Load to download.", "warn", 8000);
        } else {
          toast("Load failed: " + (j.error || r.status), "err");
          b.disabled = false;
          b.textContent = "Load";
        }
      } catch (e) {
        toast("Load failed: " + e.message, "err");
        b.disabled = false;
        b.textContent = "Load";
      }
    }));
  }

  // ── Auto-download UI ──────────────────────────────────────────────
  // Triggered when /v1/models/load returns fetchable:true. Prompts the
  // user to pick a manifest variant, kicks off POST /v1/models/fetch, and
  // polls GET /v1/models/fetch/status until the job finishes, then retries
  // the load. The status widget at the top of the Models tab renders
  // active downloads for all in-flight jobs (see render_fetch_status).
  async function open_fetch_dialog(modelId, family) {
    // Resolve a sensible default variant from the model id. The model id
    // (from server.json) often matches a manifest variant key, but not
    // always — when it doesn't, we leave the field blank for the user.
    const guessVariant = (() => {
      const id = (modelId || "").toLowerCase();
      // ace-step-turbo → ace-step-1.5-turbo is the common default
      if (id === "ace-step-turbo") return "ace-step-1.5-turbo";
      if (id === "ace-step-base")  return "ace-step-1.5-base";
      if (id === "moss-tts")       return "moss-tts-q4-k-m";
      if (id === "moss-sfx")       return "moss-sfx-f16";
      if (id === "moss-sfx-v2")    return "default";
      if (id === "moss-voicegen")  return "moss-voicegen-q8_0";
      // Qwen3-TTS proxy models have variants in HF format; leave blank.
      if (id.indexOf("qwen3") >= 0) return "";
      return modelId;
    })();
    const variant = prompt(
      "Download model files for: " + modelId + "\n" +
      "Family: " + family + "\n\n" +
      "Enter the manifest variant name (see models/manifest.json):",
      guessVariant
    );
    if (!variant) return;
    try {
      const r = await fetch("/v1/models/fetch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ family, variant }),
      });
      const j = await r.json();
      if (r.status === 409) {
        toast("Already downloading: " + j.id, "warn");
      } else if (!r.ok) {
        toast("Fetch failed: " + (j.error || r.status), "err");
        return;
      } else {
        toast("Download started: " + j.id + " — poll status below.", "ok", 6000);
      }
      // Start polling status until the job finishes, then retry the load.
      await poll_fetch_then_load(j.id || (family + "/" + variant), modelId);
    } catch (e) {
      toast("Fetch error: " + e.message, "err");
    }
  }

  // Poll /v1/models/fetch/status every 2s until `jobId` leaves `active`.
  // On success, retries /v1/models/load for `modelId`. On failure, surfaces
  // the log tail so the user can see why it broke.
  async function poll_fetch_then_load(jobId, modelId) {
    const deadline = Date.now() + 30 * 60 * 1000;  // 30 min cap
    while (Date.now() < deadline) {
      await new Promise(r => setTimeout(r, 2000));
      let st;
      try {
        st = await (await fetch("/v1/models/fetch/status")).json();
      } catch (e) { continue; }
      render_fetch_status(st);
      const done = (st.recent || []).find(j => j.id === jobId && j.finished);
      if (done) {
        if (done.exit_code === 0) {
          toast("Download complete — loading " + modelId, "ok");
          // Re-hit /v1/models/load — should succeed now.
          const lr = await fetch("/v1/models/load", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ id: modelId }),
          });
          if (lr.ok) {
            toast("Loaded " + modelId, "ok");
            await load_models();
          } else {
            const lj = await lr.json().catch(() => ({}));
            toast("Load after fetch failed: " + (lj.error || lr.status), "err");
          }
        } else {
          toast("Download failed (exit " + done.exit_code + ") — see server log.", "err", 10000);
          console.error("Fetch log tail:", done.log_tail);
        }
        return;
      }
    }
    toast("Download timed out after 30 min", "err");
  }

  // Render the active/recent downloads widget at the top of the Models tab.
  // No-op if the widget element isn't present. Called on a 2s interval while
  // the Models tab is visible so progress updates without user interaction.
  function render_fetch_status(st) {
    const box = $("#model-fetch-status");
    if (!box) return;
    const active = st.active || [];
    const recent = st.recent || [];
    if (!active.length && !recent.length) { box.style.display = "none"; return; }
    box.style.display = "";
    let html = "";
    for (const j of active) {
      const secs = Math.round((Date.now() / 1000) - j.started);
      html += '<div class="fetch-row fetch-active">' +
        '<span class="fetch-spinner"></span>' +
        '<strong>' + esc(j.id) + '</strong> · downloading… (' + secs + 's)' +
        '</div>';
    }
    for (const j of recent.slice(-5)) {
      const ok = j.exit_code === 0;
      html += '<div class="fetch-row ' + (ok ? "fetch-ok" : "fetch-fail") + '">' +
        '<code>' + (ok ? "✓" : "✗") + '</code> ' +
        '<strong>' + esc(j.id) + '</strong> · ' +
        (ok ? "finished" : ("failed (exit " + j.exit_code + ")")) +
        '</div>';
    }
    box.innerHTML = html;
  }

  // Periodic refresh of the downloads widget while the Models tab is open.
  let _fetch_poll_handle = null;
  function start_fetch_polling() {
    if (_fetch_poll_handle) return;
    _fetch_poll_handle = setInterval(async () => {
      const modelsTab = $("#tab-models");
      if (!modelsTab || !modelsTab.classList.contains("active")) return;
      try {
        const st = await (await fetch("/v1/models/fetch/status")).json();
        render_fetch_status(st);
      } catch (e) { /* server down — silent */ }
    }, 3000);
  }
  start_fetch_polling();

  const _model_refresh = $("#model-refresh");
  if (_model_refresh) _model_refresh.addEventListener("click", load_models);

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
        sfx:    "#sfx-btn",
      };
      const sel = map[active];
      if (sel) { e.preventDefault(); $(sel)?.click(); }
      return;
    }

    if (typing) return;

    // 1–8: tabs
    if (/^[1-8]$/.test(e.key)) {
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
  // Server-side uptime + version, polled from /health every 30 s. Filled
  // in async; until the first response resolves the footer just shows the
  // client session uptime (which is still useful).
  let SERVER_INFO = null;  // {version, uptime_s, models_total, models_loaded}
  async function load_health() {
    try {
      const r = await fetch("/health");
      if (!r.ok) throw new Error(r.status);
      SERVER_INFO = await r.json();
      // If the backend advertises zero loaded models, the engine dots are
      // meaningless — let the user know via the badge.
      if (SERVER_INFO.models_loaded === 0 && SERVER_INFO.models_total === 0) {
        const badge = $("#model-badge");
        badge.textContent = "no models configured";
        badge.className = "badge err";
      }
    } catch (_) {
      SERVER_INFO = null;
    }
    render_footer();
  }
  function fmt_uptime(s) {
    s = Math.max(0, Math.floor(s));
    const d = Math.floor(s / 86400); s %= 86400;
    const h = Math.floor(s / 3600);  s %= 3600;
    const m = Math.floor(s / 60);    const sec = s % 60;
    if (d > 0) return d + "d " + h + "h " + m + "m";
    if (h > 0) return h + "h " + m + "m " + sec + "s";
    return m + "m " + sec + "s";
  }
  function render_footer() {
    const el = $("#footer-uptime");
    if (!el) return;
    const session_s = Math.floor((Date.now() - t0) / 1000);
    const parts = [];
    if (SERVER_INFO) {
      parts.push("v" + (SERVER_INFO.version || "?"));
      parts.push("server " + fmt_uptime(SERVER_INFO.uptime_s || 0));
      parts.push(SERVER_INFO.models_loaded + "/" + SERVER_INFO.models_total + " models");
    } else {
      parts.push("backend offline");
    }
    parts.push("session " + fmt_uptime(session_s));
    el.textContent = parts.join(" · ");
  }
  function tick_uptime() { render_footer(); }
  setInterval(tick_uptime, 1000);
  tick_uptime();
  load_health();
  setInterval(load_health, 30000);

  // ── Init ───────────────────────────────────────────────────────────
  sync_all_sliders();
  Promise.all([load_models()]).then(async () => {
    // The cockpit polygon is the hero of the landing tab — load the voice
    // library and run PCA immediately so it's live on arrival, not buried.
    try { await load_voices(); } catch (e) {}
    auto_pca_analyze();
  });
})();
