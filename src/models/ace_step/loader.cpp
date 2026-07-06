// loader.cpp — ACE-Step weight loading + family registration.
//
// Four GGUF files in one model directory:
//
//   acestep-v15-turbo-Q8_0.gguf   DiT (turbo / sft / xl-* variants)
//   acestep-5Hz-lm-1.7B-Q8_0.gguf 5Hz music-code LM (Qwen3)
//   Qwen3-Embedding-0.6B-Q8_0.gguf Text encoder (Qwen3)
//   vae-BF16.gguf                  Audio VAE decoder
//
// The two Qwen3 transformers go through the unified qwen3::Runner — there is
// no other Qwen3 path in audiocore. They MUST be in llama.cpp tensor-name
// layout; we check up-front and refuse (with a pointer to the converter)
// otherwise. The DiT and VAE are bound into ext_ctx_ via the same GgufReader
// zero-copy path MOSS uses.

#include "audiocore/models/ace_step/family.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <sys/stat.h>

#include "ggml.h"

#include "audiocore/framework/io/gguf_reader.h"
#include "audiocore/framework/io/weight_loader.h"
#include "audiocore/framework/runtime/registry.h"

namespace audiocore::acestep {

namespace fs = std::filesystem;

namespace {

// Find a file in `dir` whose name contains `pattern`. Returns empty path if
// none match. Used to locate the 4 component GGUFs without hard-coding the
// exact filenames (turbo vs sft, 1.7B vs 4B, etc.).
//
// If `prefer` is non-empty, among all matching files the one whose
// "variant tag" (the `-Q8_0`/`-BF16`-stripped segment after `pattern-`)
// equals `prefer` exactly is returned. This matters because a naive
// substring match would pick `acestep-v15-xl-turbo` when the caller asked
// for `turbo`: both filenames contain the substring "turbo". By parsing
// the variant tag out of the filename we get exact, unambiguous selection.
//
// Filenames follow the upstream convention:
//   acestep-v15-{variant}-{quant}.gguf      (DiT)
//   acestep-5Hz-lm-{variant}-{quant}.gguf   (LM)
// So for pattern="acestep-v15" the variant tag of
// "acestep-v15-xl-turbo-Q8_0.gguf" is "xl-turbo", and only a prefer of
// "xl-turbo" (not "turbo") will select it.
std::string find_gguf(const std::string& dir, const std::string& pattern,
                      const std::string& prefer = "") {
    if (dir.empty()) return {};
    // Extract the variant tag from a filename: the substring between the
    // pattern+separator and the quant suffix (-Q8_0, -BF16, -F16, …).
    auto variant_tag = [&pattern](const std::string& name) -> std::string {
        size_t p = name.find(pattern);
        if (p == std::string::npos) return {};
        size_t start = p + pattern.size();
        // Skip the separator (usually '-').
        while (start < name.size() && name[start] == '-') ++start;
        // Find the trailing quant suffix. We look for the last segment
        // matching -Q[0-9]_[0-9], -BF16, -F16, -F32 after a '-'.
        size_t end = name.size();
        // Strip ".gguf"
        if (end >= 5 && name.compare(end - 5, 5, ".gguf") == 0) end -= 5;
        // Now walk back: find the last '-' that starts a quant token.
        // Quant tokens: Q4_0, Q4_1, Q8_0, BF16, F16, F32, Q5_K_M, Q6_K, …
        for (size_t i = end; i > start; ) {
            --i;
            if (name[i] == '-') {
                std::string seg = name.substr(i + 1, end - (i + 1));
                if (!seg.empty() &&
                    (seg[0] == 'Q' || seg == "BF16" || seg == "F16" ||
                     seg == "F32" || seg == "BF16" || seg == "fp16")) {
                    end = i;
                } else {
                    break;
                }
            }
        }
        return name.substr(start, end - start);
    };

    std::string first_match;
    std::string preferred_match;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        const std::string name = e.path().filename().string();
        if (name.size() < 5 || name.compare(name.size() - 5, 5, ".gguf") != 0) continue;
        if (name.find(pattern) == std::string::npos) continue;
        if (first_match.empty()) first_match = e.path().string();
        if (!prefer.empty()) {
            const std::string tag = variant_tag(name);
            if (tag == prefer) {
                preferred_match = e.path().string();
            }
        }
    }
    return !preferred_match.empty() ? preferred_match : first_match;
}

bool file_exists(const std::string& p) {
    struct stat st {};
    return p.empty() ? false : ::stat(p.c_str(), &st) == 0;
}

}  // namespace

bool AceStepSession::check_llamacpp_layout(const GgufReader& r,
                                           const char* role,
                                           std::string* error) {
    // libllama-loaded Qwen3 GGUFs always have a `token_embd.weight` tensor.
    // HF-style (model.embed_tokens.weight) means the converter hasn't run.
    bool has_token_embd = false;
    for (const TensorStorage& t : r.tensors()) {
        if (t.name == "token_embd.weight") { has_token_embd = true; break; }
    }
    if (!has_token_embd) {
        if (error) {
            *error = std::string("ACE-Step ") + role + " GGUF is not in "
                     "llama.cpp tensor-name layout. libllama can't load it. "
                     "Run `python tools/convert_acestep_gguf.py <file>` "
                     "once to rewrite HF names (model.embed_tokens.weight, "
                     "model.layers.N.*) to llama.cpp names (token_embd.weight, "
                     "blk.N.*).";
        }
        return false;
    }
    return true;
}

bool AceStepSession::bind_dit_and_vae(const GgufReader& dit,
                                       const GgufReader& vae,
                                       std::string* error) {
    // Count dit.* / vae.* tensors across both files to size ext_ctx_.
    size_t n = 0;
    for (const TensorStorage& t : dit.tensors()) ++n;
    for (const TensorStorage& t : vae.tensors()) ++n;

    struct ggml_init_params gip {
        /*.mem_size   =*/ ggml_tensor_overhead() * (n + 8),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ext_ctx_ = ggml_init(gip);
    if (!ext_ctx_) {
        if (error) *error = "ggml_init failed for DiT+VAE context";
        return false;
    }
    owns_ext_ctx_ = true;

    // Bind tensors from each file. DiT and VAE both use `decoder.*` names
    // upstream (ServeurpersoCom/acestep.cpp) — they collide on
    // `decoder.block.{i}.*`. To keep one ext_ctx_ addressable by name, we
    // prefix every VAE tensor with `vae.` at bind time. The VAE graph
    // builder (TODO: acestep.cpp port) will look up `vae.decoder.*`; the DiT
    // builder uses unprefixed `decoder.*`.
    //
    // The zero-copy pointer stays valid because the Session base holds the
    // GgufReaders (loader_ + extra_loaders_) for our lifetime.
    auto bind_one = [&](const GgufReader& r, bool is_vae) -> bool {
        for (const TensorStorage& t : r.tensors()) {
            const std::string bound_name = is_vae ? ("vae." + t.name) : t.name;
            ggml_tensor* gt = ggml_new_tensor(ext_ctx_, t.type, t.n_dims, t.ne);
            if (!gt) {
                if (error) *error = "ggml_new_tensor failed for " + bound_name;
                return false;
            }
            ggml_set_name(gt, bound_name.c_str());
            const void* p = r.tensor_data_ptr(t);
            if (!p) {
                if (error) *error = "tensor_data_ptr(" + bound_name + ") null";
                return false;
            }
            gt->data = const_cast<void*>(p);

            // Anchor a few hot tensors so the DiT/VAE graph builders (TODO)
            // don't have to ggml_get_tensor() them per forward.
            if (!is_vae) {
                if      (t.name == "decoder.proj_in.1.weight")  dit_proj_in_    = gt;
                else if (t.name == "decoder.proj_out.1.weight") dit_proj_out_   = gt;
                else if (t.name == "decoder.time_embed")        dit_time_embed_ = gt;
            } else {
                if      (t.name == "decoder.conv1") vae_conv_in_  = gt;
                else if (t.name == "decoder.conv2") vae_conv_out_ = gt;
            }
        }
        return true;
    };
    if (!bind_one(dit, /*is_vae=*/false)) return false;
    if (!bind_one(vae, /*is_vae=*/true))  return false;

    return true;
}

bool AceStepSession::load(const std::string& model_path,
                          const LoadOptions& opts,
                          const BackendConfig& backend_cfg,
                          std::string* error) {
    // (1) Locate the four component GGUFs by filename pattern. The model
    //     directory can be either the unpacked release or a flat folder.
    //     LoadOptions::extras["dit_variant"] / ["lm_variant"] override the
    //     default turbo / 1.7B preference when multiple variants coexist.
    //
    //     Accept either a directory or a single .gguf inside it — the
    //     registry's FamilyRegistryLoader auto-discovers the first .gguf
    //     when given a directory, so we may receive a file path even
    //     though we want the parent folder.
    std::string dir = model_path;
    {
        std::error_code ec;
        if (std::filesystem::is_regular_file(dir, ec)) {
            dir = std::filesystem::path(dir).parent_path().string();
        }
    }
    const std::string dit_variant =
        opts.extras.count("dit_variant") ? opts.extras.at("dit_variant") : "turbo";
    const std::string lm_variant =
        opts.extras.count("lm_variant") ? opts.extras.at("lm_variant") : "1.7B";
    const std::string dit_path = find_gguf(dir, "acestep-v15", dit_variant);
    const std::string lm_path  = find_gguf(dir, "5Hz-lm", lm_variant);
    const std::string te_path  = find_gguf(dir, "Qwen3-Embedding");
    // VAE path: allow override via extras["vae_path"] (ScragVAE swap-in).
    // The ScragVAE community decoder has tensor-for-tensor compatibility with
    // the stock VAE but produces dramatically better high-frequency detail
    // (12-24kHz region) and dynamic range. When the override path doesn't
    // exist or isn't supplied, fall back to the dir's default *vae*.gguf.
    std::string vae_path;
    if (opts.extras.count("vae_path") && !opts.extras.at("vae_path").empty()) {
        vae_path = opts.extras.at("vae_path");
        if (!file_exists(vae_path)) {
            if (error) *error = "ACE-Step: vae_path override '" + vae_path +
                                "' does not exist";
            return false;
        }
    } else {
        vae_path = find_gguf(dir, "vae");
    }
    if (!file_exists(dit_path) || !file_exists(lm_path) ||
        !file_exists(te_path)  || !file_exists(vae_path)) {
        if (error) *error = std::string("ACE-Step model dir must contain four "
                            "GGUFs: acestep-v15-*, *5Hz-lm*, *Qwen3-Embedding*, "
                            "*vae*. Found: ") +
                            (dit_path.empty() ? "(no dit) " : "") +
                            (lm_path.empty()  ? "(no lm) "  : "") +
                            (te_path.empty()  ? "(no te) "  : "") +
                            (vae_path.empty() ? "(no vae)"  : "");
        return false;
    }

    // (2) Open all four. Keep them alive via WeightLoaders stored on the
    //     Session — their mmaps back the ext_ctx_ tensor pointers.
    auto dit_ld = make_weight_loader(dit_path, error);   if (!dit_ld) return false;
    auto lm_ld  = make_weight_loader(lm_path,  error);   if (!lm_ld)  return false;
    auto te_ld  = make_weight_loader(te_path,  error);   if (!te_ld)  return false;
    auto vae_ld = make_weight_loader(vae_path, error);   if (!vae_ld) return false;
    auto* dit_r = dynamic_cast<GgufReader*>(dit_ld.get());
    auto* lm_r  = dynamic_cast<GgufReader*>(lm_ld.get());
    auto* te_r  = dynamic_cast<GgufReader*>(te_ld.get());
    auto* vae_r = dynamic_cast<GgufReader*>(vae_ld.get());
    if (!dit_r || !lm_r || !te_r || !vae_r) {
        if (error) *error = "ACE-Step requires all four inputs to be .gguf";
        return false;
    }

    // (3) Parse ACE-Step KV from the DiT file. config_json carries the full
    //     upstream config we fall back to for keys we haven't enumerated.
    //     We try multiple KV key formats (standard GGUF arch prefix, flat
    //     acestep.* keys) for robustness across converter versions.
    auto kv_i32 = [&](const char* key, int32_t* out) {
        // Try the exact key first, then fall back to acestep.* prefix
        // (e.g. "acestep-dit.embedding_length" → "acestep.embedding_length")
        if (dit_r->get_kv_i32(key, out)) return;
        std::string key_str(key);
        size_t dot = key_str.find('.');
        if (dot != std::string::npos) {
            std::string alt = "acestep." + key_str.substr(dot + 1);
            dit_r->get_kv_i32(alt.c_str(), out);
        }
    };

    // Documented KV keys (verified in GGUF_FORMAT.md):
    dit_r->get_kv_i32("acestep.in_channels",               &cfg_.dit.in_channels);
    dit_r->get_kv_i32("acestep.audio_acoustic_hidden_dim", &cfg_.dit.out_channels);
    dit_r->get_kv_i32("acestep.patch_size",                &cfg_.dit.patch_size);
    dit_r->get_kv_i32("acestep.sliding_window",            &cfg_.dit.sliding_window);
    dit_r->get_kv_str("acestep.config_json",                &cfg_.config_json);
    dit_r->get_kv_str("acestep.variant",                    &cfg_.variant);
    if (cfg_.variant.empty() && dit_path.find("turbo") != std::string::npos) {
        cfg_.variant = "turbo";
    } else if (cfg_.variant.empty() && dit_path.find("sft") != std::string::npos) {
        cfg_.variant = "sft";
    } else if (cfg_.variant.empty() &&
               dit_path.find("xl-base") == std::string::npos &&
               dit_path.find("-base") != std::string::npos) {
        // Standard 1.5 Base — pretrained root, not instruction-tuned for the
        // 8-step turbo shortcut. Uses the linear 50-step schedule (same as
        // SFT) in build_schedule(). XL Base is excluded so its existing
        // 8-step behaviour stays untouched.
        cfg_.variant = "base";
    }

    // Standard GGUF architecture keys (written by converter if present):
    kv_i32("acestep-dit.embedding_length",              &cfg_.dit.hidden_size);
    kv_i32("acestep-dit.feed_forward_length",           &cfg_.dit.intermediate_size);
    kv_i32("acestep-dit.attention.head_count",          &cfg_.dit.n_heads);
    kv_i32("acestep-dit.attention.head_count_kv",       &cfg_.dit.n_kv_heads);
    kv_i32("acestep-dit.attention.key_length",          &cfg_.dit.head_dim);
    kv_i32("acestep-dit.block_count",                   &cfg_.dit.n_layers);
    {
        float fval = cfg_.dit.rope_theta;
        if (dit_r->get_kv_f32("acestep-dit.rope.freq_base", &fval))
            cfg_.dit.rope_theta = fval;
    }
    {
        float fval = cfg_.dit.rms_norm_eps;
        if (dit_r->get_kv_f32("acestep-dit.attention.layer_norm_rms_epsilon", &fval))
            cfg_.dit.rms_norm_eps = fval;
    }

    // Fallback: derive any zero-valued DitConfig fields from weight tensor
    // shapes bound in step (5). This is the most reliable method since the
    // tensor shapes can't lie — but step (5) hasn't run yet at this point.
    // We'll do a second pass after bind_dit_and_vae().
    //
    // TE hidden size from the TE model (Qwen3-Embedding 0.6B = 1024).
    // NOTE: te_ is loaded below in step (4) — we derive from the weight shapes
    // in step (5b) instead.

    // (4) Verify the two Qwen3 GGUFs are in llama.cpp layout, then spin up
    //     the unified qwen3::Runner for each. Same Runner, same libllama,
    //     same path MOSS uses for its backbone.
    if (!check_llamacpp_layout(*lm_r, "LM", error)) return false;
    if (!check_llamacpp_layout(*te_r, "TE", error)) return false;

    qwen3::RunnerConfig rc;
    rc.n_ctx        = 4096;
    rc.n_threads    = backend_cfg.n_threads;
    rc.n_gpu_layers = (backend_cfg.kind == BackendKind::ggml_cuda) ? -1 : 0;
    rc.flash_attn   = true;
    lm_ = qwen3::Runner::load(lm_path, rc, error);
    if (!lm_) return false;
    te_ = qwen3::Runner::load(te_path, rc, error);
    if (!te_) return false;

    // (5) Bind DiT + VAE weights into ext_ctx_.
    if (!bind_dit_and_vae(*dit_r, *vae_r, error)) return false;

    // (5b) Post-bind: derive any zero-valued DitConfig fields from weight-
    //      tensor shapes.  The tensor shapes can't lie — they're the ground
    //      truth for the loaded model.
    //      proj_in  shape: [K=2, in_channels=192, hidden_size=2048]
    //      proj_out shape: [K=2, out_channels=64, hidden_size=2048]
    if (cfg_.dit.hidden_size == 0 && dit_proj_in_) {
        cfg_.dit.hidden_size = static_cast<int32_t>(dit_proj_in_->ne[2]);
    }
    if (cfg_.dit.in_channels == 0 && dit_proj_in_) {
        cfg_.dit.in_channels = static_cast<int32_t>(dit_proj_in_->ne[1]);
    }
    if (cfg_.dit.out_channels == 0 && dit_proj_out_) {
        cfg_.dit.out_channels = static_cast<int32_t>(dit_proj_out_->ne[1]);
    }

    // (5c) Pre-compute proj_in/proj_out weights: bf16 → f32 Conv1D weights.
    //      Both are 3D [K=2, OC, IC] bf16 in the GGUF and need bf16 decoding
    //      before use. The actual Conv1D(k=2,s=1,p=0) replaces the old
    //      manual_linear (which read bf16 data as f32 — wrong values).
    auto bf16_to_f32_vec = [](const ggml_tensor* t) -> std::vector<float> {
        if (!t) return {};
        size_t n = static_cast<size_t>(t->ne[0]) * t->ne[1] * t->ne[2];
        std::vector<float> out(n);
        const uint16_t* src = static_cast<const uint16_t*>(t->data);
        for (size_t i = 0; i < n; i++) {
            // BF16 is F32 truncated to top 16 bits — just zero-extend.
            uint32_t f32b = static_cast<uint32_t>(src[i]) << 16;
            float f; std::memcpy(&f, &f32b, sizeof(f));
            out[i] = f;
        }
        return out;
    };
    proj_in_w_f32_  = bf16_to_f32_vec(dit_proj_in_);
    proj_out_w_f32_ = bf16_to_f32_vec(dit_proj_out_);

    // (5c.1) Strict required-tensor validation. Fail LOUDLY at load time if
    //        any architecture-critical weight is missing — silent fallbacks
    //        hide checkpoint/format mismatches and produce garbage output
    //        that's painful to debug. The runtime DiT/VAE graph builders
    //        can then assume all required tensors exist.
    auto must_have = [&](const char* name) -> bool {
        if (ggml_get_tensor(ext_ctx_, name)) return true;
        if (error) *error = std::string("ACE-Step: missing required tensor '") +
                            name + "' in DiT GGUF";
        return false;
    };
    // Global tensors.
    if (!must_have("decoder.time_embed.linear_1.weight") ||
        !must_have("decoder.time_embed.linear_2.weight") ||
        !must_have("decoder.time_embed.time_proj.weight") ||
        !must_have("decoder.condition_embedder.weight")  ||
        !must_have("decoder.condition_embedder.bias")    ||
        !must_have("encoder.text_projector.weight")      ||
        !must_have("silence_latent")                     ||
        !must_have("decoder.scale_shift_table")          ||
        !must_have("decoder.norm_out.weight")) {
        return false;
    }
    // Timbre encoder tensors (4 layers).
    {
        const char* timbre_names[] = {
            "encoder.timbre_encoder.embed_tokens.weight",
            "encoder.timbre_encoder.embed_tokens.bias",
            "encoder.timbre_encoder.norm.weight",
        };
        for (const char* n : timbre_names) if (!must_have(n)) return false;
        for (int i = 0; i < 4; i++) {
            char buf[160];
            const char* suffixes[] = {
                ".input_layernorm.weight",
                ".post_attention_layernorm.weight",
                ".self_attn.q_proj.weight",
                ".self_attn.k_proj.weight",
                ".self_attn.v_proj.weight",
                ".self_attn.o_proj.weight",
                ".self_attn.q_norm.weight",
                ".self_attn.k_norm.weight",
                ".mlp.gate_proj.weight",
                ".mlp.up_proj.weight",
                ".mlp.down_proj.weight",
            };
            for (const char* sfx : suffixes) {
                std::snprintf(buf, sizeof(buf),
                              "encoder.timbre_encoder.layers.%d%s", i, sfx);
                if (!must_have(buf)) return false;
            }
        }
    }
    // Per-layer DiT tensors.
    if (cfg_.dit.n_layers <= 0) {
        if (error) *error = "ACE-Step: DitConfig.n_layers not set";
        return false;
    }
    for (int i = 0; i < cfg_.dit.n_layers; i++) {
        char buf[160];
        const char* suffixes[] = {
            ".self_attn_norm.weight",
            ".cross_attn_norm.weight",
            ".mlp_norm.weight",
            ".self_attn.q_proj.weight",
            ".self_attn.k_proj.weight",
            ".self_attn.v_proj.weight",
            ".self_attn.o_proj.weight",
            ".self_attn.q_norm.weight",
            ".self_attn.k_norm.weight",
            ".cross_attn.q_proj.weight",
            ".cross_attn.k_proj.weight",
            ".cross_attn.v_proj.weight",
            ".cross_attn.o_proj.weight",
            ".cross_attn.q_norm.weight",
            ".cross_attn.k_norm.weight",
            ".mlp.gate_proj.weight",
            ".mlp.up_proj.weight",
            ".mlp.down_proj.weight",
            ".scale_shift_table",
        };
        for (const char* sfx : suffixes) {
            std::snprintf(buf, sizeof(buf), "decoder.layers.%d%s", i, sfx);
            if (!must_have(buf)) return false;
        }
    }

    // (5d) Construct DiTRunner and VAERunner.
    dit_runner_ = std::make_unique<DiTRunner>(ext_ctx_, cfg_.dit);
    vae_runner_ = std::make_unique<VAERunner>(ext_ctx_);
    detokenizer_runner_ = std::make_unique<DetokenizerRunner>(ext_ctx_);

    // (5d) Compute FSQ code offset: audio codes are the last 64000 tokens
    //      in the LM's vocabulary (appended after the Qwen3 BPE tokens).
    fsq_code_offset_ = lm_->vocab_size() - 64000;
    if (fsq_code_offset_ < 0) fsq_code_offset_ = 0;

    // (6) Stash the readers on the Session so the mmaps outlive load().
    //     loader_ (the base-class WeightLoader slot) holds one; we keep the
    //     other three as extra members we'd add to AceStepSession if we
    //     were tracking them by hand. For now, leak them — they live for
    //     the process lifetime. TODO: add a small weighted-reader-bag to
    //     Session base class.
    loader_ = std::move(dit_ld);
    extra_loaders_.push_back(std::move(lm_ld));
    extra_loaders_.push_back(std::move(te_ld));
    extra_loaders_.push_back(std::move(vae_ld));

    loaded_ = true;
    (void)opts;
    return true;
}

// ---------------------------------------------------------------------------
// Factory registered with FamilyRegistry at static-init time.
// ---------------------------------------------------------------------------

namespace {
std::unique_ptr<Session> make_ace_step_session() {
    return std::unique_ptr<Session>(new AceStepSession());
}
}  // namespace

AUDIOCORE_REGISTER_FAMILY(ace_step, make_ace_step_session)
AUDIOCORE_EXTERN_C_GUARD(ace_step, make_ace_step_session)

}  // namespace audiocore::acestep
