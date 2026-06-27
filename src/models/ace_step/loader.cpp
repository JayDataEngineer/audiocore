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
std::string find_gguf(const std::string& dir, const std::string& pattern) {
    if (dir.empty()) return {};
    for (const auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        const std::string name = e.path().filename().string();
        if (name.size() < 5 || name.compare(name.size() - 5, 5, ".gguf") != 0) continue;
        if (name.find(pattern) != std::string::npos) return e.path().string();
    }
    return {};
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
    const std::string dir = model_path;
    const std::string dit_path = find_gguf(dir, "acestep-v15");
    const std::string lm_path  = find_gguf(dir, "5Hz-lm");
    const std::string te_path  = find_gguf(dir, "Qwen3-Embedding");
    const std::string vae_path = find_gguf(dir, "vae");
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
    dit_r->get_kv_i32("acestep.in_channels",               &cfg_.in_channels);
    dit_r->get_kv_i32("acestep.audio_acoustic_hidden_dim", &cfg_.audio_acoustic_hidden_dim);
    dit_r->get_kv_i32("acestep.patch_size",                &cfg_.patch_size);
    dit_r->get_kv_i32("acestep.sliding_window",            &cfg_.sliding_window);
    dit_r->get_kv_str("acestep.config_json",               &cfg_.config_json);
    dit_r->get_kv_str("acestep.variant",                   &cfg_.variant);
    if (cfg_.variant.empty() && dit_path.find("turbo") != std::string::npos) {
        cfg_.variant = "turbo";
    } else if (cfg_.variant.empty() && dit_path.find("sft") != std::string::npos) {
        cfg_.variant = "sft";
    }

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

// Explicit registration anchor — see comment in moss_tts/loader.cpp.
extern "C" void audiocore_register_ace_step() {
    static bool done = false;
    if (!done) {
        FamilyRegistry::instance().register_family("ace_step", make_ace_step_session);
        done = true;
    }
}

}  // namespace audiocore::acestep
