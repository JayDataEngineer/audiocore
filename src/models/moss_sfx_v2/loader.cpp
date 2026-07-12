// loader.cpp — MOSS-SoundEffect-v2 GGUF weight binding + family registration.
//
// Loads the DiT GGUF (moss_sfx_v2.* tensors), parses architecture + scheduler
// KV metadata, binds every weight into a no_alloc ggml_context backed by the
// mmap'd GGUF data. The VAE decoder currently requires a sidecar GGUF
// (--extras vae.gguf) or falls back to a Python pre-processing step.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "audiocore/framework/core/session.h"
#include "audiocore/framework/core/backend.h"
#include "audiocore/framework/io/gguf_reader.h"
#include "audiocore/framework/io/weight_loader.h"
#include "audiocore/framework/runtime/registry.h"
#include "audiocore/models/moss_sfx_v2/family.h"
#include "audiocore/models/qwen3/runner.h"

#include "ggml.h"
#include "gguf.h"

namespace fs = std::filesystem;
namespace ac = audiocore;

// The GGUF_MAX_NAME default (64) is too short for our tensor names
// (e.g. "moss_sfx_v2.blocks.29.cross_attn.norm_k.weight" ≈ 55 chars).
// The project already defines GGML_MAX_NAME=128 in CMakeLists.txt.

namespace audiocore::moss_sfx_v2 {

// ── SfxSession implementation ─────────────────────────────────────────────

SfxSession::~SfxSession() {
    dit_runner_.reset();
    vae_runner_.reset();
    te_.reset();
    if (ext_ctx_ && owns_ext_ctx_) {
        ggml_free(ext_ctx_);
        ext_ctx_ = nullptr;
    }
}

bool SfxSession::load(const std::string& model_path,
                       const LoadOptions& opts,
                       const BackendConfig& backend_cfg,
                       std::string* error) {
    auto _err = [&](const char* msg) {
        if (error) *error = msg;
        std::fprintf(stderr, "[moss_sfx_v2] %s\n", msg);
    };

    // ── 1. Find GGUF files ──────────────────────────────────────────────
    fs::path dir(model_path);
    std::string dit_path;
    std::string vae_path;

    if (!fs::is_directory(dir)) {
        // model_path is a single file — treat as DiT GGUF
        dit_path = model_path;
    } else {
        for (auto& e : fs::directory_iterator(dir)) {
            auto p = e.path().string();
            if (p.size() < 5) continue;
            if (p.substr(p.size() - 5) != ".gguf") continue;
            if (p.find(".vae.") != std::string::npos ||
                p.find("vae.") != std::string::npos ||
                p.find("-vae") != std::string::npos) {
                vae_path = p;
            } else if (p.find("extras") == std::string::npos) {
                dit_path = p;
            }
        }
    }

    // Check --extras for explicit VAE / TE path
    auto it = opts.extras.find("vae_path");
    if (it != opts.extras.end()) {
        vae_path = it->second;
    }
    std::string te_path;
    it = opts.extras.find("te_path");
    if (it != opts.extras.end()) {
        te_path = it->second;
    }

    if (dit_path.empty()) {
        _err("no DiT GGUF found");
        return false;
    }

    // ── 2. Open DiT GGUF ────────────────────────────────────────────────
    std::fprintf(stderr, "[moss_sfx_v2] opening DiT GGUF: %s\n", dit_path.c_str());
    auto dit_reader = std::make_unique<ac::GgufReader>();
    std::string load_err;
    if (!dit_reader->load(dit_path, &load_err)) {
        _err(load_err.c_str());
        return false;
    }
    std::fprintf(stderr, "[moss_sfx_v2] DiT GGUF opened OK\n");

    // ── 3. Parse KV metadata ────────────────────────────────────────────
    std::fprintf(stderr, "[moss_sfx_v2] parsing KV metadata...\n");
    SfxConfig cfg;
    auto kv_i32 = [&](const char* key, int32_t def) {
        int32_t v = def; dit_reader->get_kv_i32(key, &v); return v;
    };
    auto kv_f32 = [&](const char* key, float def) {
        float v = def; dit_reader->get_kv_f32(key, &v); return v;
    };
    auto kv_bool = [&](const char* key, bool def) {
        bool v = def; dit_reader->get_kv_bool(key, &v); return v;
    };
    cfg.dit.n_layers   = kv_i32("moss_sfx_v2.n_layers", 30);
    cfg.dit.dim        = kv_i32("moss_sfx_v2.dim", 1536);
    cfg.dit.ffn_dim    = kv_i32("moss_sfx_v2.ffn_dim", 8960);
    cfg.dit.n_heads    = kv_i32("moss_sfx_v2.n_heads", 12);
    cfg.dit.in_dim     = kv_i32("moss_sfx_v2.in_dim", 128);
    cfg.dit.out_dim    = kv_i32("moss_sfx_v2.out_dim", 128);
    cfg.dit.text_dim   = kv_i32("moss_sfx_v2.text_dim", 2048);
    cfg.dit.freq_dim   = kv_i32("moss_sfx_v2.freq_dim", 256);
    cfg.dit.eps        = kv_f32("moss_sfx_v2.eps", 1e-6f);
    cfg.dit.patch_size = kv_i32("moss_sfx_v2.patch_size", 1);
    cfg.dit.head_dim   = cfg.dit.dim / cfg.dit.n_heads;
    cfg.scheduler_shift          = kv_i32("moss_sfx_v2.scheduler.shift", 5);
    cfg.scheduler_sigma_min      = kv_f32("moss_sfx_v2.scheduler.sigma_min", 0.0f);
    cfg.scheduler_extra_one_step = kv_bool("moss_sfx_v2.scheduler.extra_one_step", true);
    cfg.scheduler_num_train_timesteps = kv_i32("moss_sfx_v2.scheduler.num_train_timesteps", 1000);
    cfg_ = cfg;
    std::fprintf(stderr, "[moss_sfx_v2] KV parsed: dim=%d n_layers=%d\n",
                 cfg.dit.dim, cfg.dit.n_layers);

    auto& tensors = dit_reader->tensors();
    std::fprintf(stderr, "[moss_sfx_v2] total tensors in GGUF: %zu\n", tensors.size());
    size_t n_dit_tensors = 0;
    for (auto& t : tensors) {
        if (t.name.find("moss_sfx_v2.") == 0) n_dit_tensors++;
    }

    // ── 4. Create ext_ctx_ and bind DiT tensors ─────────────────────────
    std::fprintf(stderr, "[moss_sfx_v2] %zu DiT tensors to bind\n", n_dit_tensors);
    size_t n_total = n_dit_tensors + (vae_path.empty() ? 0 : 256);  // VAE: up to 256

    ggml_init_params gip = {
        ggml_tensor_overhead() * n_total,
        nullptr,
        true,  // no_alloc — data lives in mmap
    };
    ext_ctx_ = ggml_init(gip);
    if (!ext_ctx_) { _err("ggml_init failed"); return false; }
    owns_ext_ctx_ = true;

    // ── 5. Materialize DiT tensors ──────────────────────────────────────
    std::fprintf(stderr, "[moss_sfx_v2] ggml_init OK, binding tensors...\n");
    {
        std::size_t n_bound = 0;
    for (auto& t : tensors) {
        if (t.name.find("moss_sfx_v2.") != 0) continue;

        ggml_tensor* gt = ggml_new_tensor(ext_ctx_, t.type, t.n_dims, t.ne);
            if (!gt) { _err("ggml_new_tensor failed"); return false; }
            ggml_set_name(gt, t.name.c_str());

            if (n_bound < 3) {
                std::fprintf(stderr, "  [%zu] %s n_dims=%d ne=[%ld,%ld,%ld] type=%d\n",
                             n_bound, t.name.c_str(), t.n_dims,
                             (long)t.ne[0], (long)t.ne[1], (long)t.ne[2], (int)t.type);
            }
            if (t.name.find("text_embedding") != std::string::npos) {
                std::fprintf(stderr, "  [TE] %s n_dims=%d ne=[%ld,%ld,%ld]\n",
                             t.name.c_str(), t.n_dims,
                             (long)t.ne[0], (long)t.ne[1], (long)t.ne[2]);
            }

            // Materialize into owned buffer (reader goes out of scope after load)
            dit_buffers_.emplace_back(t.nbytes_to_read());
            if (!dit_reader->materialize(t, dit_buffers_.back().data(), &load_err)) {
                std::fprintf(stderr, "[moss_sfx_v2] materialize FAILED for %s: %s\n",
                             t.name.c_str(), load_err.c_str());
                dit_buffers_.pop_back();
                continue;
            }
            gt->data = dit_buffers_.back().data();
            n_bound++;
        }
        std::fprintf(stderr, "[moss_sfx_v2] bound %zu DiT tensors\n", n_bound);
    }

    // ── 6. Bind VAE tensors (if VAE GGUF found) ─────────────────────────
    bool vae_continuous = true;  // default for DAC v2
    int32_t vae_decoder_dim = 2048;
    int32_t vae_latent_dim = -1;
    int32_t vae_hop_length = 960;
    int32_t vae_sample_rate = 48000;
    if (!vae_path.empty()) {
        std::fprintf(stderr, "[moss_sfx_v2] opening VAE GGUF: %s\n", vae_path.c_str());
        auto vae_reader = std::make_unique<ac::GgufReader>();
        if (!vae_reader->load(vae_path, &load_err)) {
            _err(load_err.c_str());
            return false;
        }
        std::fprintf(stderr, "[moss_sfx_v2] VAE GGUF opened OK\n");

        // Read VAE architecture KV metadata (written by tools/convert_vae.py).
        vae_reader->get_kv_i32("moss_sfx_v2.vae.latent_dim", &vae_latent_dim);
        vae_reader->get_kv_i32("moss_sfx_v2.vae.decoder_dim", &vae_decoder_dim);
        vae_reader->get_kv_i32("moss_sfx_v2.vae.hop_length", &vae_hop_length);
        vae_reader->get_kv_i32("moss_sfx_v2.vae.sample_rate", &vae_sample_rate);
        vae_reader->get_kv_bool("moss_sfx_v2.vae.continuous", &vae_continuous);
        std::fprintf(stderr,
                     "[moss_sfx_v2] VAE KV: latent_dim=%d decoder_dim=%d "
                     "hop=%d sr=%d continuous=%d\n",
                     vae_latent_dim, vae_decoder_dim, vae_hop_length,
                     vae_sample_rate, (int)vae_continuous);

        auto& vae_tensors = vae_reader->tensors();
        std::size_t n_vae = 0;
        for (auto& t : vae_tensors) {
            ggml_tensor* gt = ggml_new_tensor(ext_ctx_, t.type, t.n_dims, t.ne);
            if (!gt) { _err("ggml_new_tensor for VAE failed"); return false; }
            ggml_set_name(gt, t.name.c_str());

            // Materialize into owned buffer
            vae_buffers_.emplace_back(t.nbytes_to_read());
            if (!vae_reader->materialize(t, vae_buffers_.back().data(), &load_err)) {
                std::fprintf(stderr, "[moss_sfx_v2] VAE materialize FAILED for %s: %s\n",
                             t.name.c_str(), load_err.c_str());
                vae_buffers_.pop_back();
                continue;
            }
            gt->data = vae_buffers_.back().data();
            n_vae++;
        }
        std::fprintf(stderr, "[moss_sfx_v2] bound %zu VAE tensors\n", n_vae);
    }

    // ── 7. Load text encoder (optional) ─────────────────────────────────
    if (!te_path.empty()) {
        qwen3::RunnerConfig te_cfg;
        te_cfg.n_ctx        = 2048;
        te_cfg.n_batch      = 512;
        te_cfg.n_threads    = 0;
        te_cfg.n_gpu_layers = (backend_cfg.kind == BackendKind::ggml_cuda) ? -1 : 0;
        te_cfg.flash_attn   = true;
        auto te = qwen3::Runner::load(te_path, te_cfg, &load_err);
        if (!te) {
            _err(load_err.c_str());
            return false;
        }
        te_ = std::move(te);
        std::printf("[moss_sfx_v2] text encoder loaded from %s\n",
                     te_path.c_str());
    } else {
        std::printf("[moss_sfx_v2] no text encoder (--extras te_path=...)\n");
    }

    // ── 8. Create runners ───────────────────────────────────────────────
    std::fprintf(stderr, "[moss_sfx_v2] creating DiTRunner...\n");
    dit_runner_ = std::make_unique<DiTRunner>(ext_ctx_, cfg.dit);
    std::fprintf(stderr, "[moss_sfx_v2] DiTRunner created\n");

    VAEConfig vcfg;
    // latent_dim from DiT in_dim (matches VAE latent_dim for this checkpoint).
    // Prefer the value parsed from the VAE GGUF KV when available.
    vcfg.latent_dim  = (vae_latent_dim > 0) ? vae_latent_dim : cfg.dit.in_dim;
    vcfg.decoder_dim = vae_decoder_dim;
    vcfg.hop_length  = vae_hop_length;
    vcfg.sample_rate = vae_sample_rate;
    vcfg.continuous  = vae_continuous;
    std::fprintf(stderr, "[moss_sfx_v2] creating VAERunner...\n");
    vae_runner_ = std::make_unique<VAERunner>(ext_ctx_, vcfg);
    std::fprintf(stderr, "[moss_sfx_v2] VAERunner created\n");

    loaded_ = true;
    return true;
}

bool SfxSession::run_tts(const void* request, void* response,
                          std::string* error) {
    auto* req = static_cast<const TtsRequest*>(request);
    auto* rsp = static_cast<TtsResponse*>(response);

    std::vector<float> pcm;
    if (!run_sfx(*req, &pcm, error)) return false;

    rsp->pcm_mono = std::move(pcm);
    rsp->sampling_rate = cfg_.sample_rate;
    return true;
}

// ── Factory ────────────────────────────────────────────────────────────────

static std::unique_ptr<Session> make_session() {
    return std::make_unique<SfxSession>();
}

}  // namespace audiocore::moss_sfx_v2

AUDIOCORE_REGISTER_FAMILY(moss_sfx_v2, audiocore::moss_sfx_v2::make_session);
AUDIOCORE_EXTERN_C_GUARD(moss_sfx_v2, audiocore::moss_sfx_v2::make_session);
