// test_mse2_e2e.cpp — Load MOSS-SoundEffect-v2, run one SFX request, verify output.
//
// Requires the following model files (download via fetch_models.sh or
// manually) somewhere on disk, pointed at by AUDIOCORE_MSE2_DIR:
//   $AUDIOCORE_MSE2_DIR/mse2-dit.gguf        (DiT weights)
//   $AUDIOCORE_MSE2_DIR/mse2-vae.gguf        (DAC VAE decoder weights)
//   $AUDIOCORE_MSE2_DIR/qwen3-te.gguf        (Qwen3 text encoder, optional)
//
// Set AUDIOCORE_MSE2_DIR to the directory holding the GGUFs to run this test.
// Skips cleanly (exit 0) when AUDIOCORE_MSE2_DIR is unset.

#include "e2e_common.h"
#include "audiocore/framework/core/backend.h"
#include "audiocore/framework/core/session.h"
#include "audiocore/framework/io/weight_loader.h"
#include "audiocore/framework/runtime/registry.h"
#include "audiocore/models/moss_sfx_v2/family.h"

extern "C" void audiocore_register_moss_sfx_v2();

int main() {
    using namespace audiocore;

    audiocore_register_moss_sfx_v2();

    const char* env_dir = std::getenv("AUDIOCORE_MSE2_DIR");
    if (!env_dir) {
        std::fprintf(stderr,
            "[SKIP] set AUDIOCORE_MSE2_DIR to the directory holding the "
            "MSE2 GGUFs to run this e2e test\n");
        return 0;
    }
    std::string model_dir = env_dir;

    // ── Find GGUF files ──────────────────────────────────────────────
    std::string dit_path;
    std::string vae_path;
    std::string te_path;

    // Auto-discover GGUFs in the directory
    for (auto& e : std::filesystem::directory_iterator(model_dir)) {
        auto p = e.path().string();
        if (p.size() < 5) continue;
        if (p.substr(p.size() - 5) != ".gguf") continue;
        if (p.find(".vae.") != std::string::npos ||
            p.find("vae.") != std::string::npos) {
            vae_path = p;
        } else if (p.find("te.") != std::string::npos ||
                   p.find("text_encoder") != std::string::npos ||
                   p.find("qwen3") != std::string::npos) {
            te_path = p;
        } else {
            dit_path = p;
        }
    }

    // Allow explicit overrides via env vars
    const char* env_dit = std::getenv("AUDIOCORE_MSE2_DIT");
    if (env_dit) dit_path = env_dit;
    const char* env_vae = std::getenv("AUDIOCORE_MSE2_VAE");
    if (env_vae) vae_path = env_vae;
    const char* env_te = std::getenv("AUDIOCORE_MSE2_TE");
    if (env_te) te_path = env_te;

    std::fprintf(stderr, "[INFO] DiT:  %s\n", dit_path.c_str());
    std::fprintf(stderr, "[INFO] VAE:  %s\n", vae_path.c_str());
    std::fprintf(stderr, "[INFO] TE:   %s\n",
                 te_path.empty() ? "(none)" : te_path.c_str());

    CHECK(!dit_path.empty(), "no DiT GGUF found in AUDIOCORE_MSE2_DIR");
    {
        std::ifstream f(dit_path);
        CHECK(f.good(), "DiT GGUF not found");
    }
    if (!vae_path.empty()) {
        std::ifstream f(vae_path);
        CHECK(f.good(), "VAE GGUF not found");
    }

    // ── Create session ───────────────────────────────────────────────
    auto sess = FamilyRegistry::instance().create("moss_sfx_v2");
    CHECK(sess != nullptr, "FamilyRegistry::create(moss_sfx_v2) failed");
    std::fprintf(stderr, "[INFO] session created\n");

    LoadOptions opts;
    if (!vae_path.empty()) opts.extras["vae_path"] = vae_path;
    if (!te_path.empty())  opts.extras["te_path"]  = te_path;

    const char* env_backend = std::getenv("AUDIOCORE_BACKEND");
    BackendKind backend_kind = BackendKind::ggml_cuda;
    if (env_backend && std::string(env_backend) == "cpu") {
        backend_kind = BackendKind::ggml_cpu;
    }
    BackendConfig bc = {
        .kind      = backend_kind,
        .device_id = 0,
        .n_threads = 12,
    };

    std::string load_err;
    std::fprintf(stderr, "[INFO] loading model (this may take ~30s)...\n");
    bool loaded = sess->load(dit_path, opts, bc, &load_err);
    CHECK(loaded, ("load failed: " + load_err).c_str());
    std::fprintf(stderr, "[INFO] model loaded\n");

    // ── Run SFX generation ───────────────────────────────────────────
    TtsRequest req;
    const char* env_text = std::getenv("MSE2_TEXT");
    req.text = env_text ? env_text
        : "Rain falling on a tin roof, gentle thunder in the distance";
    req.seed = 42;
    req.duration_tokens = 125;  // ~10 seconds (125 * 0.08 = 10.0)
    req.guidance_scale = 4.0f;  // Match Python pipeline default cfg_scale

    TtsResponse resp;
    std::string run_err;
    std::fprintf(stderr, "[INFO] running SFX generation (this may take ~60s)...\n");
    bool ok = sess->run_tts(&req, &resp, &run_err);
    CHECK(ok, ("run_tts failed: " + run_err).c_str());

    CHECK(!resp.pcm_mono.empty(), "output PCM is empty");
    CHECK(resp.sampling_rate > 0, "sampling rate is zero");
    std::fprintf(stderr, "[INFO] output: %zu samples @ %d Hz (%.2f sec)\n",
                 resp.pcm_mono.size(), resp.sampling_rate,
                 static_cast<double>(resp.pcm_mono.size()) / resp.sampling_rate);

    // ── Verify non-silent output ─────────────────────────────────────
    double sum_sq = 0.0;
    for (float s : resp.pcm_mono) sum_sq += static_cast<double>(s) * s;
    double rms = std::sqrt(sum_sq / resp.pcm_mono.size());
    std::fprintf(stderr, "[INFO] RMS level: %.6f\n", rms);
    CHECK(rms > 1e-6, "output is silent — regression in pipeline");
    std::fprintf(stderr, "[PASS] non-silent output\n");

    // ── Write output WAV ─────────────────────────────────────────────
    CHECK(write_wav("test_mse2_output.wav", resp.pcm_mono.data(),
                    resp.pcm_mono.size(), resp.sampling_rate) == 0,
          "failed to write output WAV");
    std::fprintf(stderr, "[PASS] output written to test_mse2_output.wav\n");

    std::fprintf(stderr, "\n=== MSE2 e2e test PASSED ===\n");
    return 0;
}
