// test_acestep_e2e.cpp — ACE-Step text-to-music e2e test.
//
// Loads four GGUFs (DiT + LM + TE + VAE), runs one text-to-music inference,
// and verifies non-silent stereo PCM output.
//
// Prerequisites:
//   Download from https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF:
//     acestep-v15-turbo-Q8_0.gguf        DiT
//     acestep-5Hz-lm-1.7B-Q8_0.gguf     5Hz music-code LM (→ convert_acestep)
//     Qwen3-Embedding-0.6B-Q8_0.gguf    Text encoder (→ convert_acestep)
//     vae-BF16.gguf                      Audio VAE decoder
//
//   LM and TE must be post-processed with convert_acestep before use.
//
// Environment variables:
//   ACESTEP_DIR        Model directory  (default: weights/ace_step/ace-step-1.5-turbo/)
//   ACESTEP_NGPU       GPU layers       (default: -1 = all)
//   ACESTEP_DEVICE     Backend kind     (default: ggml_cuda)
//   ACESTEP_NTHREADS   CPU threads      (default: 4)

#include "e2e_common.h"
#include "audiocore/framework/core/backend.h"
#include "audiocore/framework/core/session.h"
#include "audiocore/framework/runtime/registry.h"
#include "audiocore/models/ace_step/family.h"

extern "C" void audiocore_register_ace_step();

static const char* OUTPUT_WAV = "test_acestep_output.wav";

int main() {
    using namespace audiocore;
    using namespace audiocore::acestep;

    audiocore_register_ace_step();

    const char* env_dir = std::getenv("ACESTEP_DIR");
    std::string model_dir = env_dir ? env_dir
        : "/home/user/Documents/programs/my-stuff/audiocore/weights/ace_step/ace-step-1.5-turbo/";

    const char* env_ngpu = std::getenv("ACESTEP_NGPU");
    std::string ngpu_str = env_ngpu ? env_ngpu : "-1";

    const char* env_dev = std::getenv("ACESTEP_DEVICE");
    std::string backend_dev = env_dev ? env_dev : "ggml_cuda";

    const char* env_nthr = std::getenv("ACESTEP_NTHREADS");
    std::string nthr_str = env_nthr ? env_nthr : "4";

    std::fprintf(stderr, "[INFO] ACE-Step dir:    %s\n", model_dir.c_str());
    std::fprintf(stderr, "[INFO] n_gpu_layers:    %s\n", ngpu_str.c_str());
    std::fprintf(stderr, "[INFO] backend:         %s\n", backend_dev.c_str());
    std::fprintf(stderr, "[INFO] n_threads:       %s\n", nthr_str.c_str());

    auto sess = FamilyRegistry::instance().create("ace_step");
    CHECK(sess != nullptr, "FamilyRegistry::create(ace_step) failed");
    std::fprintf(stderr, "[INFO] ace_step session created\n");

    BackendKind kind = BackendKind::ggml_cpu;
    if (backend_dev == "ggml_cuda")  kind = BackendKind::ggml_cuda;
    if (backend_dev == "ggml_vulkan") kind = BackendKind::ggml_vulkan;

    BackendConfig bc = {
        .kind      = kind,
        .device_id = 0,
        .n_threads = std::max(1, std::stoi(nthr_str)),
    };

    LoadOptions opts;
    opts.extras["n_gpu_layers"] = ngpu_str;

    // Optional variant selection for exercising dormant checkpoints:
    //   ACESTEP_DIT_VARIANT  "turbo" | "sft" | "xl-base" | "xl-sft" | "xl-turbo"
    //   ACESTEP_LM_VARIANT   "1.7B"  | "4B"
    // When unset, loader defaults to turbo / 1.7B.
    if (const char* dv = std::getenv("ACESTEP_DIT_VARIANT")) opts.extras["dit_variant"] = dv;
    if (const char* lv = std::getenv("ACESTEP_LM_VARIANT"))  opts.extras["lm_variant"]  = lv;

    std::string load_err;
    std::fprintf(stderr, "[INFO] loading ACE-Step (DiT + LM + TE + VAE) ...\n");
    bool loaded = sess->load(model_dir, opts, bc, &load_err);
    if (!loaded) {
        // Exit code 77 signals "skip" to CI runners.
        std::fprintf(stderr, "[SKIP] load failed: %s\n", load_err.c_str());
        return 77;
    }
    std::fprintf(stderr, "[INFO] model loaded\n");

    // ── Run text-to-music inference ───────────────────────────────────────
    // Env-var overrides let the same binary exercise many configurations:
    //   ACESTEP_CAPTION   prompt text
    //   ACESTEP_DURATION  seconds (1.0–60.0)
    //   ACESTEP_SEED      RNG seed
    //   ACESTEP_GUIDANCE  CFG guidance scale (1.0 = no CFG)
    //   ACESTEP_TEMP      sampling temperature (0.0 = argmax)
    MusicRequest req;
    const char* env_cap = std::getenv("ACESTEP_CAPTION");
    const char* env_dur = std::getenv("ACESTEP_DURATION");
    const char* env_seed = std::getenv("ACESTEP_SEED");
    const char* env_guid = std::getenv("ACESTEP_GUIDANCE");
    const char* env_temp = std::getenv("ACESTEP_TEMP");
    req.caption     = env_cap ? env_cap : "lo-fi ambient piano with soft rain";
    req.duration    = env_dur ? std::stof(env_dur) : 10.0f;
    req.seed        = env_seed ? std::stoull(env_seed) : 42ULL;
    req.guidance_scale = env_guid ? std::stof(env_guid) : 7.5f;
    req.temperature = env_temp ? std::stof(env_temp) : 0.0f;
    req.top_p       = 1.0f;
    req.mode        = "text_to_music";
    std::fprintf(stderr, "[INFO] caption:    %s\n", req.caption.c_str());
    std::fprintf(stderr, "[INFO] duration:   %.1f s\n", req.duration);
    std::fprintf(stderr, "[INFO] seed:       %llu\n", (unsigned long long)req.seed);
    std::fprintf(stderr, "[INFO] guidance:   %.2f\n", req.guidance_scale);
    std::fprintf(stderr, "[INFO] temperature:%.2f\n", req.temperature);

    MusicResponse resp;
    std::string run_err;
    std::fprintf(stderr, "[INFO] running text-to-music (this may take ~30s)...\n");
    bool ok = sess->run_music(&req, &resp, &run_err);
    CHECK(ok, ("run_music failed: " + run_err).c_str());

    CHECK(!resp.pcm_stereo.empty(), "output PCM stereo is empty");
    CHECK(resp.sampling_rate == 48000, "sampling rate should be 48000");
    CHECK(resp.channels == 2, "output should be stereo");

    size_t n_frames = resp.pcm_stereo.size() / 2;
    double actual_dur = static_cast<double>(n_frames) / resp.sampling_rate;
    std::fprintf(stderr, "[INFO] output: %zu samples (%zu frames) @ %d Hz (%.2f sec)\n",
                 resp.pcm_stereo.size(), n_frames,
                 resp.sampling_rate, actual_dur);

    // Verify non-silent output with RMS across both channels.
    double sum_sq = 0.0;
    for (float s : resp.pcm_stereo)
        sum_sq += static_cast<double>(s) * s;
    double rms = std::sqrt(sum_sq / resp.pcm_stereo.size());
    std::fprintf(stderr, "[INFO] RMS=%.6f\n", rms);
    CHECK(rms > 1e-6, "output is silent (RMS ~ 0) — DiT/VAE decode issue");

    // Write stereo WAV using the server helper.
    std::string wav = pcm_stereo_to_wav(resp.pcm_stereo, resp.sampling_rate);
    std::ofstream f(OUTPUT_WAV, std::ios::binary);
    CHECK(f.is_open(), "failed to open output WAV for writing");
    f.write(wav.data(), static_cast<std::streamsize>(wav.size()));
    std::fprintf(stderr, "[PASS] output written to %s (%.2f sec, %d channels)\n",
                 OUTPUT_WAV, actual_dur, resp.channels);

    return 0;
}
