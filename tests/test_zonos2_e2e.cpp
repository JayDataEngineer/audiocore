// test_zonos2_e2e.cpp — ZONOS2 subprocess e2e test.
//
// Prerequisites:
//   pip install torch transformers zonos dac
//   (or `uv sync` in the ZONOS2 repo checkout)
//
// Environment variables:
//   ZONOS2_MODEL_PATH   HuggingFace model ID or local path (default: Zyphra/ZONOS2)
//   ZONOS2_PYTHON       Python interpreter path       (default: python3)
//   ZONOS2_DEVICE       Device string for subprocess  (default: cpu)
//   ZONOS2_WEIGHTS_DIR  Local cache dir (optional, for pre-downloaded weights)

#include "e2e_common.h"
#include "audiocore/framework/core/backend.h"
#include "audiocore/framework/core/session.h"
#include "audiocore/framework/runtime/registry.h"
#include "audiocore/models/zonos2/family.h"

extern "C" void audiocore_register_zonos2();

static const char* OUTPUT_WAV = "test_zonos2_output.wav";

int main() {
    using namespace audiocore;
    using namespace audiocore::zonos2;

    audiocore_register_zonos2();

    // Model path: env override or default.
    const char* env_model = std::getenv("ZONOS2_MODEL_PATH");
    std::string model_path = env_model ? env_model : "Zyphra/ZONOS2";
    std::fprintf(stderr, "[INFO] ZONOS2 model: %s\n", model_path.c_str());

    // Create session via FamilyRegistry.
    auto sess = FamilyRegistry::instance().create("zonos2");
    CHECK(sess != nullptr, "FamilyRegistry::create(zonos2) failed");
    std::fprintf(stderr, "[INFO] zonos2 session created\n");

    // Configure load options.
    LoadOptions opts;

    const char* env_python = std::getenv("ZONOS2_PYTHON");
    if (env_python) opts.extras["python_bin"] = env_python;

    const char* env_device = std::getenv("ZONOS2_DEVICE");
    opts.extras["device"] = env_device ? env_device : "cpu";

    BackendConfig bc = {
        .kind = BackendKind::ggml_cpu,
        .device_id = 0,
        .n_threads = 4,
    };

    // Load model (starts subprocess).
    std::string load_err;
    std::fprintf(stderr, "[INFO] loading ZONOS2 (this may take ~60-180s) ...\n");
    bool loaded = sess->load(model_path, opts, bc, &load_err);
    CHECK(loaded, ("load failed: " + load_err).c_str());
    std::fprintf(stderr, "[INFO] model loaded\n");

    // Build TTS request.
    TtsRequest req;
    req.text        = "Hello, this is a test of ZONOS2 TTS on audiocore.";
    req.speed       = 1.0f;
    req.temperature = 0.7f;
    req.top_p       = 0.9f;

    TtsResponse resp;
    std::string run_err;
    std::fprintf(stderr, "[INFO] running TTS (this may take ~10-30s)...\n");
    bool ok = sess->run_tts(&req, &resp, &run_err);
    CHECK(ok, ("run_tts failed: " + run_err).c_str());

    // Verify output.
    CHECK(!resp.pcm_mono.empty(), "output PCM is empty");
    CHECK(resp.sampling_rate == 44100, "expected 44100 Hz sampling rate");
    std::fprintf(stderr, "[INFO] output: %zu samples @ %d Hz (%.2f sec)\n",
                 resp.pcm_mono.size(), resp.sampling_rate,
                 static_cast<double>(resp.pcm_mono.size()) / resp.sampling_rate);

    double sum_sq = 0.0;
    for (float s : resp.pcm_mono) sum_sq += static_cast<double>(s) * s;
    double rms = std::sqrt(sum_sq / resp.pcm_mono.size());
    CHECK(rms > 1e-6, "output is silent (RMS ~ 0)");
    std::fprintf(stderr, "[INFO] RMS level: %.6f\n", rms);

    // Write output WAV.
    CHECK(write_wav(OUTPUT_WAV, resp.pcm_mono.data(),
                    resp.pcm_mono.size(), resp.sampling_rate) == 0,
          "failed to write output WAV");
    std::fprintf(stderr, "[PASS] output written to %s\n", OUTPUT_WAV);

    return 0;
}
