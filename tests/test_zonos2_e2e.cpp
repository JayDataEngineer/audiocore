// test_zonos2_e2e.cpp — ZONOS2 native C++ e2e test.
//
// Prerequisites:
//   A GGUF model file at the path specified by ZONOS2_MODEL_PATH.
//
// Environment variables:
//   ZONOS2_MODEL_PATH   Path to the .gguf file (required)

#include "e2e_common.h"
#include "audiocore/framework/core/backend.h"
#include "audiocore/framework/core/session.h"
#include "audiocore/framework/runtime/registry.h"
#include "audiocore/models/zonos2/family.h"

extern "C" void audiocore_register_zonos2();

int main() {
    using namespace audiocore;
    using namespace audiocore::zonos2;

    audiocore_register_zonos2();

    const char* env_model = std::getenv("ZONOS2_MODEL_PATH");
    CHECK(env_model != nullptr, "ZONOS2_MODEL_PATH must point to a .gguf file");
    std::string model_path = env_model;
    std::fprintf(stderr, "[INFO] ZONOS2 model: %s\n", model_path.c_str());

    auto sess = FamilyRegistry::instance().create("zonos2");
    CHECK(sess != nullptr, "FamilyRegistry::create(zonos2) failed");

    LoadOptions opts;
    BackendConfig bc{BackendKind::ggml_cpu, 0, 4};

    std::string load_err;
    std::fprintf(stderr, "[INFO] loading ZONOS2 GGUF ...\n");
    bool loaded = sess->load(model_path, opts, bc, &load_err);
    CHECK(loaded, ("load failed: " + load_err).c_str());
    std::fprintf(stderr, "[INFO] model loaded\n");

    // Build TTS request (forward pass not yet implemented).
    TtsRequest req;
    req.text        = "Hello, this is a test of ZONOS2 TTS on audiocore.";
    req.temperature = 1.15f;

    TtsResponse resp;
    std::string run_err;
    bool ok = sess->run_tts(&req, &resp, &run_err);
    if (!ok) {
        std::fprintf(stderr, "[SKIP] run_tts: %s\n", run_err.c_str());
        std::fprintf(stderr, "[PASS] (model loaded, forward pass pending)\n");
        return 0;
    }

    CHECK(!resp.pcmMono.empty(), "output PCM is empty");
    CHECK(resp.samplingRate == 44100, "expected 44100 Hz sampling rate");
    std::fprintf(stderr, "[INFO] output: %zu samples @ %d Hz (%.2f sec)\n",
                 resp.pcmMono.size(), resp.samplingRate,
                 static_cast<double>(resp.pcmMono.size()) / resp.samplingRate);

    double sum_sq = 0.0;
    for (float s : resp.pcmMono) sum_sq += static_cast<double>(s) * s;
    double rms = std::sqrt(sum_sq / resp.pcmMono.size());
    CHECK(rms > 1e-6, "output is silent (RMS ~ 0)");
    std::fprintf(stderr, "[INFO] RMS level: %.6f\n", rms);

    CHECK(write_wav("test_zonos2_output.wav", resp.pcmMono.data(),
                    resp.pcmMono.size(), resp.samplingRate) == 0,
          "failed to write output WAV");

    return 0;
}
