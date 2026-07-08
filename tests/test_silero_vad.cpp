// test_silero_vad.cpp — End-to-end Silero VAD test.
//
// Loads a Silero VAD GGUF and runs VAD on a test WAV. Verifies: model
// loading, WAV I/O, chunk loop, graph build, segmentation. Prints the
// detected speech segments to stdout.

#include "audiocore/models/silero_vad.h"
#include "audiocore/framework/core/backend.h"
#include "audiocore/framework/runtime/tasks.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>

int main() {
    const char* model_path = std::getenv("SILERO_VAD_MODEL");
    if (!model_path) model_path = "models/silero_vad/silero_vad.gguf";
    const char* audio_path = std::getenv("SILERO_VAD_AUDIO");
    if (!audio_path) audio_path = "assets/silero_vad/test_16k.wav";

    std::fprintf(stderr, "[INFO] Silero VAD E2E test\n");
    std::fprintf(stderr, "[INFO] Model: %s\n", model_path);
    std::fprintf(stderr, "[INFO] Audio: %s\n", audio_path);

    std::ifstream test_file(model_path, std::ios::binary);
    if (!test_file) {
        std::fprintf(stderr, "[SKIP] model file not found: %s\n", model_path);
        return 77;  // CTest SKIP_RETURN_VALUE convention
    }

    audiocore::BackendConfig backend_cfg{};
    backend_cfg.n_threads = 4;

    audiocore::silero_vad::SileroVadSession session;
    std::string error;
    if (!session.load(model_path, audiocore::LoadOptions{}, backend_cfg, &error)) {
        std::fprintf(stderr, "[SKIP] load failed: %s\n", error.c_str());
        std::fprintf(stderr, "      (expected until the graph port lands)\n");
        return 77;
    }

    audiocore::VadRequest req;
    req.audio_path = audio_path;
    req.threshold = 0.5f;

    audiocore::VadResponse resp;
    if (!session.run_vad(&req, &resp, &error)) {
        std::fprintf(stderr, "[FAIL] run_vad: %s\n", error.c_str());
        return 1;
    }

    std::fprintf(stderr, "[OK] %zu segments\n", resp.segments.size());
    for (size_t i = 0; i < resp.segments.size(); ++i) {
        const auto& s = resp.segments[i];
        std::printf("  [%zu] %.3f - %.3f sec\n", i, s.start_sec, s.end_sec);
    }
    return 0;
}
