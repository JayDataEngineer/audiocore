// test_moss_e2e.cpp — Load MOSS-TTS, run one TTS request, verify output.
//
// Compiles into the existing test infrastructure. The test runner handles
// the weight-path fixtures. This test skips if the model files are absent.

#include "audiocore/framework/core/backend.h"
#include "audiocore/framework/core/session.h"
#include "audiocore/framework/io/weight_loader.h"
#include "audiocore/framework/runtime/registry.h"
#include "audiocore/models/moss_tts/family.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

// Pull in the static registrars so FamilyRegistry knows about moss_tts.
extern "C" void audiocore_register_moss_tts();

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

// Slightly-simpler Check macro that doesn't depend on a test framework.
#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "[FAIL] %s:%d: %s\n", __FILE__, __LINE__, msg); \
        return 1; \
    } \
} while(0)

#define CHECK_OK(x) do { \
    std::string _err; \
    if (!(x)) { \
        std::fprintf(stderr, "[FAIL] %s:%d: %s\n", __FILE__, __LINE__, _err.c_str()); \
        return 1; \
    } \
} while(0)

// Write a WAV file matching the test harness expectations.
static int write_wav(const std::string& path,
                     const float* pcm, size_t n_samples, int sr) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "can't open %s\n", path.c_str()); return 1; }

    auto w16 = [&](uint16_t v) { f.put(v & 0xff); f.put((v >> 8) & 0xff); };
    auto w32 = [&](uint32_t v) {
        for (int i = 0; i < 4; ++i) f.put((v >> (i * 8)) & 0xff);
    };

    const uint16_t channels = 1, bps = 16;
    const uint32_t data_bytes = static_cast<uint32_t>(n_samples) * 2;
    const uint32_t byte_rate = static_cast<uint32_t>(sr) * channels * bps / 8;

    f.write("RIFF", 4);
    w32(36 + data_bytes);
    f.write("WAVE", 4);
    f.write("fmt ", 4); w32(16); w16(1);
    w16(channels); w32(static_cast<uint32_t>(sr)); w32(byte_rate);
    w16(channels * bps / 8); w16(bps);
    f.write("data", 4); w32(data_bytes);
    for (size_t i = 0; i < n_samples; ++i) {
        float s = std::clamp(pcm[i], -1.0f, 1.0f);
        w16(static_cast<int16_t>(s * 32767.0f));
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------
// These match the weights/ directory layout.
static const char* GGUF_PATH =
    "<AUDIOCORE_ROOT>/weights/MOSS-TTS-GGUF/MOSS_TTS_Q4_K_M.gguf";
static const char* EMB_DIR =
    "<AUDIOCORE_ROOT>/weights/MOSS-TTS-GGUF/embeddings";
static const char* LM_DIR =
    "<AUDIOCORE_ROOT>/weights/MOSS-TTS-GGUF/lm_heads";
static const char* DECODER_ONNX =
    "<AUDIOCORE_ROOT>/weights/MOSS-Audio-Tokenizer-ONNX/decoder.onnx";
static const char* OUTPUT_WAV =
    "<AUDIOCORE_ROOT>/build/test_moss_output.wav";

// ---------------------------------------------------------------------------
// Reference — simple "prove by existence" comparison
// ---------------------------------------------------------------------------
// We don't pixel-match the reference_output.wav (stochastic sampling produces
// different audio per run). Instead we verify:
//   1. Load succeeds
//   2. run_tts returns true
//   3. Output PCM has plausible sample rate and length
//   4. RMS > 0 (not silence)
//   5. WAV writes without error

int main() {
    using namespace audiocore;
    using namespace audiocore::moss;

    audiocore_register_moss_tts();

    // Verify model files exist
    {
        std::ifstream f(GGUF_PATH);
        CHECK(f.good(), "GGUF file not found — download weights first");
    }
    {
        std::ifstream f(DECODER_ONNX);
        CHECK(f.good(), "decoder.onnx not found");
    }
    std::fprintf(stderr, "[INFO] model files present\n");

    // (1) Create session via FamilyRegistry
    auto sess = FamilyRegistry::instance().create("moss_tts");
    CHECK(sess != nullptr, "FamilyRegistry::create(moss_tts) failed");
    std::fprintf(stderr, "[INFO] moss_tts session created\n");

    // (2) Load with npy paths + ONNX decoder
    LoadOptions opts;
    opts.extras["embeddings_dir"] = EMB_DIR;
    opts.extras["lm_heads_dir"]   = LM_DIR;
    opts.extras["decoder_onnx"]   = DECODER_ONNX;
    BackendConfig bc = { .kind = BackendKind::ggml_cuda, .device_id = 0, .n_threads = 4 };

    std::string load_err;
    std::fprintf(stderr, "[INFO] loading model (this may take ~30s)...\n");
    bool loaded = sess->load(GGUF_PATH, opts, bc, &load_err);
    CHECK(loaded, ("load failed: " + load_err).c_str());
    std::fprintf(stderr, "[INFO] model loaded\n");

    // (3) Build request
    TtsRequest req;
    req.text = "Hello, this is a test of the MOSS-TTS system.";
    req.temperature = 0.8f;
    req.top_p = 0.9f;
    req.max_tokens = 0;  // default: auto-compute (n_vq * 60 * 30 ≈ 14s)

    TtsResponse resp;
    std::string run_err;
    std::fprintf(stderr, "[INFO] running TTS (this may take ~60s)...\n");
    bool ok = sess->run_tts(&req, &resp, &run_err);
    CHECK(ok, ("run_tts failed: " + run_err).c_str());

    // (4) Verify output
    CHECK(!resp.pcm_mono.empty(), "output PCM is empty");
    CHECK(resp.sampling_rate > 0, "sampling rate is zero");
    std::fprintf(stderr, "[INFO] output: %zu samples @ %d Hz (%.2f sec)\n",
                 resp.pcm_mono.size(), resp.sampling_rate,
                 static_cast<double>(resp.pcm_mono.size()) / resp.sampling_rate);

    // Check RMS > 0 (not silence)
    double sum_sq = 0.0;
    for (float s : resp.pcm_mono) sum_sq += static_cast<double>(s) * s;
    double rms = std::sqrt(sum_sq / resp.pcm_mono.size());
    CHECK(rms > 1e-6, "output is silent (RMS ~ 0)");
    std::fprintf(stderr, "[INFO] RMS level: %.6f\n", rms);

    // (5) Write WAV
    CHECK(write_wav(OUTPUT_WAV, resp.pcm_mono.data(),
                    resp.pcm_mono.size(), resp.sampling_rate) == 0,
          "failed to write output WAV");
    std::fprintf(stderr, "[PASS] output written to %s\n", OUTPUT_WAV);

    return 0;
}
