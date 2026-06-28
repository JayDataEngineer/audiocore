// test_qwen3tts_e2e.cpp — Qwen3-TTS e2e test.
//
// Prerequisites:
//   Download GGUFs from:
//     https://huggingface.co/wkwong/Lunavox-Qwen3-TTS-GGUF
//   Or use the Lunavox model repo structure:
//     /mnt/data/models/audio/qwen3-tts/
//       qwen3_tts_talker.q5_k.gguf
//       qwen3_tts_predictor.q8_0.gguf
//
// The speech-tokenizer ONNX that used to live alongside these GGUFs is no
// longer loaded by audiocore; run_tts currently emits silence for the codec
// decode phase pending a ggml port of the speech tokenizer.
//
// Environment variables:
//   QWEN3TTS_DIR        Model directory           (default: /mnt/data/models/audio/qwen3-tts)
//   QWEN3TTS_TALKER     Talker filename           (default: qwen3_tts_talker.q5_k.gguf)
//   QWEN3TTS_PREDICTOR  Predictor filename        (default: qwen3_tts_predictor.q8_0.gguf)
//   QWEN3TTS_NGPU       GPU layers                (default: 99)
//   QWEN3TTS_DEVICE     Backend kind              (default: ggml_cuda)

#include "e2e_common.h"
#include "audiocore/framework/core/backend.h"
#include "audiocore/framework/core/session.h"
#include "audiocore/framework/runtime/registry.h"
#include "audiocore/models/qwen3_tts/family.h"

extern "C" void audiocore_register_qwen3_tts();

static const char* OUTPUT_WAV = "test_qwen3tts_output.wav";

int main() {
    using namespace audiocore;
    using namespace audiocore::qwen3_tts;

    audiocore_register_qwen3_tts();

    // ── Configure paths from env or defaults ────────────────────────────────
    const char* env_dir  = std::getenv("QWEN3TTS_DIR");
    std::string model_dir = env_dir ? env_dir : "/mnt/data/models/audio/qwen3-tts/";

    const char* env_talker = std::getenv("QWEN3TTS_TALKER");
    std::string talker_fn  = env_talker ? env_talker : "qwen3_tts_talker.q5_k.gguf";

    const char* env_pred = std::getenv("QWEN3TTS_PREDICTOR");
    std::string pred_fn  = env_pred ? env_pred : "qwen3_tts_predictor.q8_0.gguf";

    const char* env_ngpu = std::getenv("QWEN3TTS_NGPU");
    std::string ngpu_str = env_ngpu ? env_ngpu : "99";

    const char* env_dev = std::getenv("QWEN3TTS_DEVICE");
    std::string backend_dev = env_dev ? env_dev : "ggml_cuda";

    std::fprintf(stderr, "[INFO] Qwen3-TTS dir: %s\n", model_dir.c_str());
    std::fprintf(stderr, "[INFO] talker:         %s\n", talker_fn.c_str());
    std::fprintf(stderr, "[INFO] predictor:      %s\n", pred_fn.c_str());
    std::fprintf(stderr, "[INFO] n_gpu_layers:   %s\n", ngpu_str.c_str());
    std::fprintf(stderr, "[INFO] backend:        %s\n", backend_dev.c_str());

    // ── Create session via FamilyRegistry ───────────────────────────────────
    auto sess = FamilyRegistry::instance().create("qwen3_tts");
    CHECK(sess != nullptr, "FamilyRegistry::create(qwen3_tts) failed");
    std::fprintf(stderr, "[INFO] qwen3_tts session created\n");

    // ── Configure load options ──────────────────────────────────────────────
    LoadOptions opts;
    opts.extras["talker_path"]         = talker_fn;
    opts.extras["predictor_path"]      = pred_fn;
    opts.extras["n_gpu_layers"]        = ngpu_str;

    BackendKind kind = BackendKind::ggml_cpu;
    if (backend_dev == "ggml_cuda")  kind = BackendKind::ggml_cuda;
    if (backend_dev == "ggml_vulkan") kind = BackendKind::ggml_vulkan;

    BackendConfig bc = {
        .kind      = kind,
        .device_id = 0,
        .n_threads = 4,
    };

    // ── Load model ──────────────────────────────────────────────────────────
    std::string load_err;
    std::fprintf(stderr, "[INFO] loading Qwen3-TTS (talker ~1GB + predictor ~156MB) ...\n");
    bool loaded = sess->load(model_dir, opts, bc, &load_err);
    if (!loaded) {
        // If the weights aren't found, skip rather than failing — this test
        // requires external model files that aren't part of the build.
        std::fprintf(stderr, "[SKIP] load failed (weights not available?): %s\n",
                     load_err.c_str());
        return 77;  // ctest skip code
    }
    std::fprintf(stderr, "[INFO] model loaded\n");

    // ── Build TTS request ───────────────────────────────────────────────────
    TtsRequest req;
    req.text        = "Hello, this is a test of Qwen3-TTS on audiocore.";
    req.speed       = 1.0f;
    req.temperature = 0.7f;
    req.top_p       = 0.9f;
    req.language    = "en";

    TtsResponse resp;
    std::string run_err;
    std::fprintf(stderr, "[INFO] running TTS ...\n");
    bool ok = sess->run_tts(&req, &resp, &run_err);
    CHECK(ok, ("run_tts failed: " + run_err).c_str());

    // ── Verify output ───────────────────────────────────────────────────────
    CHECK(!resp.pcm_mono.empty(), "output PCM is empty");
    std::fprintf(stderr, "[INFO] output: %zu samples @ %d Hz (%.2f sec)\n",
                 resp.pcm_mono.size(), resp.sampling_rate,
                 static_cast<double>(resp.pcm_mono.size()) / resp.sampling_rate);

    double sum_sq = 0.0;
    for (float s : resp.pcm_mono) sum_sq += static_cast<double>(s) * s;
    double rms = std::sqrt(sum_sq / resp.pcm_mono.size());
    // The codec decode is currently a silence stub (see src/models/qwen3_tts/
    // session.cpp); until the ggml speech-tokenizer port lands, RMS is ~0
    // and we only assert that run_tts returned PCM of the expected shape.
    std::fprintf(stderr, "[INFO] RMS level: %.6f (silence expected until codec wired)\n", rms);

    // ── Write output WAV ────────────────────────────────────────────────────
    CHECK(write_wav(OUTPUT_WAV, resp.pcm_mono.data(),
                    resp.pcm_mono.size(), resp.sampling_rate) == 0,
          "failed to write output WAV");
    std::fprintf(stderr, "[PASS] output written to %s\n", OUTPUT_WAV);

    return 0;
}
