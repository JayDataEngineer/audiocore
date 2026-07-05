// test_qwen3tts_e2e.cpp — Qwen3-TTS e2e test.
//
// Prerequisites:
//   Download GGUFs from:
//     https://huggingface.co/wkwong/Lunavox-Qwen3-TTS-GGUF  (talker + predictor)
//     https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF  (codec sidecar)
//   Place them in any directory and point QWEN3TTS_DIR at it.
//
// Environment variables:
//   QWEN3TTS_DIR        Model directory           (REQUIRED — test skips if unset)
//   QWEN3TTS_TALKER     Talker filename           (default: talker.q5_k.gguf)
//   QWEN3TTS_PREDICTOR  Predictor filename        (default: predictor.q8_0.gguf)
//   QWEN3TTS_CODEC      Codec sidecar filename    (default: tokenizer-q8_0.gguf)
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

    const char* env_dir  = std::getenv("QWEN3TTS_DIR");
    if (!env_dir) {
        std::fprintf(stderr,
            "[SKIP] set QWEN3TTS_DIR to the directory holding the Qwen3-TTS "
            "talker/predictor/codec GGUFs to run this e2e test\n");
        return 0;
    }
    std::string model_dir = env_dir;

    const char* env_talker = std::getenv("QWEN3TTS_TALKER");
    std::string talker_fn  = env_talker ? env_talker : "talker.q5_k.gguf";

    const char* env_pred = std::getenv("QWEN3TTS_PREDICTOR");
    std::string pred_fn  = env_pred ? env_pred : "predictor.q8_0.gguf";

    const char* env_codec = std::getenv("QWEN3TTS_CODEC");
    std::string codec_fn  = env_codec ? env_codec : "tokenizer-q8_0.gguf";

    const char* env_ngpu = std::getenv("QWEN3TTS_NGPU");
    std::string ngpu_str = env_ngpu ? env_ngpu : "99";

    const char* env_dev = std::getenv("QWEN3TTS_DEVICE");
    std::string backend_dev = env_dev ? env_dev : "ggml_cuda";

    std::fprintf(stderr, "[INFO] Qwen3-TTS dir: %s\n", model_dir.c_str());
    std::fprintf(stderr, "[INFO] talker:         %s\n", talker_fn.c_str());
    std::fprintf(stderr, "[INFO] predictor:      %s\n", pred_fn.c_str());
    std::fprintf(stderr, "[INFO] codec:          %s\n", codec_fn.c_str());
    std::fprintf(stderr, "[INFO] n_gpu_layers:   %s\n", ngpu_str.c_str());
    std::fprintf(stderr, "[INFO] backend:        %s\n", backend_dev.c_str());

    auto sess = FamilyRegistry::instance().create("qwen3_tts");
    CHECK(sess != nullptr, "FamilyRegistry::create(qwen3_tts) failed");
    std::fprintf(stderr, "[INFO] qwen3_tts session created\n");

    LoadOptions opts;
    opts.extras["talker_path"]    = talker_fn;
    opts.extras["predictor_path"] = pred_fn;
    opts.extras["codec_path"]     = codec_fn;
    opts.extras["n_gpu_layers"]   = ngpu_str;

    BackendKind kind = BackendKind::ggml_cpu;
    if (backend_dev == "ggml_cuda")  kind = BackendKind::ggml_cuda;
    if (backend_dev == "ggml_vulkan") kind = BackendKind::ggml_vulkan;

    BackendConfig bc = {
        .kind      = kind,
        .device_id = 0,
        .n_threads = 4,
    };

    std::string load_err;
    std::fprintf(stderr, "[INFO] loading Qwen3-TTS (talker + predictor + codec) ...\n");
    bool loaded = sess->load(model_dir, opts, bc, &load_err);
    if (!loaded) {
        std::fprintf(stderr, "[SKIP] load failed (weights not available?): %s\n",
                     load_err.c_str());
        return 77;
    }
    std::fprintf(stderr, "[INFO] model loaded\n");

    TtsRequest req;
    req.text           = "Hello, this is a test of Qwen3-TTS on audiocore.";
    req.max_new_tokens = 50;
    req.speed          = 1.0f;
    req.temperature    = 0.7f;
    req.top_p          = 0.9f;
    req.language       = "en";

    // CustomVoice models need a named speaker; Base models need a reference
    // audio (voice cloning).  Allow the test to pick a speaker via env var.
    if (const char* spk = std::getenv("QWEN3TTS_SPEAKER")) {
        req.speaker_name = spk;
        std::fprintf(stderr, "[INFO] speaker:       %s\n", spk);
    }

    TtsResponse resp;
    std::string run_err;
    std::fprintf(stderr, "[INFO] running TTS ...\n");
    bool ok = sess->run_tts(&req, &resp, &run_err);
    CHECK(ok, ("run_tts failed: " + run_err).c_str());

    CHECK(!resp.pcm_mono.empty(), "output PCM is empty");
    std::fprintf(stderr, "[INFO] output: %zu samples @ %d Hz (%.2f sec)\n",
                 resp.pcm_mono.size(), resp.sampling_rate,
                 static_cast<double>(resp.pcm_mono.size()) / resp.sampling_rate);

    double sum_sq = 0.0;
    for (float s : resp.pcm_mono)
        sum_sq += static_cast<double>(s) * s;
    double rms = std::sqrt(sum_sq / resp.pcm_mono.size());
    std::fprintf(stderr, "[INFO] RMS=%.6f\n", rms);
    // Codec is wired — expect real audio, not silence.
    CHECK(rms > 1e-6, "output is silent (RMS ~ 0) — codec decode issue");

    CHECK(write_wav(OUTPUT_WAV, resp.pcm_mono.data(),
                    resp.pcm_mono.size(), resp.sampling_rate) == 0,
          "failed to write output WAV");
    std::fprintf(stderr, "[PASS] output written to %s\n", OUTPUT_WAV);

    return 0;
}
