// test_moss_e2e.cpp — Load MOSS-TTS, run one TTS request, verify output.
//
// Requires the following model files (download via fetch_models.sh or
// manually):
//   /mnt/data/models/audio/moss-tts/moss-tts-v1.5-q8_0.gguf          (backbone)
//   /mnt/data/models/audio/moss-tts/moss-tts-v1.5-q8_0.extras.gguf   (codec +
//                                                                      audio
//                                                                      embeds +
//                                                                      LM heads)
//
// Sets AUDIOCORE_MOSS_DIR to override the model directory.

#include "e2e_common.h"
#include "audiocore/framework/core/backend.h"
#include "audiocore/framework/core/session.h"
#include "audiocore/framework/io/weight_loader.h"
#include "audiocore/framework/runtime/registry.h"
#include "audiocore/models/moss_tts/family.h"

extern "C" void audiocore_register_moss_tts();

int main() {
    using namespace audiocore;
    using namespace audiocore::moss;

    audiocore_register_moss_tts();

    const char* env_dir = std::getenv("AUDIOCORE_MOSS_DIR");
    std::string model_dir = env_dir ? env_dir
                                    : "/mnt/data/models/audio/moss-tts/";
    std::string backbone_path = model_dir + "/moss-tts-v1.5-q8_0.gguf";

    // Try a prioritized list of extras GGUFs. The full extras (moss-tts.extras.gguf)
    // has ALL codec tensors with valid data pointers. The v1.5-specific extras is
    // truncated and codec tensors at offsets beyond the file can't be read.
    std::string extras_path;
    for (const char* candidate : {"moss-tts.extras.gguf",
                                  "moss-tts-v1.5-q8_0.extras.gguf"}) {
        std::string p = model_dir + "/" + candidate;
        std::ifstream f(p);
        if (f.good()) { extras_path = p; break; }
    }

    std::fprintf(stderr, "[INFO] MOSS-TTS backbone: %s\n", backbone_path.c_str());
    std::fprintf(stderr, "[INFO] MOSS-TTS extras:   %s\n", extras_path.c_str());

    // Verify model files exist
    {
        std::ifstream f(backbone_path);
        CHECK(f.good(), "backbone GGUF not found");
    }
    {
        std::ifstream f(extras_path);
        CHECK(f.good(), "extras GGUF not found");
    }
    std::fprintf(stderr, "[INFO] model files present\n");

    auto sess = FamilyRegistry::instance().create("moss_tts");
    CHECK(sess != nullptr, "FamilyRegistry::create(moss_tts) failed");
    std::fprintf(stderr, "[INFO] moss_tts session created\n");

    LoadOptions opts;
    opts.extras["extras_gguf_path"] = extras_path;
    // Use CPU backend to avoid CUDA graph + memory issues during testing.
    // The GGUF backbone is Q8_0 and fits within 24GB VRAM on a 4090, but
    // llama.cpp's CUDA graph replay can fail with "failed to find a memory
    // slot" on certain kernel combinations. Override with ggml_cuda if you
    // have a supported system.
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
    bool loaded = sess->load(backbone_path, opts, bc, &load_err);
    CHECK(loaded, ("load failed: " + load_err).c_str());
    std::fprintf(stderr, "[INFO] model loaded\n");

    TtsRequest req;
    req.text = "Hello, this is a test of the MOSS-TTS system with the ggml "
               "codec path.";
    req.temperature = 0.8f;
    req.top_p = 0.9f;
    req.max_new_tokens = 512;

    TtsResponse resp;
    std::string run_err;
    std::fprintf(stderr, "[INFO] running TTS (this may take ~60s)...\n");
    bool ok = sess->run_tts(&req, &resp, &run_err);
    CHECK(ok, ("run_tts failed: " + run_err).c_str());

    CHECK(!resp.pcm_mono.empty(), "output PCM is empty");
    CHECK(resp.sampling_rate > 0, "sampling rate is zero");
    std::fprintf(stderr, "[INFO] output: %zu samples @ %d Hz (%.2f sec)\n",
                 resp.pcm_mono.size(), resp.sampling_rate,
                 static_cast<double>(resp.pcm_mono.size()) / resp.sampling_rate);

    double sum_sq = 0.0;
    for (float s : resp.pcm_mono) sum_sq += static_cast<double>(s) * s;
    double rms = std::sqrt(sum_sq / resp.pcm_mono.size());
    std::fprintf(stderr, "[INFO] RMS level: %.6f\n", rms);
    // Upstream-honest pipeline (§1.3 of the gap tracker has been ported);
    // there is no longer a silence fallback to hide behind. If the codec
    // produced silent output, that is a regression.
    CHECK(rms > 1e-6, "output is silent — regression in AR or codec path");
    std::fprintf(stderr, "[PASS] non-silent output — full codec path works\n");

    CHECK(write_wav("test_moss_output.wav", resp.pcm_mono.data(),
                    resp.pcm_mono.size(), resp.sampling_rate) == 0,
          "failed to write output WAV");
    std::fprintf(stderr, "[PASS] output written to test_moss_output.wav\n");

    return 0;
}
