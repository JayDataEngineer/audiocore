// test_moss_voice_clone_e2e.cpp — Real voice-cloning e2e test.
//
// Verifies the upstream-honest voice-cloning pipeline:
//   1. Generate a short reference clip with mode="tts" (any MOSS output works
//      as a reference; the encoder doesn't care what voice it is).
//   2. Run mode="voice_clone" against that reference.
//
// This exercises:
//   • audiocore::io::read_wav_mono (resamples if needed; 24 kHz here)
//   • MossCodecGraphs::encode (the ported openmoss codec encoder)
//   • build_reference_audio_block + build_prompt_grid splicing codes into the
//     [S1]: block of the upstream <user_inst> template
//   • The full AR + codec decode path on the spliced prompt
//
// Precondition: ./test_moss_e2e must have produced test_moss_output.wav in
// the current directory. The CMake test harness runs test_moss_e2e first.

#include "e2e_common.h"
#include "audiocore/framework/core/backend.h"
#include "audiocore/framework/core/session.h"
#include "audiocore/framework/io/weight_loader.h"
#include "audiocore/framework/runtime/registry.h"
#include "audiocore/models/moss_tts/family.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <cmath>

extern "C" void audiocore_register_moss_tts();

int main() {
    using namespace audiocore;
    using namespace audiocore::moss;

    audiocore_register_moss_tts();

    const char* env_dir = std::getenv("AUDIOCORE_MOSS_DIR");
    if (!env_dir) {
        std::fprintf(stderr,
            "[SKIP] set AUDIOCORE_MOSS_DIR to the directory holding the "
            "MOSS GGUFs to run this voice-clone e2e test\n");
        return 0;
    }
    std::string model_dir = env_dir;
    std::string backbone_path = model_dir + "/moss-tts-v1.5-q8_0.gguf";

    // Prefer the full extras (codec tensors + encoder) over the v1.5 truncated
    // one — the encoder requires moss.codec.enc.* tensors to be present with
    // valid data pointers.
    std::string extras_path;
    for (const char* candidate : {"moss-tts.extras.gguf",
                                  "moss-tts-v1.5-q8_0.extras.gguf"}) {
        std::string p = model_dir + "/" + candidate;
        std::ifstream f(p);
        if (f.good()) { extras_path = p; break; }
    }

    std::string ref_wav = "test_moss_output.wav";
    std::ifstream ref_check(ref_wav);
    if (!ref_check.good()) {
        std::fprintf(stderr, "[SKIP] %s not found — run test_moss_e2e first\n",
                     ref_wav.c_str());
        return 0;  // soft-skip so the harness doesn't fail when run out of order
    }
    ref_check.close();

    std::fprintf(stderr, "[INFO] MOSS-TTS backbone: %s\n", backbone_path.c_str());
    std::fprintf(stderr, "[INFO] MOSS-TTS extras:   %s\n", extras_path.c_str());
    std::fprintf(stderr, "[INFO] reference WAV:     %s\n", ref_wav.c_str());

    std::ifstream fb(backbone_path);
    CHECK(fb.good(), "backbone GGUF not found");
    std::ifstream fe(extras_path);
    CHECK(fe.good(), "extras GGUF not found");

    auto sess = FamilyRegistry::instance().create("moss_tts");
    CHECK(sess != nullptr, "FamilyRegistry::create(moss_tts) failed");

    LoadOptions opts;
    opts.extras["extras_gguf_path"] = extras_path;
    const char* env_backend = std::getenv("AUDIOCORE_BACKEND");
    BackendKind backend_kind = (env_backend && std::string(env_backend) == "cpu")
        ? BackendKind::ggml_cpu : BackendKind::ggml_cuda;
    BackendConfig bc = { .kind = backend_kind, .device_id = 0, .n_threads = 12 };

    std::string load_err;
    std::fprintf(stderr, "[INFO] loading model...\n");
    bool loaded = sess->load(backbone_path, opts, bc, &load_err);
    CHECK(loaded, ("load failed: " + load_err).c_str());
    std::fprintf(stderr, "[INFO] model loaded\n");

    TtsRequest req;
    req.mode = "voice_clone";
    req.reference_audio = ref_wav;
    req.text = "This is a voice cloning test, replicating the reference timbre.";
    req.temperature = 0.8f;
    req.top_p = 0.9f;
    req.max_new_tokens = 512;

    TtsResponse resp;
    std::string run_err;
    std::fprintf(stderr, "[INFO] running voice_clone...\n");
    bool ok = sess->run_tts(&req, &resp, &run_err);
    CHECK(ok, ("run_tts voice_clone failed: " + run_err).c_str());

    CHECK(!resp.pcm_mono.empty(), "voice_clone output PCM is empty");
    CHECK(resp.sampling_rate > 0, "sampling rate is zero");
    std::fprintf(stderr, "[INFO] output: %zu samples @ %d Hz (%.2f sec)\n",
                 resp.pcm_mono.size(), resp.sampling_rate,
                 static_cast<double>(resp.pcm_mono.size()) / resp.sampling_rate);

    double sum_sq = 0.0;
    for (float s : resp.pcm_mono) sum_sq += static_cast<double>(s) * s;
    double rms = std::sqrt(sum_sq / resp.pcm_mono.size());
    std::fprintf(stderr, "[INFO] RMS level: %.6f\n", rms);
    CHECK(rms > 1e-6, "voice_clone output is silent — codec encoder or splice regression");

    CHECK(write_wav("test_moss_voice_clone_output.wav", resp.pcm_mono.data(),
                    resp.pcm_mono.size(), resp.sampling_rate) == 0,
          "failed to write voice_clone output WAV");
    std::fprintf(stderr, "[PASS] voice_clone output written to test_moss_voice_clone_output.wav\n");
    return 0;
}
