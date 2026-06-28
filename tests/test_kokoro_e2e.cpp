// test_kokoro_e2e.cpp — Load Kokoro ONNX TTS, run one TTS request, verify output.
//
// Downloads weight files from GitHub Releases on first run if not cached locally.
// Following the same pattern as test_moss_e2e.cpp.

#include "e2e_common.h"
#include "audiocore/framework/core/backend.h"
#include "audiocore/framework/core/session.h"
#include "audiocore/framework/runtime/registry.h"
#include "audiocore/models/kokoro/family.h"

extern "C" void audiocore_register_kokoro();

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

static bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

// Download a file from URL to local path using curl.
static bool download_file(const std::string& url, const std::string& path) {
    std::fprintf(stderr, "[INFO] downloading %s ...\n", url.c_str());
    std::string cmd = "curl -sL -o \"" + path + "\" \"" + url + "\"";
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::fprintf(stderr, "[FAIL] download failed (ret=%d)\n", ret);
        return false;
    }
    // Verify it's not an HTML error page (HuggingFace returns HTML on error)
    std::ifstream f(path);
    std::string first_line;
    std::getline(f, first_line);
    if (first_line.find("<!DOCTYPE") != std::string::npos ||
        first_line.find("<html") != std::string::npos) {
        std::fprintf(stderr, "[FAIL] downloaded an HTML page instead of model file\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Paths (auto-downloaded to a cache dir)
// ---------------------------------------------------------------------------
static std::string CACHE_DIR = ".";
static const char* ONNX_FILE = "kokoro-v1.0.fp16.onnx";
static const char* VOICES_FILE = "voices-v1.0.bin";
static const char* OUTPUT_WAV = "test_kokoro_output.wav";

int main() {
    using namespace audiocore;
    using namespace audiocore::kokoro;

    audiocore_register_kokoro();

    // Cache directory: use env KOKORO_WEIGHTS_DIR or default to ./weights/
    const char* env_dir = std::getenv("KOKORO_WEIGHTS_DIR");
    if (env_dir) CACHE_DIR = env_dir;
    else CACHE_DIR = "weights";

    // Ensure cache dir exists
    std::system(("mkdir -p " + CACHE_DIR).c_str());

    const std::string onnx_path  = CACHE_DIR + "/" + ONNX_FILE;
    const std::string voices_path = CACHE_DIR + "/" + VOICES_FILE;

    // Download model files if not cached.
    if (!file_exists(onnx_path)) {
        std::fprintf(stderr, "[INFO] ONNX model not found at %s\n", onnx_path.c_str());
        bool ok = download_file(
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.fp16.onnx",
            onnx_path);
        CHECK(ok, "failed to download ONNX model");
    } else {
        std::fprintf(stderr, "[INFO] ONNX model found at %s\n", onnx_path.c_str());
    }

    if (!file_exists(voices_path)) {
        std::fprintf(stderr, "[INFO] voices file not found at %s\n", voices_path.c_str());
        bool ok = download_file(
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
            voices_path);
        CHECK(ok, "failed to download voices file");
    } else {
        std::fprintf(stderr, "[INFO] voices file found at %s\n", voices_path.c_str());
    }

    // (1) Create session via FamilyRegistry
    auto sess = FamilyRegistry::instance().create("kokoro");
    CHECK(sess != nullptr, "FamilyRegistry::create(kokoro) failed");
    std::fprintf(stderr, "[INFO] kokoro session created\n");

    // (2) Load model
    LoadOptions opts;
    opts.extras["voices_path"] = voices_path;
    opts.extras["voice"] = "af_heart";
    BackendConfig bc = { .kind = BackendKind::onnxruntime, .device_id = 0, .n_threads = 4 };

    std::string load_err;
    std::fprintf(stderr, "[INFO] loading model (this may take ~10s)...\n");
    bool loaded = sess->load(onnx_path, opts, bc, &load_err);
    CHECK(loaded, ("load failed: " + load_err).c_str());
    std::fprintf(stderr, "[INFO] model loaded\n");

    // List available voices
    auto voices = static_cast<KokoroSession*>(sess.get())->list_voices();
    std::fprintf(stderr, "[INFO] %zu voices available\n", voices.size());

    // (3) Build TTS request
    TtsRequest req;
    req.text     = "Hello, this is a test of Kokoro TTS on audiocore.";
    req.voice    = "af_heart";
    req.speed    = 1.0f;
    req.language = "en-us";

    TtsResponse resp;
    std::string run_err;
    std::fprintf(stderr, "[INFO] running TTS (this may take ~5s)...\n");
    bool ok = sess->run_tts(&req, &resp, &run_err);
    CHECK(ok, ("run_tts failed: " + run_err).c_str());

    // (4) Verify output
    CHECK(!resp.pcm_mono.empty(), "output PCM is empty");
    CHECK(resp.sampling_rate == 24000, "expected 24kHz sampling rate");
    std::fprintf(stderr, "[INFO] output: %zu samples @ %d Hz (%.2f sec)\n",
                 resp.pcm_mono.size(), resp.sampling_rate,
                 static_cast<double>(resp.pcm_mono.size()) / resp.sampling_rate);

    // Check for non-silent audio
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
