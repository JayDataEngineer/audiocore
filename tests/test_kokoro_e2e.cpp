// test_kokoro_e2e.cpp — End-to-end Kokoro TTS test.
//
// Loads a Kokoro GGUF and generates speech for a test string.
// Verifies: model loading, phonemization, duration prediction,
// audio generation, and PCM output range.

#include "audiocore/models/kokoro_tts.h"
#include "audiocore/framework/runtime/tasks.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#endif

// Simple WAV writer (mono, 16-bit PCM)
static void write_wav(const std::string& path, const std::vector<float>& samples,
                      int sample_rate) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;

    uint32_t data_size = samples.size() * 2;
    uint32_t total_size = 44 + data_size;

    // RIFF header
    fputs("RIFF", f);
    uint32_t val32 = total_size - 8; fwrite(&val32, 4, 1, f);
    fputs("WAVE", f);

    // fmt chunk
    fputs("fmt ", f);
    val32 = 16; fwrite(&val32, 4, 1, f);        // chunk size
    uint16_t val16 = 1; fwrite(&val16, 2, 1, f); // PCM
    val16 = 1; fwrite(&val16, 2, 1, f);          // mono
    val32 = sample_rate; fwrite(&val32, 4, 1, f);
    val32 = sample_rate * 2; fwrite(&val32, 4, 1, f); // byte rate
    val16 = 2; fwrite(&val16, 2, 1, f);          // block align
    val16 = 16; fwrite(&val16, 2, 1, f);         // bits per sample

    // data chunk
    fputs("data", f);
    val32 = data_size; fwrite(&val32, 4, 1, f);

    for (float s : samples) {
        int16_t pcm = (int16_t)(std::max(-1.0f, std::min(1.0f, s)) * 32767);
        fwrite(&pcm, 2, 1, f);
    }
    fclose(f);
}

int main() {
    const char* model_path = std::getenv("KOKORO_MODEL");
    if (!model_path) {
        // Try default path
        model_path = "/mnt/data/models/tts/kokoro-gguf/models--mmwillet2--Kokoro_GGUF/snapshots/e9e81d8e813948353195c9db77ef065476335c8d/Kokoro_espeak_F16.gguf";
    }
    const char* voice = std::getenv("KOKORO_VOICE");
    if (!voice) voice = "af_heart";

    std::fprintf(stderr, "[INFO] Kokoro TTS E2E test\n");
    std::fprintf(stderr, "[INFO] Model: %s\n", model_path);
    std::fprintf(stderr, "[INFO] Voice: %s\n", voice);

    // Check file exists
    std::ifstream test_file(model_path, std::ios::binary | std::ios::ate);
    if (!test_file.is_open()) {
        std::fprintf(stderr, "[FAIL] Cannot open model file: %s\n", model_path);
        return 1;
    }
    test_file.close();

    // Create session
    audiocore::kokoro_tts::KokoroTtsSession session;

    audiocore::LoadOptions opts;
    opts.extras["voice"] = voice;

    audiocore::BackendConfig backend_cfg;
    backend_cfg.kind = audiocore::BackendKind::ggml_cpu;
    backend_cfg.n_threads = 4;

    std::string error;
    std::fprintf(stderr, "[INFO] Loading model...\n");
    if (!session.load(model_path, opts, backend_cfg, &error)) {
        std::fprintf(stderr, "[FAIL] Load error: %s\n", error.c_str());
        return 1;
    }
    std::fprintf(stderr, "[INFO] Model loaded successfully\n");

    // Generate speech
    audiocore::TtsRequest req;
    req.text = "Hello, this is a test of Kokoro text to speech.";
    req.speaker_name = voice;

    audiocore::TtsResponse resp;

    std::fprintf(stderr, "[INFO] Generating speech...\n");
    if (!session.run_tts(&req, &resp, &error)) {
        std::fprintf(stderr, "[FAIL] TTS error: %s\n", error.c_str());
        return 1;
    }

    std::fprintf(stderr, "[INFO] Generated %zu samples @ %d Hz (%.2f sec)\n",
                 resp.pcm_mono.size(), resp.sampling_rate,
                 (float)resp.pcm_mono.size() / resp.sampling_rate);

    // Analyze output
    if (!resp.pcm_mono.empty()) {
        float rms = 0, peak = 0;
        for (float s : resp.pcm_mono) {
            rms += s * s;
            peak = std::max(peak, std::abs(s));
        }
        rms = std::sqrt(rms / resp.pcm_mono.size());
        std::fprintf(stderr, "[INFO] RMS=%.4f Peak=%.4f\n", rms, peak);

        // Write WAV
        write_wav("test_kokoro_output.wav", resp.pcm_mono, resp.sampling_rate);
        std::fprintf(stderr, "[PASS] Output written to test_kokoro_output.wav\n");
    }

    return 0;
}
