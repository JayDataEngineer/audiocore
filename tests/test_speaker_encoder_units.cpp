// test_speaker_encoder_units.cpp — Unit tests for the Qwen3-TTS speaker encoder.
//
// Tests the pure-function components of Qwen3TtsSpeakerEncoder:
//   compute_mel() — mel spectrogram from PCM (exercises the custom FFT, Hann
//                    window, reflect-pad, Slaney mel filterbank, and log
//                    compression)
//   load_wav()    — minimal WAV file parser
//
// These tests do NOT require model weights — they use synthetic PCM buffers
// and tiny hand-crafted WAV files.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "audiocore/models/qwen3_tts/speaker_encoder.h"

using namespace audiocore::qwen3_tts;

// ── Test macros (void-safe, accumulate failures) ────────────────────────────
static int g_n_pass = 0;
static int g_n_fail = 0;

#define CHECK(cond, msg) do {                                          \
    if (!(cond)) {                                                     \
        std::fprintf(stderr, "  [FAIL] %s:%d: %s\n",                   \
                     __FILE__, __LINE__, msg);                         \
        g_n_fail++;                                                    \
    } else {                                                           \
        g_n_pass++;                                                    \
    }                                                                  \
} while(0)

#define CHECK_CLOSE(a, b, tol, msg) do {                               \
    double _a = (double)(a);                                           \
    double _b = (double)(b);                                           \
    if (std::fabs(_a - _b) > (tol)) {                                  \
        std::fprintf(stderr, "  [FAIL] %s:%d: %s (%.8f vs %.8f, tol=%g)\n", \
                     __FILE__, __LINE__, msg, _a, _b, (double)(tol));  \
        g_n_fail++;                                                    \
    } else {                                                           \
        g_n_pass++;                                                    \
    }                                                                  \
} while(0)

#define CHECK_EQUAL(a, b, msg) CHECK((a) == (b), msg)

// =========================================================================
//  Helper: write a minimal 24 kHz mono 16-bit WAV to a temp file
// =========================================================================
static std::string write_test_wav(const std::vector<float>& pcm) {
    std::string path = "/tmp/test_speaker_encoder.wav";

    auto w16 = [](std::ofstream& o, uint16_t v) {
        o.put(v & 0xff); o.put((v >> 8) & 0xff);
    };
    auto w32 = [](std::ofstream& o, uint32_t v) {
        o.put(v & 0xff); o.put((v >> 8) & 0xff);
        o.put((v >> 16) & 0xff); o.put((v >> 24) & 0xff);
    };

    std::ofstream f(path, std::ios::binary);
    uint32_t data_bytes = (uint32_t)pcm.size() * 2;
    uint32_t byte_rate  = 24000 * 1 * 16 / 8;

    f.write("RIFF", 4);  w32(f, 36 + data_bytes);
    f.write("WAVE", 4);
    f.write("fmt ", 4);  w32(f, 16);  w16(f, 1);  // PCM
    w16(f, 1);           // mono
    w32(f, 24000);       // sample rate
    w32(f, byte_rate);
    w16(f, 2);           // block align
    w16(f, 16);          // bits per sample
    f.write("data", 4);  w32(f, data_bytes);
    for (float s : pcm) {
        if (s >  1.0f) s =  1.0f;
        if (s < -1.0f) s = -1.0f;
        w16(f, (uint16_t)((int16_t)(s * 32767.0f)));
    }
    return path;
}

// =========================================================================
//  compute_mel tests
// =========================================================================

static void test_mel_shape_sine() {
    // 1 second of 440 Hz sine at 24 kHz → mel shape should be (T, 128)
    const int sr = 24000, dur = 1, n_samples = sr * dur;
    std::vector<float> pcm((size_t)n_samples);
    for (int i = 0; i < n_samples; i++)
        pcm[(size_t)i] = 0.5f * sinf(2.0f * (float)M_PI * 440.0f * i / sr);

    int T = 0;
    auto mel = Qwen3TtsSpeakerEncoder::compute_mel(pcm.data(), n_samples, &T);
    CHECK(T > 0, "T > 0 for 1s sine");
    CHECK(!mel.empty(), "mel not empty");
    CHECK_EQUAL((int)mel.size(), T * 128, "mel size == T * 128");

    // Expected T: (n_samples + 2*pad - n_fft) / hop + 1
    // pad = (1024 - 256) / 2 = 384
    // audio_pad size = 24000 + 768 = 24768
    // T = (24768 - 1024) / 256 + 1 = 93.6875 → 93
    int expected_T = ((n_samples + 2 * 384) - 1024) / 256 + 1;
    CHECK_EQUAL(T, expected_T, "T matches expected frame count");
}

static void test_mel_silence_gives_low_energy() {
    // Pure silence → all frames should have very low log-mel values.
    // The FFT of all-zeros gives mag = sqrt(0+0+1e-9) ≈ 3.16e-5 from the
    // sqrt-epsilon in the magnitude computation, which after the area-normalized
    // filterbank gives log ≈ -10.36. This is well below the -5.0 threshold for
    // "active" speech energy.
    const int sr = 24000, n_samples = sr;  // 1 second
    std::vector<float> pcm((size_t)n_samples, 0.0f);

    int T = 0;
    auto mel = Qwen3TtsSpeakerEncoder::compute_mel(pcm.data(), n_samples, &T);
    CHECK(T > 0, "T > 0 for silence");

    double avg = 0.0;
    for (float v : mel) avg += v;
    avg /= mel.size();
    CHECK(avg < -5.0, "silence average well below typical speech energy");

    // Count how many values are at the absolute floor (log(1e-5) ≈ -11.51)
    // vs. near the sqrt-epsilon level (~ -10.36). Some narrow filters may fail
    // to reach the area sum of 1.0 due to discrete truncation, causing partial
    // floor hits — but the vast majority should be uniform.
    int n_floor = 0;
    for (float v : mel) if (v <= -11.0f) n_floor++;
    CHECK(n_floor < (int)(mel.size() * 0.1),
          "at most 10% of silence values at absolute floor");
}

static void test_mel_dc_signal() {
    // Pure DC (all samples = 1.0) → energy concentrated at FFT bin 0 →
    // the lowest mel band (filter ~0-15 Hz) should have significant energy,
    // while the highest bands (~11-12 kHz) should be near the noise floor.
    const int sr = 24000, n_samples = sr / 10;  // 0.1s
    std::vector<float> pcm((size_t)n_samples, 1.0f);

    int T = 0;
    auto mel = Qwen3TtsSpeakerEncoder::compute_mel(pcm.data(), n_samples, &T);
    CHECK(T > 0, "T > 0 for DC signal");

    // Check just band 0 (DC → FFT bin 0 → mel filter 0 at ~0-16 Hz) which
    // should strongly respond, and band 127 (~11-12 kHz) which should be
    // near the noise floor for a DC signal.
    double band0_sum = 0.0, band127_sum = 0.0;
    for (int t = 0; t < T; t++) {
        band0_sum  += mel[(size_t)t * 128];
        band127_sum += mel[(size_t)t * 128 + 127];
    }
    double band0_avg  = band0_sum / T;
    double band127_avg = band127_sum / T;

    // Band 0 captures near-DC energy (sum of Hann window ≈ 512 → log ≈ 6.2)
    CHECK(band0_avg > 0.0, "DC produces positive log-mel in band 0");
    // Band 127 sees only numerical noise → near the epsilon floor
    CHECK(band127_avg < -5.0, "high band near noise floor for DC signal");
}

static void test_mel_energy_conservation() {
    // Half-scale sine vs full-scale: the implementation uses magnitude spectrum,
    // so doubling the input amplitude doubles the mel output linearly. In the
    // linear domain (exp(mel)) the ratio should be ~2× for 2× amplitude.
    const int sr = 24000, n_samples = sr / 10;  // 0.1s
    std::vector<float> pcm1((size_t)n_samples);
    std::vector<float> pcm2((size_t)n_samples);
    for (int i = 0; i < n_samples; i++) {
        float v = sinf(2.0f * (float)M_PI * 1000.0f * i / sr);
        pcm1[(size_t)i] = 0.5f * v;
        pcm2[(size_t)i] = 1.0f * v;
    }

    int T1 = 0, T2 = 0;
    auto mel1 = Qwen3TtsSpeakerEncoder::compute_mel(pcm1.data(), n_samples, &T1);
    auto mel2 = Qwen3TtsSpeakerEncoder::compute_mel(pcm2.data(), n_samples, &T2);
    CHECK_EQUAL(T1, T2, "same T for different amplitude");

    // Compare in linear domain (exp(log_mel) recovers mag * fb).
    // Magnitude spectrum → doubling amplitude doubles mag → 2x linear sum.
    double sum1 = 0.0, sum2 = 0.0;
    for (size_t i = 0; i < mel1.size(); i++) {
        sum1 += exp(mel1[i]);
        sum2 += exp(mel2[i]);
    }
    double ratio = sum2 / sum1;
    CHECK_CLOSE(ratio, 2.0, 0.5, "doubling amplitude → 2x linear magnitude response");
}

static void test_mel_small_input() {
    // Very short input (less than n_fft) — should still produce at least 1 frame
    std::vector<float> pcm(512, 0.0f);
    pcm[0] = 1.0f;  // single impulse

    int T = 0;
    auto mel = Qwen3TtsSpeakerEncoder::compute_mel(pcm.data(), (int)pcm.size(), &T);
    CHECK(T > 0, "short input still produces frames");
    CHECK(!mel.empty(), "short input produces non-empty mel");
}

// =========================================================================
//  load_wav tests
// =========================================================================

static void test_load_wav_valid() {
    std::vector<float> pcm(4800);  // 0.2s at 24 kHz
    for (size_t i = 0; i < pcm.size(); i++)
        pcm[i] = 0.3f * sinf(2.0f * (float)M_PI * 1000.0f * i / 24000.0f);

    std::string path = write_test_wav(pcm);
    std::string err;
    auto loaded = Qwen3TtsSpeakerEncoder::load_wav(path, &err);
    CHECK(!loaded.empty(), "valid WAV loads successfully");
    CHECK(err.empty(), "no error for valid WAV");
    CHECK_EQUAL((int)loaded.size(), (int)pcm.size(), "loaded PCM has correct length");

    // Verify content (within float tolerance)
    int mismatches = 0;
    for (size_t i = 0; i < loaded.size() && i < pcm.size(); i++) {
        if (std::fabs(loaded[i] - pcm[i]) > 1e-4) mismatches++;
    }
    CHECK(mismatches == 0, "loaded PCM content matches original within 1e-4");
}

static void test_load_wav_not_found() {
    std::string err;
    auto loaded = Qwen3TtsSpeakerEncoder::load_wav("/nonexistent/file.wav", &err);
    CHECK(loaded.empty(), "nonexistent file returns empty PCM");
    CHECK(!err.empty(), "error message for nonexistent file");
}

static void test_load_wav_wrong_samplerate() {
    // Write a WAV with the wrong sample rate (should be 24000)
    std::vector<float> pcm(100, 0.5f);
    // We'll just write a 48 kHz WAV instead:
    std::string path = "/tmp/test_speaker_encoder_bad_sr.wav";
    {
        std::ofstream f(path, std::ios::binary);
        auto w16 = [&](uint16_t v) { f.put(v & 0xff); f.put((v >> 8) & 0xff); };
        auto w32 = [&](uint32_t v) {
            f.put(v & 0xff); f.put((v >> 8) & 0xff);
            f.put((v >> 16) & 0xff); f.put((v >> 24) & 0xff);
        };
        uint32_t data_bytes = (uint32_t)pcm.size() * 2;
        f.write("RIFF", 4);  w32(36 + data_bytes);
        f.write("WAVE", 4);
        f.write("fmt ", 4);  w32(16);  w16(1); w16(1);
        w32(48000);   // <-- WRONG sample rate!
        w32(48000 * 1 * 16 / 8);
        w16(2);  w16(16);
        f.write("data", 4);  w32(data_bytes);
        for (float s : pcm) w16((uint16_t)((int16_t)(s * 32767.0f)));
    }

    std::string err;
    auto loaded = Qwen3TtsSpeakerEncoder::load_wav(path, &err);
    CHECK(loaded.empty(), "48 kHz WAV returns empty (need 24 kHz)");
    CHECK(!err.empty(), "error for wrong sample rate");
}

// =========================================================================
//  main
// =========================================================================

int main() {
    std::fprintf(stderr, "=== Speaker Encoder Unit Tests ===\n");

    struct { const char* name; void (*fn)(); } tests[] = {
        {"mel_shape_sine",            test_mel_shape_sine},
        {"mel_silence_gives_low_energy", test_mel_silence_gives_low_energy},
        {"mel_dc_signal",             test_mel_dc_signal},
        {"mel_energy_conservation",   test_mel_energy_conservation},
        {"mel_small_input",           test_mel_small_input},
        {"load_wav_valid",            test_load_wav_valid},
        {"load_wav_not_found",        test_load_wav_not_found},
        {"load_wav_wrong_samplerate", test_load_wav_wrong_samplerate},
    };

    for (auto& t : tests) {
        std::fprintf(stderr, "  %s ... ", t.name);
        int before = g_n_fail;
        t.fn();
        std::fprintf(stderr, "%s\n", (g_n_fail == before) ? "PASS" : "FAIL");
    }

    std::fprintf(stderr, "\n  %d/%d checks passed\n", g_n_pass, g_n_pass + g_n_fail);
    return (g_n_fail == 0) ? 0 : 1;
}
