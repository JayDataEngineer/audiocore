// e2e_common.h — shared helpers for end-to-end tests.
//
// Each e2e test is a standalone executable (not registered in ctest) that
// links audiocore_framework and has its own main(). This header provides
// common boilerplate: CHECK macro, WAV writer, RMS verifier.

#ifndef AUDIOCORE_TESTS_E2E_COMMON_H
#define AUDIOCORE_TESTS_E2E_COMMON_H

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "audiocore/server/server.h"

// Fail-fast check that prints and returns.
#define CHECK(cond, msg) do {                                              \
    if (!(cond)) {                                                         \
        std::fprintf(stderr, "[FAIL] %s:%d: %s\n",                         \
                     __FILE__, __LINE__, msg);                             \
        return 1;                                                          \
    }                                                                      \
} while(0)

// Write PCM float samples to a 16-bit mono WAV file using the shared encoder.
static inline int write_wav(const std::string& path,
                            const float* pcm, size_t n_samples, int sr) {
    std::vector<float> buf(pcm, pcm + n_samples);
    std::string wav = audiocore::pcm_mono_to_wav(buf, sr);
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "can't open %s\n", path.c_str()); return 1; }
    f.write(wav.data(), static_cast<std::streamsize>(wav.size()));
    return 0;
}

#endif  // AUDIOCORE_TESTS_E2E_COMMON_H
