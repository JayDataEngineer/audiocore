// vocab_data.h — Kokoro phoneme vocabulary (auto-generated from kokoro-onnx config.json).
//
// Maps IPA phoneme characters (as UTF-32 codepoints) to token IDs.
// 114 entries covering IPA extensions, stress marks, tone letters,
// punctuation, and ASCII letters.
//
// Source: kokoro-onnx v1.0 config.json vocab
// Generated: 2026-06-28

#ifndef AUDIOCORE_MODELS_KOKORO_VOCAB_DATA_H
#define AUDIOCORE_MODELS_KOKORO_VOCAB_DATA_H

#include <cstdint>
#include <unordered_map>

namespace audiocore::kokoro {

static const std::unordered_map<char32_t, int32_t> kKokoroVocab = {
    // Punctuation & spacing
    {U';', 1},
    {U':', 2},
    {U',', 3},
    {U'.', 4},
    {U'!', 5},
    {U'?', 6},
    {U'—', 9},   // em-dash —
    {U'…', 10},  // ellipsis …
    {U'"', 11},
    {U'(', 12},
    {U')', 13},
    {U'“', 14},  // left double quote "
    {U'”', 15},  // right double quote "
    {U' ', 16},

    // Diacritics & special IPA
    {0x0303, 17},   // combining tilde ̃
    {0x02A3, 18},   // dz digraph ʣ
    {0x02A5, 19},   // dʒ digraph ʥ
    {0x02A6, 20},   // ts digraph ʦ
    {0x02A8, 21},   // tʃ digraph ʨ
    {0x1D5D, 22},   // ɝ
    {0xAB67, 23},   // ꭧ

    // IPA uppercase (small caps)
    {U'A', 24},
    {U'I', 25},
    {U'O', 31},
    {U'Q', 33},
    {U'S', 35},
    {U'T', 36},
    {U'W', 39},
    {U'Y', 41},
    {0x1D4A, 42},   // ᵊ

    // IPA lowercase (basic Latin)
    {U'a', 43},
    {U'b', 44},
    {U'c', 45},
    {U'd', 46},
    {U'e', 47},
    {U'f', 48},
    {U'h', 50},
    {U'i', 51},
    {U'j', 52},
    {U'k', 53},
    {U'l', 54},
    {U'm', 55},
    {U'n', 56},
    {U'o', 57},
    {U'p', 58},
    {U'q', 59},
    {U'r', 60},
    {U's', 61},
    {U't', 62},
    {U'u', 63},
    {U'v', 64},
    {U'w', 65},
    {U'x', 66},
    {U'y', 67},
    {U'z', 68},

    // IPA extensions
    {0x0251, 69},   // ɑ
    {0x0250, 70},   // ɐ
    {0x0252, 71},   // ɒ
    {0x00E6, 72},   // æ
    {0x03B2, 75},   // β
    {0x0254, 76},   // ɔ
    {0x0255, 77},   // ɕ
    {0x00E7, 78},   // ç
    {0x0256, 80},   // ɖ
    {0x00F0, 81},   // ð
    {0x02A4, 82},   // ʤ
    {0x0259, 83},   // ə
    {0x025A, 85},   // ɚ
    {0x025B, 86},   // ɛ
    {0x025C, 87},   // ɜ
    {0x025F, 90},   // ɟ
    {0x0261, 92},   // ɡ
    {0x0265, 99},   // ɥ
    {0x0268, 101},  // ɨ
    {0x026A, 102},  // ɪ
    {0x029D, 103},  // ʝ
    {0x026F, 110},  // ɯ
    {0x0270, 111},  // ɰ
    {0x014B, 112},  // ŋ
    {0x0273, 113},  // ɳ
    {0x0272, 114},  // ɲ
    {0x0274, 115},  // ɴ
    {0x00F8, 116},  // ø
    {0x0278, 118},  // ɸ
    {0x03B8, 119},  // θ
    {0x0153, 120},  // œ
    {0x0279, 123},  // ɹ
    {0x027E, 125},  // ɾ
    {0x027B, 126},  // ɻ
    {0x0281, 128},  // ʁ
    {0x027D, 129},  // ɽ
    {0x0282, 130},  // ɂ (sʡ)
    {0x0283, 131},  // ʃ
    {0x0288, 132},  // ʈ
    {0x02A7, 133},  // ʧ
    {0x028A, 135},  // ʊ
    {0x028B, 136},  // ʋ
    {0x028C, 138},  // ʌ
    {0x0263, 139},  // ɣ
    {0x0264, 140},  // ɤ
    {0x03C7, 142},  // χ
    {0x028E, 143},  // ʒ
    {0x0292, 147},  // ʐ (voiced retroflex fricative)
    {0x0294, 148},  // ʔ

    // Suprasegmentals (stress, length, tone)
    {0x02C8, 156},  // ˈ primary stress
    {0x02CC, 157},  // ˌ secondary stress
    {0x02D0, 158},  // ː length mark
    {0x02B0, 162},  // ʰ aspiration
    {0x02B2, 164},  // ʲ palatalization
    {0x2193, 169},  // ↓ downstep
    {0x2192, 171},  // →
    {0x2197, 172},  // ↗ upstep
    {0x2198, 173},  // ↘ global rise
    {0x1D7B, 177},  // ᵻ
};

// Build reverse mapping (token ID → character).
// Initialized once on first access.
inline const std::unordered_map<int32_t, char32_t>& reverseVocab() {
    static const auto* rev = new std::unordered_map<int32_t, char32_t>([]() {
        std::unordered_map<int32_t, char32_t> m;
        for (const auto& [cp, id] : kKokoroVocab) {
            m[id] = cp;
        }
        return m;
    }());
    return *rev;
}

}  // namespace audiocore::kokoro

#endif  // AUDIOCORE_MODELS_KOKORO_VOCAB_DATA_H
