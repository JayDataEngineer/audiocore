// tokenizer.cpp — Kokoro phoneme tokenizer (espeak-ng + vocab mapping).

#include "tokenizer.h"
#include "vocab_data.h"

#include <algorithm>
#include <cstring>
#include <espeak-ng/speak_lib.h>
#include <string>
#include <vector>

namespace audiocore::kokoro {

// ===========================================================================
// UTF-8 helpers
// ===========================================================================

// Decode one UTF-8 codepoint from `p`, advance `p` past the sequence.
// Returns U+FFFD on invalid sequence.
static char32_t utf8_decode(const char*& p) {
    const auto c = static_cast<unsigned char>(*p);
    if (c < 0x80) {
        char32_t cp = static_cast<char32_t>(c);
        p += 1;
        return cp;
    }
    if ((c & 0xE0) == 0xC0) {
        char32_t cp = c & 0x1F;
        if ((static_cast<unsigned char>(p[1]) & 0xC0) != 0x80) goto bad;
        cp = (cp << 6) | (static_cast<unsigned char>(p[1]) & 0x3F);
        p += 2;
        return cp;
    }
    if ((c & 0xF0) == 0xE0) {
        char32_t cp = c & 0x0F;
        if ((static_cast<unsigned char>(p[1]) & 0xC0) != 0x80) goto bad;
        if ((static_cast<unsigned char>(p[2]) & 0xC0) != 0x80) goto bad;
        cp = (cp << 6) | (static_cast<unsigned char>(p[1]) & 0x3F);
        cp = (cp << 6) | (static_cast<unsigned char>(p[2]) & 0x3F);
        p += 3;
        return cp;
    }
    if ((c & 0xF8) == 0xF0) {
        char32_t cp = c & 0x07;
        if ((static_cast<unsigned char>(p[1]) & 0xC0) != 0x80) goto bad;
        if ((static_cast<unsigned char>(p[2]) & 0xC0) != 0x80) goto bad;
        if ((static_cast<unsigned char>(p[3]) & 0xC0) != 0x80) goto bad;
        cp = (cp << 6) | (static_cast<unsigned char>(p[1]) & 0x3F);
        cp = (cp << 6) | (static_cast<unsigned char>(p[2]) & 0x3F);
        cp = (cp << 6) | (static_cast<unsigned char>(p[3]) & 0x3F);
        p += 4;
        return cp;
    }
bad:
    p += 1;
    return 0xFFFD;  // replacement character
}

// Encode one UTF-32 codepoint into a UTF-8 string.
static std::string utf8_encode(char32_t cp) {
    if (cp < 0x80) {
        return {static_cast<char>(static_cast<unsigned char>(cp)), 1};
    }
    if (cp < 0x800) {
        return {static_cast<char>(0xC0 | (cp >> 6)),
                static_cast<char>(0x80 | (cp & 0x3F))};
    }
    if (cp < 0x10000) {
        return {static_cast<char>(0xE0 | (cp >> 12)),
                static_cast<char>(0x80 | ((cp >> 6) & 0x3F)),
                static_cast<char>(0x80 | (cp & 0x3F))};
    }
    return {static_cast<char>(0xF0 | (cp >> 18)),
            static_cast<char>(0x80 | ((cp >> 12) & 0x3F)),
            static_cast<char>(0x80 | ((cp >> 6) & 0x3F)),
            static_cast<char>(0x80 | (cp & 0x3F))};
}

// ===========================================================================
// PhonemeTokenizer implementation
// ===========================================================================

PhonemeTokenizer::PhonemeTokenizer() = default;

PhonemeTokenizer::~PhonemeTokenizer() {
    if (initialized_) {
        espeak_Terminate();
    }
}

bool PhonemeTokenizer::initialize(std::string* error) {
    if (initialized_) return true;

    sample_rate_ = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS,
                                     espeakINITIALIZE_PHONEME_IPA,
                                     nullptr, 0);
    if (sample_rate_ < 0) {
        if (error) *error = "espeak_Initialize failed";
        return false;
    }

    initialized_ = true;
    return true;
}

std::string PhonemeTokenizer::phonemize(const std::string& text,
                                         const std::string& lang) {
    if (!initialized_ || text.empty()) return {};

    // Set the voice/language.
    espeak_SetVoiceByName(lang.c_str());

    std::string result;

    // Process the text clause-by-clause via espeak_TextToPhonemes.
    // Each call returns phonemes for one clause (up to sentence break
    // or comma/semicolon/colon), advancing the text pointer.
    const void* ptr = text.c_str();
    while (ptr) {
        const char* phonemes = espeak_TextToPhonemes(&ptr, espeakCHARS_UTF8, 2);
        if (!phonemes || !*phonemes) continue;

        // Filter through vocab: keep only characters present in kKokoroVocab,
        // matching the Python behavior: filter(lambda p: p in vocab, phonemes).
        const char* p = phonemes;
        while (*p) {
            const char* start = p;
            char32_t cp = utf8_decode(p);
            if (cp == 0xFFFD) continue;

            auto it = kKokoroVocab.find(cp);
            if (it != kKokoroVocab.end()) {
                // Re-encode to UTF-8 via the start → p bytes.
                // (Fast path: just copy the raw bytes we already scanned.)
                result.append(start, static_cast<size_t>(p - start));
            }
        }
    }

    return result;
}

std::vector<int64_t> PhonemeTokenizer::tokenize(const std::string& phonemes) {
    std::vector<int64_t> tokens;
    tokens.reserve(phonemes.size());  // upper bound

    for (const char* p = phonemes.c_str(); *p; ) {
        char32_t cp = utf8_decode(p);
        if (cp == 0xFFFD) continue;

        auto it = kKokoroVocab.find(cp);
        if (it != kKokoroVocab.end()) {
            tokens.push_back(static_cast<int64_t>(it->second));
        }
    }

    return tokens;
}

std::vector<int64_t> PhonemeTokenizer::phonemize_and_tokenize(
        const std::string& text, const std::string& lang, int32_t max_len) {
    // Step 1: phonemize.
    std::string phonemes = phonemize(text, lang);

    // Truncate phoneme string length to max_len chars (in bytes — approximate).
    // The Python code checks len(phonemes) > MAX_PHONEME_LENGTH and truncates
    // by character index. For simplicity, we truncate after tokenization.
    if (static_cast<int32_t>(phonemes.size()) > max_len * 4) {
        // Very rough guard — actual char count may differ from byte count.
        // We rely on the token-level truncation below.
    }

    // Step 2: tokenize.
    auto tokens = tokenize(phonemes);

    // Step 3: truncate to max_len (minus room for pad tokens).
    const int32_t max_content = max_len - 2;  // leave room for [0, ..., 0]
    if (static_cast<int32_t>(tokens.size()) > max_content) {
        tokens.resize(max_content);
    }

    // Step 4: wrap with pad tokens — [0, ...tokens, 0].
    std::vector<int64_t> padded;
    padded.reserve(tokens.size() + 2);
    padded.push_back(0);  // pad token
    padded.insert(padded.end(), tokens.begin(), tokens.end());
    padded.push_back(0);  // pad token

    return padded;
}

}  // namespace audiocore::kokoro
