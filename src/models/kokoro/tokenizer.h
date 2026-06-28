// tokenizer.h — Kokoro phoneme tokenizer wrapping espeak-ng.
//
// Provides text → phonemes → token-ID conversion matching the
// kokoro-onnx Python pipeline. Uses espeak-ng directly for
// IPA phoneme generation.

#ifndef AUDIOCORE_MODELS_KOKORO_TOKENIZER_H
#define AUDIOCORE_MODELS_KOKORO_TOKENIZER_H

#include <cstdint>
#include <string>
#include <vector>

namespace audiocore::kokoro {

class PhonemeTokenizer {
public:
    PhonemeTokenizer();
    ~PhonemeTokenizer();

    PhonemeTokenizer(const PhonemeTokenizer&) = delete;
    PhonemeTokenizer& operator=(const PhonemeTokenizer&) = delete;

    // Initialize espeak-ng. May be called multiple times (re-entrant).
    // Returns true on success. Call before phonemize()/tokenize().
    bool initialize(std::string* error = nullptr);

    // Convert text to IPA phoneme string using espeak-ng.
    //   text:  input text (UTF-8)
    //   lang:  espeak voice name, e.g. "en-us", "en-gb"
    // Returns phoneme string with stress marks, filtered to vocab chars only.
    std::string phonemize(const std::string& text, const std::string& lang);

    // Convert phoneme string to token IDs.
    // Each phoneme character is mapped through the Kokoro vocab.
    // Characters not in the vocab are silently dropped.
    std::vector<int64_t> tokenize(const std::string& phonemes);

    // Convenience: phonemize + tokenize in one call.
    // Returns token IDs wrapped with pad tokens [0, ...tokens, 0].
    //   max_len: maximum allowed token count (default 510).
    //            Phonemes are truncated before tokenization if needed.
    std::vector<int64_t> phonemize_and_tokenize(const std::string& text,
                                                 const std::string& lang,
                                                 int32_t max_len = 510);

    // Check if initialization succeeded.
    bool initialized() const { return initialized_; }

    // Get the current sample rate (espeak internal — not the model's rate).
    int sample_rate() const { return sample_rate_; }

private:
    bool initialized_ = false;
    int sample_rate_ = 0;
};

}  // namespace audiocore::kokoro

#endif  // AUDIOCORE_MODELS_KOKORO_TOKENIZER_H
