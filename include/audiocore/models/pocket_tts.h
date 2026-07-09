// pocket_tts.h — Kyutai PocketTTS-100M text-to-speech family.
//
// PocketTTS is a multilingual (en/de/it/pt/es) flow-matching TTS model:
//   1. Text → SentencePiece tokens
//   2. Voice conditioning: reference audio → Mimi encoder → voice state
//   3. Flow-matching LM (6-layer transformer, KV cache) → latents
//   4. Acoustic model (transformer) refines latents
//   5. Mimi decoder (Seanet conv + 2-layer transformer) → 24 kHz PCM
//
// Output: mono float32 PCM at 24 kHz. Fits the existing run_tts surface.
//
// Ported from 0xShug0/audio.cpp (release-0.1, MIT license). See
// src/models/pocket_tts/README.md for the phased port status — the
// scaffolding is in place but the graph code is a multi-stage port.
//
// Weights source: kyutai/pocket-tts on HuggingFace (gated), converted to
// GGUF via tools/convert_pocket_tts.py. Stored as pocket_tts.* tensors.

#ifndef AUDIOCORE_MODELS_POCKET_TTS_H
#define AUDIOCORE_MODELS_POCKET_TTS_H

#include <memory>
#include <string>

#include "audiocore/framework/core/session.h"

namespace audiocore::pocket_tts {

class PocketTtsSession : public Session {
public:
    PocketTtsSession();
    ~PocketTtsSession() override;

    std::string family() const override { return "pocket_tts"; }

    bool load(const std::string& model_path,
              const LoadOptions& opts,
              const BackendConfig& backend_cfg,
              std::string* error = nullptr) override;

    bool run_tts(const void* request, void* response,
                 std::string* error = nullptr) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace audiocore::pocket_tts

#endif  // AUDIOCORE_MODELS_POCKET_TTS_H
