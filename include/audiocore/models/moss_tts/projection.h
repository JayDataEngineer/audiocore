// projection.h — MOSS-TTS audio-head projection.
//
// The projection multiplies Qwen3 hidden states (n_tokens × hidden_size) by
// n_vq audio-head weight matrices, producing logits over the codec vocab for
// each of n_vq RVQ streams:
//
//   logits[t, s, v] = sum_j hidden[t, j] · audio_head[s][v, j]
//
// Two implementations share this signature:
//   • project_logits_reference — plain C++ loops. F32/F16 weights only.
//     O(n_tokens × n_vq × hs × vocab). Slow but obvious; serves as the
//     correctness oracle for the cgraph path.
//   • project_logits_cgraph    — one ggml_cgraph per stream, dispatched via
//     ggml_backend. Handles quantized weights natively via ggml's dequant.
//     This is the production path; CUDA/CPU come for free from the backend.
//
// Keeping them as free functions (not MossSession methods) means tests can
// exercise both paths with synthetic weights — no real Qwen3 backbone needed.

#ifndef AUDIOCORE_MODELS_MOSS_TTS_PROJECTION_H
#define AUDIOCORE_MODELS_MOSS_TTS_PROJECTION_H

#include <cstdint>
#include <string>

struct ggml_tensor;

namespace audiocore::moss {

// Inputs to the projection. logits_out must hold n_tokens × n_vq × vocab
// float32s, row-major with vocab innermost.
struct ProjectionRefs {
    int32_t n_tokens    = 0;
    int32_t hidden_size = 0;
    int32_t n_vq        = 0;
    int32_t vocab       = 0;          // audio_vocab_size + 1 (pad slot)

    const float*                 hidden = nullptr;  // (n_tokens, hidden_size)
    const ggml_tensor* const*    heads  = nullptr;  // [n_vq], each (vocab, hs)
};

// Reference C++ loop version. Refuses quantized weights with a clean error
// so callers can fall back to the cgraph path.
bool project_logits_reference(const ProjectionRefs& refs,
                              float* logits_out,
                              std::string* error);

// Production ggml_cgraph version. Supports any ggml_type via backend dequant.
// Falls back to CPU backend when no accelerator is available.
bool project_logits_cgraph(const ProjectionRefs& refs,
                           float* logits_out,
                           std::string* error);

}  // namespace audiocore::moss

#endif  // AUDIOCORE_MODELS_MOSS_TTS_PROJECTION_H
