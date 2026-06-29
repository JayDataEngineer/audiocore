// speaker_encoder.h — Qwen3-TTS ECAPA-TDNN speaker encoder, ggml port.
//
// Adapted from CrispStrobe/CrispASR (MIT) src/qwen3_tts.cpp into audiocore's
// Backend / TensorStorage / WeightLoader abstractions.
//
// Architecture: 128-mel→Conv1d→3×SE-Res2Net→MFA→ASP→FC→speaker embedding.
// The speaker encoder weights live in the TALKER GGUF (not a separate file),
// under the `speaker.*` namespace. llama.cpp skips these tensors during the
// talker load, so we open the talker GGUF with a second GgufReader to resolve
// them here.
//
// Output dimension matches the talker's d_model (1024 for 0.6B, 2048 for 1.7B),
// so the embedding can be injected directly into the codec bridge as a
// speaker-position token — no extra projection needed.

#ifndef AUDIOCORE_MODELS_QWEN3_TTS_SPEAKER_ENCODER_H
#define AUDIOCORE_MODELS_QWEN3_TTS_SPEAKER_ENCODER_H

#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
using ggml_backend_t = ggml_backend*;
struct ggml_backend_sched;
using ggml_backend_sched_t = ggml_backend_sched*;

namespace audiocore::qwen3_tts {

class Qwen3TtsSpeakerEncoder {
public:
    Qwen3TtsSpeakerEncoder() = default;
    ~Qwen3TtsSpeakerEncoder();

    Qwen3TtsSpeakerEncoder(const Qwen3TtsSpeakerEncoder&) = delete;
    Qwen3TtsSpeakerEncoder& operator=(const Qwen3TtsSpeakerEncoder&) = delete;

    // Hyperparameters. Populated from GGUF KV before calling bind().
    struct HP {
        uint32_t enc_dim      = 1024;   // embedding dimension
        uint32_t sample_rate  = 24000;  // native audio rate
    };
    HP hp;

    // Resolve every speaker-encoder tensor by name from `source_ctx` (which
    // should be the GgufReader's meta_ctx for the talker GGUF), allocate the
    // schedule, and prepare for compute_embedding(). Returns false + *error
    // if any required tensor is missing.
    bool bind(ggml_context* source_ctx,
              ggml_backend_t backend,
              std::string* error);

    bool is_loaded() const { return loaded_; }

    // Full pipeline: load WAV → compute mel → ECAPA forward → embedding.
    // `audio_path` must be 24 kHz mono 16-bit PCM WAV.
    // Returns a vector of float[hp.enc_dim] on success, empty on failure.
    std::vector<float> compute_embedding(const std::string& audio_path);

    // Lower-level: mel → speaker embedding. `mel_TC` is (T_mel, 128) row-major.
    std::vector<float> run_on_mel(const float* mel_TC, int T_mel);

    // Mel spectrogram: 24 kHz mono PCM → (T, 128) log-mel for ECAPA input.
    // Parameters: n_fft=1024, hop=256, n_mels=128, fmin=0, fmax=12000,
    // reflect-pad, periodic Hann, magnitude STFT, Slaney mel filterbank,
    // natural log. Output is (T, 128) row-major.
    static std::vector<float> compute_mel(const float* audio, int n_samples,
                                          int* T_out);

    // Minimal WAV loader: 24 kHz mono 16-bit PCM.
    // Returns mono float samples. Empty on error.
    static std::vector<float> load_wav(const std::string& path,
                                       std::string* error);

private:
    // ── ECAPA sub-structures (mirroring CrispASR's g3t_spk_*) ────────
    struct TDNN { ggml_tensor* w = nullptr; ggml_tensor* b = nullptr; };
    struct SE   { ggml_tensor* c1w = nullptr; ggml_tensor* c1b = nullptr;
                  ggml_tensor* c2w = nullptr; ggml_tensor* c2b = nullptr; };
    struct Res2Net  { TDNN blocks[7]; };
    struct SERes2Net { TDNN tdnn1, tdnn2; Res2Net res2net; SE se; };
    struct ASP     { TDNN tdnn; ggml_tensor* conv_w = nullptr;
                     ggml_tensor* conv_b = nullptr; };

    // ── Resolved weights ─────────────────────────────────────────────
    TDNN     blk0_;       // initial: 128→512, k=5, d=1
    SERes2Net blk_[3];    // 3 SE-Res2Net blocks, d=2/3/4
    TDNN     mfa_;        // multi-layer feature aggregation: 1536→1536, k=1
    ASP      asp_;        // attentive-statistics pooling
    ggml_tensor* fc_w_ = nullptr;  // 3072→enc_dim
    ggml_tensor* fc_b_ = nullptr;

    // ── Runtime state ────────────────────────────────────────────────
    bool                loaded_   = false;
    ggml_backend_t      backend_  = nullptr;  // not owned
    ggml_backend_sched_t sched_   = nullptr;  // owned
    std::vector<uint8_t> compute_meta_;       // scratch for ggml_init

    // ── Graph helpers (adapted from CrispASR) ────────────────────────
    static ggml_tensor* same_conv1d_(ggml_context* ctx, ggml_tensor* x,
                                      ggml_tensor* w, ggml_tensor* b,
                                      int dilation);
    static ggml_tensor* tdnn_block_(ggml_context* ctx, ggml_tensor* x,
                                     const TDNN& t, int dilation);
    static ggml_tensor* se_block_(ggml_context* ctx, ggml_tensor* x,
                                   const SE& se);
    static ggml_tensor* res2net_block_(ggml_context* ctx, ggml_tensor* x,
                                        const Res2Net& r, int dilation);
    static ggml_tensor* se_res2net_(ggml_context* ctx, ggml_tensor* x,
                                     const SERes2Net& blk, int d);
    static ggml_tensor* asp_block_(ggml_context* ctx, ggml_tensor* x,
                                    const ASP& asp);
};

}  // namespace audiocore::qwen3_tts

#endif  // AUDIOCORE_MODELS_QWEN3_TTS_SPEAKER_ENCODER_H
