// codec.h — MOSS-Audio-Tokenizer encoder + decoder, ggml port.
//
// Adapted from pwilkin/openmoss (Apache-2.0) src/codec.cpp into audiocore's
// Backend / TensorStorage / WeightLoader abstractions. Architecture overview
// and port-plan in docs/CODEC_PORTS.md §1.
//
// Decoder path: 32-RVQ code matrix (32, T_audio) → 24 kHz mono PCM of
// length T_audio * 1920. The four-stage ProjectedTransformer front-end
// runs entirely in ggml on the same backend as the Qwen3 backbone.
//
// Encoder path (mirror): 24 kHz mono PCM → (32, T_audio) code matrix.
// The encoder runs the same 4-stage ProjectedTransformer in reverse
// (patch=240 downsample BEFORE the first stage, then patch=2 between
// stages), then a 32-step residual LFQ quantizer. Used for voice cloning:
// a reference WAV is encoded and spliced into the user-side audio block
// of the chat prompt.

#ifndef AUDIOCORE_MODELS_MOSS_TTS_CODEC_H
#define AUDIOCORE_MODELS_MOSS_TTS_CODEC_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
using ggml_backend_t = ggml_backend*;
struct ggml_gallocr;
using ggml_gallocr_t = ggml_gallocr*;
struct ggml_backend_buffer;
using ggml_backend_buffer_t = ggml_backend_buffer*;

namespace audiocore::moss {

// MossCodecGraphs owns no weights itself — the moss.codec.* tensors live
// in the MossSession's ext_ctx_ (where bind_extension_tensors already
// placed every tensor whose name starts with "moss."). This class resolves
// the codec-specific tensors by name, materialises the post-weight-norm
// effective weights on the backend once at bind time, then builds + runs
// the decode graph on demand.
class MossCodecGraphs {
public:
    MossCodecGraphs() = default;
    ~MossCodecGraphs();

    MossCodecGraphs(const MossCodecGraphs&) = delete;
    MossCodecGraphs& operator=(const MossCodecGraphs&) = delete;

    // Resolve every decoder tensor by name from `source_ctx` (which must
    // outlive this object — typically MossSession::ext_ctx_), materialise
    // the post-weight-norm effective weights onto `backend`, and create
    // the ggml_gallocr used by decode(). On failure, returns false and
    // sets *error; is_present() stays false.
    //
    // Idempotent: returns true without rebinding if already bound.
    bool bind(ggml_context* source_ctx,
              ggml_backend_t backend,
              std::string* error);

    // True once bind() succeeded. decode() throws if you call it while
    // this is false.
    bool is_present() const { return present_; }

    // Decode (n_vq, T_audio) codes (n_vq must be 32) → mono PCM float32
    // at the codec's native 24 kHz. Output length = T_audio * 1920.
    // Empty vector when T_audio <= 0.
    //
    // Throws std::runtime_error on shape / graph-compute failures, or if
    // the requested T_audio would need an attention mask larger than 8 GiB
    // (refuses early instead of attempting a doomed allocation).
    std::vector<float> decode(const int32_t* codes,
                              int32_t n_vq,
                              int32_t T_audio);

    // Encode mono PCM float32 @ 24 kHz → (n_vq=32, T_audio) row-major codes.
    //   waveform:  PCM samples, length n_samples (any positive length; padded
    //              up to a multiple of hop=1920 internally).
    //   returns:   (32 * T_audio) row-major int32, where T_audio_out is
    //              ceil(n_samples / 1920). T_audio_out is also written to the
    //              reference parameter.
    //
    // Throws std::runtime_error on shape / graph-compute failures, or if
    // encoder weights were not bound (no moss.codec.enc.* tensors present
    // in the GGUF).
    std::vector<int32_t> encode(const float* waveform,
                                int64_t n_samples,
                                int32_t& T_audio_out);

    // True if bind() resolved encoder tensors. decode() works without this;
    // encode() requires it. Set by bind() after resolve_encoder_() succeeds.
    bool encoder_present() const { return encoder_present_; }

    // Stage architecture spec — public so the constexpr DECODER_STAGES table
    // in codec.cpp can name the type. The matching `DECODER_STAGES` array is
    // file-scope in codec.cpp.
    struct StageSpec {
        int input_dim;
        int d_model;
        int n_heads;
        int dim_ff;
        int n_layers;
        int output_dim;
        int patch_after;     // upsample factor following the stage; 0 = none
        int gguf_idx;        // dec.<this>
    };

private:
    // ── Constants (matching openmoss/src/codec.cpp) ─────────────────────
    static constexpr int CODEC_NUM_VQ    = 32;
    static constexpr int CODEC_CB_DIM    = 8;
    static constexpr int CODEC_RVQ_DIM   = 512;
    static constexpr int CODEC_OUT_DIM   = 768;
    static constexpr int CODEC_CB_SIZE   = 1024;
    static constexpr int CODEC_PRE_PATCH = 240;   // encoder initial patch downsample

    struct Layer {
        ggml_tensor* norm1_w       = nullptr;
        ggml_tensor* norm1_b       = nullptr;
        ggml_tensor* norm2_w       = nullptr;
        ggml_tensor* norm2_b       = nullptr;
        ggml_tensor* attn_in       = nullptr;  // (d, 3d) fused QKV
        ggml_tensor* attn_out      = nullptr;  // (d, d)
        ggml_tensor* linear1       = nullptr;  // (d, dff)
        ggml_tensor* linear2       = nullptr;  // (dff, d)
        ggml_tensor* layer_scale_1 = nullptr;  // (d,)
        ggml_tensor* layer_scale_2 = nullptr;  // (d,)
    };
    struct Stage {
        StageSpec           spec;
        ggml_tensor*        iproj = nullptr;  // optional
        ggml_tensor*        oproj = nullptr;  // optional
        std::vector<Layer>  layers;
    };

    // ── State ───────────────────────────────────────────────────────────
    bool                 present_         = false;
    bool                 encoder_present_ = false;
    ggml_context*        source_ctx_ = nullptr;  // not owned
    ggml_backend_t       backend_   = nullptr;   // not owned
    ggml_gallocr_t       galloc_    = nullptr;   // owned

    // Effective-weight context + buffer (post-weight-norm materialisation).
    // ALSO contains device copies of all codec weight tensors from source_ctx_
    // (which only have host mmap data pointers). w_buf_ allocates device memory
    // for the entire w_ctx_ at once via ggml_backend_alloc_ctx_tensors.
    ggml_context*         w_ctx_ = nullptr;       // owned
    ggml_backend_buffer_t w_buf_ = nullptr;       // owned

    // All source_ctx_ tensors that the codec decoder reads. During
    // resolve_decoder_(), each resolved tensor is stored here. During
    // compute_effective_weights_(), device copies are created in w_ctx_ and
    // the copy replaces the original in the Layer/Stage member fields.
    struct TensorSrc { ggml_tensor** field_ptr; ggml_tensor* src; };
    std::vector<TensorSrc> tensor_srcs_;

    std::array<ggml_tensor*, 32> codebook_     {};   // (will be updated to device copy)
    std::array<ggml_tensor*, 32> q_oproj_w_    {};  // per-quantizer effective
    std::array<ggml_tensor*, 32> q_oproj_b_    {};
    ggml_tensor*                quant_oproj_w_ = nullptr;
    ggml_tensor*                quant_oproj_b_ = nullptr;
    std::array<Stage, 4>        stages_        {};

    // ── Encoder-only effective weights ───────────────────────────────────
    // Per-quantizer iproj: 512 → 8 (Conv1d kernel=1, weight-normed).
    // Global quantizer iproj: 768 → 512.
    // Normalized codebooks: (8, 1024) L2-normalized rows for cosine-sim argmax.
    std::array<ggml_tensor*, 32> q_iproj_w_        {};
    std::array<ggml_tensor*, 32> q_iproj_b_        {};
    ggml_tensor*                quant_iproj_w_     = nullptr;
    ggml_tensor*                quant_iproj_b_     = nullptr;
    std::array<ggml_tensor*, 32> codebook_normed_  {};
    std::array<Stage, 4>        enc_stages_        {};

    // ── Helpers ─────────────────────────────────────────────────────────
    ggml_tensor* tensor_(const std::string& name) const;
    ggml_tensor* tensor_or_null_(const std::string& name) const;

    void resolve_decoder_();
    void resolve_encoder_();
    void compute_effective_weights_();
    void compute_encoder_effective_weights_();
    void compute_normalized_codebooks_();

    // Helpers shared by compute_effective_weights_ and the encoder variant:
    //   read_f16_host_  — pull f16 bytes from a host mmap or device tensor
    //   reconstruct_wn_ — build effective Conv1d weight from wp0/wp1/bias
    void read_f16_host_(ggml_tensor* t, std::vector<uint16_t>& out) const;
    void reconstruct_wn_(const std::string& wp0_name,
                          const std::string& wp1_name,
                          const std::string& bias_name,
                          ggml_tensor* dst_w,
                          ggml_tensor* dst_b,
                          int in_dim,
                          int out_dim);

    // Allocate device-side copies of every tensor in tensor_srcs_, upload
    // their bytes, and repoint each TensorSrc::field_ptr at the device copy.
    // Called once after resolve_decoder_ + resolve_encoder_ have populated
    // tensor_srcs_ (so a single alloc_ctx_tensors handles all weights).
    void device_copy_src_tensors_(const std::string& tag);

    // Allocate one effective-weight tensor (f16) in w_ctx_. No data yet.
    ggml_tensor* make_eff_w_(const std::string& name, int in_dim, int out_dim);
    ggml_tensor* make_eff_b_(const std::string& name, int out_dim);

    // Graph builders — adapted verbatim from openmoss.
    ggml_tensor* build_layer_norm_(ggml_context* gctx, ggml_tensor* x,
                                    ggml_tensor* w, ggml_tensor* b) const;
    ggml_tensor* build_attention_(ggml_context* gctx, ggml_tensor* x,
                                   const Layer& L, int d_model, int n_heads,
                                   ggml_tensor* pos,
                                   ggml_tensor* mask) const;
    ggml_tensor* build_ffn_(ggml_context* gctx, ggml_tensor* x,
                             const Layer& L) const;
    ggml_tensor* build_layer_(ggml_context* gctx, ggml_tensor* x,
                               const Layer& L, int d_model, int n_heads,
                               ggml_tensor* pos, ggml_tensor* mask) const;
    ggml_tensor* build_stage_(ggml_context* gctx, ggml_tensor* x,
                               const Stage& S, ggml_tensor* pos,
                               ggml_tensor* mask) const;
    static ggml_tensor* make_causal_mask_(ggml_context* gctx, int64_t T);
    static void         fill_causal_mask_(ggml_tensor* mask);
    static ggml_tensor* patch_upsample_(ggml_context* gctx,
                                         ggml_tensor* x, int patch);
    static ggml_tensor* patch_downsample_(ggml_context* gctx,
                                          ggml_tensor* x, int patch);
};

}  // namespace audiocore::moss

#endif  // AUDIOCORE_MODELS_MOSS_TTS_CODEC_H
