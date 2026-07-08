// codec.cpp — MOSS-Audio-Tokenizer decoder, ggml port.
//
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from pwilkin/openmoss/src/codec.cpp (Apache-2.0). The original
// copyright belongs to the openmoss contributors; this file is the
// audiocore-side port described in docs/CODEC_PORTS.md §1.
//
// Decoder graph:
//
//   codes (32, T_audio) int32
//     │
//     │ per quantizer i ∈ [0,32):
//     │   codebook lookup → out_proj (Conv1d 8→512, kernel=1, weight-normed)
//     │   sum into single (512, T) tensor
//     ▼
//   quantizer.output_proj (Conv1d 512→768, weight-normed)
//     │
//     ▼
//   4 ProjectedTransformer stages, each followed by patch upsample:
//     dec.0: in=768  d=1280 nh=20 dff=5120 nl=32 out=1280  → patch=2 → (640, 2·T)
//     dec.2: in=640  d=768  nh=12 dff=3072 nl=12 out=768   → patch=2 → (384, 4·T)
//     dec.4: in=384  d=768  nh=12 dff=3072 nl=12 out=768   → patch=2 → (384, 8·T)
//     dec.6: in=384  d=768  nh=12 dff=3072 nl=12 out=240   → patch=240 → (1, 8·240·T)
//     │
//     ▼
//   reshape_1d → waveform (T_audio * 1920,)  ≈ 80 ms per codec frame at 24 kHz
//
// All transformer layers are pre-LN with fused QKV, RoPE (interleaved-pair,
// max_period=10000), causal self-attention via ggml_flash_attn_ext, GELU FFN,
// and per-channel LayerScale on the residual branches. The quantizer
// projections are stored upstream as weight_norm (wp0, wp1) and materialised
// here once at bind time as plain f16 weights.

#include "audiocore/models/moss_tts/codec.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace audiocore::moss {

namespace {

// ── Architecture constants (openmoss codec.cpp lines 60–84) ───────────────
constexpr float NORM_EPS       = 1e-5f;
constexpr float ROPE_FREQ_BASE = 10000.0f;

// The four decoder stages, openmoss codec.cpp line 78–84. patch_after is
// the upsample factor that follows the stage (matches the original
// PatchedPretransform layout). Uses MossCodecGraphs::StageSpec (the single
// declaration of the struct, defined in the header).
constexpr std::array<MossCodecGraphs::StageSpec, 4> DECODER_STAGES = {{
    //   in   d   nh   dff  nl  out  patch  gguf
    {  768, 1280, 20, 5120, 32, 1280,    2,    0 },  // dec.0 + dec.1 patch
    {  640,  768, 12, 3072, 12,  768,    2,    2 },  // dec.2 + dec.3 patch
    {  384,  768, 12, 3072, 12,  768,    2,    4 },  // dec.4 + dec.5 patch
    {  384,  768, 12, 3072, 12,  240,  240,    6 },  // dec.6 + dec.7 patch
}};

// Encoder mirrors the decoder. The initial patch=240 happens BEFORE the
// first transformer stage; patch_after here is the patch that follows
// the stage (openmoss codec.cpp line 89–95).
constexpr std::array<MossCodecGraphs::StageSpec, 4> ENCODER_STAGES = {{
    //   in   d   nh   dff  nl  out  patch  gguf
    {  240,  768, 12, 3072, 12,  384,    2,    1 },  // enc.1 + enc.2 patch
    {  768,  768, 12, 3072, 12,  384,    2,    3 },  // enc.3 + enc.4 patch
    {  768,  768, 12, 3072, 12,  640,    2,    5 },  // enc.5 + enc.6 patch
    { 1280, 1280, 20, 5120, 32,  768,    0,    7 },  // enc.7 (no patch after)
}};

// ── f16 helpers (openmoss codec.cpp lines 98–151) ─────────────────────────
// The host-side weight-norm reconstruction runs against f16 bit patterns
// directly, so we replicate openmoss's exact conversions to guarantee
// bit-identical results to upstream.
inline uint16_t f32_to_f16_bits(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    uint32_t sign = (u >> 31) & 0x1;
    int32_t  exp  = int32_t((u >> 23) & 0xff) - 127 + 15;
    uint32_t mant = u & 0x7fffff;
    uint16_t out;
    if (exp <= 0) {
        if (exp < -10) { out = uint16_t(sign << 15); }
        else {
            mant = (mant | 0x800000) >> uint32_t(1 - exp);
            if (mant & 0x1000) mant += 0x2000;
            out = uint16_t((sign << 15) | (mant >> 13));
        }
    } else if (exp >= 0x1f) {
        out = uint16_t((sign << 15) | (0x1f << 10) | (mant ? 0x200 : 0));
    } else {
        if (mant & 0x1000) {
            mant += 0x2000;
            if (mant & 0x800000) { mant = 0; exp += 1; }
            if (exp >= 0x1f) {
                out = uint16_t((sign << 15) | (0x1f << 10));
                return out;
            }
        }
        out = uint16_t((sign << 15) | (uint32_t(exp) << 10) | (mant >> 13));
    }
    return out;
}

inline float f16_bits_to_f32(uint16_t h) {
    uint32_t sign = uint32_t(h >> 15) & 0x1;
    int32_t  exp  = int32_t((h >> 10) & 0x1f);
    uint32_t mant = uint32_t(h & 0x3ff);
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) {
            u = sign << 31;
        } else {
            int32_t e = -1;
            do { e++; mant <<= 1; } while ((mant & 0x400) == 0);
            mant &= 0x3ff;
            u = (sign << 31) | (uint32_t(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        u = (sign << 31) | (0xffu << 23) | (mant << 13);
    } else {
        u = (sign << 31) | (uint32_t(exp - 15 + 127) << 23) | (mant << 13);
    }
    float f; std::memcpy(&f, &u, 4); return f;
}

// Cast an f16 weight constant to f32 in the graph. The CUDA bin_bcast kernel
// only supports (F32, F32, F32) for the F32 dst case; mixing f16 weights into
// an f32 activation aborts. (openmoss codec.cpp lines 560–565.)
ggml_tensor* to_f32_(ggml_context* gctx, ggml_tensor* t) {
    return (t->type == GGML_TYPE_F32) ? t : ggml_cast(gctx, t, GGML_TYPE_F32);
}

}  // namespace

// ───────────────────────────────────────────────────────────────────────────
// Lifecycle
// ───────────────────────────────────────────────────────────────────────────

MossCodecGraphs::~MossCodecGraphs() {
    if (galloc_) ggml_gallocr_free(galloc_);
    if (w_buf_)  ggml_backend_buffer_free(w_buf_);
    if (w_ctx_)  ggml_free(w_ctx_);
}

bool MossCodecGraphs::bind(ggml_context* source_ctx,
                            ggml_backend_t backend,
                            int32_t n_vq,
                            std::string* error) {
    if (present_) return true;
    source_ctx_ = source_ctx;
    backend_    = backend;
    n_vq_       = n_vq;
    if (!source_ctx_ || !backend_) {
        if (error) *error = "MossCodecGraphs::bind: nullptr source_ctx or backend";
        return false;
    }
    if (n_vq_ < 1 || n_vq_ > 32) {
        if (error) *error = "MossCodecGraphs::bind: n_vq out of range [1, 32]: "
                            + std::to_string(n_vq_);
        return false;
    }
    try {
        // Resolve source-tensor pointers first so a single
        // ggml_backend_alloc_ctx_tensors pass covers every descriptor we'll
        // add to w_ctx_ (decoder effective weights + device copies +, when
        // available, encoder effective weights + normalized codebooks).
        resolve_decoder_();

        bool want_encoder = true;
        try {
            resolve_encoder_();
        } catch (const std::exception& e) {
            // Encoder is optional — present iff moss.codec.enc.* tensors
            // live in the GGUF. openmoss's own converted GGUFs always
            // carry them; some community backbone-only splits don't.
            std::fprintf(stderr, "MossCodecGraphs::bind: encoder unavailable (%s)\n",
                         e.what());
            want_encoder = false;
        }

        // Init w_ctx_ — sized for decoder + encoder + device copies.
        {
            ggml_init_params ip{};
            ip.mem_size   = ggml_tensor_overhead() * 4000;
            ip.mem_buffer = nullptr;
            ip.no_alloc   = true;
            w_ctx_ = ggml_init(ip);
            if (!w_ctx_) throw std::runtime_error("bind: ggml_init for weight ctx failed");
        }

        // Resize per-quantizer vectors to n_vq_.
        q_oproj_w_.resize(n_vq_);
        q_oproj_b_.resize(n_vq_);

        // Allocate decoder effective-weight descriptors.
        for (int i = 0; i < n_vq_; ++i) {
            const std::string n = "q.oproj." + std::to_string(i);
            q_oproj_w_[i] = make_eff_w_(n + ".w", CODEC_CB_DIM, CODEC_RVQ_DIM);
            q_oproj_b_[i] = make_eff_b_(n + ".b", CODEC_RVQ_DIM);
        }
        quant_oproj_w_ = make_eff_w_("quant.oproj.w", CODEC_RVQ_DIM, CODEC_OUT_DIM);
        quant_oproj_b_ = make_eff_b_("quant.oproj.b", CODEC_OUT_DIM);

        // Allocate encoder effective-weight descriptors (if encoder resolved).
        if (want_encoder) {
            q_iproj_w_.resize(n_vq_);
            q_iproj_b_.resize(n_vq_);
            codebook_normed_.resize(n_vq_);

            for (int i = 0; i < n_vq_; ++i) {
                const std::string n = "q.iproj." + std::to_string(i);
                q_iproj_w_[i] = make_eff_w_(n + ".w", CODEC_RVQ_DIM, CODEC_CB_DIM);
                q_iproj_b_[i] = make_eff_b_(n + ".b", CODEC_CB_DIM);
            }
            quant_iproj_w_ = make_eff_w_("quant.iproj.w", CODEC_OUT_DIM, CODEC_RVQ_DIM);
            quant_iproj_b_ = make_eff_b_("quant.iproj.b", CODEC_RVQ_DIM);

            for (int i = 0; i < n_vq_; ++i) {
                ggml_tensor* t = ggml_new_tensor_2d(w_ctx_, GGML_TYPE_F16,
                                                     CODEC_CB_DIM, CODEC_CB_SIZE);
                ggml_set_name(t, ("cb_norm." + std::to_string(i)).c_str());
                codebook_normed_[i] = t;
            }
        }

        // Allocate device copies of all source tensors, upload bytes, repoint
        // field pointers. This also allocates w_buf_ — one pass, covers every
        // descriptor added so far (effective weights + device copies).
        device_copy_src_tensors_("MossCodecGraphs::bind");

        // Reconstruct decoder effective weights (reads from on-device wp0/wp1).
        for (int i = 0; i < n_vq_; ++i) {
            const std::string base = "moss.codec.quantizer.q." + std::to_string(i) + ".oproj.";
            reconstruct_wn_(base + "wp0", base + "wp1", base + "bias",
                            q_oproj_w_[i], q_oproj_b_[i],
                            CODEC_CB_DIM, CODEC_RVQ_DIM);
        }
        reconstruct_wn_("moss.codec.quantizer.oproj.wp0",
                        "moss.codec.quantizer.oproj.wp1",
                        "moss.codec.quantizer.oproj.bias",
                        quant_oproj_w_, quant_oproj_b_,
                        CODEC_RVQ_DIM, CODEC_OUT_DIM);

        if (want_encoder) {
            // Reconstruct encoder effective weights.
            for (int i = 0; i < n_vq_; ++i) {
                const std::string base = "moss.codec.quantizer.q." + std::to_string(i) + ".iproj.";
                reconstruct_wn_(base + "wp0", base + "wp1", base + "bias",
                                q_iproj_w_[i], q_iproj_b_[i],
                                CODEC_RVQ_DIM, CODEC_CB_DIM);
            }
            reconstruct_wn_("moss.codec.quantizer.iproj.wp0",
                            "moss.codec.quantizer.iproj.wp1",
                            "moss.codec.quantizer.iproj.bias",
                            quant_iproj_w_, quant_iproj_b_,
                            CODEC_OUT_DIM, CODEC_RVQ_DIM);
            compute_normalized_codebooks_();
            encoder_present_ = true;
        }
    } catch (const std::exception& e) {
        if (error) *error = std::string("MossCodecGraphs::bind: ") + e.what();
        return false;
    }
    galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (!galloc_) {
        if (error) *error = "MossCodecGraphs::bind: gallocr_new failed";
        return false;
    }
    present_ = true;
    return true;
}

ggml_tensor* MossCodecGraphs::tensor_(const std::string& name) const {
    ggml_tensor* t = ggml_get_tensor(source_ctx_, name.c_str());
    if (!t) {
        throw std::runtime_error("MossCodecGraphs: missing codec tensor: " + name);
    }
    return t;
}

ggml_tensor* MossCodecGraphs::tensor_or_null_(const std::string& name) const {
    return ggml_get_tensor(source_ctx_, name.c_str());
}

// ── Macro to register a tensor pointer for device copy ──────────────────────
// Called in resolve_decoder_() after assigning each member field.
#define REG_SRC(member_ptr) do { \
    if ((member_ptr)) { \
        tensor_srcs_.push_back({&(member_ptr), (member_ptr)}); \
    } \
} while(0)

// ───────────────────────────────────────────────────────────────────────────
// Tensor resolution
// ───────────────────────────────────────────────────────────────────────────

void MossCodecGraphs::resolve_decoder_() {
    codebook_.resize(n_vq_);
    // Codebooks: moss.codec.quantizer.q.{i}.codebook.weight  (8, 1024) f16
    for (int i = 0; i < n_vq_; ++i) {
        codebook_[i] = tensor_("moss.codec.quantizer.q." + std::to_string(i)
                                + ".codebook.weight");
        REG_SRC(codebook_[i]);
    }

    // Decoder stages.
    for (size_t s = 0; s < stages_.size(); ++s) {
        Stage& S = stages_[s];
        S.spec = DECODER_STAGES[s];
        const std::string base = "moss.codec.dec." + std::to_string(S.spec.gguf_idx) + ".";

        // Optional input/output projections (Identity in PyTorch when dim matches).
        S.iproj = (S.spec.d_model != S.spec.input_dim)
            ? tensor_(base + "iproj.weight") : nullptr;
        REG_SRC(S.iproj);
        S.oproj = (S.spec.d_model != S.spec.output_dim)
            ? tensor_(base + "oproj.weight") : nullptr;
        REG_SRC(S.oproj);

        S.layers.resize(size_t(S.spec.n_layers));
        for (int li = 0; li < S.spec.n_layers; ++li) {
            const std::string lb = base + "tr.l." + std::to_string(li) + ".";
            Layer& L = S.layers[size_t(li)];
            L.norm1_w       = tensor_(lb + "norm1.weight");       REG_SRC(L.norm1_w);
            L.norm1_b       = tensor_(lb + "norm1.bias");         REG_SRC(L.norm1_b);
            L.norm2_w       = tensor_(lb + "norm2.weight");       REG_SRC(L.norm2_w);
            L.norm2_b       = tensor_(lb + "norm2.bias");         REG_SRC(L.norm2_b);
            L.attn_in       = tensor_(lb + "attn.inp.0.weight");  REG_SRC(L.attn_in);
            L.attn_out      = tensor_(lb + "attn.outp.0.weight"); REG_SRC(L.attn_out);
            L.linear1       = tensor_(lb + "linear1.weight");     REG_SRC(L.linear1);
            L.linear2       = tensor_(lb + "linear2.weight");     REG_SRC(L.linear2);
            L.layer_scale_1 = tensor_(lb + "layer_scale_1.scale"); REG_SRC(L.layer_scale_1);
            L.layer_scale_2 = tensor_(lb + "layer_scale_2.scale"); REG_SRC(L.layer_scale_2);
        }
    }
}

// Resolve encoder weights (mirror of resolve_decoder_). Throws on the first
// missing tensor — bind() catches that and downgrades to "decoder-only" mode.
// Adapted from openmoss codec.cpp resolve_encoder_, lines 331–358.
void MossCodecGraphs::resolve_encoder_() {
    for (size_t s = 0; s < enc_stages_.size(); ++s) {
        Stage& S = enc_stages_[s];
        S.spec = ENCODER_STAGES[s];
        const std::string base = "moss.codec.enc." + std::to_string(S.spec.gguf_idx) + ".";

        S.iproj = (S.spec.d_model != S.spec.input_dim)
            ? tensor_(base + "iproj.weight") : nullptr;
        REG_SRC(S.iproj);
        S.oproj = (S.spec.d_model != S.spec.output_dim)
            ? tensor_(base + "oproj.weight") : nullptr;
        REG_SRC(S.oproj);

        S.layers.resize(size_t(S.spec.n_layers));
        for (int li = 0; li < S.spec.n_layers; ++li) {
            const std::string lb = base + "tr.l." + std::to_string(li) + ".";
            Layer& L = S.layers[size_t(li)];
            L.norm1_w       = tensor_(lb + "norm1.weight");       REG_SRC(L.norm1_w);
            L.norm1_b       = tensor_(lb + "norm1.bias");         REG_SRC(L.norm1_b);
            L.norm2_w       = tensor_(lb + "norm2.weight");       REG_SRC(L.norm2_w);
            L.norm2_b       = tensor_(lb + "norm2.bias");         REG_SRC(L.norm2_b);
            L.attn_in       = tensor_(lb + "attn.inp.0.weight");  REG_SRC(L.attn_in);
            L.attn_out      = tensor_(lb + "attn.outp.0.weight"); REG_SRC(L.attn_out);
            L.linear1       = tensor_(lb + "linear1.weight");     REG_SRC(L.linear1);
            L.linear2       = tensor_(lb + "linear2.weight");     REG_SRC(L.linear2);
            L.layer_scale_1 = tensor_(lb + "layer_scale_1.scale"); REG_SRC(L.layer_scale_1);
            L.layer_scale_2 = tensor_(lb + "layer_scale_2.scale"); REG_SRC(L.layer_scale_2);
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Weight-norm reconstruction
// ───────────────────────────────────────────────────────────────────────────
//
// The upstream model stores each Conv1d projection as `weight_norm`-decorated
// parameters: `wp0` (magnitude, shape (out, 1, 1)) and `wp1` (direction, shape
// (out, in, 1)). We materialise the effective weight `w[o,i] = wp0[o] *
// wp1[o,i] / sqrt(Σi wp1[o,i]^2)` on the host once at bind time and upload it
// as a fresh f16 tensor on the backend.
//
// compute_effective_weights_ and compute_encoder_effective_weights_ are now
// folded into bind() (which owns the single alloc_ctx_tensors pass that has
// to happen with every descriptor present). The shared reconstruction
// helpers live below.

void MossCodecGraphs::compute_effective_weights_() {
    // Legacy entry — retained for any external caller that only wanted the
    // decoder. bind() does this work inline so this just delegates.
    if (present_) return;
    // bind() is the single supported entry point now.
}

void MossCodecGraphs::compute_encoder_effective_weights_() {
    // Legacy entry — bind() does this work inline now.
}

// Pre-compute L2-normalized codebooks (per-row L2 normalization). Used during
// encoding to compute cosine similarity via a single matmul against the
// normalized codebook rows. Adapted verbatim from openmoss codec.cpp lines
// 528–554.
void MossCodecGraphs::compute_normalized_codebooks_() {
    for (int i = 0; i < n_vq_; ++i) {
        if (!codebook_normed_[i]) {
            throw std::runtime_error("compute_normalized_codebooks_: codebook_normed_[" +
                                      std::to_string(i) + "] not allocated");
        }
        std::vector<uint16_t> cb_f16;
        read_f16_host_(codebook_[i], cb_f16);
        if (cb_f16.size() != size_t(CODEC_CB_SIZE) * size_t(CODEC_CB_DIM)) {
            throw std::runtime_error("compute_normalized_codebooks_: shape mismatch for q." +
                                      std::to_string(i) + ".codebook");
        }
        std::vector<uint16_t> cb_norm(cb_f16.size());
        for (int r = 0; r < CODEC_CB_SIZE; ++r) {
            float ssq = 0.0f;
            for (int c = 0; c < CODEC_CB_DIM; ++c) {
                float v = f16_bits_to_f32(cb_f16[size_t(r) * CODEC_CB_DIM + size_t(c)]);
                ssq += v * v;
            }
            float inv = (ssq > 0.0f) ? 1.0f / std::sqrt(ssq) : 0.0f;
            for (int c = 0; c < CODEC_CB_DIM; ++c) {
                float v = f16_bits_to_f32(cb_f16[size_t(r) * CODEC_CB_DIM + size_t(c)]);
                cb_norm[size_t(r) * CODEC_CB_DIM + size_t(c)] = f32_to_f16_bits(v * inv);
            }
        }
        ggml_backend_tensor_set(codebook_normed_[i], cb_norm.data(), 0,
                                cb_norm.size() * sizeof(uint16_t));
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Shared helpers — file-local lambda equivalents in the original port, made
// member functions so the encoder path reuses them.
// ───────────────────────────────────────────────────────────────────────────

ggml_tensor* MossCodecGraphs::make_eff_w_(const std::string& name,
                                            int in_dim, int out_dim) {
    ggml_tensor* t = ggml_new_tensor_2d(w_ctx_, GGML_TYPE_F16, in_dim, out_dim);
    ggml_set_name(t, name.c_str());
    return t;
}

ggml_tensor* MossCodecGraphs::make_eff_b_(const std::string& name, int out_dim) {
    ggml_tensor* t = ggml_new_tensor_1d(w_ctx_, GGML_TYPE_F16, out_dim);
    ggml_set_name(t, name.c_str());
    return t;
}

void MossCodecGraphs::read_f16_host_(ggml_tensor* t,
                                      std::vector<uint16_t>& out) const {
    if (!t) throw std::runtime_error("read_f16_host_: null tensor");
    if (t->type != GGML_TYPE_F16) {
        throw std::runtime_error(std::string("read_f16_host_: expected f16 for ")
                                 + (t->name[0] ? t->name : "<unnamed>"));
    }
    const size_t n = ggml_nelements(t);
    out.resize(n);
    if (t->buffer) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(uint16_t));
    } else if (t->data) {
        std::memcpy(out.data(), t->data, n * sizeof(uint16_t));
    } else {
        throw std::runtime_error(std::string("read_f16_host_: no buffer and no data for ")
                                 + (t->name[0] ? std::string(t->name) : "<unnamed>"));
    }
}

void MossCodecGraphs::reconstruct_wn_(const std::string& wp0_name,
                                        const std::string& wp1_name,
                                        const std::string& bias_name,
                                        ggml_tensor* dst_w,
                                        ggml_tensor* dst_b,
                                        int in_dim,
                                        int out_dim) {
    ggml_tensor* wp0_t  = tensor_(wp0_name);
    ggml_tensor* wp1_t  = tensor_(wp1_name);
    ggml_tensor* bias_t = tensor_or_null_(bias_name);

    if (ggml_nelements(wp0_t) != out_dim) {
        throw std::runtime_error("reconstruct_wn: wp0 shape mismatch for " + wp0_name);
    }
    if (ggml_nelements(wp1_t) != int64_t(in_dim) * int64_t(out_dim)) {
        throw std::runtime_error("reconstruct_wn: wp1 shape mismatch for " + wp1_name);
    }

    std::vector<uint16_t> wp0_f16, wp1_f16, bias_f16;
    read_f16_host_(wp0_t, wp0_f16);
    read_f16_host_(wp1_t, wp1_f16);
    if (bias_t) read_f16_host_(bias_t, bias_f16);

    std::vector<uint16_t> w_eff(size_t(in_dim) * size_t(out_dim));
    for (int o = 0; o < out_dim; ++o) {
        float g = f16_bits_to_f32(wp0_f16[size_t(o)]);
        float ssq = 0.0f;
        for (int i = 0; i < in_dim; ++i) {
            float v = f16_bits_to_f32(wp1_f16[size_t(o) * size_t(in_dim) + size_t(i)]);
            ssq += v * v;
        }
        float inv = (ssq > 0.0f) ? 1.0f / std::sqrt(ssq) : 0.0f;
        float scale = g * inv;
        for (int i = 0; i < in_dim; ++i) {
            float v = f16_bits_to_f32(wp1_f16[size_t(o) * size_t(in_dim) + size_t(i)]);
            w_eff[size_t(o) * size_t(in_dim) + size_t(i)] = f32_to_f16_bits(scale * v);
        }
    }

    ggml_backend_tensor_set(dst_w, w_eff.data(), 0, w_eff.size() * sizeof(uint16_t));
    if (bias_t && dst_b) {
        ggml_backend_tensor_set(dst_b, bias_f16.data(), 0,
                                bias_f16.size() * sizeof(uint16_t));
    } else if (dst_b) {
        std::vector<uint16_t> zero(out_dim, 0);
        ggml_backend_tensor_set(dst_b, zero.data(), 0,
                                zero.size() * sizeof(uint16_t));
    }
}

// Allocate device copies of every tensor in tensor_srcs_, upload bytes from
// the host mmap into them, and repoint each TensorSrc::field_ptr at the
// device copy. After this runs, the graph builders automatically reference
// device memory.
//
// Must be called BEFORE reconstruct_wn_ — the wp0/wp1/bias tensors the
// reconstruct helper reads must already be on the device (or readable via
// host pointer, which the source_ctx_ mmap provides — but uploading once
// here avoids the host-read path entirely except where we explicitly
// compute effective weights).
void MossCodecGraphs::device_copy_src_tensors_(const std::string& tag) {
    const size_t n_src = tensor_srcs_.size();
    std::vector<ggml_tensor*> dst_tensors(n_src, nullptr);
    for (size_t i = 0; i < n_src; ++i) {
        ggml_tensor* src = tensor_srcs_[i].src;
        const int nd = ggml_n_dims(src);
        const int64_t ne_arr[] = {src->ne[0], src->ne[1], src->ne[2], src->ne[3]};
        ggml_tensor* dst = ggml_new_tensor(w_ctx_, src->type, nd, ne_arr);
        if (src->name[0]) {
            ggml_set_name(dst, (std::string("d_") + src->name).c_str());
        } else {
            ggml_set_name(dst, ("d_src_" + std::to_string(i)).c_str());
        }
        dst_tensors[i] = dst;
    }

    // Allocate backing storage on the backend for ALL w_ctx_ tensors at once.
    if (!w_buf_) {
        w_buf_ = ggml_backend_alloc_ctx_tensors(w_ctx_, backend_);
        if (!w_buf_) {
            throw std::runtime_error(tag + ": alloc_ctx_tensors for weight buf failed");
        }
    } else {
        // w_buf_ already exists (encoder weights added after decoder). Re-alloc
        // is not idempotent; ggml_backend_alloc_ctx_tensors must run on a fresh
        // context. So we can't extend — only the single-call pattern works.
        // The bind() sequence ensures encoder weights are allocated in the same
        // pass: compute_effective_weights_ runs first, allocating w_buf_, then
        // compute_encoder_effective_weights_ adds encoder descriptors and calls
        // device_copy_src_tensors_ which does NOT re-alloc. The descriptors
        // made by make_eff_w_ will need backing storage — handle that by
        // allocating fresh ctx tensors for just the new descriptors.
        //
        // In practice this code path is only reached if you call bind() twice
        // (which is idempotent — see the present_ guard in bind()).
    }

    for (size_t i = 0; i < n_src; ++i) {
        ggml_tensor* dst = dst_tensors[i];
        ggml_tensor* src = tensor_srcs_[i].src;
        const size_t nbytes = ggml_nbytes(src);
        ggml_backend_tensor_set(dst, src->data, 0, nbytes);
        *tensor_srcs_[i].field_ptr = dst;
    }
    tensor_srcs_.clear();
}

// ───────────────────────────────────────────────────────────────────────────
// Graph builders (verbatim from openmoss codec.cpp lines 560–742)
// ───────────────────────────────────────────────────────────────────────────

ggml_tensor* MossCodecGraphs::build_layer_norm_(ggml_context* gctx,
                                                  ggml_tensor* x,
                                                  ggml_tensor* w,
                                                  ggml_tensor* b) const {
    ggml_tensor* y = ggml_norm(gctx, x, NORM_EPS);
    y = ggml_mul(gctx, y, to_f32_(gctx, w));
    y = ggml_add(gctx, y, to_f32_(gctx, b));
    return y;
}

// Self-attention: x is (d_model, T). Returns (d_model, T).
//   - Fused QKV via a single mul_mat: (3*d_model, T)
//   - Split into Q, K, V each (head_dim, n_heads, T) via 3D views into the qkv buffer
//   - RoPE on Q, K (interleaved-pair convention)
//   - ggml_flash_attn_ext (causal) — streams the softmax, no O(T²) scores buffer
//   - Output projection
ggml_tensor* MossCodecGraphs::build_attention_(ggml_context* gctx,
                                                  ggml_tensor* x,
                                                  const Layer& L,
                                                  int d_model,
                                                  int n_heads,
                                                  ggml_tensor* pos,
                                                  ggml_tensor* mask) const {
    const int head_dim = d_model / n_heads;
    const int T        = int(x->ne[1]);

    ggml_tensor* qkv = ggml_mul_mat(gctx, L.attn_in, x);   // (3*d_model, T)

    const size_t e        = ggml_type_size(qkv->type);
    const size_t row_size = size_t(head_dim) * e;
    const size_t qkv_nb1  = qkv->nb[1];

    ggml_tensor* Q = ggml_view_3d(gctx, qkv, head_dim, n_heads, T,
                                    row_size, qkv_nb1, 0);
    ggml_tensor* K = ggml_view_3d(gctx, qkv, head_dim, n_heads, T,
                                    row_size, qkv_nb1, size_t(d_model) * e);
    ggml_tensor* V = ggml_view_3d(gctx, qkv, head_dim, n_heads, T,
                                    row_size, qkv_nb1, size_t(2 * d_model) * e);

    Q = ggml_cont(gctx, Q);
    K = ggml_cont(gctx, K);
    V = ggml_cont(gctx, V);

    // RoPE — interleaved-pair convention, matches MOSS' q.view(*, D//2, 2).
    Q = ggml_rope_ext(gctx, Q, pos, /*c=*/nullptr,
                       head_dim, GGML_ROPE_TYPE_NORMAL, /*n_ctx_orig=*/T,
                       ROPE_FREQ_BASE, /*freq_scale=*/1.0f,
                       /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
                       /*beta_fast=*/0.0f, /*beta_slow=*/0.0f);
    K = ggml_rope_ext(gctx, K, pos, nullptr,
                       head_dim, GGML_ROPE_TYPE_NORMAL, T,
                       ROPE_FREQ_BASE, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // Layout flash_attn_ext expects: q/k/v = (head_dim, T, n_heads).
    ggml_tensor* Qp = ggml_cont(gctx, ggml_permute(gctx, Q, 0, 2, 1, 3));
    ggml_tensor* Kp = ggml_cont(gctx, ggml_permute(gctx, K, 0, 2, 1, 3));
    ggml_tensor* Vp = ggml_cont(gctx, ggml_permute(gctx, V, 0, 2, 1, 3));

    // K/V cast to f16 to match the llama.cpp KV path; Q stays f32.
    ggml_tensor* Kh = ggml_cast(gctx, Kp, GGML_TYPE_F16);
    ggml_tensor* Vh = ggml_cast(gctx, Vp, GGML_TYPE_F16);

    ggml_tensor* attn = ggml_flash_attn_ext(gctx, Qp, Kh, Vh, mask,
                                              1.0f / std::sqrt(float(head_dim)),
                                              /*max_bias=*/0.0f, /*logit_softcap=*/0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    // result: (head_dim, n_heads, T) contiguous → flatten heads into d_model.
    attn = ggml_reshape_2d(gctx, attn, d_model, T);

    // output projection
    attn = ggml_mul_mat(gctx, L.attn_out, attn);
    return attn;
}

ggml_tensor* MossCodecGraphs::build_ffn_(ggml_context* gctx,
                                           ggml_tensor* x,
                                           const Layer& L) const {
    ggml_tensor* y = ggml_mul_mat(gctx, L.linear1, x);
    y = ggml_gelu(gctx, y);
    y = ggml_mul_mat(gctx, L.linear2, y);
    return y;
}

ggml_tensor* MossCodecGraphs::build_layer_(ggml_context* gctx,
                                             ggml_tensor* x,
                                             const Layer& L,
                                             int d_model,
                                             int n_heads,
                                             ggml_tensor* pos,
                                             ggml_tensor* mask) const {
    // attn block
    ggml_tensor* y = build_layer_norm_(gctx, x, L.norm1_w, L.norm1_b);
    y = build_attention_(gctx, y, L, d_model, n_heads, pos, mask);
    y = ggml_mul(gctx, y, to_f32_(gctx, L.layer_scale_1));
    x = ggml_add(gctx, x, y);

    // ffn block
    y = build_layer_norm_(gctx, x, L.norm2_w, L.norm2_b);
    y = build_ffn_(gctx, y, L);
    y = ggml_mul(gctx, y, to_f32_(gctx, L.layer_scale_2));
    x = ggml_add(gctx, x, y);

    return x;
}

ggml_tensor* MossCodecGraphs::build_stage_(ggml_context* gctx,
                                              ggml_tensor* x,
                                              const Stage& S,
                                              ggml_tensor* pos,
                                              ggml_tensor* mask) const {
    if (S.iproj) x = ggml_mul_mat(gctx, S.iproj, x);
    for (const Layer& L : S.layers) {
        x = build_layer_(gctx, x, L, S.spec.d_model, S.spec.n_heads, pos, mask);
    }
    if (S.oproj) x = ggml_mul_mat(gctx, S.oproj, x);
    return x;
}

ggml_tensor* MossCodecGraphs::make_causal_mask_(ggml_context* gctx, int64_t T) {
    ggml_tensor* m = ggml_new_tensor_2d(gctx, GGML_TYPE_F16, T, T);
    ggml_set_input(m);
    return m;
}

void MossCodecGraphs::fill_causal_mask_(ggml_tensor* mask) {
    const int64_t T = mask->ne[0];
    static const uint16_t F16_ZERO    = 0x0000;
    static const uint16_t F16_NEG_INF = 0xFC00;
    std::vector<uint16_t> buf(size_t(T) * size_t(T));
    for (int64_t q = 0; q < T; ++q) {
        uint16_t* row = buf.data() + size_t(q) * size_t(T);
        for (int64_t kv = 0; kv < T; ++kv) {
            row[kv] = (kv <= q) ? F16_ZERO : F16_NEG_INF;
        }
    }
    ggml_backend_tensor_set(mask, buf.data(), 0, buf.size() * sizeof(uint16_t));
}

// PyTorch:  x.reshape(b, d, h, l).permute(0, 1, 3, 2).reshape(b, d, l*h)
//   input  ne=(d*h, T_in)
//   output ne=(d,   T_in*h)
ggml_tensor* MossCodecGraphs::patch_upsample_(ggml_context* gctx,
                                                ggml_tensor* x, int patch) {
    const int64_t dh   = x->ne[0];
    const int64_t T_in = x->ne[1];
    if (dh % patch != 0) {
        throw std::runtime_error("patch_upsample_: channel count " + std::to_string(dh)
                                  + " not divisible by patch=" + std::to_string(patch));
    }
    const int64_t d = dh / patch;
    ggml_tensor* y = ggml_reshape_3d(gctx, x, patch, d, T_in);
    y = ggml_permute(gctx, y, 1, 0, 2, 3);                  // (d, h, T_in)
    y = ggml_cont(gctx, y);                                  // contiguous
    y = ggml_reshape_2d(gctx, y, d, T_in * patch);
    return y;
}

// Inverse of patch_upsample_. PyTorch encode:
//   x.reshape(b, d, T_in/h, h).permute(0, 1, 3, 2).reshape(b, d*h, T_in/h)
//   ⇒ out[d_out = d*h + h_idx, t_out] = in[d, t_out*h + h_idx]
// In GGML (D innermost):
//   1. ggml_reshape_3d(x, D, h, T_out)        [view ne=(D, T_in) as ne=(D, h, T_out)]
//   2. ggml_permute(_, 1, 0, 2, 3)            → ne=(h, D, T_out)
//   3. ggml_cont(_)                            → contiguous (h, D, T_out)
//   4. ggml_reshape_2d(_, D*h, T_out)
ggml_tensor* MossCodecGraphs::patch_downsample_(ggml_context* gctx,
                                                  ggml_tensor* x, int patch) {
    const int64_t D = x->ne[0];
    const int64_t T_in = x->ne[1];
    if (T_in % patch != 0) {
        throw std::runtime_error("patch_downsample_: T_in " + std::to_string(T_in)
                                  + " not divisible by patch=" + std::to_string(patch));
    }
    const int64_t T_out = T_in / patch;
    ggml_tensor* y = ggml_reshape_3d(gctx, x, D, patch, T_out);
    y = ggml_permute(gctx, y, 1, 0, 2, 3);                  // (h, D, T_out)
    y = ggml_cont(gctx, y);                                  // contiguous
    y = ggml_reshape_2d(gctx, y, D * patch, T_out);
    return y;
}

// ───────────────────────────────────────────────────────────────────────────
// Decode entry point (adapted from openmoss codec.cpp lines 770–908)
// ───────────────────────────────────────────────────────────────────────────

std::vector<float> MossCodecGraphs::decode(const int32_t* codes,
                                            int32_t n_vq,
                                            int32_t T_audio) {
    if (!present_) {
        throw std::runtime_error("MossCodecGraphs::decode: not bound (codec tensors missing?)");
    }
    if (n_vq != n_vq_) {
        throw std::runtime_error("MossCodecGraphs::decode: expected n_vq="
                                  + std::to_string(n_vq_) + ", got "
                                  + std::to_string(n_vq));
    }
    if (T_audio <= 0) return {};

    // Refuse runaway generations early. The last stage runs at 8·T_audio and
    // needs an (T_last × T_last) f16 causal mask; let that exceed 8 GiB and
    // we're in OOM-territory for a doomed allocation anyway.
    {
        const int64_t T_last     = int64_t(T_audio) * 8;
        const int64_t mask_bytes = T_last * T_last * int64_t(sizeof(uint16_t));
        const int64_t cap_bytes  = int64_t(8) * 1024 * 1024 * 1024;  // ~8 GiB
        if (mask_bytes > cap_bytes) {
            const double secs = T_audio * 1920.0 / 24000.0;
            throw std::runtime_error(
                "MossCodecGraphs::decode: refusing to decode " + std::to_string(T_audio) +
                " frames (~" + std::to_string(int(secs)) + "s): the codec attention "
                "mask alone would need ~" + std::to_string(mask_bytes >> 30) +
                " GiB. This usually means generation never emitted an end-of-speech "
                "token (try a lower --max-new-tokens / different sampling).");
        }
    }

    const int n_samples = T_audio * 1920;

    // ── Build graph ─────────────────────────────────────────────────────
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * 65536 + ggml_graph_overhead_custom(65536, false);
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context* gctx = ggml_init(ip);
    if (!gctx) throw std::runtime_error("MossCodecGraphs::decode: ggml_init failed");

    // Per-quantizer code inputs.
    std::vector<ggml_tensor*> codes_in(n_vq_);
    for (int i = 0; i < n_vq_; ++i) {
        codes_in[i] = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T_audio);
        ggml_set_name(codes_in[i], ("codes_" + std::to_string(i)).c_str());
        ggml_set_input(codes_in[i]);
    }

    // Per-stage RoPE position + causal-mask inputs. T grows by patch_after
    // at each stage boundary.
    int T_at[4];
    T_at[0] = T_audio;
    T_at[1] = T_at[0] * DECODER_STAGES[0].patch_after;
    T_at[2] = T_at[1] * DECODER_STAGES[1].patch_after;
    T_at[3] = T_at[2] * DECODER_STAGES[2].patch_after;
    std::array<ggml_tensor*, 4> pos_T  {};
    std::array<ggml_tensor*, 4> mask_T {};
    for (int s = 0; s < 4; ++s) {
        pos_T[s] = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T_at[s]);
        ggml_set_name(pos_T[s], ("pos_" + std::to_string(s)).c_str());
        ggml_set_input(pos_T[s]);
        mask_T[s] = make_causal_mask_(gctx, T_at[s]);
        ggml_set_name(mask_T[s], ("mask_" + std::to_string(s)).c_str());
    }

    // ── Quantizer.decode_codes ──────────────────────────────────────────
    ggml_tensor* sum = nullptr;
    for (int i = 0; i < n_vq_; ++i) {
        // (codebook_dim=8, T) via embedding lookup → f32 (get_rows promotes).
        // codebook_[i] now points to the device copy in w_buf_ (the generic
        // device-copy mechanism in compute_effective_weights_ updated it).
        ggml_tensor* z = ggml_get_rows(gctx, codebook_[i], codes_in[i]);
        // Conv1d 8 → 512 (kernel=1). Effective weight stored as (in=8, out=512) f16.
        z = ggml_mul_mat(gctx, q_oproj_w_[i], z);                    // (512, T) f32
        z = ggml_add(gctx, z, to_f32_(gctx, q_oproj_b_[i]));         // broadcast bias
        sum = sum ? ggml_add(gctx, sum, z) : z;
    }
    // Final rvq oproj: 512 → 768.
    ggml_tensor* x = ggml_mul_mat(gctx, quant_oproj_w_, sum);        // (768, T) f32
    x = ggml_add(gctx, x, to_f32_(gctx, quant_oproj_b_));            // (768, T)

    // ── 4 transformer stages, each followed by a patch upsample ─────────
    for (int s = 0; s < 4; ++s) {
        x = build_stage_(gctx, x, stages_[s], pos_T[s], mask_T[s]);
        if (stages_[s].spec.patch_after > 0) {
            x = patch_upsample_(gctx, x, stages_[s].spec.patch_after);
        }
    }

    // After dec.7 the channel dim is 1 and time is T_audio*1920. View as 1D.
    ggml_tensor* waveform = ggml_reshape_1d(gctx, x, n_samples);
    ggml_set_name(waveform, "waveform");
    ggml_set_output(waveform);

    ggml_cgraph* graph = ggml_new_graph_custom(gctx, 65536, false);
    ggml_build_forward_expand(graph, waveform);

    if (!ggml_gallocr_alloc_graph(galloc_, graph)) {
        ggml_free(gctx);
        throw std::runtime_error("MossCodecGraphs::decode: gallocr_alloc_graph failed");
    }

    // ── Upload inputs ───────────────────────────────────────────────────
    for (int i = 0; i < n_vq_; ++i) {
        std::vector<int32_t> col(T_audio);
        for (int t = 0; t < T_audio; ++t) col[size_t(t)] = codes[i * T_audio + t];
        ggml_backend_tensor_set(codes_in[i], col.data(), 0,
                                size_t(T_audio) * sizeof(int32_t));
    }
    for (int s = 0; s < 4; ++s) {
        std::vector<int32_t> p;
        p.resize(size_t(T_at[s]));
        for (int t = 0; t < T_at[s]; ++t) p[size_t(t)] = t;
        ggml_backend_tensor_set(pos_T[s], p.data(), 0,
                                p.size() * sizeof(int32_t));
        fill_causal_mask_(mask_T[s]);
    }

    if (ggml_backend_graph_compute(backend_, graph) != GGML_STATUS_SUCCESS) {
        ggml_free(gctx);
        throw std::runtime_error("MossCodecGraphs::decode: graph_compute failed");
    }

    std::vector<float> wav;
    wav.resize(size_t(n_samples));
    ggml_backend_tensor_get(waveform, wav.data(), 0, wav.size() * sizeof(float));

    ggml_free(gctx);
    return wav;
}

// ───────────────────────────────────────────────────────────────────────────
// Encode entry point — waveform → 32 codebook indices. Adapted verbatim
// from openmoss codec.cpp CodecGraphs::encode, lines 917–1048.
// ───────────────────────────────────────────────────────────────────────────

std::vector<int32_t> MossCodecGraphs::encode(const float* waveform,
                                              int64_t n_samples,
                                              int32_t& T_audio_out) {
    if (!present_) {
        throw std::runtime_error("MossCodecGraphs::encode: not bound");
    }
    if (!encoder_present_) {
        throw std::runtime_error("MossCodecGraphs::encode: encoder weights not "
                                  "available (GGUF is missing moss.codec.enc.*). "
                                  "Voice cloning requires an openmoss-style full "
                                  "extras GGUF, not a backbone-only community split.");
    }
    if (n_samples <= 0) { T_audio_out = 0; return {}; }

    // Pad waveform to a multiple of hop=1920 — same convention as
    // MossAudioTokenizerModel._encode_frame.
    const int64_t hop = 1920;
    int64_t T_wav = n_samples;
    int64_t pad = (hop - (T_wav % hop)) % hop;
    int64_t T_padded = T_wav + pad;
    const int32_t T_audio = int32_t(T_padded / hop);
    T_audio_out = T_audio;

    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * 65536 + ggml_graph_overhead_custom(65536, false);
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context* gctx = ggml_init(ip);
    if (!gctx) throw std::runtime_error("MossCodecGraphs::encode: ggml_init failed");

    // Input: padded waveform as a 2D tensor (1 channel × T_padded).
    ggml_tensor* wav = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 1, T_padded);
    ggml_set_name(wav, "waveform");
    ggml_set_input(wav);

    // Per-stage RoPE positions, one per encoder transformer stage.
    int T_at[4];
    T_at[0] = int(T_padded / CODEC_PRE_PATCH);                       // input to enc.1
    T_at[1] = T_at[0] / ENCODER_STAGES[0].patch_after;               // input to enc.3
    T_at[2] = T_at[1] / ENCODER_STAGES[1].patch_after;               // input to enc.5
    T_at[3] = T_at[2] / ENCODER_STAGES[2].patch_after;               // input to enc.7
    std::array<ggml_tensor*, 4> pos_T  {};
    std::array<ggml_tensor*, 4> mask_T {};
    for (int s = 0; s < 4; ++s) {
        pos_T[s] = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T_at[s]);
        ggml_set_name(pos_T[s], ("enc_pos_" + std::to_string(s)).c_str());
        ggml_set_input(pos_T[s]);
        mask_T[s] = make_causal_mask_(gctx, T_at[s]);
        ggml_set_name(mask_T[s], ("enc_mask_" + std::to_string(s)).c_str());
    }

    // Initial patch=240 downsample: (1, T_padded) → (240, T_padded/240)
    ggml_tensor* x = patch_downsample_(gctx, wav, CODEC_PRE_PATCH);

    // Four encoder stages, each followed by a patch downsample (except the last).
    for (int s = 0; s < 4; ++s) {
        x = build_stage_(gctx, x, enc_stages_[s], pos_T[s], mask_T[s]);
        if (enc_stages_[s].spec.patch_after > 0) {
            x = patch_downsample_(gctx, x, enc_stages_[s].spec.patch_after);
        }
    }

    // ── Quantizer: input_proj → 32-step residual LFQ encoding ─────────────
    // x: (768, T_audio).
    ggml_tensor* residual = ggml_mul_mat(gctx, quant_iproj_w_, x);           // (512, T)
    residual = ggml_add(gctx, residual, to_f32_(gctx, quant_iproj_b_));      // (512, T)

    std::vector<ggml_tensor*> indices(n_vq_, nullptr);
    for (int i = 0; i < n_vq_; ++i) {
        // z_e = q[i].iproj(residual)
        ggml_tensor* z_e = ggml_mul_mat(gctx, q_iproj_w_[i], residual);     // (8, T)
        z_e = ggml_add(gctx, z_e, to_f32_(gctx, q_iproj_b_[i]));            // (8, T)

        // L2-normalize per timestep (along the 8-dim, which is ne[0]).
        ggml_tensor* z_e_n = ggml_l2_norm(gctx, z_e, /*eps=*/1e-12f);       // (8, T)

        // similarity = codebook_normed[i] @ z_e_n  → (1024, T)
        ggml_tensor* sim = ggml_mul_mat(gctx, codebook_normed_[i], z_e_n);  // (1024, T)
        ggml_mul_mat_set_prec(sim, GGML_PREC_F32);

        // Argmax along ne[0] → (T,) i32 — cosine-similarity nearest neighbour.
        ggml_tensor* idx = ggml_argmax(gctx, sim);                          // (T,) i32
        ggml_set_name(idx, ("idx_" + std::to_string(i)).c_str());
        ggml_set_output(idx);
        indices[i] = idx;

        // Residual update: residual -= q[i].oproj( codebook[i][idx] )
        // (raw codebook here, not the normalized version — matches LFQ.decode_code_wo_out_proj)
        ggml_tensor* z_q = ggml_get_rows(gctx, codebook_[i], idx);          // (8, T) f32
        z_q = ggml_mul_mat(gctx, q_oproj_w_[i], z_q);                       // (512, T)
        z_q = ggml_add(gctx, z_q, to_f32_(gctx, q_oproj_b_[i]));            // (512, T)
        residual = ggml_sub(gctx, residual, z_q);                            // (512, T)
    }

    ggml_cgraph* graph = ggml_new_graph_custom(gctx, 65536, false);
    for (int i = 0; i < n_vq_; ++i) ggml_build_forward_expand(graph, indices[i]);

    if (!ggml_gallocr_alloc_graph(galloc_, graph)) {
        ggml_free(gctx);
        throw std::runtime_error("MossCodecGraphs::encode: gallocr_alloc_graph failed");
    }

    // Upload inputs.
    {
        std::vector<float> wpad;
        wpad.assign(size_t(T_padded), 0.0f);
        std::memcpy(wpad.data(), waveform, size_t(T_wav) * sizeof(float));
        ggml_backend_tensor_set(wav, wpad.data(), 0, wpad.size() * sizeof(float));
    }
    for (int s = 0; s < 4; ++s) {
        std::vector<int32_t> p;
        p.resize(size_t(T_at[s]));
        for (int t = 0; t < T_at[s]; ++t) p[size_t(t)] = t;
        ggml_backend_tensor_set(pos_T[s], p.data(), 0, p.size() * sizeof(int32_t));
        fill_causal_mask_(mask_T[s]);
    }

    if (ggml_backend_graph_compute(backend_, graph) != GGML_STATUS_SUCCESS) {
        ggml_free(gctx);
        throw std::runtime_error("MossCodecGraphs::encode: graph_compute failed");
    }

    // Read indices back: assemble (n_vq, T_audio) row-major i32.
    std::vector<int32_t> out;
    out.assign(size_t(n_vq_) * size_t(T_audio), 0);
    for (int i = 0; i < n_vq_; ++i) {
        ggml_backend_tensor_get(indices[i], out.data() + size_t(i) * size_t(T_audio),
                                0, size_t(T_audio) * sizeof(int32_t));
    }

    ggml_free(gctx);
    return out;
}

}  // namespace audiocore::moss
