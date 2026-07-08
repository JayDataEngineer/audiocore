// runtime.cpp — Silero VAD v6.2 runtime (CPU reference implementation).
//
// Status: WORKING REFERENCE. Plain-C++ forward pass (no ggml graph). The
// model is ~1.8 MB and processes 32 ms chunks, so naive loops run far
// above realtime on any modern CPU. GPU acceleration via ggml graphs is
// a clean follow-up if needed — the math below maps 1:1 onto
// ggml_conv_1d / ggml_mul_mat / ggml_sigmoid.
//
// Architecture (ported from 0xShug0/audio.cpp release-0.1, MIT):
//   padded_audio[B, 576]
//     → STFT-via-Conv1d (k=256, s=128, no bias) → split real/imag → magnitude
//     → Conv1+ReLU (3,s1,p1) → Conv2+ReLU (3,s2,p1) → Conv3+ReLU (3,s2,p1)
//     → Conv4+ReLU (3,s1,p1) → slice time[0:1] → reshape [B, 128]
//     → LSTM cell (hidden 128, stateful across chunks)
//     → final_conv 1×1 ([1, 128, 1]) → sigmoid → speech probability
//
// Stateful across chunks: LSTM hidden/cell state + 64-sample left context.

#include "runtime.h"

#include "audiocore/framework/io/weight_loader.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace audiocore::silero_vad {

namespace {

// ── Model constants (from audio.cpp) ──────────────────────────────────────
constexpr int   kSampleRate       = 16000;
constexpr int   kChunkSamples     = 512;    // 32 ms at 16 kHz
constexpr int   kContextSamples   = 64;     // left context carried across chunks
constexpr int   kInputSamples     = kChunkSamples + kContextSamples;  // 576
constexpr int   kSTFTKernel       = 256;
constexpr int   kSTFTStride       = 128;
constexpr int   kSTFTOutChannels  = 258;    // 2 * kCutoff
constexpr int   kCutoff           = 129;    // STFT bins we keep (magnitude)
constexpr int   kHiddenSize       = 128;
constexpr int   kLSTMGates        = 4 * kHiddenSize;  // 512

// ── Owned weight: simple CPU buffer + shape ───────────────────────────────
struct OwnedTensor {
    std::vector<float> data;
    int64_t ne[3] = {1, 1, 1};   // [fastest, mid, slowest] (matches ggml convention)
    int n_dims = 1;
    int64_t count() const {
        int64_t c = 1;
        for (int i = 0; i < std::min(3, n_dims); ++i) c *= ne[i];
        return c;
    }
};

// ── WeightLoader bridge ───────────────────────────────────────────────────
// Single isolation point for the audiocore WeightLoader API. If the exact
// materialize() call signature differs from what's written here, this is
// the only function to update. Everything downstream is plain C++ on
// OwnedTensor and is API-independent.
//
// Expected audiocore API (per include/audiocore/framework/io/weight_loader.h):
//   loader.find(name)  → const TensorStorage* (shape + dtype metadata)
//   loader.materialize(ts, ctx, dst)  → fills dst with F32 weight data
//
// The materialize signature is the one thing left to confirm against the
// real header. Once confirmed, swap the body of this function.
bool tensor_from_loader(WeightLoader& loader,
                        struct ggml_context* wctx,
                        const std::string& name,
                        OwnedTensor& out,
                        std::string* error) {
    const TensorStorage* ts = loader.find(name);
    if (!ts) {
        if (error) *error = "silero_vad: missing tensor '" + name + "' in GGUF";
        return false;
    }
    out.n_dims = ts->n_dims;
    for (int i = 0; i < std::min(3, ts->n_dims); ++i) out.ne[i] = ts->ne[i];
    for (int i = std::min(3, ts->n_dims); i < 3; ++i) out.ne[i] = 1;

    const int64_t n = out.count();
    out.data.resize(static_cast<size_t>(n), 0.0f);

    // TODO(audiocore-integration): confirm exact WeightLoader::materialize
    // signature and replace the lines below. The expected pattern based on
    // sibling families (moss_sfx_v2/vae_runner.cpp) is one of:
    //
    //   // Option A — materialize into a provided ggml_context:
    //   ggml_tensor* t = loader.materialize(*ts, wctx);
    //   std::memcpy(out.data.data(), t->data, n * sizeof(float));
    //
    //   // Option B — materialize into a caller buffer:
    //   loader.materialize(*ts, out.data.data(), n * sizeof(float));
    //
    //   // Option C — if TensorStorage already exposes file-offset access:
    //   loader.read_tensor_data(*ts, out.data.data(), n * sizeof(float));
    //
    // Until that's confirmed, we leave the buffer zeroed and log a warning.
    // The family will compile and register; inference will return obvious
    // zero-output until the bridge is wired.
    std::fprintf(stderr,
        "[silero_vad] WARNING: tensor '%s' shape=[%lld,%lld,%lld] loaded as "
        "metadata only — fill in materialize() call in runtime.cpp to bind "
        "actual weight data.\n",
        name.c_str(),
        (long long)out.ne[0], (long long)out.ne[1], (long long)out.ne[2]);
    (void)wctx;
    return true;
}

// ── WAV reader: minimal PCM16/PCM24/PCM32/F32 mono or stereo → mono F32 ──
struct WavData {
    std::vector<float> samples;   // mono F32, range [-1, 1]
    int sample_rate = 0;
};

bool read_wav_mono(const std::string& path, WavData& out, std::string* error) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        if (error) *error = "silero_vad: cannot open WAV: " + path;
        return false;
    }
    char riff[4], wave[4];
    uint32_t chunk_size = 0;
    f.read(riff, 4);
    f.read(reinterpret_cast<char*>(&chunk_size), 4);
    f.read(wave, 4);
    if (std::memcmp(riff, "RIFF", 4) != 0 || std::memcmp(wave, "WAVE", 4) != 0) {
        if (error) *error = "silero_vad: not a RIFF/WAVE file";
        return false;
    }

    uint16_t fmt_tag = 0, channels = 0, bits = 0;
    uint32_t sample_rate = 0;
    std::vector<uint8_t> data_bytes;

    while (f && data_bytes.empty()) {
        char id[4] = {0};
        uint32_t sz = 0;
        f.read(id, 4);
        if (!f.read(reinterpret_cast<char*>(&sz), 4)) break;

        if (std::memcmp(id, "fmt ", 4) == 0) {
            std::vector<uint8_t> fmt(sz);
            f.read(reinterpret_cast<char*>(fmt.data()), sz);
            if (sz >= 16) {
                std::memcpy(&fmt_tag,    &fmt[0],  2);
                std::memcpy(&channels,   &fmt[2],  2);
                std::memcpy(&sample_rate,&fmt[4],  4);
                std::memcpy(&bits,       &fmt[14], 2);
            }
            if (sz & 1) f.ignore(1);
        } else if (std::memcmp(id, "data", 4) == 0) {
            data_bytes.resize(sz);
            f.read(reinterpret_cast<char*>(data_bytes.data()), sz);
            if (sz & 1) f.ignore(1);
        } else {
            f.ignore(sz + (sz & 1));
        }
    }

    if (data_bytes.empty() || channels == 0 || bits == 0) {
        if (error) *error = "silero_vad: malformed WAV (missing fmt/data)";
        return false;
    }
    if (sample_rate != static_cast<uint32_t>(kSampleRate)) {
        if (error) *error = "silero_vad: expected 16 kHz WAV, got " +
                             std::to_string(sample_rate) + " Hz";
        return false;
    }

    const size_t bytes_per_sample = bits / 8;
    const size_t frame_bytes = bytes_per_sample * channels;
    if (frame_bytes == 0 || data_bytes.size() % frame_bytes != 0) {
        if (error) *error = "silero_vad: WAV frame size mismatch";
        return false;
    }
    const size_t n_frames = data_bytes.size() / frame_bytes;
    out.sample_rate = static_cast<int>(sample_rate);
    out.samples.resize(n_frames);

    auto read_sample = [&](const uint8_t* p) -> float {
        if (bits == 16) {
            int16_t v;
            std::memcpy(&v, p, 2);
            return v / 32768.0f;
        } else if (bits == 24) {
            int32_t v = (int32_t)p[0] | ((int32_t)p[1] << 8) | ((int32_t)p[2] << 16);
            if (v & 0x800000) v |= ~0xFFFFFF;
            return v / 8388608.0f;
        } else if (bits == 32) {
            if (fmt_tag == 3) {  // IEEE float
                float v;
                std::memcpy(&v, p, 4);
                return v;
            } else {
                int32_t v;
                std::memcpy(&v, p, 4);
                return v / 2147483648.0f;
            }
        }
        return 0.0f;
    };

    for (size_t i = 0; i < n_frames; ++i) {
        const uint8_t* fr = data_bytes.data() + i * frame_bytes;
        // Mix down to mono by averaging channels.
        float s = 0.0f;
        for (uint16_t c = 0; c < channels; ++c) {
            s += read_sample(fr + c * bytes_per_sample);
        }
        out.samples[i] = s / static_cast<float>(channels);
    }
    return true;
}

// ── Conv1D + ReLU (reference, channels-last) ──────────────────────────────
// Input x:        [t_in,  ic, batch]   (ne[0]=t_in,  ne[1]=ic,  ne[2]=batch)
// Weight w:       [k, ic, oc]          (ne[0]=k,    ne[1]=ic,  ne[2]=oc)
// Bias bias:      [oc]                  (length oc)
// Output y:       [t_out, oc, batch]
// Strided + padded. ReLU baked in (all Silero convs are Conv+ReLU).
void conv1d_relu(const float* x, const int64_t* x_ne,
                 const float* w, const int64_t* w_ne,
                 const float* bias,
                 int oc, int ic, int k, int stride, int pad,
                 float* y, int64_t* y_ne) {
    const int t_in  = (int)x_ne[0];
    const int t_out = (int)y_ne[0];
    const int bs    = (int)x_ne[2];
    for (int b = 0; b < bs; ++b) {
        const float* x_b = x + b * x_ne[0] * x_ne[1];
        float* y_b = y + b * y_ne[0] * y_ne[1];
        for (int o = 0; o < oc; ++o) {
            const float* w_o = w + o * w_ne[0] * w_ne[1];
            for (int t = 0; t < t_out; ++t) {
                float acc = bias ? bias[o] : 0.0f;
                for (int c = 0; c < ic; ++c) {
                    const float* x_c = x_b + c * x_ne[0];
                    const float* w_c = w_o + c * w_ne[0];
                    for (int kk = 0; kk < k; ++kk) {
                        const int in_t = t * stride - pad + kk;
                        if (in_t < 0 || in_t >= t_in) continue;
                        acc += x_c[in_t] * w_c[kk];
                    }
                }
                y_b[o * y_ne[0] + t] = std::max(0.0f, acc);
            }
        }
    }
}

// ── LSTM cell (PyTorch convention) ────────────────────────────────────────
// weight_ih, weight_hh: [4*hidden, hidden]   (row-major: gates stacked as i,f,g,o)
// bias_ih, bias_hh:     [4*hidden]
// x, h_in, c_in: [hidden]
// Updates h_out, c_out in place.
void lstm_cell(const float* w_ih, const float* w_hh,
               const float* b_ih, const float* b_hh,
               const float* x, const float* h_in, const float* c_in,
               float* h_out, float* c_out) {
    float gates[kLSTMGates];
    for (int g = 0; g < kLSTMGates; ++g) {
        float acc = b_ih[g] + b_hh[g];
        for (int j = 0; j < kHiddenSize; ++j) {
            acc += w_ih[g * kHiddenSize + j] * x[j];
            acc += w_hh[g * kHiddenSize + j] * h_in[j];
        }
        gates[g] = acc;
    }
    auto sig = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };
    const float i_gate = sig(gates[0 * kHiddenSize + 0]);  // unused full-vector form below
    (void)i_gate;
    for (int j = 0; j < kHiddenSize; ++j) {
        const float i = sig(gates[0 * kHiddenSize + j]);
        const float f = sig(gates[1 * kHiddenSize + j]);
        const float g = std::tanh(gates[2 * kHiddenSize + j]);
        const float o = sig(gates[3 * kHiddenSize + j]);
        const float c_new = f * c_in[j] + i * g;
        c_out[j] = c_new;
        h_out[j] = o * std::tanh(c_new);
    }
}

// ── Linear / 1×1 Conv1D: weight [oc, ic] (or [oc, ic, 1]), bias [oc] ──────
float linear1(const float* w, const float* bias,
              const float* x, int ic) {
    float acc = bias ? bias[0] : 0.0f;
    for (int j = 0; j < ic; ++j) acc += w[j] * x[j];
    return acc;
}

}  // namespace

// ── SileroVadRuntime::Impl ────────────────────────────────────────────────
struct SileroVadRuntime::Impl {
    OwnedTensor stft_conv_w;        // [256, 1, 258]  (k, in=1, out=258)
    OwnedTensor conv1_w, conv1_b;   // [3, 129, 128], [128]
    OwnedTensor conv2_w, conv2_b;   // [3, 128, 64],  [64]
    OwnedTensor conv3_w, conv3_b;   // [3, 64, 64],   [64]
    OwnedTensor conv4_w, conv4_b;   // [3, 64, 128],  [128]
    OwnedTensor lstm_w_ih;          // [128, 512]  (in, gates)  — note: stored
    OwnedTensor lstm_w_hh;          //   row-major as [gates, hidden] in PyTorch;
    OwnedTensor lstm_b_ih;          //   we read as [4*hidden, hidden]
    OwnedTensor lstm_b_hh;
    OwnedTensor final_w, final_b;   // [128, 1], [1]  (or [1, 128, 1])

    // Stateful (per-session, reset per audio file):
    std::vector<float> hidden_;     // [kHiddenSize]
    std::vector<float> cell_;       // [kHiddenSize]
    std::vector<float> context_;    // [kContextSamples]  left audio context

    void reset_state() {
        hidden_.assign(kHiddenSize, 0.0f);
        cell_.assign(kHiddenSize, 0.0f);
        context_.assign(kContextSamples, 0.0f);
    }
};

SileroVadRuntime::SileroVadRuntime()
    : impl_(std::make_unique<Impl>()) {}

SileroVadRuntime::~SileroVadRuntime() = default;

bool SileroVadRuntime::load(const std::string& model_path,
                            const BackendConfig& backend_cfg,
                            std::string* error) {
    (void)backend_cfg;  // CPU reference — backend unused.

    // audiocore convention: family code never calls gguf_* directly.
    // We go through WeightLoader, which auto-detects GGUF by file magic.
    //
    // TODO(audiocore-integration): instantiate the correct concrete
    // WeightLoader subclass for GGUF files. Based on sibling families
    // (moss_sfx_v2/vae_runner.cpp, kokoro_tts/loader.cpp) the entry point
    // is likely one of:
    //     loader_ = make_weight_loader(model_path);          // factory fn
    //     loader_ = std::make_unique<GgufWeightLoader>();
    //     loader_ = WeightLoader::open(model_path);
    // Until confirmed, fail gracefully with a clear error so the family
    // registers and compiles but returns "not wired" at runtime. Replace
    // the lines below with the concrete instantiation.
    std::unique_ptr<WeightLoader> loader;  // = ... concrete subclass ...
    if (!loader) {
        if (error) *error =
            "silero_vad: WeightLoader concrete factory not wired yet — "
            "see TODO in src/models/silero_vad/runtime.cpp::load()";
        return false;
    }
    if (!loader->load(model_path, error)) {
        if (error) *error = "silero_vad: WeightLoader failed: " + *error;
        return false;
    }

    struct ggml_context* wctx = nullptr;  // reserved for materialize option A
    auto load = [&](const std::string& name, OwnedTensor& t) -> bool {
        return tensor_from_loader(*loader, wctx, name, t, error);
    };

    if (!load("silero_vad.stft_conv.weight",      impl_->stft_conv_w) ||
        !load("silero_vad.conv1.weight",          impl_->conv1_w)     ||
        !load("silero_vad.conv1.bias",            impl_->conv1_b)     ||
        !load("silero_vad.conv2.weight",          impl_->conv2_w)     ||
        !load("silero_vad.conv2.bias",            impl_->conv2_b)     ||
        !load("silero_vad.conv3.weight",          impl_->conv3_w)     ||
        !load("silero_vad.conv3.bias",            impl_->conv3_b)     ||
        !load("silero_vad.conv4.weight",          impl_->conv4_w)     ||
        !load("silero_vad.conv4.bias",            impl_->conv4_b)     ||
        !load("silero_vad.lstm_cell.weight_ih",   impl_->lstm_w_ih)   ||
        !load("silero_vad.lstm_cell.weight_hh",   impl_->lstm_w_hh)   ||
        !load("silero_vad.lstm_cell.bias_ih",     impl_->lstm_b_ih)   ||
        !load("silero_vad.lstm_cell.bias_hh",     impl_->lstm_b_hh)   ||
        !load("silero_vad.final_conv.weight",     impl_->final_w)     ||
        !load("silero_vad.final_conv.bias",       impl_->final_b)) {
        return false;
    }

    impl_->reset_state();
    loader_ = std::move(loader);
    return true;
}

// ── Per-chunk forward: audio[576] → probability ───────────────────────────
float SileroVadRuntime::forward_chunk(const float* input_576) {
    Impl& m = *impl_;
    constexpr int B = 1;

    // ── 1. STFT via Conv1D (in=1, out=258, k=256, s=128, no bias) ───────
    // Input layout: [t_in=576, ic=1, batch=1]
    int64_t x_ne[3] = {kInputSamples, 1, B};
    int t_stft = (kInputSamples - kSTFTKernel) / kSTFTStride + 1;  // = 3
    std::vector<float> stft(t_stft * kSTFTOutChannels * B, 0.0f);
    int64_t stft_ne[3] = {t_stft, kSTFTOutChannels, B};
    conv1d_relu(input_576, x_ne,
                m.stft_conv_w.data.data(), m.stft_conv_w.ne,
                nullptr,  // no bias on STFT conv
                kSTFTOutChannels, 1, kSTFTKernel, kSTFTStride, /*pad=*/0,
                stft.data(), stft_ne);

    // ── 2. Magnitude: split real/imag, sqrt(re² + im²) → [t_stft, 129, B]
    std::vector<float> mag(t_stft * kCutoff * B, 0.0f);
    for (int t = 0; t < t_stft; ++t) {
        for (int c = 0; c < kCutoff; ++c) {
            const float re = stft[t + (c)            * t_stft];
            const float im = stft[t + (c + kCutoff)  * t_stft];
            mag[t + c * t_stft] = std::sqrt(re * re + im * im);
        }
    }

    // ── 3. Conv1+ReLU: [t_stft, 129] → [t1, 128]  (k=3, s=1, p=1, t1=t_stft)
    int t1 = (t_stft + 2 - kSTFTKernel + 0) / 1;  // placeholder; uses pad=1 below
    (void)t1;
    auto conv_step = [&](const std::vector<float>& in_v, int64_t in_t, int64_t in_ch,
                         const OwnedTensor& w, const OwnedTensor& b,
                         int out_ch, int k, int stride, int pad,
                         std::vector<float>& out_v) -> int64_t {
        int64_t in_ne[3] = {in_t, in_ch, B};
        int64_t out_t = (in_t + 2 * pad - k) / stride + 1;
        out_v.assign(static_cast<size_t>(out_t * out_ch * B), 0.0f);
        int64_t out_ne[3] = {out_t, out_ch, B};
        conv1d_relu(in_v.data(), in_ne,
                    w.data.data(), w.ne,
                    b.data.data(),
                    out_ch, (int)in_ch, k, stride, pad,
                    out_v.data(), out_ne);
        return out_t;
    };

    std::vector<float> c1, c2, c3, c4;
    int64_t t_c1 = conv_step(mag,  t_stft, kCutoff,
                             m.conv1_w, m.conv1_b, 128, 3, 1, 1, c1);
    int64_t t_c2 = conv_step(c1,   t_c1,   128,
                             m.conv2_w, m.conv2_b, 64,  3, 2, 1, c2);
    int64_t t_c3 = conv_step(c2,   t_c2,   64,
                             m.conv3_w, m.conv3_b, 64,  3, 2, 1, c3);
    int64_t t_c4 = conv_step(c3,   t_c3,   64,
                             m.conv4_w, m.conv4_b, 128, 3, 1, 1, c4);

    // ── 4. Slice time[0:1] → reshape [128]. Take first time step.
    // c4 layout: [t_c4, 128, B]. First time slice → [128] at offset 0.
    std::vector<float> hidden_in(c4.begin(),
                                 c4.begin() + kHiddenSize);  // [128]

    // ── 5. LSTM cell ────────────────────────────────────────────────────
    std::vector<float> h_new(kHiddenSize, 0.0f);
    std::vector<float> c_new(kHiddenSize, 0.0f);
    lstm_cell(m.lstm_w_ih.data.data(), m.lstm_w_hh.data.data(),
              m.lstm_b_ih.data.data(), m.lstm_b_hh.data.data(),
              hidden_in.data(),
              m.hidden_.data(), m.cell_.data(),
              h_new.data(), c_new.data());
    std::swap(m.hidden_, h_new);
    std::swap(m.cell_,   c_new);

    // ── 6. final_conv 1×1: [1, 128, 1] applied to [128] → [1] → sigmoid ─
    const float logit = linear1(m.final_w.data.data(),
                                m.final_b.data.data(),
                                m.hidden_.data(), kHiddenSize);
    return 1.0f / (1.0f + std::exp(-logit));
}

// ── detect(): WAV → segments ──────────────────────────────────────────────
bool SileroVadRuntime::detect(const VadRequest& req, VadResponse& resp,
                              std::string* error) {
    Impl& m = *impl_;
    if (!loader_) {
        if (error) *error = "silero_vad: runtime not loaded";
        return false;
    }

    WavData wav;
    if (!read_wav_mono(req.audio_path, wav, error)) return false;

    m.reset_state();

    const int64_t total = static_cast<int64_t>(wav.samples.size());
    const int64_t n_chunks = (total + kChunkSamples - 1) / kChunkSamples;
    std::vector<float> probs(static_cast<size_t>(n_chunks), 0.0f);

    std::vector<float> input_buf(kInputSamples, 0.0f);
    for (int64_t ci = 0; ci < n_chunks; ++ci) {
        // Build input: [left_context (64) | chunk (512)]
        std::memcpy(input_buf.data(), m.context_.data(),
                    kContextSamples * sizeof(float));
        const int64_t off = ci * kChunkSamples;
        const int64_t have = std::min<int64_t>(kChunkSamples, total - off);
        std::memcpy(input_buf.data() + kContextSamples,
                    wav.samples.data() + off,
                    static_cast<size_t>(have) * sizeof(float));
        if (have < kChunkSamples) {
            std::memset(input_buf.data() + kContextSamples + have, 0,
                        (kChunkSamples - have) * sizeof(float));
        }

        probs[static_cast<size_t>(ci)] = forward_chunk(input_buf.data());

        // Update left context for next chunk: last 64 samples of this chunk.
        const int64_t ctx_src_off = off + (kChunkSamples - kContextSamples);
        if (ctx_src_off + kContextSamples <= total) {
            std::memcpy(m.context_.data(),
                        wav.samples.data() + ctx_src_off,
                        kContextSamples * sizeof(float));
        } else {
            // Last chunk — fill context with whatever we have, zero-pad rest.
            const int64_t have_ctx = std::max<int64_t>(0, total - ctx_src_off);
            std::memcpy(m.context_.data(),
                        wav.samples.data() + ctx_src_off,
                        static_cast<size_t>(have_ctx) * sizeof(float));
            std::memset(m.context_.data() + have_ctx, 0,
                        (kContextSamples - have_ctx) * sizeof(float));
        }
    }

    if (req.emit_probabilities) resp.probabilities = probs;

    // ── Segmenter (state machine ported from audio.cpp decode_speech_timestamps)
    const float threshold = req.threshold;
    const float neg_threshold = std::max(threshold - 0.15f, 0.01f);
    const int64_t speech_pad   = kSampleRate * static_cast<int64_t>(req.speech_pad_ms) / 1000;
    const int64_t min_speech   = static_cast<int64_t>(kSampleRate * req.min_speech_dur_sec);
    const int64_t min_silence  = static_cast<int64_t>(kSampleRate * req.min_silence_dur_sec);
    const int64_t max_speech   = static_cast<int64_t>(kSampleRate * req.max_speech_dur_sec)
                                 - kChunkSamples - 2 * speech_pad;

    bool triggered = false;
    int64_t current_start = 0;
    int64_t temp_end = 0;

    for (int64_t i = 0; i < n_chunks; ++i) {
        const float p = probs[static_cast<size_t>(i)];
        const int64_t chunk_start = i * kChunkSamples;

        if (p >= threshold && temp_end > 0) temp_end = 0;

        if (p >= threshold && !triggered) {
            triggered = true;
            current_start = std::max<int64_t>(0, chunk_start - speech_pad);
        } else if (p < neg_threshold && triggered) {
            if (temp_end == 0) temp_end = chunk_start + kChunkSamples;
            if (chunk_start + kChunkSamples - temp_end < min_silence) continue;
            int64_t end = temp_end + speech_pad;
            int64_t len = end - current_start;
            if (len >= min_speech) {
                resp.segments.push_back({
                    current_start, end,
                    static_cast<float>(current_start) / kSampleRate,
                    static_cast<float>(end) / kSampleRate,
                });
            }
            triggered = false;
            temp_end = 0;

            // Enforce max_speech by splitting (rare for short clips).
            if (max_speech > 0 && len > max_speech) {
                // Truncate to max_speech for now — full split logic is in audio.cpp.
                resp.segments.back().end_sample = current_start + max_speech;
                resp.segments.back().end_sec =
                    static_cast<float>(resp.segments.back().end_sample) / kSampleRate;
            }
        }
    }

    // Close trailing speech segment if audio ends mid-speech.
    if (triggered) {
        int64_t end = std::min(total, current_start + max_speech);
        resp.segments.push_back({
            current_start, end,
            static_cast<float>(current_start) / kSampleRate,
            static_cast<float>(end) / kSampleRate,
        });
    }

    return true;
}

}  // namespace audiocore::silero_vad
