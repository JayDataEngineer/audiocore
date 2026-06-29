// speaker_encoder.cpp — Qwen3-TTS ECAPA-TDNN speaker encoder implementation.
//
// Adapted from CrispStrobe/CrispASR (MIT) src/qwen3_tts.cpp — the ECAPA-TDNN
// section only (~lines 5096-5423). Architecture:
//
//   mel (128, T_mel)
//    → blk0: Conv1d 128→512 k=5 d=1 reflect-pad ReLU
//    → 3× SE-Res2Net (d=2/3/4): tdnn1→res2net→tdnn2→SE→+residual
//    → concat[blk0_out, blk1_out, blk2_out] → mfa: Conv1d 1536→1536 k=1 ReLU
//    → ASP: concat(x, mean, std) → tdnn→tanh→conv→softmax→weighted stats
//    → fc: 3072→enc_dim (1024 / 2048)
//    → (enc_dim,) speaker embedding
//
// Tensor source: the talker GGUF carries `speaker.*` tensors alongside the
// talker and code-predictor weights. llama.cpp skips them; we open the same
// file with a separate GgufReader and load only speaker.* tensors.
//
// Mel front-end: 128-band log-mel spectrogram computed in host code (not
// ggml) — the ECAPA reference uses Slaney mel filterbank + magnitude STFT
// with specific parameters (n_fft=1024, hop=256, fmin=0, fmax=12000).

#include "audiocore/models/qwen3_tts/speaker_encoder.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

namespace audiocore::qwen3_tts {

// ═══════════════════════════════════════════════════════════════════════════
//  Minimal WAV loader (24 kHz mono 16-bit PCM)
// ═══════════════════════════════════════════════════════════════════════════

std::vector<float> Qwen3TtsSpeakerEncoder::load_wav(const std::string& path,
                                                     std::string* error) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        if (error) *error = "cannot open WAV file: " + path;
        return {};
    }

    auto cleanup = [&] { std::fclose(f); };

    char riff[4];
    if (std::fread(riff, 1, 4, f) != 4 || std::memcmp(riff, "RIFF", 4) != 0) {
        if (error) *error = "not a RIFF WAV";
        cleanup(); return {};
    }
    std::fseek(f, 4, SEEK_CUR); // skip chunk size
    char wave[4];
    if (std::fread(wave, 1, 4, f) != 4 || std::memcmp(wave, "WAVE", 4) != 0) {
        if (error) *error = "not a WAVE file";
        cleanup(); return {};
    }

    int sr = 0, channels = 0, bits = 0;
    bool found_data = false;
    std::vector<float> samples;

    while (!found_data) {
        char id[4];
        uint32_t sz;
        if (std::fread(id, 1, 4, f) != 4 || std::fread(&sz, 4, 1, f) != 1) break;
        if (std::memcmp(id, "fmt ", 4) == 0) {
            uint16_t fmt, ch;
            uint32_t srate;
            uint16_t bps;
            std::fread(&fmt, 2, 1, f);
            std::fread(&ch, 2, 1, f);
            std::fread(&srate, 4, 1, f);
            std::fseek(f, 6, SEEK_CUR); // skip byte rate + block align
            std::fread(&bps, 2, 1, f);
            if (sz > 16) std::fseek(f, sz - 16, SEEK_CUR);
            sr = (int)srate;
            channels = (int)ch;
            bits = (int)bps;
        } else if (std::memcmp(id, "data", 4) == 0) {
            if (sr != 24000) {
                if (error) *error = "reference audio must be 24 kHz, got " + std::to_string(sr) + " Hz";
                cleanup(); return {};
            }
            const int n_frames = (int)(sz / (channels * (bits / 8)));
            samples.reserve((size_t)n_frames);
            if (bits == 16) {
                std::vector<int16_t> raw((size_t)n_frames * channels);
                std::fread(raw.data(), 2, raw.size(), f);
                for (int i = 0; i < n_frames; i++) {
                    float s = 0.0f;
                    for (int c = 0; c < channels; c++)
                        s += raw[(size_t)i * channels + c] / 32768.0f;
                    samples.push_back(s / channels);
                }
            } else if (bits == 32) {
                std::vector<float> raw((size_t)n_frames * channels);
                std::fread(raw.data(), 4, raw.size(), f);
                for (int i = 0; i < n_frames; i++) {
                    float s = 0.0f;
                    for (int c = 0; c < channels; c++)
                        s += raw[(size_t)i * channels + c];
                    samples.push_back(s / channels);
                }
            } else {
                if (error) *error = "unsupported WAV bit depth: " + std::to_string(bits);
                cleanup(); return {};
            }
            found_data = true;
        } else {
            std::fseek(f, sz, SEEK_CUR);
        }
    }
    cleanup();

    if (samples.empty()) {
        if (error) *error = "no audio data found in WAV";
        return {};
    }
    return samples;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Mel spectrogram (128-band, 24 kHz, Slaney-scale, magnitude STFT)
// ═══════════════════════════════════════════════════════════════════════════
//
// Matches CrispASR's compute_spk_mel() exactly — same FFT, reflect pad, Hann,
// magnitude spectrum, Slaney FB, log. Independent of the codec's mel path.

static void fft_r2c(const float* in, int N, float* out) {
    int bits = 0;
    for (int n = N; n > 1; n >>= 1) bits++;
    for (int i = 0; i < N; i++) {
        int rev = 0;
        for (int b = 0; b < bits; b++) rev = (rev << 1) | ((i >> b) & 1);
        out[(size_t)2 * rev] = in[i];
        out[(size_t)2 * rev + 1] = 0.0f;
    }
    for (int len = 2; len <= N; len <<= 1) {
        float ang = -2.0f * (float)M_PI / (float)len;
        float wre = cosf(ang), wim = sinf(ang);
        for (int i = 0; i < N; i += len) {
            float ure = 1.0f, uim = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                int a = i + j, bj = i + j + len / 2;
                float are = out[(size_t)2 * a], aim = out[(size_t)2 * a + 1];
                float bre = out[(size_t)2 * bj], bim = out[(size_t)2 * bj + 1];
                float tre = ure * bre - uim * bim, tim = ure * bim + uim * bre;
                out[(size_t)2 * a] = are + tre;
                out[(size_t)2 * a + 1] = aim + tim;
                out[(size_t)2 * bj] = are - tre;
                out[(size_t)2 * bj + 1] = aim - tim;
                float nr = ure * wre - uim * wim;
                uim = ure * wim + uim * wre;
                ure = nr;
            }
        }
    }
}

// Slaney-style mel filterbank: n_mels triangular filters on the mel scale,
// area-normalized. Returns (n_freqs, n_mels) row-major.
static std::vector<float> build_slaney_fb(int sr, int n_fft, int n_mels,
                                           float fmin, float fmax) {
    const int n_freqs = n_fft / 2 + 1;
    auto hz_to_mel = [](float h) { return 2595.0f * log10f(1.0f + h / 700.0f); };
    auto mel_to_hz = [](float m) { return 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f); };

    float mel_min = hz_to_mel(fmin);
    float mel_max = hz_to_mel(fmax);
    std::vector<float> mel_pts((size_t)n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++)
        mel_pts[(size_t)i] = mel_min + (mel_max - mel_min) * i / (n_mels + 1);

    std::vector<float> fb((size_t)n_freqs * n_mels, 0.0f);
    for (int m = 0; m < n_mels; m++) {
        float hz_l = mel_to_hz(mel_pts[(size_t)m]);
        float hz_c = mel_to_hz(mel_pts[(size_t)m + 1]);
        float hz_r = mel_to_hz(mel_pts[(size_t)m + 2]);
        int bin_l = (int)(hz_l * (n_fft + 1) / sr);
        int bin_c = (int)(hz_c * (n_fft + 1) / sr);
        int bin_r = (int)(hz_r * (n_fft + 1) / sr);
        for (int k = bin_l; k <= bin_c && k < n_freqs; k++) {
            fb[(size_t)k * n_mels + m] = (float)(k - bin_l) / (bin_c - bin_l + 1);
        }
        for (int k = bin_c; k <= bin_r && k < n_freqs; k++) {
            fb[(size_t)k * n_mels + m] = (float)(bin_r - k) / (bin_r - bin_c + 1);
        }
        // Area normalization: each triangular filter sums to 1.0.
        float area = 0.0f;
        for (int k = 0; k < n_freqs; k++) area += fb[(size_t)k * n_mels + m];
        if (area > 0) {
            float inv = 1.0f / area;
            for (int k = 0; k < n_freqs; k++)
                fb[(size_t)k * n_mels + m] *= inv;
        }
    }
    return fb;
}

std::vector<float> Qwen3TtsSpeakerEncoder::compute_mel(const float* audio,
                                                        int n_samples,
                                                        int* T_out) {
    const int n_fft = 1024, hop = 256, n_mels = 128, sr = 24000;
    const int pad = (n_fft - hop) / 2; // 384

    // Reflect-pad audio (PyTorch-style reflect: excludes boundary element)
    std::vector<float> audio_p((size_t)n_samples + 2 * pad, 0.0f);
    for (int i = 0; i < pad; i++)
        audio_p[(size_t)i] = audio[(size_t)std::min(pad - i, n_samples - 1)];
    std::memcpy(audio_p.data() + pad, audio, (size_t)n_samples * sizeof(float));
    for (int i = 0; i < pad; i++)
        audio_p[(size_t)pad + n_samples + i] = audio[(size_t)std::max(n_samples - 2 - i, 0)];

    // Periodic Hann window
    std::vector<float> hann((size_t)n_fft);
    for (int i = 0; i < n_fft; i++)
        hann[(size_t)i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (float)n_fft));

    const int n_freqs = n_fft / 2 + 1;
    auto fb = build_slaney_fb(sr, n_fft, n_mels, 0.0f, 12000.0f);

    // STFT + magnitude spectrum + mel filterbank
    const int T = ((int)audio_p.size() - n_fft) / hop + 1;
    std::vector<float> mel((size_t)T * n_mels, 0.0f);
    std::vector<float> fft_buf((size_t)2 * n_fft);

    for (int t = 0; t < T; t++) {
        const float* frame = audio_p.data() + (size_t)t * hop;
        // Apply Hann
        std::vector<float> windowed((size_t)n_fft);
        for (int i = 0; i < n_fft; i++)
            windowed[(size_t)i] = frame[i] * hann[(size_t)i];
        // FFT
        fft_r2c(windowed.data(), n_fft, fft_buf.data());
        // Magnitude spectrum
        for (int k = 0; k < n_freqs; k++) {
            float re = fft_buf[(size_t)2 * k];
            float im = fft_buf[(size_t)2 * k + 1];
            float mag = sqrtf(re * re + im * im + 1e-9f);
            // Apply mel filterbank
            for (int m = 0; m < n_mels; m++) {
                mel[(size_t)t * n_mels + m] += mag * fb[(size_t)k * n_mels + m];
            }
        }
    }

    // Log: clamp to 1e-5, natural log
    for (size_t i = 0; i < mel.size(); i++) {
        if (mel[i] < 1e-5f) mel[i] = 1e-5f;
        mel[i] = logf(mel[i]);
    }

    if (T_out) *T_out = T;
    return mel;
}

// ═══════════════════════════════════════════════════════════════════════════
//  ECAPA graph builders (adapted from CrispASR)
// ═══════════════════════════════════════════════════════════════════════════

// Conv1d with symmetric REFLECT padding ("same", matching PyTorch
// padding_mode='reflect'). Input/output: [C, T] channels-first.
ggml_tensor* Qwen3TtsSpeakerEncoder::same_conv1d_(ggml_context* ctx,
                                                    ggml_tensor* x,
                                                    ggml_tensor* w,
                                                    ggml_tensor* b,
                                                    int dilation) {
    const int K = (int)w->ne[0];
    const int pad = (K - 1) * dilation / 2;
    // ggml_conv_1d expects [T, C] layout, w is [K, C_out, C_in]
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // [T, C]
    if (pad > 0)
        x = ggml_pad_reflect_1d(ctx, x, pad, pad);
    x = ggml_conv_1d(ctx, w, x, 1, 0, dilation);
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // [C_out, T]
    if (b) x = ggml_add(ctx, x, b);
    return x;
}

ggml_tensor* Qwen3TtsSpeakerEncoder::tdnn_block_(ggml_context* ctx,
                                                   ggml_tensor* x,
                                                   const TDNN& t,
                                                   int dilation) {
    return ggml_relu(ctx, same_conv1d_(ctx, x, t.w, t.b, dilation));
}

ggml_tensor* Qwen3TtsSpeakerEncoder::se_block_(ggml_context* ctx,
                                                ggml_tensor* x,
                                                const SE& se) {
    const int T = (int)x->ne[1];
    // Global mean over T for each C channel
    ggml_tensor* m = ggml_cont(ctx, ggml_transpose(ctx,
        ggml_scale(ctx, ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, x))),
                   1.0f / T))); // [C, 1]
    auto w1 = ggml_reshape_2d(ctx, se.c1w, se.c1w->ne[1], se.c1w->ne[2]);
    ggml_tensor* h = ggml_relu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w1, m), se.c1b));
    auto w2 = ggml_reshape_2d(ctx, se.c2w, se.c2w->ne[1], se.c2w->ne[2]);
    ggml_tensor* sc = ggml_sigmoid(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w2, h), se.c2b));
    return ggml_mul(ctx, x, ggml_repeat(ctx, sc, x));
}

ggml_tensor* Qwen3TtsSpeakerEncoder::res2net_block_(ggml_context* ctx,
                                                     ggml_tensor* x,
                                                     const Res2Net& r,
                                                     int dilation) {
    const int T = (int)x->ne[1];
    const int chunk = 64; // C/8
    ggml_tensor* outs[8];
    for (int i = 0; i < 8; i++) {
        ggml_tensor* ci = ggml_cont(ctx, ggml_view_2d(ctx, x, chunk, T,
            x->nb[1], (size_t)i * chunk * sizeof(float)));
        if (i == 0) { outs[i] = ci; continue; }
        ggml_tensor* in = (i == 1) ? ci : ggml_add(ctx, ci, outs[i - 1]);
        outs[i] = tdnn_block_(ctx, in, r.blocks[i - 1], dilation);
    }
    ggml_tensor* out = outs[0];
    for (int i = 1; i < 8; i++)
        out = ggml_concat(ctx, out, outs[i], 0);
    return out;
}

ggml_tensor* Qwen3TtsSpeakerEncoder::se_res2net_(ggml_context* ctx,
                                                   ggml_tensor* x,
                                                   const SERes2Net& blk,
                                                   int d) {
    ggml_tensor* res = x;
    x = tdnn_block_(ctx, x, blk.tdnn1, 1);
    x = res2net_block_(ctx, x, blk.res2net, d);
    x = tdnn_block_(ctx, x, blk.tdnn2, 1);
    x = se_block_(ctx, x, blk.se);
    return ggml_add(ctx, x, res);
}

ggml_tensor* Qwen3TtsSpeakerEncoder::asp_block_(ggml_context* ctx,
                                                  ggml_tensor* x,
                                                  const ASP& asp) {
    const int T = (int)x->ne[1];
    // Global statistics for attention input
    ggml_tensor* xT = ggml_cont(ctx, ggml_transpose(ctx, x));
    ggml_tensor* m1C = ggml_scale(ctx, ggml_sum_rows(ctx, xT), 1.0f / T); // [1, C]
    ggml_tensor* mC1 = ggml_cont(ctx, ggml_transpose(ctx, m1C));          // [C, 1]
    ggml_tensor* mCT = ggml_repeat(ctx, mC1, x);                          // [C, T]
    ggml_tensor* d2 = ggml_mul(ctx, ggml_sub(ctx, x, mCT),
                                ggml_sub(ctx, x, mCT));
    ggml_tensor* s1C = ggml_sqrt(ctx,
        ggml_scale(ctx, ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, d2))),
                   1.0f / T));                                            // [1, C]
    ggml_tensor* sCT = ggml_repeat(ctx, ggml_cont(ctx, ggml_transpose(ctx, s1C)), x); // [C, T]
    // [x, mean, std] → TDNN → tanh → k=1-conv → softmax over T
    ggml_tensor* att = ggml_concat(ctx, ggml_concat(ctx, x, mCT, 0), sCT, 0);
    att = tdnn_block_(ctx, att, asp.tdnn, 1);
    att = ggml_tanh(ctx, att);
    auto cw = ggml_reshape_2d(ctx, asp.conv_w, asp.conv_w->ne[1], asp.conv_w->ne[2]);
    att = ggml_add(ctx, ggml_mul_mat(ctx, cw, att), asp.conv_b); // [C, T]
    att = ggml_cont(ctx, ggml_transpose(ctx, att));              // [T, C]
    att = ggml_soft_max(ctx, att);                               // softmax over T (ne[0])
    att = ggml_cont(ctx, ggml_transpose(ctx, att));              // [C, T]
    // Weighted mean & std → [2C, 1]
    ggml_tensor* wx = ggml_mul(ctx, att, x);
    ggml_tensor* wm = ggml_cont(ctx, ggml_transpose(ctx,
        ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, wx))))); // [C,1]
    ggml_tensor* wmCT = ggml_repeat(ctx, wm, x);
    ggml_tensor* dd = ggml_sub(ctx, x, wmCT);
    ggml_tensor* ws = ggml_sqrt(ctx, ggml_cont(ctx, ggml_transpose(ctx,
        ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx,
            ggml_mul(ctx, att, ggml_mul(ctx, dd, dd)))))))); // [C,1]
    return ggml_concat(ctx, wm, ws, 0); // [2C, 1]
}

// ═══════════════════════════════════════════════════════════════════════════
//  bind() — resolve tensor pointers from source_ctx
// ═══════════════════════════════════════════════════════════════════════════

static ggml_tensor* req_tensor(ggml_context* ctx, const char* name,
                                std::string* error) {
    ggml_tensor* t = ggml_get_tensor(ctx, name);
    if (!t && error) *error += std::string("missing speaker encoder tensor: ") + name + "\n";
    return t;
}

bool Qwen3TtsSpeakerEncoder::bind(ggml_context* source_ctx,
                                   ggml_backend_t backend,
                                   std::string* error) {
    if (loaded_) return true;

    backend_ = backend;

    // Resolve all tensors. Order from CrispASR's load_spk_enc().
    auto T = [&](const char* n) { return req_tensor(source_ctx, n, error); };

    blk0_.w = T("speaker.blocks.0.conv.weight");
    blk0_.b = T("speaker.blocks.0.conv.bias");
    if (!blk0_.w) { loaded_ = false; return false; }

    for (int i = 0; i < 3; i++) {
        int bi = i + 1;
        char buf[128];
        auto& blk = blk_[i];

        std::snprintf(buf, sizeof(buf), "speaker.blocks.%d.tdnn1.conv.weight", bi);
        blk.tdnn1.w = T(buf);
        std::snprintf(buf, sizeof(buf), "speaker.blocks.%d.tdnn1.conv.bias", bi);
        blk.tdnn1.b = T(buf);
        std::snprintf(buf, sizeof(buf), "speaker.blocks.%d.tdnn2.conv.weight", bi);
        blk.tdnn2.w = T(buf);
        std::snprintf(buf, sizeof(buf), "speaker.blocks.%d.tdnn2.conv.bias", bi);
        blk.tdnn2.b = T(buf);
        std::snprintf(buf, sizeof(buf), "speaker.blocks.%d.se_block.conv1.weight", bi);
        blk.se.c1w = T(buf);
        std::snprintf(buf, sizeof(buf), "speaker.blocks.%d.se_block.conv1.bias", bi);
        blk.se.c1b = T(buf);
        std::snprintf(buf, sizeof(buf), "speaker.blocks.%d.se_block.conv2.weight", bi);
        blk.se.c2w = T(buf);
        std::snprintf(buf, sizeof(buf), "speaker.blocks.%d.se_block.conv2.bias", bi);
        blk.se.c2b = T(buf);

        for (int j = 0; j < 7; j++) {
            std::snprintf(buf, sizeof(buf),
                "speaker.blocks.%d.res2net_block.blocks.%d.conv.weight", bi, j);
            blk.res2net.blocks[j].w = T(buf);
            std::snprintf(buf, sizeof(buf),
                "speaker.blocks.%d.res2net_block.blocks.%d.conv.bias", bi, j);
            blk.res2net.blocks[j].b = T(buf);
            if (!blk.res2net.blocks[j].w) { loaded_ = false; return false; }
        }
    }

    mfa_.w = T("speaker.mfa.conv.weight");
    mfa_.b = T("speaker.mfa.conv.bias");
    asp_.tdnn.w = T("speaker.asp.tdnn.conv.weight");
    asp_.tdnn.b = T("speaker.asp.tdnn.conv.bias");
    asp_.conv_w = T("speaker.asp.conv.weight");
    asp_.conv_b = T("speaker.asp.conv.bias");
    fc_w_ = T("speaker.fc.weight");
    fc_b_ = T("speaker.fc.bias");

    if (!mfa_.w || !asp_.tdnn.w || !fc_w_) {
        loaded_ = false;
        return false;
    }

    // Allocate scheduler scratch space
    compute_meta_.resize(
        ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(4096, false));

    // Create scheduler
    {
        ggml_backend_t be_list[1] = {backend};
        sched_ = ggml_backend_sched_new(be_list, /*bufts=*/nullptr, 1, 4096, /*parallel=*/false, /*op_offload=*/false);
        if (!sched_) {
            if (error) *error += "failed to create backend scheduler for speaker encoder";
            loaded_ = false;
            return false;
        }
    }

    loaded_ = true;
    return true;
}

Qwen3TtsSpeakerEncoder::~Qwen3TtsSpeakerEncoder() {
    if (sched_) ggml_backend_sched_free(sched_);
}

// ═══════════════════════════════════════════════════════════════════════════
//  run_on_mel() — mel → embedding via ECAPA graph
// ═══════════════════════════════════════════════════════════════════════════

std::vector<float> Qwen3TtsSpeakerEncoder::run_on_mel(const float* mel_TC,
                                                       int T_mel) {
    if (!loaded_ || T_mel <= 0) return {};

    // Convert mel (T, 128) row-major → ggml [C=128, T] flat layout
    std::vector<float> mel_CT((size_t)128 * T_mel);
    for (int t = 0; t < T_mel; t++) {
        for (int c = 0; c < 128; c++) {
            mel_CT[(size_t)c + (size_t)t * 128] = mel_TC[(size_t)t * 128 + c];
        }
    }

    // Build graph
    ggml_init_params ip = {compute_meta_.size(), compute_meta_.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    ggml_tensor* h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 128, T_mel); // [128, T]
    ggml_set_name(h, "spk_mel");
    ggml_set_input(h);

    h = tdnn_block_(ctx0, h, blk0_, 1);
    ggml_set_name(h, "spk_blk0_out");
    ggml_set_output(h);

    static const int dilations[3] = {2, 3, 4};
    ggml_tensor* blk_outs[3];
    for (int i = 0; i < 3; i++) {
        h = se_res2net_(ctx0, h, blk_[i], dilations[i]);
        blk_outs[i] = h;
    }

    ggml_tensor* mfa_in = ggml_concat(ctx0,
        ggml_concat(ctx0, blk_outs[0], blk_outs[1], 0), blk_outs[2], 0);
    h = tdnn_block_(ctx0, mfa_in, mfa_, 1);
    ggml_set_name(h, "spk_mfa_out");
    ggml_set_output(h);

    h = asp_block_(ctx0, h, asp_); // [3072, 1]

    const int enc_dim = (int)hp.enc_dim;
    auto fcw = ggml_reshape_2d(ctx0, fc_w_, fc_w_->ne[1], fc_w_->ne[2]);
    h = ggml_add(ctx0, ggml_mul_mat(ctx0, fcw, h), fc_b_); // [enc_dim, 1]
    h = ggml_reshape_1d(ctx0, h, enc_dim);
    ggml_set_name(h, "spk_emb");
    ggml_build_forward_expand(gf, h);

    // Compute
    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) return {};

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "spk_mel"),
                            mel_CT.data(), 0, mel_CT.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(sched_, gf) != GGML_STATUS_SUCCESS)
        return {};

    std::vector<float> emb((size_t)enc_dim);
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "spk_emb"),
                            emb.data(), 0, (size_t)enc_dim * sizeof(float));

    ggml_free(ctx0);
    return emb;
}

// ═══════════════════════════════════════════════════════════════════════════
//  compute_embedding() — full pipeline: audio → mel → ECAPA → embedding
// ═══════════════════════════════════════════════════════════════════════════

std::vector<float> Qwen3TtsSpeakerEncoder::compute_embedding(
    const std::string& audio_path) {
    if (!loaded_) return {};

    std::string wav_err;
    auto audio = load_wav(audio_path, &wav_err);
    if (audio.empty()) {
        std::fprintf(stderr, "qwen3_tts: voice clone: %s\n", wav_err.c_str());
        return {};
    }

    int T_mel = 0;
    auto mel = compute_mel(audio.data(), (int)audio.size(), &T_mel);
    if (mel.empty() || T_mel <= 0) {
        std::fprintf(stderr, "qwen3_tts: voice clone: mel computation failed\n");
        return {};
    }

    auto emb = run_on_mel(mel.data(), T_mel);
    if (emb.empty()) {
        std::fprintf(stderr, "qwen3_tts: voice clone: ECAPA forward failed\n");
        return {};
    }

    return emb;
}

}  // namespace audiocore::qwen3_tts
