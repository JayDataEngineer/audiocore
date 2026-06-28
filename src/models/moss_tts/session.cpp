// session.cpp — MOSS-TTS full embedding-based generation pipeline.
//
// Pipeline:
//   1. Tokenize text via Qwen3 chat template + append <|audio_start|>
//   2. Build summed text+audio embeddings for the prompt
//   3. Prefill: forward embeddings through Qwen3 backbone
//   4. Autoregressive loop:
//      a. Get last hidden state from backbone
//      b. Project through audio_head.{i}.weight -> per-stream codec logits
//      c. Get text logits from backbone output
//      d. delay_step -> sample next (1+N_VQ) token vector
//      e. Embed the token vector (text_embed + sum audio_embed[i])
//      f. Forward single embedding through backbone (incremental KV)
//   5. Apply de-delay pattern to audio channels
//   6. Decode codec tokens -> PCM via ONNX Runtime
//   7. Write PCM into response
//
// Everything transformer-shaped goes through qwen3::Runner (libllama). This
// file is everything that ISN'T the transformer: the audio-head projection,
// the sampler, the delay state machine, and the codec decode.

#include "audiocore/models/moss_tts/family.h"
#include "audiocore/models/moss_tts/projection.h"
#include "audiocore/models/moss_tts/delay_state.h"
#include "audiocore/models/moss_tts/codec.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

#include "ggml.h"

namespace audiocore::moss {

namespace {

// Loudness normalization: target -20 dBFS with +/- 3 dB gain range.
// Matches upstream pipeline.py loudness_normalize().
void loudness_normalize(std::vector<float>& wav) {
    if (wav.empty()) return;
    double sum_sq = 0.0;
    for (float s : wav) sum_sq += static_cast<double>(s) * s;
    double rms = std::sqrt(sum_sq / wav.size() + 1e-9);
    double current_dbfs = 20.0 * std::log10(rms);
    double gain = std::clamp(-20.0 - current_dbfs, -3.0, 3.0);
    float factor = static_cast<float>(std::pow(10.0, gain / 20.0));
    for (float& s : wav) s *= factor;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// embed_one_step -- sum text_embed + audio_embed[stream] for one token vector
// ---------------------------------------------------------------------------
bool MossSession::embed_one_step(const int32_t* tokens,
                                  std::vector<float>* embd_out,
                                  std::string* error) {
    // tokens[0] = text token, tokens[1..N_VQ] = audio codec tokens.
    const int32_t hs = backbone_->hidden_size();
    (void)error;
    embd_out->assign(static_cast<size_t>(hs), 0.0f);

    // Text token embedding
    int32_t text_tok = tokens[0];
    if (text_tok >= 0 && token_embd_) {
        float* row = embd_out->data();
        const size_t row_bytes = static_cast<size_t>(hs) *
            (token_embd_->type == GGML_TYPE_F32 ? sizeof(float)
                                                 : sizeof(ggml_fp16_t));
        if (static_cast<size_t>(text_tok) * row_bytes + row_bytes <= ggml_nbytes(token_embd_)) {
            const char* src = static_cast<const char*>(token_embd_->data)
                              + static_cast<size_t>(text_tok) * row_bytes;
            if (token_embd_->type == GGML_TYPE_F32) {
                std::memcpy(row, src, row_bytes);
            } else if (token_embd_->type == GGML_TYPE_F16) {
                const ggml_fp16_t* s = reinterpret_cast<const ggml_fp16_t*>(src);
                for (int j = 0; j < hs; ++j)
                    row[j] = ggml_fp16_to_fp32(s[j]);
            }
        }
    }

    // Sum audio embeddings for each stream
    for (int s = 0; s < cfg_.n_vq; ++s) {
        int32_t code = tokens[1 + s];
        if (code < 0) continue;
        ggml_tensor* W = audio_embed_[s];
        if (!W) continue;
        if (W->ne[0] != hs) continue;
        float* row = embd_out->data();
        const size_t row_off = static_cast<size_t>(code) * hs;
        if (row_off + static_cast<size_t>(hs) > ggml_nbytes(W) /
            (W->type == GGML_TYPE_F32 ? sizeof(float) : sizeof(ggml_fp16_t)))
            continue;
        if (W->type == GGML_TYPE_F32) {
            const float* wrow = static_cast<const float*>(W->data) + row_off;
            for (int j = 0; j < hs; ++j)
                row[j] += wrow[j];
        } else if (W->type == GGML_TYPE_F16) {
            const ggml_fp16_t* wrow = static_cast<const ggml_fp16_t*>(W->data) + row_off;
            for (int j = 0; j < hs; ++j)
                row[j] += ggml_fp16_to_fp32(wrow[j]);
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// run_tts
// ---------------------------------------------------------------------------
bool MossSession::run_tts(const void* request, void* response,
                          std::string* error) {
    if (!loaded_) {
        if (error) *error = "MossSession not loaded";
        return false;
    }
    const auto* req = static_cast<const TtsRequest*>(request);
    auto*       res = static_cast<TtsResponse*>(response);
    if (!req || !res) {
        if (error) *error = "null request/response";
        return false;
    }
    res->sampling_rate = cfg_.sampling_rate;

    if (req->text.empty()) {
        if (error) *error = "empty text";
        return false;
    }

    const int32_t hs = backbone_->hidden_size();
    const int32_t text_vocab = backbone_->vocab_size();
    const int32_t audio_vocab = cfg_.audio_vocab_size + 1;  // +1 for pad code

    // ======================================================================
    // (1) Tokenize prompt
    // ======================================================================
    std::string templated;
    if (!backbone_->apply_chat_template(
            {{"system", "You are a helpful voice assistant."},
             {"user",   req->text}},
            /*add_assistant_prompt=*/true,
            &templated, error)) {
        return false;
    }

    // Get the string form of <|audio_start|>
    std::string audio_start_str;
    if (!backbone_->token_to_piece(AUDIO_START_TOKEN_ID, &audio_start_str, error))
        return false;
    templated += audio_start_str;

    std::vector<int32_t> text_tokens;
    if (!backbone_->tokenize(templated, /*add_special=*/false,
                             /*parse_special=*/true, &text_tokens,
                             nullptr, error)) {
        return false;
    }
    if (text_tokens.empty()) {
        if (error) *error = "tokenize returned zero tokens";
        return false;
    }
    const int32_t S = static_cast<int32_t>(text_tokens.size());

    // ======================================================================
    // (2) Build prompt embedding
    // ======================================================================
    // Zero-shot TTS: prompt has text tokens only — no reference audio, so no
    // audio embedding contributions to add. (In few-shot mode, reference audio
    // tokens would be summed via embed_one_step per position.)
    std::vector<float> prompt_embeds(static_cast<size_t>(S) * hs, 0.0f);

    for (int32_t i = 0; i < S; ++i) {
        float* row = prompt_embeds.data() + static_cast<size_t>(i) * hs;

        // Gather text token embedding
        int32_t tok = text_tokens[i];
        if (tok >= 0 && token_embd_) {
            const size_t row_bytes = static_cast<size_t>(hs) *
                (token_embd_->type == GGML_TYPE_F32 ? sizeof(float)
                                                     : sizeof(ggml_fp16_t));
            if (static_cast<size_t>(tok) * row_bytes + row_bytes <= ggml_nbytes(token_embd_)) {
                const char* src = static_cast<const char*>(token_embd_->data)
                                  + static_cast<size_t>(tok) * row_bytes;
                if (token_embd_->type == GGML_TYPE_F32) {
                    std::memcpy(row, src, row_bytes);
                } else if (token_embd_->type == GGML_TYPE_F16) {
                    const ggml_fp16_t* s = reinterpret_cast<const ggml_fp16_t*>(src);
                    for (int j = 0; j < hs; ++j)
                        row[j] = ggml_fp16_to_fp32(s[j]);
                }
            }
        }
        // No audio embedding added for zero-shot prompt positions.
        // (audio_embed tables have 1024 entries 0-1023; AUDIO_PAD_CODE=1024
        //  is outside the valid range and would be a buffer over-read.)
    }

    // ======================================================================
    // (3) Prefill through backbone
    // ======================================================================
    std::vector<float> hidden_buf(static_cast<size_t>(S) * hs);
    if (!backbone_->forward_embeddings(prompt_embeds.data(), S, 0,
                                        hidden_buf.data(), error)) {
        return false;
    }

    // ======================================================================
    // (4) Build multi-channel prompt for delay state
    // ======================================================================
    std::vector<std::vector<int32_t>> prompt_ids(
        S, std::vector<int32_t>(1 + N_VQ, AUDIO_PAD_CODE));
    for (int32_t i = 0; i < S; ++i)
        prompt_ids[i][0] = text_tokens[i];

    DelayState state = init_delay_state(prompt_ids);

    SamplingConfig samp_cfg;
    samp_cfg.text_temperature = 1.5f;
    samp_cfg.text_top_p = 1.0f;
    samp_cfg.text_top_k = 50;
    samp_cfg.audio_temperature = req->temperature > 0.0f
        ? req->temperature : 1.7f;
    samp_cfg.audio_top_p = req->top_p > 0.0f
        ? req->top_p : 0.8f;
    samp_cfg.audio_top_k = 25;
    samp_cfg.audio_repetition_penalty = 1.0f;

    // ======================================================================
    // (5) Autoregressive generation loop
    // ======================================================================
    const int32_t max_steps = req->max_tokens > 0
        ? req->max_tokens
        : cfg_.n_vq * 60 * 30;  // default: 60 fps * 30 s

    std::vector<float> step_embd(static_cast<size_t>(hs));
    std::vector<float> step_hidden(static_cast<size_t>(hs));
    std::vector<float> audio_logits_buf(
        static_cast<size_t>(cfg_.n_vq) * audio_vocab);

    int32_t pos = S;
    bool generated = false;

    for (int32_t step = 0; step < max_steps; ++step) {
        // (a) Get last hidden state
        const float* last_hidden = (step == 0)
            ? (hidden_buf.data() + static_cast<size_t>(S - 1) * hs)
            : step_hidden.data();

        // (b) Get text logits from backbone (from the just-decoded position)
        const float* text_logits = backbone_->get_logits_ith(
            (step == 0) ? (S - 1) : 0);
        if (!text_logits) {
            if (error) *error = "get_logits_ith returned null";
            return false;
        }

        // (c) Project last hidden through audio_head
        ProjectionRefs refs;
        refs.n_tokens    = 1;
        refs.hidden_size = hs;
        refs.n_vq        = cfg_.n_vq;
        refs.vocab       = audio_vocab;
        refs.hidden      = last_hidden;
        refs.heads       = audio_head_;

        bool proj_ok = project_logits_cgraph(refs, audio_logits_buf.data(), error);
        if (!proj_ok) {
            std::string ref_err;
            proj_ok = project_logits_reference(refs, audio_logits_buf.data(), &ref_err);
            if (!proj_ok) {
                if (error) *error = "audio head projection failed: " + ref_err;
                return false;
            }
        }

        // (d) Delay step -> sample next tokens
        std::vector<int32_t> next_ids = delay_step(
            state, text_logits, text_vocab,
            audio_logits_buf.data(), audio_vocab,
            samp_cfg);

        // (e) Embed the sampled token vector
        embed_one_step(next_ids.data(), &step_embd, error);

        // (f) Forward single embedding (incremental KV)
        if (!backbone_->forward_embeddings(step_embd.data(), 1, pos,
                                            step_hidden.data(), error)) {
            return false;
        }
        pos++;
        generated = true;

        // (g) Termination check
        if (state.is_stopping) break;
    }

    if (!generated) {
        if (error) *error = "zero generation steps";
        return false;
    }

    // ======================================================================
    // (6) Extract audio from generation output
    // ======================================================================
    // Build audio channel view from the delay state's audio buffer
    // (everything that was appended during the loop)
    auto audio_channels = state.audio_buf;  // copy

    // De-delay + extract non-padding segments
    auto segments = extract_audio_segments(audio_channels);

    if (segments.empty()) {
        if (error) *error = "no audio tokens generated";
        return false;
    }

    // Replace any remaining AUDIO_PAD_CODE (1024) with 0 — the codec embedding
    // table has 1024 entries (0-1023) and pad 1024 is out of range.
    for (auto& frame : segments) {
        for (auto& code : frame) {
            if (code >= cfg_.audio_vocab_size) code = 0;
        }
    }

    // ======================================================================
    // (7) Decode codec -> PCM via ONNX
    // ======================================================================
    if (decoder_onnx_path_.empty()) {
        if (error) *error = "decoder_onnx path not configured";
        return false;
    }

    OnnxDecoder decoder;
    if (!decoder.load(decoder_onnx_path_, /*use_gpu=*/false, error))
        return false;

    if (!decoder.decode(segments, &res->pcm_mono, error))
        return false;

    loudness_normalize(res->pcm_mono);

    return true;
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
MossSession::~MossSession() {
    if (owns_ext_ctx_ && ext_ctx_) {
        ggml_free(ext_ctx_);
        ext_ctx_ = nullptr;
    }
}

}  // namespace audiocore::moss
