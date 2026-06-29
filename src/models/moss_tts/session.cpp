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
//   6. (Stage 16) Decode codec tokens -> PCM via MossCodecGraphs (ggml port
//      of openmoss/src/codec.cpp). Falls back to 1 s silence when the GGUF
//      carries no moss.codec.* tensors.
//   7. Write PCM into response
//
// Per-frame streaming: when mode="realtime"/"streaming" and a non-null
// stream callback is provided, the AR loop decodes newly available codec
// frames incrementally (via apply_de_delay_pattern on the growing delay
// buffer) and emits PCM through the callback in real-time. The first frame
// becomes available after N_VQ delay steps (~1.3s at 80ms/frame); subsequent
// frames emit at the codec frame rate (80ms). Response PCM is empty in
// streaming mode — all audio flows through the callback. Non-streaming use
// with a callback still works via post-hoc chunked emission (Stage 18).
//
// Everything transformer-shaped goes through qwen3::Runner (libllama). This
// file is everything that ISN'T the transformer: the audio-head projection,
// the sampler, the delay state machine, and the codec decode.

#include "audiocore/models/moss_tts/family.h"
#include "audiocore/models/moss_tts/projection.h"
#include "audiocore/models/moss_tts/delay_state.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

// ── Load codec tokens from a .codes binary file ─────────────────────────────
// Format: [n_frames: i32le] [codes: n_frames * N_VQ * i32le]
// Returns empty vector on error.
std::vector<std::vector<int32_t>> load_codec_tokens(const std::string& path,
                                                     int32_t expected_n_vq,
                                                     std::string* error) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) {
        if (error) *error = "voice_clone: cannot open " + path;
        return {};
    }
    int32_t n_frames = 0;
    if (fread(&n_frames, sizeof(int32_t), 1, fp) != 1) {
        if (error) *error = "voice_clone: " + path + " too short (n_frames)";
        fclose(fp); return {};
    }
    const int32_t n_vq = expected_n_vq;
    std::vector<std::vector<int32_t>> result;
    result.reserve(static_cast<size_t>(n_frames));
    for (int32_t i = 0; i < n_frames; i++) {
        std::vector<int32_t> frame(static_cast<size_t>(n_vq));
        size_t nread = fread(frame.data(), sizeof(int32_t),
                             static_cast<size_t>(n_vq), fp);
        if (static_cast<int32_t>(nread) != n_vq) {
            if (error)
                *error = "voice_clone: " + path + " truncated at frame " +
                         std::to_string(i);
            fclose(fp); return {};
        }
        result.push_back(std::move(frame));
    }
    fclose(fp);
    return result;
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

    if (req->text.empty() && req->messages.empty()) {
        if (error) *error = "empty text (or messages)";
        return false;
    }

    const int32_t hs = backbone_->hidden_size();
    const int32_t text_vocab = backbone_->vocab_size();
    const int32_t audio_vocab = cfg_.audio_vocab_size + 1;  // +1 for pad code

    // Validate mode
    const bool is_sfx          = (req->mode == "sfx");
    const bool is_clone        = (req->mode == "voice_clone");
    const bool is_dialogue     = (req->mode == "dialogue");
    const bool is_voice_design = (req->mode == "voice_design");
    const bool is_realtime     = (req->mode == "realtime" ||
                                  req->mode == "streaming");

    // ── Realtime / streaming modes ───────────────────────────────────────────
    // Per-frame streaming: when req->stream is provided (with on_audio
    // callback), the AR loop decodes codec frames incrementally and emits
    // PCM through the callback as they become available. The response PCM
    // is empty in streaming mode — all audio goes through the callback.
    // When req->stream is null, the fail-fast is preserved so callers know
    // to enable streaming in their request.
    if (is_realtime && (!req->stream || !req->stream->on_audio)) {
        if (error) *error =
            "moss_tts realtime / streaming requires a non-null stream "
            "callback (set stream.on_audio) for incremental codec decode. "
            "See GAPS.md §1.2.";
        return false;
    }

    // ── Voice Design (a.k.a. VoiceGenerator) mode ────────────────────────────
    // The dedicated VoiceGenerator model (1.7B) isn't loaded here. As a
    // best-effort fallback we route the voice description through the
    // flagship backbone's instruct slot with a voice-design system prompt.
    // Output is intelligible TTS with the requested voice characteristics
    // when the flagship can produce them; for parity with the upstream
    // VoiceGenerator you'd point this same request at a VoiceGenerator
    // GGUF instead.
    if (is_voice_design && req->instruct.empty()) {
        if (error) *error =
            "moss_tts voice_design mode requires a voice description in "
            "the 'instruct' field (e.g. 'a calm, deep female voice')";
        return false;
    }

    // ── System prompt by mode ────────────────────────────────────────────────
    std::string system_prompt;
    if (is_sfx) {
        system_prompt = "You are a sound effects generator.";
    } else if (is_dialogue && req->messages.empty()) {
        // Default dialogue system prompt when no messages array is provided.
        // The user's single `text` becomes the opening turn.
        system_prompt =
            "You are a spoken-dialogue assistant. Continue the conversation "
            "the user begins, alternating between speakers with natural "
            "conversational pacing.";
    } else if (is_voice_design) {
        system_prompt =
            "You are a voice cloning assistant. Speak the user's text in "
            "a voice matching this description: " + req->instruct + ".";
    } else {
        system_prompt = "You are a helpful voice assistant.";
    }

    // ======================================================================
    // (1) Tokenize text prompt
    // ======================================================================
    std::string templated;

    if (!req->messages.empty()) {
        // Multi-turn: use the messages array directly. The first message may
        // be a system prompt; if not, prepend the mode's default system prompt.
        std::vector<std::pair<std::string, std::string>> msgs;
        if (req->messages[0].role != "system") {
            msgs.emplace_back("system", system_prompt);
        }
        for (const auto& m : req->messages) {
            msgs.emplace_back(m.role, m.content);
        }
        if (!backbone_->apply_chat_template(msgs, /*add_assistant_prompt=*/true,
                                             &templated, error))
            return false;
    } else {
        if (req->text.empty()) {
            if (error) *error = "empty text";
            return false;
        }
        std::vector<std::pair<std::string, std::string>> msgs;
        msgs.emplace_back("system", system_prompt);
        msgs.emplace_back("user", req->text);
        if (!backbone_->apply_chat_template(msgs, /*add_assistant_prompt=*/true,
                                             &templated, error))
            return false;
    }

    // Append <|audio_start|> (signals the model to begin audio)
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
    const int32_t n_text = static_cast<int32_t>(text_tokens.size());

    // ======================================================================
    // (1b) Voice cloning: load reference audio codec tokens
    // ======================================================================
    std::vector<std::vector<int32_t>> ref_audio;
    if (is_clone) {
        if (req->voice_path.empty()) {
            if (error) *error = "voice_clone mode requires voice_path";
            return false;
        }
        ref_audio = load_codec_tokens(req->voice_path, cfg_.n_vq, error);
        if (ref_audio.empty()) {
            // error already set by load_codec_tokens
            if (error && error->empty())
                *error = "voice_clone: failed to load " + req->voice_path;
            return false;
        }
    }

    // ======================================================================
    // (2) Build multi-channel prompt: all token positions
    //     Each row = [text_token, code_0, code_1, ..., code_31]
    //     Text-only rows have AUDIO_PAD_CODE in audio channels.
    //     Audio-reference rows have GEN_SLOT as text + codec tokens in audio.
    // ======================================================================
    const int32_t n_audio_ref = is_clone ? static_cast<int32_t>(ref_audio.size()) : 0;
    const int32_t total_S = n_text + n_audio_ref;

    std::vector<std::vector<int32_t>> all_ids(
        static_cast<size_t>(total_S), std::vector<int32_t>(1 + N_VQ, AUDIO_PAD_CODE));

    // Fill text token row-channel 0
    for (int32_t i = 0; i < n_text; i++)
        all_ids[static_cast<size_t>(i)][0] = text_tokens[i];

    // Fill reference audio rows (after text)
    for (int32_t i = 0; i < n_audio_ref; i++) {
        const size_t row = static_cast<size_t>(n_text + i);
        all_ids[row][0] = AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID;
        for (int s = 0; s < N_VQ; s++)
            all_ids[row][1 + s] = ref_audio[i][static_cast<size_t>(s)];
    }

    // ======================================================================
    // (3) Build prompt embeddings
    // ======================================================================
    std::vector<float> prompt_embeds(static_cast<size_t>(total_S) * hs, 0.0f);

    for (int32_t i = 0; i < total_S; i++) {
        float* row = prompt_embeds.data() + static_cast<size_t>(i) * hs;
        const int32_t text_tok = all_ids[static_cast<size_t>(i)][0];

        // Gather text token embedding
        if (text_tok >= 0 && token_embd_) {
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

        // Sum audio embeddings for each stream (only if this is a ref audio frame)
        for (int s = 0; s < cfg_.n_vq && i >= n_text; ++s) {
            int32_t code = all_ids[static_cast<size_t>(i)][1 + s];
            if (code < 0) continue;
            ggml_tensor* W = audio_embed_[s];
            if (!W) continue;
            if (W->ne[0] != hs) continue;
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
    }

    // ======================================================================
    // (4) Prefill through backbone
    // ======================================================================
    std::vector<float> hidden_buf(static_cast<size_t>(total_S) * hs);
    if (!backbone_->forward_embeddings(prompt_embeds.data(), total_S, 0,
                                        hidden_buf.data(), error)) {
        return false;
    }

    // ======================================================================
    // (5) Delay state from full multi-channel prompt
    // ======================================================================
    DelayState state = init_delay_state(all_ids);

    // ── Sampling parameters by mode ──────────────────────────────────────────
    SamplingConfig samp_cfg;
    if (is_sfx) {
        // Sound effects: lower temps, higher top-k
        samp_cfg.text_temperature = 1.2f;
        samp_cfg.text_top_p       = 1.0f;
        samp_cfg.text_top_k       = 30;
        samp_cfg.audio_temperature = req->temperature > 0.0f
            ? req->temperature : 1.2f;
        samp_cfg.audio_top_p       = req->top_p > 0.0f
            ? req->top_p : 0.9f;
        samp_cfg.audio_top_k       = 20;
        samp_cfg.audio_repetition_penalty = 1.0f;
    } else if (is_dialogue) {
        // Dialogue (TTSD): higher text temperature for varied responses,
        // tighter audio nucleus to keep voices distinct.
        samp_cfg.text_temperature = 1.6f;
        samp_cfg.text_top_p       = 0.95f;
        samp_cfg.text_top_k       = 50;
        samp_cfg.audio_temperature = req->temperature > 0.0f
            ? req->temperature : 1.6f;
        samp_cfg.audio_top_p       = req->top_p > 0.0f
            ? req->top_p : 0.85f;
        samp_cfg.audio_top_k       = 25;
        samp_cfg.audio_repetition_penalty = 1.05f;  // dampen echo across turns
    } else if (is_voice_design) {
        // Voice Design: stay close to the standard TTS defaults so the
        // backbone spends its degrees of freedom on the voice, not on
        // sampling exploration.
        samp_cfg.text_temperature = 1.4f;
        samp_cfg.text_top_p       = 1.0f;
        samp_cfg.text_top_k       = 50;
        samp_cfg.audio_temperature = req->temperature > 0.0f
            ? req->temperature : 1.5f;
        samp_cfg.audio_top_p       = req->top_p > 0.0f
            ? req->top_p : 0.85f;
        samp_cfg.audio_top_k       = 30;
        samp_cfg.audio_repetition_penalty = 1.0f;
    } else {
        // TTS defaults
        samp_cfg.text_temperature = 1.5f;
        samp_cfg.text_top_p       = 1.0f;
        samp_cfg.text_top_k       = 50;
        samp_cfg.audio_temperature = req->temperature > 0.0f
            ? req->temperature : 1.7f;
        samp_cfg.audio_top_p       = req->top_p > 0.0f
            ? req->top_p : 0.8f;
        samp_cfg.audio_top_k       = 25;
        samp_cfg.audio_repetition_penalty = 1.0f;
    }

    // ======================================================================
    // (5) Autoregressive generation loop
    // ======================================================================
    const int32_t max_steps = req->max_new_tokens > 0
        ? req->max_new_tokens
        : cfg_.n_vq * 60 * 30;  // default: 60 fps * 30 s

    std::vector<float> step_embd(static_cast<size_t>(hs));
    std::vector<float> step_hidden(static_cast<size_t>(hs));
    std::vector<float> audio_logits_buf(
        static_cast<size_t>(cfg_.n_vq) * audio_vocab);

    int32_t pos = total_S;
    bool generated = false;
    bool streaming = is_realtime && req->stream && req->stream->on_audio;

    // Per-frame streaming: track how many de-delayed frames have been
    // decoded and emitted via the streaming callback.  Incremental decode
    // starts after at least N_VQ delay-buffer frames are accumulated
    // (the first de-delayed frame requires N_VQ delay steps).
    int32_t decoded_frames = 0;

    for (int32_t step = 0; step < max_steps; ++step) {
        const float* last_hidden = (step == 0)
            ? (hidden_buf.data() + static_cast<size_t>(total_S - 1) * hs)
            : step_hidden.data();

        const float* text_logits = backbone_->get_logits_ith(
            (step == 0) ? (total_S - 1) : 0);
        if (!text_logits) {
            if (error) *error = "get_logits_ith returned null";
            return false;
        }

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

        std::vector<int32_t> next_ids = delay_step(
            state, text_logits, text_vocab,
            audio_logits_buf.data(), audio_vocab,
            samp_cfg);

        embed_one_step(next_ids.data(), &step_embd, error);

        if (!backbone_->forward_embeddings(step_embd.data(), 1, pos,
                                            step_hidden.data(), error)) {
            return false;
        }
        pos++;
        generated = true;

        if (state.is_stopping) break;

        // ── Per-frame streaming: decode newly available frames ─────────────
        if (streaming && codec_graphs_.is_present()) {
            const int32_t n_delayed = static_cast<int32_t>(state.audio_buf.size());
            if (n_delayed >= N_VQ) {
                auto dedelayed = apply_de_delay_pattern(state.audio_buf);
                const int32_t n_available = static_cast<int32_t>(dedelayed.size());
                if (n_available > decoded_frames) {
                    // Clamp out-of-range codes
                    for (int32_t i = decoded_frames; i < n_available; i++) {
                        for (int32_t s = 0; s < N_VQ; s++) {
                            if (dedelayed[size_t(i)][size_t(s)] >= cfg_.audio_vocab_size)
                                dedelayed[size_t(i)][size_t(s)] = 0;
                        }
                    }
                    const int32_t n_new = n_available - decoded_frames;
                    std::vector<int32_t> flat(
                        static_cast<size_t>(cfg_.n_vq) * static_cast<size_t>(n_new));
                    for (int32_t t = 0; t < n_new; ++t) {
                        const auto& frame = dedelayed[decoded_frames + t];
                        for (int32_t v = 0; v < cfg_.n_vq; ++v) {
                            flat[size_t(v) * size_t(n_new) + size_t(t)] = frame[size_t(v)];
                        }
                    }
                    try {
                        auto pcm = codec_graphs_.decode(flat.data(), cfg_.n_vq, n_new);
                        if (!pcm.empty()) {
                            if (!req->stream->on_audio(pcm.data(), pcm.size())) {
                                if (error) *error = "moss_tts: streaming aborted by client";
                                return false;
                            }
                        }
                    } catch (const std::exception& e) {
                        std::fprintf(stderr, "moss_tts: streaming codec decode failed "
                                             "at step %d (%s); continuing\n",
                                     step, e.what());
                    }
                    decoded_frames = n_available;
                }
            }
        }
    }

    if (!generated) {
        if (error) *error = "zero generation steps";
        return false;
    }

    // ======================================================================
    // (6) Extract audio from generation output
    // ======================================================================
    auto audio_channels = state.audio_buf;
    auto segments = extract_audio_segments(audio_channels);

    // ── Streaming: flush any remaining frames ─────────────────────────────
    if (streaming && codec_graphs_.is_present() && !segments.empty()) {
        const int32_t n_remaining = static_cast<int32_t>(segments.size()) - decoded_frames;
        if (n_remaining > 0) {
            for (int32_t i = decoded_frames; i < static_cast<int32_t>(segments.size()); i++) {
                for (int32_t s = 0; s < N_VQ; s++) {
                    if (segments[size_t(i)][size_t(s)] >= cfg_.audio_vocab_size)
                        segments[size_t(i)][size_t(s)] = 0;
                }
            }
            const int32_t n_new = n_remaining;
            std::vector<int32_t> flat(
                static_cast<size_t>(cfg_.n_vq) * static_cast<size_t>(n_new));
            for (int32_t t = 0; t < n_new; ++t) {
                const auto& frame = segments[decoded_frames + t];
                for (int32_t v = 0; v < cfg_.n_vq; ++v) {
                    flat[size_t(v) * size_t(n_new) + size_t(t)] = frame[size_t(v)];
                }
            }
            try {
                auto pcm = codec_graphs_.decode(flat.data(), cfg_.n_vq, n_new);
                if (!pcm.empty()) {
                    if (!req->stream->on_audio(pcm.data(), pcm.size())) {
                        if (error) *error = "moss_tts: streaming aborted by client";
                        return false;
                    }
                }
            } catch (const std::exception& e) {
                std::fprintf(stderr, "moss_tts: streaming final codec decode failed (%s)\n", e.what());
            }
        }
        // Streaming: response PCM is empty (all audio went through callback)
        res->pcm_mono.clear();
        res->sampling_rate = cfg_.sampling_rate;
        return true;
    }

    if (segments.empty()) {
        if (error) *error = "no audio tokens generated";
        return false;
    }

    for (auto& frame : segments) {
        for (auto& code : frame) {
            if (code >= cfg_.audio_vocab_size) code = 0;
        }
    }

    // ======================================================================
    // (7) Decode codec -> PCM.
    //
    // Stage 16: if MossCodecGraphs is bound (the GGUF carries the
    // moss.codec.* tensors), route the de-delayed codec tokens through the
    // ggml port of openmoss/src/codec.cpp. Otherwise fall back to 1 s of
    // silence — the documented behavior for community backbone-only GGUFs
    // (see GAPS.md §1.3 and docs/CODEC_PORTS.md §1).
    // ======================================================================
    if (codec_graphs_.is_present()) {
        const int32_t T_audio = static_cast<int32_t>(segments.size());
        if (T_audio > 0) {
            std::vector<int32_t> flat(
                static_cast<size_t>(cfg_.n_vq) * static_cast<size_t>(T_audio));
            for (int32_t t = 0; t < T_audio; ++t) {
                const auto& frame = segments[size_t(t)];
                for (int32_t v = 0; v < cfg_.n_vq; ++v) {
                    flat[size_t(v) * size_t(T_audio) + size_t(t)] = frame[size_t(v)];
                }
            }
            try {
                res->pcm_mono = codec_graphs_.decode(flat.data(), cfg_.n_vq, T_audio);
                res->sampling_rate = cfg_.sampling_rate;
                // Non-streaming post-hoc: emit audio in 64 KiB chunks via callback
                if (req->stream && req->stream->on_audio) {
                    constexpr size_t kChunk = 32768;
                    const float* pcm = res->pcm_mono.data();
                    size_t total = res->pcm_mono.size();
                    for (size_t off = 0; off < total; off += kChunk) {
                        size_t n = std::min(kChunk, total - off);
                        if (!req->stream->on_audio(pcm + off, n)) {
                            if (error) *error = "moss_tts: streaming aborted by client";
                            return false;
                        }
                    }
                }
                (void)loudness_normalize;
                return true;
            } catch (const std::exception& e) {
                if (error) *error = std::string("moss_tts codec decode failed: ") + e.what();
                return false;
            }
        }
        // Empty segment list — emit empty PCM, not silence.
        res->pcm_mono.clear();
        res->sampling_rate = cfg_.sampling_rate;
        return true;
    }

    // Silence fallback — codec tensors not present in this GGUF.
    std::fprintf(stderr, "moss_tts: codec tensors not bound "
                         "(no moss.codec.* in GGUF); emitting 1s silence\n");
    (void)loudness_normalize;
    (void)segments;
    res->pcm_mono.assign(static_cast<size_t>(cfg_.sampling_rate), 0.0f);
    res->sampling_rate = cfg_.sampling_rate;
    return true;
}

bool MossSession::decode_codec(const int32_t* codec_tokens, int32_t n_tokens,
                                std::vector<float>* pcm_out,
                                std::string* error) {
    // Stage 16: route through the ggml port when codec tensors are bound.
    // `codec_tokens` here is (n_tokens, n_vq) row-major — flatten into the
    // (n_vq, T_audio) layout MossCodecGraphs::decode expects.
    if (codec_graphs_.is_present() && n_tokens > 0) {
        std::vector<int32_t> flat(static_cast<size_t>(cfg_.n_vq) *
                                   static_cast<size_t>(n_tokens));
        for (int32_t v = 0; v < cfg_.n_vq; ++v) {
            for (int32_t t = 0; t < n_tokens; ++t) {
                flat[size_t(v) * size_t(n_tokens) + size_t(t)] =
                    codec_tokens[size_t(t) * size_t(cfg_.n_vq) + size_t(v)];
            }
        }
        try {
            *pcm_out = codec_graphs_.decode(flat.data(), cfg_.n_vq, n_tokens);
            return true;
        } catch (const std::exception& e) {
            if (error) *error = std::string("codec decode failed: ") + e.what();
            return false;
        }
    }

    // Silence fallback — codec tensors not present in this GGUF.
    (void)error;
    pcm_out->assign(static_cast<size_t>(cfg_.sampling_rate), 0.0f);
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
