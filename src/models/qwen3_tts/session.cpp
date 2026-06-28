// session.cpp — Qwen3-TTS inference pipeline.
//
// Pipeline (official Qwen3-TTS):
//   text / instruct ─→ tokenize ─→ text_embedding + text_projection ─→
//   sum with codec prefix (codec_pad × T + codec_bos) ─→
//   Talker forward ─→ hidden @ codec_head ─→ codebook 0 ─→
//   Code Predictor MTP loop (31 fine codebooks) ─→
//   [32 × T] code matrix ─→ Speech Tokenizer ─→ PCM audio
//
// Both transformers (talker + predictor) run through the unified
// qwen3::Runner — the same class MOSS and ACE-Step use. There is no longer
// a separate TalkerRunner or PredictorRunner in audiocore.

#include "audiocore/models/qwen3_tts/family.h"
#include "audiocore/models/qwen3/runner.h"
#include "audiocore/framework/sampling/sampler.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace audiocore::qwen3_tts {

using audiocore::qwen3::Runner;
using audiocore::qwen3::RunnerConfig;
using audiocore::sampler::Params;
using audiocore::sampler::sample_token;

// ── Codec special tokens (from official Qwen3-TTS config) ────────────────
static constexpr int32_t CODEC_PAD    = 4196;
static constexpr int32_t CODEC_BOS    = 4197;
static constexpr int32_t CODEC_EOS    = 4198;
static constexpr int32_t CODEC_THINK  = 4202;
static constexpr int32_t CODEC_NOT    = 4203;
static constexpr int32_t CODEC_TBOS   = 4204;
static constexpr int32_t CODEC_TEOS   = 4205;

// Default speaker name → codec token mapping (from official config)
// These are approximate; real values come from the model's config.json.
static int32_t speaker_to_token(const std::string& name) {
    static const std::unordered_map<std::string, int32_t> spk_map = {
        {"vivian", 4206}, {"ryan", 4207}, {"sarah", 4208},
        {"alex", 4209}, {"emma", 4210}, {"james", 4211},
        {"olivia", 4212}, {"liam", 4213}, {"sophia", 4214},
    };
    auto it = spk_map.find(name);
    if (it != spk_map.end()) return it->second;
    // Unknown speaker: use default
    return -1;
}

// Language → codec token mapping (from official config)
static int32_t language_to_token(const std::string& lang) {
    static const std::unordered_map<std::string, int32_t> lang_map = {
        {"zh", 4206}, {"en", 4207}, {"ja", 4208}, {"ko", 4209},
        {"de", 4210}, {"fr", 4211}, {"ru", 4212}, {"pt", 4213},
        {"es", 4214}, {"it", 4215},
    };
    auto it = lang_map.find(lang);
    if (it != lang_map.end()) return it->second;
    return -1;
}

// Parse a mode string from the unified TtsRequest.mode field into the
// Qwen3-TTS-specific enum. Unknown / empty / "tts" → plain batch TTS, the
// safe default that works on every variant.
Qwen3TtsMode parse_mode(const std::string& s) {
    std::string m;
    m.reserve(s.size());
    for (char c : s) m.push_back(static_cast<char>(std::tolower(c)));
    if (m == "voice_design" || m == "voicedesign") return Qwen3TtsMode::VoiceDesign;
    if (m == "voice_clone"  || m == "voiceclone")  return Qwen3TtsMode::VoiceClone;
    if (m == "streaming")                        return Qwen3TtsMode::Streaming;
    return Qwen3TtsMode::TtsBatch;
}

// Voice Design mode system prompt. The official VoiceDesign variant trains
// the model to accept a natural-language voice description in the instruct
// slot, prefixed with a "Generate a voice with the following characteristics"
// template. We use the same template on the Base / CustomVoice backbones as
// a best-effort fallback — output will be intelligible but less polished
// than the dedicated VoiceDesign weights.
static const char* kVoiceDesignInstructPrefix =
    "Generate a voice with the following characteristics: ";

// ── Sampling ────────────────────────────────────────────────────────────
// qwen3_tts uses the unified audiocore::sampler. A small lambda below at
// each call site builds a Params struct from the request fields and hands
// the session RNG to it.

// ── Constructor / Destructor ────────────────────────────────────────────

Qwen3TtsSession::Qwen3TtsSession() = default;
Qwen3TtsSession::~Qwen3TtsSession() = default;

// ── Inference pipeline ──────────────────────────────────────────────────

bool Qwen3TtsSession::run_inference(const TtsRequest& req, TtsResponse& resp,
                                    std::string* error) {
    if (!talker_ || !predictor_) {
        if (error) *error = "session not loaded";
        return false;
    }

    // ── Mode routing ──────────────────────────────────────────────────────
    //
    // qwen3_tts understands four modes from the unified TtsRequest.mode
    // field. voice_clone + streaming are NOT implemented and fail fast with
    // a pointer at GAPS.md so callers know it's a known gap, not a bug.
    const Qwen3TtsMode mode = parse_mode(req.mode);
    if (mode == Qwen3TtsMode::VoiceClone) {
        if (error) *error =
            "qwen3_tts voice_clone requires the ECAPA-TDNN speaker encoder "
            "(not yet ported to ggml). See GAPS.md §2.3 for the port plan.";
        return false;
    }
    if (mode == Qwen3TtsMode::Streaming) {
        if (error) *error =
            "qwen3_tts streaming requires chunked HTTP + Dual-Track "
            "incremental decode (not implemented). See GAPS.md §2.2.";
        return false;
    }

    // Voice Design mode: the instruct field carries the voice description.
    // We prefix it with the VoiceDesign template so the model knows to
    // synthesize the voice before speaking. If instruct is empty we still
    // proceed (the dedicated VoiceDesign variant will produce a generic
    // voice; the Base variant will just do plain TTS).
    std::string effective_instruct = req.instruct;
    if (mode == Qwen3TtsMode::VoiceDesign) {
        if (effective_instruct.empty()) {
            std::fprintf(stderr,
                "qwen3_tts: voice_design mode with empty instruct — "
                "no voice description provided; proceeding with defaults\n");
        } else {
            effective_instruct = std::string(kVoiceDesignInstructPrefix)
                                 + effective_instruct;
        }
    }

    const int32_t codec_vocab = talker_->codec_vocab();
    const int32_t n_embd = talker_->n_embd();
    std::mt19937 rng(42);

    // ═══════════════════════════════════════════════════════════════════
    //  Phase 1: Build input embeddings
    // ═══════════════════════════════════════════════════════════════════

    // Determine total prefill length:
    // [instruct tokens (if any)] + [text tokens] + [speaker token (if any)]
    //                                + [codec_bos]
    // Each text token is summed with codec_pad. codec_bos gets zero text.

    // Tokenize main text
    std::vector<int32_t> text_tokens;
    if (!talker_->tokenize(req.text, /*add_special=*/true,
                           /*parse_special=*/false, &text_tokens,
                           /*needed=*/nullptr, error)) {
        return false;
    }
    if (text_tokens.empty()) {
        if (error) *error = "empty input after tokenization";
        return false;
    }
    std::fprintf(stderr, "qwen3_tts: %zu text tokens (mode=%d, variant=%s)\n",
                 text_tokens.size(),
                 static_cast<int>(mode),
                 variant_name(config_.variant));

    // Resolve speaker_name → codec token. The CustomVoice variant defines
    // nine default speakers; the Base variant ignores the field. Unknown
    // names warn and skip — same behavior as before this stage but no
    // longer silent.
    int32_t speaker_token = -1;
    if (!req.speaker_name.empty()) {
        speaker_token = speaker_to_token(req.speaker_name);
        if (speaker_token < 0) {
            std::fprintf(stderr,
                "qwen3_tts: unknown speaker '%s' — ignoring (known: vivian, "
                "ryan, sarah, alex, emma, james, olivia, liam, sophia)\n",
                req.speaker_name.c_str());
        } else {
            std::fprintf(stderr,
                "qwen3_tts: speaker '%s' → codec token %d\n",
                req.speaker_name.c_str(), speaker_token);
        }
    }

    // Tokenize instruct (optional)
    std::vector<int32_t> instruct_tokens;
    bool has_instruct = !effective_instruct.empty();
    if (has_instruct) {
        if (!talker_->tokenize(effective_instruct, /*add_special=*/true,
                               /*parse_special=*/false, &instruct_tokens,
                               /*needed=*/nullptr, error)) {
            return false;
        }
        std::fprintf(stderr, "qwen3_tts: %zu instruct tokens\n", instruct_tokens.size());
    }

    // Build text embedding sequence: [instruct...] + [text...]
    std::string full_text;
    if (has_instruct) full_text = effective_instruct;
    full_text += req.text;

    std::vector<float> text_embd;
    int32_t n_text_tokens = 0;
    if (talker_->has_text_embedding()) {
        if (!talker_->compute_text_embedding(full_text, &text_embd,
                                              &n_text_tokens, error)) {
            return false;
        }
    } else {
        // Fallback for Lunavox GGUFs: no text_embedding — use token IDs directly
        n_text_tokens = (int32_t)text_tokens.size();
    }

    // Build codec prefix sequence.
    //   [codec_pad × n_text_tokens] + [speaker_token?] + [codec_bos]
    // The optional speaker_token slot only appears when the request names
    // a known default speaker (CustomVoice variant) — that injects the
    // "<|spk_NAME|>" codec token Qwen3-TTS uses to switch voice profile.
    const bool has_speaker = (speaker_token >= 0);
    const int32_t prefix_len = n_text_tokens + (has_speaker ? 1 : 0) + 1;

    std::vector<float> codec_prefix((size_t)prefix_len * n_embd, 0.0f);

    if (talker_->codec_embedding()) {
        // codec_pad for all text+instruct positions
        for (int32_t i = 0; i < n_text_tokens; i++) {
            const float* pad_row = talker_->codec_embedding() + (size_t)CODEC_PAD * n_embd;
            std::memcpy(&codec_prefix[(size_t)i * n_embd], pad_row,
                        (size_t)n_embd * sizeof(float));
        }
        int32_t cursor = n_text_tokens;
        // Optional speaker slot (CustomVoice variant): one codec token row.
        if (has_speaker) {
            if (speaker_token < talker_->codec_vocab()) {
                const float* spk_row =
                    talker_->codec_embedding() + (size_t)speaker_token * n_embd;
                std::memcpy(&codec_prefix[(size_t)cursor * n_embd], spk_row,
                            (size_t)n_embd * sizeof(float));
            } else {
                std::fprintf(stderr,
                    "qwen3_tts: speaker token %d out of codec vocab range %d; "
                    "skipping speaker slot\n",
                    speaker_token, talker_->codec_vocab());
            }
            cursor += 1;
        }
        // codec_bos for the final position.
        const float* bos_row = talker_->codec_embedding() + (size_t)CODEC_BOS * n_embd;
        std::memcpy(&codec_prefix[(size_t)cursor * n_embd], bos_row,
                    (size_t)n_embd * sizeof(float));
    } else {
        std::fprintf(stderr, "qwen3_tts: no codec embedding table!\n");
    }

    // Sum text embeddings + codec prefix
    // Positions 0..n_text_tokens-1: text_emb + codec_pad_emb
    // Position n_text_tokens: 0 + codec_bos_emb
    std::vector<float> talker_input((size_t)prefix_len * n_embd);

    if (talker_->has_text_embedding() && !text_embd.empty()) {
        for (int32_t i = 0; i < n_text_tokens; i++) {
            for (int32_t j = 0; j < n_embd; j++) {
                size_t idx = (size_t)i * n_embd + j;
                talker_input[idx] = text_embd[idx] + codec_prefix[idx];
            }
        }
    } else {
        // Lunavox fallback: zero text embedding, just codec prefix
        std::memcpy(talker_input.data(), codec_prefix.data(),
                    (size_t)prefix_len * n_embd * sizeof(float));
    }

    // Copy codec_bos position (position n_text_tokens)
    std::memcpy(&talker_input[(size_t)n_text_tokens * n_embd],
                &codec_prefix[(size_t)n_text_tokens * n_embd],
                (size_t)n_embd * sizeof(float));

    // ═══════════════════════════════════════════════════════════════════
    //  Phase 2: Talker forward pass (prefill)
    // ═══════════════════════════════════════════════════════════════════

    std::vector<float> talker_hidden((size_t)prefix_len * n_embd);
    if (talker_->has_text_embedding()) {
        // Use embedding-mode forward (text_embd + codec_embd summed)
        if (!talker_->forward_embeddings(talker_input.data(), prefix_len, 0,
                                         talker_hidden.data(), error)) {
            return false;
        }
    } else {
        // Lunavox fallback: use token-mode with text tokens + codec_bos
        std::vector<int32_t> token_input(text_tokens);
        token_input.push_back(CODEC_BOS);
        if (!talker_->forward_get_hidden(token_input.data(), (int32_t)token_input.size(),
                                         0, talker_hidden.data(), error)) {
            return false;
        }
    }

    // The last token's hidden state (codec_bos position) is the conditioning
    // vector for the code predictor.
    const float* last_hidden = &talker_hidden[(size_t)(prefix_len - 1) * n_embd];
    std::fprintf(stderr, "qwen3_tts: talker prefill done (%d tokens)\n", prefix_len);

    // ═══════════════════════════════════════════════════════════════════
    //  Phase 3: Autoregressive decode
    // ═══════════════════════════════════════════════════════════════════
    //
    // At each step:
    //   1. Apply codec_head to last hidden → codebook 0 logits → sample
    //   2. MTP loop → 31 fine codebook IDs
    //   3. Store [32 × step] codes
    //   4. If EOS detected for all codebooks, stop
    //   5. Build next input: sum of all 32 code embeddings + tts_pad
    //   6. Run talker on 1 token → new hidden state

    const int32_t max_steps = req.max_new_tokens > 0
        ? req.max_new_tokens : config_.max_new_tokens;
    const int32_t n_total_books = 32;
    const int32_t n_fine_books = 31;

    // Code matrix: [n_total_books × max_steps]
    std::vector<int32_t> code_matrix((size_t)n_total_books * max_steps, 0);

    // Previous step's fine codes (31 zeros for first step)
    std::vector<int32_t> prev_fine((size_t)n_fine_books, 0);

    // Current hidden state (starts as last_hidden from prefill)
    std::vector<float> cur_hidden(last_hidden, last_hidden + n_embd);

    for (int32_t step = 0; step < max_steps; step++) {
        // ── 3a. Sample codebook 0 via codec_head ───
        int32_t code_0 = 0;
        if (talker_->codec_head()) {
            const float* head = talker_->codec_head();  // [codec_vocab, n_embd]
            std::vector<float> c0_logits((size_t)codec_vocab);
            for (int j = 0; j < codec_vocab; j++) {
                float s = 0;
                for (int d = 0; d < n_embd; d++) {
                    s += cur_hidden[(size_t)d] * head[(size_t)j * n_embd + d];
                }
                c0_logits[(size_t)j] = s;
            }
            code_0 = [&] {
                Params sp;
                sp.temperature = req.temperature;
                sp.top_p       = req.top_p;
                return sample_token(c0_logits.data(), codec_vocab, sp,
                                    /*prev_tokens=*/nullptr, /*n_prev=*/0, &rng);
            }();
        }
        code_matrix[(size_t)0 * max_steps + step] = code_0;

        // ── 3b. MTP loop: predict 31 fine codebooks ───
        std::vector<int32_t> fine_codes((size_t)n_fine_books, 0);

        // Build prev_codes for the predictor:
        //   [code_0, prev_fine[0], ..., prev_fine[30]]
        std::vector<int32_t> pred_input((size_t)n_total_books);
        pred_input[0] = code_0;
        for (int i = 0; i < n_fine_books; i++) {
            pred_input[(size_t)(i + 1)] = prev_fine[(size_t)i];
        }

        bool mtp_ok = false;
        if (predictor_->has_mtp()) {
            if (predictor_->predict_one_step(cur_hidden.data(),
                                             pred_input.data(),
                                             n_fine_books,
                                             fine_codes.data(), error)) {
                mtp_ok = true;
            }
        }

        if (!mtp_ok) {
            // Fallback: use standard predictor forward
            std::vector<float> logits((size_t)config_.n_codebooks * predictor_->vocab_size());
            if (predictor_->forward_tokens(pred_input.data(), n_total_books,
                                           step * n_total_books,
                                           logits.data(), error)) {
                for (int cb = 0; cb < n_fine_books; cb++) {
                    int32_t cb_vocab = predictor_->vocab_size() / n_total_books;
                    const float* cb_logits = logits.data() +
                        (size_t)(cb + 1) * cb_vocab;
                    Params sp;
                    sp.temperature = req.temperature;
                    sp.top_p       = req.top_p;
                    fine_codes[(size_t)cb] = sample_token(
                        cb_logits, cb_vocab, sp,
                        /*prev_tokens=*/nullptr, /*n_prev=*/0, &rng);
                }
            }
        }

        // Store fine codes
        for (int i = 0; i < n_fine_books; i++) {
            code_matrix[(size_t)(i + 1) * max_steps + step] = fine_codes[(size_t)i];
        }

        // Update prev_fine for next step
        prev_fine = fine_codes;

        // ── 3c. Check for EOS ───
        bool all_eos = true;
        for (int cb = 0; cb < n_total_books; cb++) {
            if (code_matrix[(size_t)cb * max_steps + step] != 0) {
                all_eos = false;
                break;
            }
        }
        if (all_eos && step > 0) {
            std::fprintf(stderr, "qwen3_tts: EOS at step %d\n", step);
            break;
        }

        // ── 3d. Build next input embedding ───
        // Sum all 32 code embeddings + tts_pad
        std::vector<float> next_embd((size_t)n_embd, 0.0f);

        if (talker_->codec_embedding()) {
            // Coarse code (codebook 0) uses talker's codec_embedding
            int32_t c0 = std::max(0, code_0);
            if (c0 < talker_->codec_vocab()) {
                const float* c0_row = talker_->codec_embedding() + (size_t)c0 * n_embd;
                for (int d = 0; d < n_embd; d++) next_embd[(size_t)d] += c0_row[d];
            }

            // Fine codes use predictor's fine_embd tables
            for (int i = 0; i < n_fine_books; i++) {
                int32_t cid = fine_codes[(size_t)i];
                if (predictor_->has_mtp()) {
                    const float* fi_row = predictor_->fine_embedding(i);
                    if (fi_row && cid >= 0) {
                        cid = cid % predictor_->vocab_size();
                        const float* row = fi_row + (size_t)cid * n_embd;
                        for (int d = 0; d < n_embd; d++) next_embd[(size_t)d] += row[d];
                    }
                }
            }
        }

        // Run talker on this single token to get next hidden
        if (!talker_->forward_embeddings(next_embd.data(), 1,
                                         prefix_len + step,
                                         cur_hidden.data(), error)) {
            return false;
        }
    }

    // ── 3e. Trim to actual number of steps ───
    // (We'll determine the actual steps during EOS detection)

    // ═══════════════════════════════════════════════════════════════════
    //  Phase 4: Codec decode → PCM
    // ═══════════════════════════════════════════════════════════════════

    // Find the actual number of generated steps (stop at first all-zero column)
    int32_t n_frames = 0;
    for (int s = 0; s < max_steps; s++) {
        bool all_zero = true;
        for (int cb = 0; cb < n_total_books; cb++) {
            if (code_matrix[(size_t)cb * max_steps + s] != 0) { all_zero = false; break; }
        }
        if (all_zero) break;
        n_frames = s + 1;
    }

    if (n_frames <= 0) {
        std::fprintf(stderr, "qwen3_tts: no frames generated\n");
        resp.pcm_mono.assign(48000, 0.0f);  // 1 second silence
        resp.sampling_rate = 24000;
        return true;
    }

    std::fprintf(stderr, "qwen3_tts: %d frames generated\n", n_frames);

    // ── Codec decode → PCM ─────────────────────────────────────────────────
    // The ONNX speech tokenizer has been removed from audiocore. Codec
    // decode will be a ggml port of the speech-tokenizer graph that reads
    // its weights from the talker GGUF. Until that port lands we emit a
    // 1-second silence buffer so the rest of the pipeline stays exercisable.
    std::fprintf(stderr, "qwen3_tts: codec->PCM decoder not wired "
                         "(ggml speech-tokenizer port pending); emitting silence\n");
    resp.pcm_mono.assign(24000, 0.0f);
    resp.sampling_rate = 24000;
    return true;
}

// ── run_tts ─────────────────────────────────────────────────────────────

bool Qwen3TtsSession::run_tts(const void* request, void* response,
                               std::string* error) {
    if (!loaded_) {
        if (error) *error = "session not loaded";
        return false;
    }
    const auto& req = *static_cast<const TtsRequest*>(request);
    auto& resp = *static_cast<TtsResponse*>(response);
    return run_inference(req, resp, error);
}

}  // namespace audiocore::qwen3_tts
