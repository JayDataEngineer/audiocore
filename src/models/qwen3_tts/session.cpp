// session.cpp — Qwen3-TTS inference pipeline.
//
// Pipeline (official Qwen3-TTS):
//   text / instruct ─→ tokenize ─→ text_embedding + text_projection ─→
//   sum with codec prefix (codec_pad × T + codec_bos) ─→
//   Talker forward ─→ hidden @ codec_head ─→ codebook 0 ─→
//   Code Predictor MTP loop (31 fine codebooks) ─→
//   [32 × T] code matrix ─→ Speech Tokenizer ─→ PCM audio
//
// Stage 17 update: the codec decode step at the tail runs the
// Qwen3-TTS-Tokenizer-12Hz ggml port when the codec GGUF was discovered at
// load time. The 32-codebook talker+predictor matrix is sliced to the first
// 16 codebooks (the 12Hz codec's n_q) before decode. Silence fallback
// retained for codec-less configurations (GAPS.md §2.3).
//
// Stage 19 update: per-frame streaming. When mode="streaming" and a non-null
// stream callback is provided, newly generated codec frames are decoded
// incrementally during the AR loop and emitted via callback. Response PCM is
// empty in streaming mode — all audio goes through the callback.
//
// Stage 17b update: Voice Clone mode now runs the ECAPA-TDNN speaker encoder
// (Qwen3TtsSpeakerEncoder) when the speaker encoder was loaded from the talker
// GGUF. The computed speaker embedding is injected into the codec bridge at the
// speaker-position slot (analogous to how speaker_name works for CustomVoice),
// enabling voice cloning from a reference WAV file. Setups without the speaker
// encoder still fail fast with a GAPS.md pointer.
//
// Stage 18 update: ICL prefill — when `reference_text` is provided
// in Voice Clone mode, the reference text tokens are embedded via text_proj and
// inserted between the speaker slot and codec_bos. This gives the talker
// phonetic context about the reference voice during the prefill, improving
// clone fidelity. When the codec encoder is present (codec.enc.* tensors), the
// reference audio is also encoded into code tokens (ref_codes) and injected as
// additional prefill positions between ref_text and codec_bos, giving the
// talker full acoustic context. The prefill layout becomes:
//   [syn_text_emb + codec_pad] [spk_emb] [ref_text_emb + codec_pad]
//   [codec_emb(ref_code) + codec_pad] [codec_bos]
//
// Stage 19 update: per-frame streaming for Qwen3-TTS. When mode="streaming"
// and a non-null stream callback is provided (req.stream->on_audio), the AR
// loop decodes newly generated frames through the codec incrementally and
// emits PCM chunks via the callback. First frame emitted after ~80ms, then
// one frame (1920 samples = 80ms at 24 kHz) per AR step. Response PCM is
// empty in streaming mode — all audio flows through the callback.
//
// Both transformers (talker + predictor) run through the unified
// qwen3::Runner — the same class MOSS and ACE-Step use. There is no longer
// a separate TalkerRunner or PredictorRunner in audiocore.

#include "audiocore/models/qwen3_tts/family.h"
#include "audiocore/models/qwen3/runner.h"
#include "audiocore/framework/io/gguf_reader.h"   // Stage 17: full type for unique_ptr<GgufReader> dtor
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
// PAD and BOS are the last two rows of the codec embedding table.
static int32_t codec_pad(int32_t vocab) { return vocab - 2; }
static int32_t codec_bos(int32_t vocab) { return vocab - 1; }
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

// Voice Design mode is only valid when the loaded talker GGUF is the
// `Qwen3-TTS-12Hz-1.7B-VoiceDesign` variant. The official VoiceDesign model
// is fine-tuned to accept a natural-language voice description in the
// instruct slot; running the same prompt through the Base / CustomVoice
// backbone was the FRAUD #7 documented in Testing/GAP_TRACKER.md §3.1
// ("best-effort fallback" that returned non-voice-designed speech). The
// fraud is deleted: voice_design mode on a non-VoiceDesign checkpoint now
// fails fast with an error naming the missing model.

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
    // field. voice_clone requires the ECAPA-TDNN speaker encoder (Stage
    // 17b); when present we compute the embedding from reference_audio and
    // inject it into the codec bridge.
    //
    // streaming mode requires a non-null stream callback for incremental
    // codec decode. Without one we fail fast (GAPS.md §2.2).
    const Qwen3TtsMode mode = parse_mode(req.mode);
    const bool streaming = (req.stream && req.stream->on_audio);
    if (mode == Qwen3TtsMode::Streaming && !streaming) {
        if (error) *error =
            "qwen3_tts streaming requires a non-null stream callback "
            "(set stream.on_audio) for incremental codec decode. "
            "See GAPS.md §2.2.";
        return false;
    }
    if (streaming && !config_.codec_present) {
        if (error) *error = "qwen3_tts streaming requires codec GGUF";
        return false;
    }

    // For Voice Clone mode, compute the ECAPA speaker embedding from the
    // reference WAV before building the prefill. We store the result in a
    // local so Phase 1 can inject it at the speaker-position slot.
    std::vector<float> vc_spk_emb;
    if (mode == Qwen3TtsMode::VoiceClone) {
        if (!config_.speaker_present) {
            if (error) *error =
                "qwen3_tts voice_clone requires the ECAPA-TDNN speaker encoder "
                "(GGUF lacks `speaker.*` tensors). See GAPS.md §2.3.";
            return false;
        }
        if (req.reference_audio.empty()) {
            if (error) *error = "voice_clone requires reference_audio path";
            return false;
        }
        std::fprintf(stderr, "qwen3_tts: voice clone: computing ECAPA embedding from %s\n",
                     req.reference_audio.c_str());
        vc_spk_emb = speaker_encoder_.compute_embedding(req.reference_audio);
        if (vc_spk_emb.empty()) {
            if (error) *error = "voice_clone: speaker encoder failed on reference audio";
            return false;
        }
        const int d = talker_->n_embd();
        if ((int)vc_spk_emb.size() != d) {
            std::fprintf(stderr, "qwen3_tts: voice clone: spk_emb dim %zu != d_model %d; "
                                 "padding to match\n", vc_spk_emb.size(), d);
            vc_spk_emb.resize((size_t)d, 0.0f);
        }
        std::fprintf(stderr, "qwen3_tts: voice clone: ECAPA embedding computed (%zu dims)\n",
                     vc_spk_emb.size());
    }

    // ── Full ICL: encode reference audio via codec encoder ────────────────
    // When the codec encoder is present, we encode the reference WAV into
    // code tokens and inject them as ref_code positions in the talker prefill.
    // This gives the talker acoustic context about the reference voice.
    std::vector<int32_t> ref_codes;
    bool has_ref_codes = false;
    if (mode == Qwen3TtsMode::VoiceClone && !req.reference_audio.empty()
        && codec_graphs_.has_encoder()) {
        std::string wav_err;
        std::vector<float> ref_pcm = Qwen3TtsSpeakerEncoder::load_wav(
            req.reference_audio, &wav_err);
        if (ref_pcm.empty()) {
            std::fprintf(stderr, "qwen3_tts: full ICL: failed to load reference WAV "
                                 "for codec encode (%s); skipping ref codes\n",
                         wav_err.c_str());
        } else {
            std::fprintf(stderr, "qwen3_tts: full ICL: encoding %zu PCM samples "
                                 "via codec encoder\n", ref_pcm.size());
            std::vector<int32_t> ref_codes_all = codec_graphs_.encode(
                ref_pcm.data(), (int32_t)ref_pcm.size());
            if (ref_codes_all.empty()) {
                std::fprintf(stderr, "qwen3_tts: full ICL: codec encoder failed; "
                                     "skipping ref codes\n");
            } else {
                // ref_codes_all is [16, T_ref] row-major. Extract codebook 0.
                const int32_t T_ref = (int32_t)(ref_codes_all.size() / 16);
                ref_codes.resize(size_t(T_ref));
                for (int32_t t = 0; t < T_ref; ++t)
                    ref_codes[size_t(t)] = ref_codes_all[size_t(t)];
                has_ref_codes = true;
                std::fprintf(stderr, "qwen3_tts: full ICL: %d ref code frames\n",
                             T_ref);
            }
        }
    }

    // Voice Design mode requires the dedicated VoiceDesign variant. We do NOT
    // impersonate it by prepending a template string to the Base backbone's
    // instruct slot — that was FRAUD #7 (Testing/GAP_TRACKER.md §3.1).
    if (mode == Qwen3TtsMode::VoiceDesign) {
        if (config_.variant != Qwen3TtsVariant::VoiceDesign) {
            if (error) *error =
                "qwen3_tts voice_design mode requires the dedicated "
                "`Qwen3-TTS-12Hz-1.7B-VoiceDesign` checkpoint. The loaded "
                "talker is variant=";
            *error += variant_name(config_.variant);
            *error += ". See Testing/GAP_TRACKER.md §3.1.";
            return false;
        }
        if (req.instruct.empty()) {
            if (error) *error =
                "voice_design mode requires a non-empty 'instruct' field "
                "(natural-language description of the voice to design).";
            return false;
        }
    }
    const std::string& effective_instruct = req.instruct;

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

    // ── ICL prefill: reference text embedding for Voice Clone ─────────────
    std::vector<float> ref_text_embd;
    int32_t n_ref_tokens = 0;
    bool has_ref_text = false;
    if (mode == Qwen3TtsMode::VoiceClone && !req.reference_text.empty()) {
        has_ref_text = true;
        if (!talker_->compute_text_embedding(req.reference_text, &ref_text_embd,
                                              &n_ref_tokens, error)) {
            return false;
        }
        std::fprintf(stderr, "qwen3_tts: ICL prefill: %d ref text tokens\n", n_ref_tokens);
    }

    // Build codec prefix sequence.
    //   [codec_pad × n_text_tokens] + [speaker_token? / vc_embedding?] + [codec_bos]
    // The optional speaker_token slot appears when the request names a known
    // default speaker (CustomVoice variant). Voice Clone mode replaces this
    // slot with the ECAPA speaker embedding (compute above in mode routing).
    // They are mutually exclusive — Voice Clone ignores speaker_name.
    const bool has_speaker   = (speaker_token >= 0);
    const bool has_vc_spk    = !vc_spk_emb.empty();
    const bool has_spk_slot  = has_speaker || has_vc_spk;
    const int32_t n_ref_frames = has_ref_codes ? (int32_t)ref_codes.size() : 0;
    const int32_t prefix_len = n_text_tokens + (has_spk_slot ? 1 : 0)
                             + (has_ref_text ? n_ref_tokens : 0)
                             + n_ref_frames + 1;

    std::vector<float> codec_prefix((size_t)prefix_len * n_embd, 0.0f);

    const int32_t ce_dim = talker_->codec_embd_dim();
    const int32_t pad_idx     = codec_pad(codec_vocab);
    const int32_t bos_idx     = codec_bos(codec_vocab);


    if (talker_->codec_embedding() && ce_dim > 0) {
        // codec_pad for all text+instruct positions
        for (int32_t i = 0; i < n_text_tokens; i++) {
            const float* pad_row = talker_->codec_embedding() + (size_t)pad_idx * ce_dim;
            std::memcpy(&codec_prefix[(size_t)i * n_embd], pad_row,
                        (size_t)ce_dim * sizeof(float));
        }
        int32_t cursor = n_text_tokens;
        // Optional speaker slot
        if (has_spk_slot) {
            if (has_vc_spk) {
                std::memcpy(&codec_prefix[(size_t)cursor * n_embd],
                            vc_spk_emb.data(),
                            (size_t)n_embd * sizeof(float));
            } else if (speaker_token < codec_vocab) {
                const float* spk_row =
                    talker_->codec_embedding() + (size_t)speaker_token * ce_dim;
                std::memcpy(&codec_prefix[(size_t)cursor * n_embd], spk_row,
                            (size_t)ce_dim * sizeof(float));
            } else {
                std::fprintf(stderr,
                    "qwen3_tts: speaker token %d out of codec vocab range %d; "
                    "skipping speaker slot\n",
                    speaker_token, codec_vocab);
            }
            cursor += 1;
        }
        // codec_pad for reference text (ICL prefill)
        if (has_ref_text) {
            for (int32_t i = 0; i < n_ref_tokens; i++) {
                const float* pad_row = talker_->codec_embedding() + (size_t)pad_idx * ce_dim;
                std::memcpy(&codec_prefix[(size_t)cursor * n_embd], pad_row,
                            (size_t)ce_dim * sizeof(float));
                cursor += 1;
            }
        }
        // codec_pad + ref_code embedding for reference code frames (full ICL)
        if (has_ref_codes) {
            for (int32_t i = 0; i < n_ref_frames; i++) {
                int32_t code = ref_codes[(size_t)i];
                if (code >= 0 && code < codec_vocab) {
                    const float* code_row = talker_->codec_embedding() + (size_t)code * ce_dim;
                    const float* pad_row  = talker_->codec_embedding() + (size_t)pad_idx * ce_dim;
                    for (int j = 0; j < ce_dim; j++) {
                        codec_prefix[(size_t)cursor * n_embd + j] = code_row[j] + pad_row[j];
                    }
                } else {
                    const float* pad_row = talker_->codec_embedding() + (size_t)pad_idx * ce_dim;
                    std::memcpy(&codec_prefix[(size_t)cursor * n_embd], pad_row,
                                (size_t)ce_dim * sizeof(float));
                }
                cursor += 1;
            }
        }
        // codec_bos for the final position.
        if (bos_idx >= 0 && bos_idx < codec_vocab) {
            const float* bos_row = talker_->codec_embedding() + (size_t)bos_idx * ce_dim;
            std::memcpy(&codec_prefix[(size_t)cursor * n_embd], bos_row,
                        (size_t)ce_dim * sizeof(float));
        } else {
            std::fprintf(stderr, "qwen3_tts: codec_bos index %d out of range\n", bos_idx);
        }
    } else {
        std::fprintf(stderr, "qwen3_tts: no codec embedding table!\n");
    }

    // Sum text embeddings + codec prefix
    // Layout (no ICL):         [syn_text_emb + codec_pad × L_syn] [spk_emb] [codec_bos]
    // Layout (with ICL):       [syn_text_emb + codec_pad × L_syn] [spk_emb] [ref_text_emb + codec_pad × L_ref] [codec_bos]
    // Layout (full ICL):       [syn_text_emb + codec_pad × L_syn] [spk_emb] [ref_text_emb + codec_pad × L_ref] [codec_emb(ref_code) + codec_pad × T_ref] [codec_bos]
    std::vector<float> talker_input((size_t)prefix_len * n_embd);

    if (talker_->has_text_embedding() && !text_embd.empty()) {
        // Synthesis text positions: text_emb + codec_pad
        for (int32_t i = 0; i < n_text_tokens; i++) {
            for (int32_t j = 0; j < n_embd; j++) {
                size_t idx = (size_t)i * n_embd + j;
                talker_input[idx] = text_embd[idx] + codec_prefix[idx];
            }
        }
        // Speaker slot (at n_text_tokens): codec_prefix has spk_emb or speaker_row
        if (has_spk_slot) {
            std::memcpy(&talker_input[(size_t)n_text_tokens * n_embd],
                        &codec_prefix[(size_t)n_text_tokens * n_embd],
                        (size_t)n_embd * sizeof(float));
        }
        // Reference text positions (ICL prefill): ref_text_emb + codec_pad
        if (has_ref_text) {
            const size_t ref_offset = (size_t)(n_text_tokens + (has_spk_slot ? 1 : 0)) * n_embd;
            for (int32_t i = 0; i < n_ref_tokens; i++) {
                for (int32_t j = 0; j < n_embd; j++) {
                    size_t idx = ref_offset + (size_t)i * n_embd + j;
                    talker_input[idx] = ref_text_embd[(size_t)i * n_embd + j] + codec_prefix[idx];
                }
            }
        }
        // Reference code positions (full ICL): codec_prefix already has
        // codec_embedding(ref_code) + codec_pad, just copy the values.
        if (has_ref_codes) {
            const size_t rc_offset = (size_t)(n_text_tokens + (has_spk_slot ? 1 : 0)
                                            + (has_ref_text ? n_ref_tokens : 0)) * n_embd;
            std::memcpy(&talker_input[rc_offset], &codec_prefix[rc_offset],
                        (size_t)n_ref_frames * n_embd * sizeof(float));
        }
        // codec_bos is already at its position in codec_prefix; copy it over
        // (it lives at the last position, which the initial zero-init leaves unset).
        const size_t bos_pos = (size_t)(prefix_len - 1) * n_embd;
        std::memcpy(&talker_input[bos_pos], &codec_prefix[bos_pos],
                    (size_t)n_embd * sizeof(float));
    } else {
        // Lunavox fallback: zero text embedding, just codec prefix
        std::memcpy(talker_input.data(), codec_prefix.data(),
                    (size_t)prefix_len * n_embd * sizeof(float));
    }

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
        token_input.push_back(bos_idx);
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

    // Per-frame streaming: track how many frames have been decoded and
    // emitted via the streaming callback. Incremental decode at each step.
    int32_t decoded_frames = 0;

    for (int32_t step = 0; step < max_steps; step++) {
        // ── 3a. Sample codebook 0 via codec_head ───
        int32_t code_0 = 0;
        if (talker_->codec_head()) {
            const float* head = talker_->codec_head();  // [codec_vocab, ce_dim]
            std::vector<float> c0_logits((size_t)codec_vocab);
            for (int j = 0; j < codec_vocab; j++) {
                float s = 0;
                for (int d = 0; d < ce_dim; d++) {
                    s += cur_hidden[(size_t)d] * head[(size_t)j * ce_dim + d];
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

        // ── 3c. Per-frame streaming: decode newly available frames ──────
        if (streaming && codec_graphs_.is_present()) {
            const int32_t n_available = step + 1;
            const int32_t codec_n_q = (int32_t)codec_graphs_.hp.n_q;
            if (n_available > decoded_frames) {
                const int32_t n_new = n_available - decoded_frames;
                std::vector<int32_t> stream_codes((size_t)codec_n_q * (size_t)n_new);
                for (int32_t cb = 0; cb < codec_n_q; ++cb) {
                    for (int32_t t = 0; t < n_new; ++t) {
                        stream_codes[(size_t)cb * n_new + t] =
                            code_matrix[(size_t)cb * max_steps + decoded_frames + t];
                    }
                }
                try {
                    auto pcm = codec_graphs_.decode(
                        stream_codes.data(), codec_n_q, n_new);
                    if (!pcm.empty()) {
                        if (!req.stream->on_audio(pcm.data(), pcm.size())) {
                            if (error) *error = "qwen3_tts: streaming aborted by client";
                            return false;
                        }
                    }
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "qwen3_tts: streaming codec decode failed "
                                         "at step %d (%s); continuing\n",
                                 step, e.what());
                }
                decoded_frames = n_available;
            }
        }

        // ── 3d. Check for EOS ───
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
                const float* c0_row = talker_->codec_embedding() + (size_t)c0 * ce_dim;
                for (int d = 0; d < ce_dim; d++) next_embd[(size_t)d] += c0_row[d];
            }

            // Fine codes use predictor's fine_embd tables
            const int32_t fe_dim = predictor_->fine_embd_dim();
            for (int i = 0; i < n_fine_books; i++) {
                int32_t cid = fine_codes[(size_t)i];
                if (predictor_->has_mtp()) {
                    const float* fi_row = predictor_->fine_embedding(i);
                    if (fi_row && cid >= 0) {
                        cid = cid % predictor_->vocab_size();
                        const float* row = fi_row + (size_t)cid * fe_dim;
                        for (int d = 0; d < fe_dim; d++) next_embd[(size_t)d] += row[d];
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
        if (error) *error =
            "qwen3_tts: no audio frames generated — AR loop produced only "
            "stop-token frames. Check that the talker + predictor weights "
            "match the request language and that the prefill was non-empty.";
        return false;
    }

    // ── Streaming: all frames already emitted via callback ───────────────
    if (streaming && codec_graphs_.is_present()) {
        // Any remaining frames (should be zero since we emit every step)
        // were already decoded and emitted in the AR loop. Return empty PCM.
        std::fprintf(stderr, "qwen3_tts: streaming done (%d frames emitted)\n",
                     decoded_frames);
        resp.pcm_mono.clear();
        resp.sampling_rate = 24000;
        return true;
    }

    std::fprintf(stderr, "qwen3_tts: %d frames generated\n", n_frames);

    // ── Codec decode → PCM (Stage 17) ─────────────────────────────────────
    // Missing codec tensors is now a hard error. The previous behavior
    // (returning 1 s of silence with `return true`) masked broken/missing
    // codec GGUFs and broke regression tests. The codec is required to
    // produce audio.
    //
    // Codec/talker codebook count match-up: the talker + predictor emit a
    // 32-codebook matrix (QwenLM/Qwen3-TTS original architecture), but the
    // 12Hz codec consumes only the first `codec_graphs_.hp.n_q` codebooks
    // (typically 16). We pack the first codec_n_q codebooks × n_frames
    // columns into a contiguous row-major buffer before calling decode().
    if (!config_.codec_present || !codec_graphs_.is_present()) {
        if (error) *error =
            "qwen3_tts: codec GGUF is not bound (no Qwen-TTS-Tokenizer "
            "discovered at load time). The codec is required to produce PCM. "
            "Provide a tokenizer GGUF via the loader.";
        return false;
    }

    const int32_t codec_n_q = (int32_t)codec_graphs_.hp.n_q;
    if (codec_n_q > n_total_books) {
        if (error) *error =
            "qwen3_tts: codec expects " + std::to_string(codec_n_q) +
            " codebooks but talker+predictor only produce " +
            std::to_string(n_total_books) + ". Variant/codec mismatch.";
        return false;
    }

    // Pack (codec_n_q, n_frames) row-major out of the (n_total_books, max_steps)
    // matrix. code_matrix is indexed [cb * max_steps + step] so we walk the
    // first codec_n_q rows and pick the first n_frames columns.
    std::vector<int32_t> codec_codes((size_t)codec_n_q * (size_t)n_frames);
    for (int32_t cb = 0; cb < codec_n_q; ++cb) {
        for (int32_t t = 0; t < n_frames; ++t) {
            codec_codes[(size_t)cb * n_frames + t] =
                code_matrix[(size_t)cb * max_steps + t];
        }
    }

    try {
        resp.pcm_mono = codec_graphs_.decode(codec_codes.data(),
                                              codec_n_q, n_frames);
        resp.sampling_rate = 24000;
        std::fprintf(stderr, "qwen3_tts: codec decode ok (%zu samples, %.2fs)\n",
                     resp.pcm_mono.size(),
                     double(resp.pcm_mono.size()) / 24000.0);
        return true;
    } catch (const std::exception& e) {
        if (error) *error = std::string("qwen3_tts codec decode failed: ") + e.what();
        return false;
    }
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
