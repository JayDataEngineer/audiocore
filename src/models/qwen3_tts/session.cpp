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
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

namespace audiocore::qwen3_tts {

using audiocore::qwen3::Runner;
using audiocore::qwen3::RunnerConfig;
using audiocore::sampler::Params;
using audiocore::sampler::sample_token;

// ── Simple phase timer ───────────────────────────────────────────────────
struct PhaseTimer {
    using Clock = std::chrono::steady_clock;
    Clock::time_point start;
    PhaseTimer() : start(Clock::now()) {}
    double elapsed() const {
        auto now = Clock::now();
        return std::chrono::duration<double>(now - start).count();
    }
    void log(const char* label) const {
        std::fprintf(stderr, "qwen3_tts: [%.2fs] %s\n", elapsed(), label);
    }
};

// ── Quick linear resample for reference WAVs (16/22.05/48 → 24 kHz) ─────
static std::vector<float> resample_to_24k(const float* src, int n_src, int src_rate) {
    if (src_rate == 24000) return {src, src + n_src};
    const int n_dst = (int)((double)n_src * 24000.0 / src_rate);
    std::vector<float> dst((size_t)n_dst);
    for (int i = 0; i < n_dst; i++) {
        double pos = (double)i * src_rate / 24000.0;
        int i0 = (int)pos;
        int i1 = std::min(i0 + 1, n_src - 1);
        double frac = pos - i0;
        dst[(size_t)i] = (float)(src[i0] * (1.0 - frac) + src[i1] * frac);
    }
    return dst;
}

// ── Codec special tokens (from official Qwen3-TTS config) ────────────────
// PAD and BOS are the last two rows of the codec embedding table.
static int32_t codec_pad(int32_t vocab) { return vocab - 2; }
static int32_t codec_bos(int32_t vocab) { return vocab - 1; }
static constexpr int32_t CODEC_EOS    = 2150;  // 12Hz tokenizer default
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
// instruct slot. An earlier revision of this file impersonated VoiceDesign
// by prepending a template string to the Base / CustomVoice instruct slot
// (a "best-effort fallback" that returned non-voice-designed speech); that
// path is deleted. voice_design mode on a non-VoiceDesign checkpoint now
// fails fast with an error naming the missing model. See GAPS.md §2.2.

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

    // Clear KV caches so the new request starts from a clean slate.
    if (!talker_->clear_kv_cache(error) || !predictor_->clear_kv_cache(error))
        return false;

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
    // reference WAV (or use a pre-computed embedding) before building the
    // prefill. We store the result in a local so Phase 1 can inject it at
    // the speaker-position slot.
    std::vector<float> vc_spk_emb;
    if (mode == Qwen3TtsMode::VoiceClone) {
        if (!config_.speaker_present) {
            if (error) *error =
                "qwen3_tts voice_clone requires the ECAPA-TDNN speaker encoder "
                "(GGUF lacks `speaker.*` tensors). See GAPS.md §2.3.";
            return false;
        }
        const int d = talker_->n_embd();

        // Pre-computed embedding takes priority (no WAV load needed)
        if (!req.speaker_embedding.empty()) {
            vc_spk_emb = req.speaker_embedding;
            std::fprintf(stderr, "qwen3_tts: voice clone: using pre-computed "
                                 "embedding (%zu dims)\n", vc_spk_emb.size());
        } else if (!req.reference_audio.empty()) {
            std::fprintf(stderr, "qwen3_tts: voice clone: computing ECAPA embedding from %s\n",
                         req.reference_audio.c_str());
            vc_spk_emb = speaker_encoder_.compute_embedding(req.reference_audio);
            if (vc_spk_emb.empty()) {
                if (error) *error = "voice_clone: speaker encoder failed on reference audio";
                return false;
            }
        } else {
            if (error) *error = "voice_clone requires reference_audio or speaker_embedding";
            return false;
        }

        if ((int)vc_spk_emb.size() != d) {
            std::fprintf(stderr, "qwen3_tts: voice clone: spk_emb dim %zu != d_model %d; "
                                 "padding to match\n", vc_spk_emb.size(), d);
            vc_spk_emb.resize((size_t)d, 0.0f);
        }
        std::fprintf(stderr, "qwen3_tts: voice clone: ECAPA embedding ready (%zu dims)\n",
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
    // instruct slot — that earlier "best-effort fallback" path was deleted.
    if (mode == Qwen3TtsMode::VoiceDesign) {
        if (config_.variant != Qwen3TtsVariant::VoiceDesign) {
            if (error) *error =
                "qwen3_tts voice_design mode requires the dedicated "
                "`Qwen3-TTS-12Hz-1.7B-VoiceDesign` checkpoint. The loaded "
                "talker is variant=";
            *error += variant_name(config_.variant);
            *error += ". See GAPS.md §2.2 for the variant/mode matrix.";
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

    // ── Progress callback helper ──────────────────────────────────────────
    auto prog = [&](const char* phase, float pct, const char* msg) {
        if (req.stream && req.stream->on_progress)
            req.stream->on_progress(phase, pct, msg);
    };

    // ── Phase timer ───────────────────────────────────────────────────────
    PhaseTimer session_timer;

    // ═══════════════════════════════════════════════════════════════════
    //  Phase 1: Build input embeddings
    // ═══════════════════════════════════════════════════════════════════
    prog("prefill", 0.0f, "tokenizing");
    //
    // Layout (matching predict-woo/qwen3-tts.cpp reference):
    //   [role(3): text_proj(bos, assistant, \n)]
    //   [codec_think/nothink + tts_pad_overlay]
    //   [codec_think_bos + tts_pad_overlay]
    //   [language_id + tts_pad_overlay]  (or absent)
    //   [codec_think_eos + tts_pad_overlay]
    //   [speaker_embd]                    (voice clone or named speaker)
    //   [ref_text + codec_pad × L_ref]   (ICL)
    //   [codec_emb(ref_code) + codec_pad × T_ref] (ICL)
    //   [codec_pad + tts_bos_embed]      ← transition to text region
    //   [syn_text_0 + codec_bos]         ← first syn text summed with codec_bos
    //
    // The trailing text tokens (syn_text_1..N) + tts_eos are NOT in the
    // prefill — they are overlaid during the AR loop via trailing_text_hidden.

    const int32_t ce_dim  = talker_->codec_embd_dim();
    const int32_t pad_idx = codec_pad(codec_vocab);
    const int32_t bos_idx = codec_bos(codec_vocab);

    // ── Resolve special token IDs ────────────────────────────────────────
    int32_t asst_tok = 77091, nl_tok = 198; // Qwen3 defaults
    {
        std::vector<int32_t> tmp;
        if (talker_->tokenize("assistant", false, false, &tmp) && !tmp.empty())
            asst_tok = tmp[0];
        tmp.clear();
        if (talker_->tokenize("\n", false, false, &tmp) && !tmp.empty())
            nl_tok = tmp[0];
    }

    // ── encode_for_tts: wrap text in chat template ───────────────────────
    auto encode_for_tts = [&](const std::string& txt) -> std::vector<int32_t> {
        std::vector<int32_t> toks;
        // Tokenize text without special tokens
        talker_->tokenize(txt, false, false, &toks);
        // <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
        std::vector<int32_t> result;
        result.push_back(talker_->bos_token_id());   // <|im_start|>
        result.push_back(asst_tok);                  // assistant
        result.push_back(nl_tok);                    // \n
        result.insert(result.end(), toks.begin(), toks.end());  // text
        result.push_back(talker_->eos_token_id());   // <|im_end|>
        result.push_back(nl_tok);                    // \n
        result.push_back(talker_->bos_token_id());   // <|im_start|>
        result.push_back(asst_tok);                  // assistant
        result.push_back(nl_tok);                    // \n
        return result;
    };

    // ── Tokenize synthesis text in chat format ──────────────────────────
    std::string syn_text = effective_instruct + req.text;
    std::vector<int32_t> syn_tokens = encode_for_tts(syn_text);
    if (syn_tokens.empty()) {
        if (error) *error = "empty input after tokenization";
        return false;
    }
    std::fprintf(stderr, "qwen3_tts: %zu syn tokens (chat format, mode=%d, variant=%s)\n",
                 syn_tokens.size(), static_cast<int>(mode), variant_name(config_.variant));

    // ── Resolve speaker token ────────────────────────────────────────────
    int32_t speaker_token = -1;
    if (!req.speaker_name.empty()) {
        speaker_token = speaker_to_token(req.speaker_name);
        if (speaker_token < 0) {
            std::fprintf(stderr,
                "qwen3_tts: unknown speaker '%s' — ignoring\n", req.speaker_name.c_str());
        } else {
            std::fprintf(stderr, "qwen3_tts: speaker '%s' → codec token %d\n",
                         req.speaker_name.c_str(), speaker_token);
        }
    }
    const bool has_speaker  = (speaker_token >= 0);
    const bool has_vc_spk   = !vc_spk_emb.empty();
    const bool has_spk_slot = has_speaker || has_vc_spk;

    // ── ICL: ref text and ref codes ──────────────────────────────────────
    std::vector<float> ref_text_embd;
    int32_t n_ref_tokens = 0;
    bool has_ref_text = false;
    if (mode == Qwen3TtsMode::VoiceClone && !req.reference_text.empty()) {
        has_ref_text = true;
        if (!talker_->compute_text_embedding(req.reference_text, &ref_text_embd,
                                              &n_ref_tokens, error))
            return false;
        std::fprintf(stderr, "qwen3_tts: ICL prefill: %d ref text tokens\n", n_ref_tokens);
    }
    const int32_t n_ref_frames = has_ref_codes ? (int32_t)ref_codes.size() : 0;

    // ── Language → codec language token ID ───────────────────────────────
    // The reference maps language strings to int IDs (2050=en, 2055=zh, etc.)
    int32_t lang_codec_id = -1;
    if (!req.language.empty()) {
        // Try common language codes
        static const std::unordered_map<std::string, int32_t> lang_ids = {
            {"en", 2050}, {"zh", 2055}, {"ja", 2058}, {"ko", 2064},
            {"de", 2053}, {"fr", 2061}, {"es", 2054}, {"ru", 2069},
            {"pt", 2067}, {"it", 2056},
        };
        auto it = lang_ids.find(req.language);
        if (it != lang_ids.end()) lang_codec_id = it->second;
    }

    // ── Build codec bridge ───────────────────────────────────────────────
    // Composed of:
    //   • think/nothink + think_bos + [lang_id?] + think_eos  (3 or 4 tokens)
    //   • speaker slot (optional)
    //   • ICL ref text (optional)
    //   • ICL ref codes (optional)
    //   • codec_pad + codec_bos
    //
    // The last position (codec_bos) is the "start generating" marker.
    // The second-to-last (codec_pad) is overlaid with tts_bos_embed.
    // Everything else in the bridge is overlaid with tts_pad_embed.

    // Codec bridge: think/nothink family
    int32_t n_codec_bridge = (lang_codec_id >= 0) ? 4 : 3;
    std::vector<int32_t> bridge_ids;
    if (lang_codec_id >= 0) {
        bridge_ids = {CODEC_THINK, CODEC_TBOS, lang_codec_id, CODEC_TEOS};
    } else {
        bridge_ids = {CODEC_NOT, CODEC_TBOS, CODEC_TEOS};
    }

    const int32_t  codec_input_len = n_codec_bridge
                                   + (has_spk_slot ? 1 : 0)
                                   + (has_ref_text ? n_ref_tokens : 0)
                                   + n_ref_frames + 2;   // +2 for codec_pad + codec_bos
    const int32_t  overlay_len     = codec_input_len - 1; // all except codec_bos

    // ── Project special text tokens (bos, eos, pad) ─────────────────────
    std::vector<float> special_proj(static_cast<size_t>(3) * n_embd);
    {
        int32_t special_ids[3] = {talker_->bos_token_id(),
                                  talker_->eos_token_id(),
                                  static_cast<int32_t>(pad_idx)};
        // Project pad through text_proj as well — fallback only
        talker_->project_text_tokens(special_ids, 3, special_proj.data(), error);
    }
    const float* tts_bos_embed = &special_proj[0];
    const float* tts_eos_embed = &special_proj[static_cast<size_t>(1) * n_embd];
    const float* tts_pad_embed = &special_proj[static_cast<size_t>(2) * n_embd];

    // ── Project role tokens (first 3 tokens of syn_tokens) ──────────────
    // These are text_tokens[0..2] = [bos, assistant, \n]
    std::vector<float> role_embed(static_cast<size_t>(3) * n_embd);
    talker_->project_text_tokens(syn_tokens.data(), 3, role_embed.data(), error);

    // ── Look up codec embeddings for bridge + tail tokens ────────────────
    auto codec_row = [&](int32_t id) -> const float* {
        if (id >= 0 && id < codec_vocab)
            return talker_->codec_embedding() + static_cast<size_t>(id) * ce_dim;
        return nullptr;
    };

    // Build the codec overlay part: [bridge_tokens...] + [spk?] + [icl_ref_text?]
    //                             + [icl_ref_codes?] + [codec_pad]
    // Each position is a codec embedding + a text overlay.
    std::vector<float> codec_overlay(static_cast<size_t>(overlay_len) * n_embd, 0.0f);
    {
        size_t pos = 0;

        // Bridge tokens: codec_embedding + tts_pad_overlay
        for (int i = 0; i < n_codec_bridge; i++) {
            const float* cb_row = codec_row(bridge_ids[static_cast<size_t>(i)]);
            float* dst = &codec_overlay[pos * n_embd];
            for (int j = 0; j < ce_dim; j++)
                dst[j] = (cb_row ? cb_row[j] : 0.0f) + tts_pad_embed[j];
            pos++;
        }

        // Speaker slot
        if (has_spk_slot) {
            float* dst = &codec_overlay[pos * n_embd];
            if (has_vc_spk) {
                std::memcpy(dst, vc_spk_emb.data(), static_cast<size_t>(n_embd) * sizeof(float));
            } else if (speaker_token < codec_vocab) {
                const float* spk_row = codec_row(speaker_token);
                if (spk_row) std::memcpy(dst, spk_row, static_cast<size_t>(ce_dim) * sizeof(float));
            }
            pos++;
        }

        // ICL ref text: codec_pad + ref_text_emb
        if (has_ref_text) {
            for (int32_t i = 0; i < n_ref_tokens; i++) {
                const float* pad_row = codec_row(pad_idx);
                float* dst = &codec_overlay[pos * n_embd];
                for (int j = 0; j < n_embd; j++) {
                    dst[j] = (pad_row ? pad_row[j] : 0.0f) + ref_text_embd[static_cast<size_t>(i) * n_embd + j];
                }
                pos++;
            }
        }

        // ICL ref codes: codec_embedding(ref_code) + codec_pad
        if (has_ref_codes) {
            for (int32_t i = 0; i < n_ref_frames; i++) {
                int32_t code = ref_codes[static_cast<size_t>(i)];
                const float* cr = codec_row(code >= 0 && code < codec_vocab ? code : pad_idx);
                const float* pr = codec_row(pad_idx);
                float* dst = &codec_overlay[pos * n_embd];
                for (int j = 0; j < ce_dim; j++) {
                    dst[j] = (cr ? cr[j] : 0.0f) + (pr ? pr[j] : 0.0f);
                }
                pos++;
            }
        }

        // codec_pad + tts_bos_embed (the transition to text)
        {
            const float* pr = codec_row(pad_idx);
            float* dst = &codec_overlay[pos * n_embd];
            for (int j = 0; j < ce_dim; j++)
                dst[j] = (pr ? pr[j] : 0.0f) + tts_bos_embed[j];
        }
    }

    // ── First synthesis text token + codec_bos ──────────────────────────
    // syn_tokens[3] is the first content token (after role at 0-2)
    std::vector<float> first_text_plus_codec_bos(static_cast<size_t>(n_embd));
    {
        float first_text_proj[8192]; // max hidden_size
        talker_->project_single_token(syn_tokens[3], first_text_proj, error);
        const float* codec_bos_row = codec_row(bos_idx);
        for (int j = 0; j < n_embd; j++) {
            first_text_plus_codec_bos[static_cast<size_t>(j)] =
                first_text_proj[j] + (codec_bos_row ? codec_bos_row[j] : 0.0f);
        }
    }

    // ── Assemble prefill input ───────────────────────────────────────────
    // [role(3)] [codec_overlay(overlay_len)] [first_text_plus_codec_bos(1)]
    const int32_t prefill_len = 3 + overlay_len + 1;
    std::vector<float> talker_input(static_cast<size_t>(prefill_len) * n_embd);
    std::memcpy(talker_input.data(), role_embed.data(),
                static_cast<size_t>(3) * n_embd * sizeof(float));
    std::memcpy(&talker_input[static_cast<size_t>(3) * n_embd],
                codec_overlay.data(),
                static_cast<size_t>(overlay_len) * n_embd * sizeof(float));
    std::memcpy(&talker_input[static_cast<size_t>(prefill_len - 1) * n_embd],
                first_text_plus_codec_bos.data(),
                static_cast<size_t>(n_embd) * sizeof(float));

    // ── Trailing text hidden states (for AR loop overlay) ───────────────
    // syn_tokens[4..n_tokens-1] are projected through text_proj,
    // then tts_eos is appended. The trailing positions from the chat template
    // after the content tokens are dropped (the reference uses n_tokens-9
    // to omit the template-closing tokens).
    const int32_t n_syn_content = static_cast<int32_t>(syn_tokens.size()) - 9; // drop template closing
    std::vector<float> trailing_text_hidden;
    {
        // trailing content tokens start at index 4 (after role + first content)
        const int32_t n_trail = std::max(0, n_syn_content);
        trailing_text_hidden.resize(static_cast<size_t>(n_trail + 1) * n_embd);
        if (n_trail > 0) {
            talker_->project_text_tokens(&syn_tokens[4], n_trail,
                                          trailing_text_hidden.data(), error);
        }
        // Append tts_eos at the end
        std::memcpy(&trailing_text_hidden[static_cast<size_t>(n_trail) * n_embd],
                    tts_eos_embed, static_cast<size_t>(n_embd) * sizeof(float));
    }
    const int32_t trailing_len = static_cast<int32_t>(trailing_text_hidden.size() / n_embd);

    session_timer.log("prefill assembly done");
    std::fprintf(stderr, "qwen3_tts: prefill len=%d, trailing len=%d\n",
                 prefill_len, trailing_len);
    prog("prefill", 0.1f, "embeddings built");

    // ═══════════════════════════════════════════════════════════════════
    //  Phase 2: Talker forward pass (prefill)
    // ═══════════════════════════════════════════════════════════════════
    prog("talker_forward", 0.1f, "starting talker prefill");

    std::vector<float> talker_hidden(static_cast<size_t>(prefill_len) * n_embd);
    if (!talker_->forward_embeddings(talker_input.data(), prefill_len, 0,
                                     talker_hidden.data(), error)) {
        return false;
    }

    // The last token's hidden state (first_text + codec_bos position) is
    // the conditioning vector for the code predictor.
    const float* last_hidden = &talker_hidden[static_cast<size_t>(prefill_len - 1) * n_embd];
    session_timer.log("talker prefill done");
    prog("talker_forward", 0.3f, "talker prefill done");
    std::fprintf(stderr, "qwen3_tts: talker prefill done (%d tokens)\n", prefill_len);

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

    // Repetition penalty: track previously generated CB0 tokens.
    std::unordered_set<int32_t> generated_cb0;

    const int32_t suppress_start = codec_vocab - 1024;
    const int32_t progress_interval = std::max(1, max_steps / 20); // ~5% steps
    prog("ar_decode", 0.3f, "starting AR loop");

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

            // Suppress tokens in [codec_vocab - 1024, codec_vocab) except CODEC_EOS
            for (int32_t i = suppress_start; i < codec_vocab; i++) {
                if (i != CODEC_EOS) c0_logits[(size_t)i] = -INFINITY;
            }

            // Repetition penalty on previously generated CB0 tokens
            const float rp = req.repetition_penalty;
            if (rp != 1.0f) {
                for (int32_t tok : generated_cb0) {
                    if (tok >= 0 && tok < codec_vocab) {
                        float& l = c0_logits[(size_t)tok];
                        l = (l > 0.0f) ? (l / rp) : (l * rp);
                    }
                }
            }

            // Sample via unified sampler with top-k, top-p, temperature
            code_0 = [&] {
                Params sp;
                sp.temperature        = req.temperature;
                sp.top_p              = req.top_p;
                sp.top_k              = req.text_top_k;
                sp.repetition_penalty = 1.0f;  // already applied above
                return sample_token(c0_logits.data(), codec_vocab, sp,
                                    /*prev_tokens=*/nullptr, /*n_prev=*/0, &rng);
            }();

            // Early break on EOS
            if (code_0 == CODEC_EOS) break;
        }
        code_matrix[(size_t)0 * max_steps + step] = code_0;
        generated_cb0.insert(code_0);

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

        // Periodic progress
        if ((step % progress_interval) == 0 && step > 0) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "step %d/%d", step, max_steps);
            prog("ar_decode", 0.3f + 0.4f * ((float)step / max_steps), buf);
        }

        // ── 3e. Build next input embedding ───
        // Sum all 32 code embeddings + trailing text overlay
        std::vector<float> next_embd(static_cast<size_t>(n_embd), 0.0f);

        if (talker_->codec_embedding()) {
            // Coarse code (codebook 0) uses talker's codec_embedding
            int32_t c0 = std::max(0, code_0);
            if (c0 < talker_->codec_vocab()) {
                const float* c0_row = talker_->codec_embedding() + static_cast<size_t>(c0) * ce_dim;
                for (int d = 0; d < ce_dim; d++) next_embd[static_cast<size_t>(d)] += c0_row[d];
            }

            // Fine codes use predictor's fine_embd tables
            const int32_t fe_dim = predictor_->fine_embd_dim();
            for (int i = 0; i < n_fine_books; i++) {
                int32_t cid = fine_codes[static_cast<size_t>(i)];
                if (predictor_->has_mtp()) {
                    const float* fi_row = predictor_->fine_embedding(i);
                    if (fi_row && cid >= 0) {
                        cid = cid % predictor_->vocab_size();
                        const float* row = fi_row + static_cast<size_t>(cid) * fe_dim;
                        for (int d = 0; d < fe_dim; d++) next_embd[static_cast<size_t>(d)] += row[d];
                    }
                }
            }
        }

        // Add trailing text overlay (temporal alignment: one text token per frame)
        const float* trail_row = (step < trailing_len)
            ? &trailing_text_hidden[static_cast<size_t>(step) * n_embd]
            : tts_pad_embed;
        for (int j = 0; j < n_embd; j++)
            next_embd[static_cast<size_t>(j)] += trail_row[j];

        // Run talker on this single token to get next hidden
        if (!talker_->forward_embeddings(next_embd.data(), 1,
                                         prefill_len + step,
                                         cur_hidden.data(), error)) {
            return false;
        }
    }

    session_timer.log("AR loop done");
    prog("ar_decode", 0.7f, "AR loop done");

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

    // ── Chunked codec decode ──────────────────────────────────────────────
    // The full n_frames × codec_n_q matrix can produce very large intermediate
    // tensors in the codec decoder (especially the 4 decoder blocks with
    // strided upsampling). Decode in chunks to stay within GPU VRAM.
    prog("codec_decode", 0.8f, "decoding codec frames");
    const int32_t max_decode_chunk = 512;  // ~43 seconds per chunk
    resp.pcm_mono.reserve(static_cast<size_t>(n_frames) * 1920);

    for (int32_t chunk_start = 0; chunk_start < n_frames; ) {
        const int32_t chunk_size = std::min(max_decode_chunk, n_frames - chunk_start);

        // Pack (codec_n_q, chunk_size) row-major from the code_matrix
        std::vector<int32_t> chunk_codes(static_cast<size_t>(codec_n_q) * chunk_size);
        for (int32_t cb = 0; cb < codec_n_q; ++cb) {
            for (int32_t t = 0; t < chunk_size; ++t) {
                chunk_codes[static_cast<size_t>(cb) * chunk_size + t] =
                    code_matrix[static_cast<size_t>(cb) * max_steps + chunk_start + t];
            }
        }

        try {
            auto chunk_pcm = codec_graphs_.decode(chunk_codes.data(),
                                                   codec_n_q, chunk_size);
            if (!chunk_pcm.empty()) {
                resp.pcm_mono.insert(resp.pcm_mono.end(),
                                     chunk_pcm.begin(), chunk_pcm.end());
            }
        } catch (const std::exception& e) {
            if (error) *error = std::string("qwen3_tts codec decode failed at frame ")
                              + std::to_string(chunk_start) + ": " + e.what();
            return false;
        }

        chunk_start += chunk_size;
        char buf[64];
        std::snprintf(buf, sizeof(buf), "decoded %d/%d frames", chunk_start, n_frames);
        prog("codec_decode", 0.8f + 0.2f * ((float)chunk_start / n_frames), buf);
    }

    resp.sampling_rate = 24000;
    session_timer.log("codec decode done");
    prog("codec_decode", 1.0f, "done");
    std::fprintf(stderr, "qwen3_tts: codec decode ok (%zu samples, %.2fs)\n",
                 resp.pcm_mono.size(),
                 double(resp.pcm_mono.size()) / 24000.0);
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

// ── extract_speaker_embedding ─────────────────────────────────────────────

std::vector<float> Qwen3TtsSession::extract_speaker_embedding(
    const std::string& wav_path, std::string* error) {
    if (!config_.speaker_present) {
        if (error) *error =
            "qwen3_tts: speaker encoder not loaded (GGUF lacks `speaker.*` tensors)";
        return {};
    }
    PhaseTimer t;
    std::fprintf(stderr, "qwen3_tts: extracting speaker embedding from %s\n",
                 wav_path.c_str());
    auto emb = speaker_encoder_.compute_embedding(wav_path);
    if (emb.empty()) {
        if (error) *error = "speaker encoder failed on " + wav_path;
        return {};
    }
    t.log("speaker embedding extract done");
    return emb;
}

// ── synthesize_with_embedding ─────────────────────────────────────────────

bool Qwen3TtsSession::synthesize_with_embedding(
    const TtsRequest& req_base, const float* embedding, size_t emb_dim,
    TtsResponse& resp, std::string* error) {
    if (!loaded_) {
        if (error) *error = "session not loaded";
        return false;
    }
    if (!embedding || emb_dim == 0) {
        if (error) *error = "synthesize_with_embedding: null or empty embedding";
        return false;
    }
    // Clone the base request and inject the pre-computed embedding
    TtsRequest req = req_base;
    req.mode = "voice_clone";
    req.speaker_embedding.assign(embedding, embedding + emb_dim);
    return run_inference(req, resp, error);
}

}  // namespace audiocore::qwen3_tts
