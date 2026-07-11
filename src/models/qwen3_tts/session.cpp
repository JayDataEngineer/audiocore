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
#include "audiocore/framework/audio/dsp.h"         // trim_silence for over-generation fix

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
using audiocore::sampler::PhiloxRng;
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

// ── Codec special tokens (from official Qwen3-TTS config.json) ───────────
// These IDs are FIXED in the original Qwen3-TTS-1.7B / 0.6B config:
//   codec_pad_id           = 2148
//   codec_bos_id           = 2149
//   codec_eos_token_id     = 2150
//   codebook_vocab_size    = 2151  (real codec tokens + 3 special)
// The codec embedding table in our split GGUF is padded to 3072 rows
// (the original `codec_embedding.weight` shape), but tokens 2151..3071
// are unused padding and MUST NOT be addressed. Using `vocab - 2` /
// `vocab - 1` would feed token 3070/3071 which the model never saw in
// training — the resulting garbage prefill caused severe distortion.
static constexpr int32_t CODEC_PAD   = 2148;
static constexpr int32_t CODEC_BOS   = 2149;
static constexpr int32_t CODEC_EOS   = 2150;
// Per canonical qwen3-tts.cpp tts_transformer.cpp (defaults match
// Qwen3-TTS-1.7B / 0.6B config.json exactly):
//   codec_think_id      = 2154
//   codec_nothink_id    = 2155
//   codec_think_bos_id  = 2156
//   codec_think_eos_id  = 2157
// The earlier 4202-4205 values were TEXT-vocab IDs that fell outside
// the codec_embedding table — codec_row() returned nullptr and every
// bridge position collapsed to tts_pad_embed, stripping the prefill
// of its think/nothink mode marker. On 0.6B the model is robust
// enough to recover; on 1.7B it hallucinates foreign-language speech
// because it never sees the nothink signal.
static constexpr int32_t CODEC_THINK = 2154;
static constexpr int32_t CODEC_NOT   = 2155;
static constexpr int32_t CODEC_TBOS  = 2156;
static constexpr int32_t CODEC_TEOS  = 2157;
// Back-compat shims (no longer depend on vocab).
static inline int32_t codec_pad(int32_t /*vocab*/) { return CODEC_PAD; }
static inline int32_t codec_bos(int32_t /*vocab*/) { return CODEC_BOS; }

// Default speaker name → codec token mapping.
// Canonical IDs from Qwen3-TTS-12Hz-{0.6B,1.7B}-CustomVoice config.json:
//   talker_config.spk_id = {"serena":3066, "vivian":3065, "uncle_fu":3010,
//   "ryan":3061, "aiden":2861, "ono_anna":2873, "sohee":2864, "eric":2875,
//   "dylan":2878}
// Lookup is case-insensitive (the upstream Python uses speaker.lower()).
// The previously-hardcoded values 4206..4214 were WRONG and caused the
// CustomVoice variants to produce "wawa wawa wubba wubba" garbage — see
// GAPS.md §B1.
static int32_t speaker_to_token(const std::string& name) {
    static const std::unordered_map<std::string, int32_t> spk_map = {
        {"vivian", 3065}, {"serena", 3066}, {"ryan", 3061},
        {"aiden", 2861}, {"eric", 2875}, {"dylan", 2878},
        {"sohee", 2864}, {"ono_anna", 2873}, {"uncle_fu", 3010},
        // Aliases for the previous (incorrect) names so existing scripts
        // don't break hard — they'll resolve to a real speaker now.
        {"sarah", 3066}, {"alex", 2861}, {"emma", 3065},
        {"james", 3061}, {"olivia", 2873}, {"liam", 2878},
        {"sophia", 2864},
    };
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    auto it = spk_map.find(lower);
    if (it != spk_map.end()) return it->second;
    return -1;
}

// Language → codec token mapping. Canonical IDs from Qwen3-TTS config.json:
//   talker_config.codec_language_id = {"chinese":2055, "english":2050,
//   "german":2053, "italian":2070, "portuguese":2071, "spanish":2054,
//   "japanese":2058, "korean":2064, "french":2061, "russian":2069,
//   "beijing_dialect":2074, "sichuan_dialect":2062}
// The previous map (4206..4215) was incorrect — those IDs fall inside the
// spk_id namespace, not the language namespace.
static int32_t language_to_token(const std::string& lang) {
    static const std::unordered_map<std::string, int32_t> lang_map = {
        {"zh", 2055}, {"en", 2050}, {"ja", 2058}, {"ko", 2064},
        {"de", 2053}, {"fr", 2061}, {"ru", 2069}, {"pt", 2071},
        {"es", 2054}, {"it", 2070},
        // Long-form names (matches config.json keys)
        {"chinese", 2055}, {"english", 2050}, {"japanese", 2058},
        {"korean", 2064}, {"german", 2053}, {"french", 2061},
        {"russian", 2069}, {"portuguese", 2071}, {"spanish", 2054},
        {"italian", 2070},
    };
    std::string lower = lang;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    auto it = lang_map.find(lower);
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
    //
    // Third valid path: a named speaker (`req.speaker_name`) resolves to a
    // codec-token slot inside the talker's `token_embd` table. This is the
    // canonical CustomVoice path and needs neither an ECAPA encoder nor a
    // raw embedding. We still allow voice_clone mode here and let the
    // prefill builder inject the speaker-token row instead of `vc_spk_emb`.
    std::vector<float> vc_spk_emb;
    if (mode == Qwen3TtsMode::VoiceClone) {
        const int d = talker_->n_embd();

        // Pre-computed embedding takes priority (no WAV load needed, no
        // speaker encoder required). This mirrors ht-vllm-omni PR #1227:
        // passing a raw `speaker_embedding` vector bypasses ref_audio /
        // ref_text entirely and implies x_vector_only_mode=True.
        if (!req.speaker_embedding.empty()) {
            vc_spk_emb = req.speaker_embedding;
            std::fprintf(stderr, "qwen3_tts: voice clone: using pre-computed "
                                 "embedding (%zu dims)\n", vc_spk_emb.size());
        } else if (!req.reference_audio.empty()) {
            // ECAPA-TDNN encoder is only required when we must COMPUTE the
            // embedding from a reference WAV at runtime.
            if (!config_.speaker_present) {
                if (error) *error =
                    "qwen3_tts voice_clone from reference_audio requires the "
                    "ECAPA-TDNN speaker encoder (GGUF lacks `speaker.*` "
                    "tensors). See GAPS.md §2.3.";
                return false;
            }
            std::fprintf(stderr, "qwen3_tts: voice clone: computing ECAPA embedding from %s\n",
                         req.reference_audio.c_str());
            vc_spk_emb = speaker_encoder_.compute_embedding(req.reference_audio);
            if (vc_spk_emb.empty()) {
                if (error) *error = "voice_clone: speaker encoder failed on reference audio";
                return false;
            }
        } else if (!req.speaker_name.empty()) {
            // CustomVoice named-speaker path: no embedding needed. The
            // prefill builder resolves `speaker_name` to a codec-token slot
            // and injects the row from `token_embd` directly. Proceed with
            // an empty `vc_spk_emb`; `has_vc_spk` will be false but
            // `has_speaker` (set later from `speaker_to_token`) will be true.
            std::fprintf(stderr, "qwen3_tts: voice clone: using named speaker '%s' "
                                 "(no embedding required)\n",
                         req.speaker_name.c_str());
        } else {
            if (error) *error = "voice_clone requires reference_audio, speaker_embedding, or speaker_name";
            return false;
        }

        if (!vc_spk_emb.empty() && (int)vc_spk_emb.size() != d) {
            std::fprintf(stderr, "qwen3_tts: voice clone: spk_emb dim %zu != d_model %d; "
                                 "padding to match\n", vc_spk_emb.size(), d);
            vc_spk_emb.resize((size_t)d, 0.0f);
        }
        if (!vc_spk_emb.empty()) {
            // Apply embedding strength scaling. 1.0 = original (unchanged).
            // <1.0 = softer/more neutral voice, >1.0 = stronger/more pronounced.
            // Scales the embedding vector magnitude, which controls how strongly
            // the speaker identity influences generation vs. the instruct text.
            if (req.embedding_strength != 1.0f && req.embedding_strength > 0.0f) {
                std::fprintf(stderr, "qwen3_tts: voice clone: applying embedding_strength=%.3f\n",
                             req.embedding_strength);
                for (auto& v : vc_spk_emb) v *= req.embedding_strength;
            }
            std::fprintf(stderr, "qwen3_tts: voice clone: ECAPA embedding ready (%zu dims)\n",
                         vc_spk_emb.size());
        }
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
    PhiloxRng rng{42};

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
    int32_t asst_tok = 77091, user_tok = 872, sys_tok = 8948, nl_tok = 198; // Qwen3 defaults
    {
        std::vector<int32_t> tmp;
        if (talker_->tokenize("assistant", false, false, &tmp) && !tmp.empty())
            asst_tok = tmp[0];
        tmp.clear();
        if (talker_->tokenize("user", false, false, &tmp) && !tmp.empty())
            user_tok = tmp[0];
        tmp.clear();
        if (talker_->tokenize("system", false, false, &tmp) && !tmp.empty())
            sys_tok = tmp[0];
        tmp.clear();
        if (talker_->tokenize("\n", false, false, &tmp) && !tmp.empty())
            nl_tok = tmp[0];
    }

    // ── encode_for_tts: wrap text in chat template ───────────────────────
    //
    // Produces ONLY the assistant turn:
    //   <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
    //
    // The instruct text is NOT concatenated here — it's tokenized
    // separately and injected as a separate prefill segment (see
    // `instruct_overlay` below) so it doesn't shift the role-token
    // alignment of `syn_tokens[0..2]` that downstream `role_embed`
    // depends on.
    //
    // CRITICAL: a prior bug concatenated `instruct + req.text` here,
    // causing the model to literally read the instruct string aloud.
    auto encode_for_tts = [&](const std::string& txt) -> std::vector<int32_t> {
        std::vector<int32_t> text_toks;
        talker_->tokenize(txt, false, false, &text_toks);
        // <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
        std::vector<int32_t> result;
        result.push_back(talker_->bos_token_id());   // <|im_start|>
        result.push_back(asst_tok);                  // assistant
        result.push_back(nl_tok);                    // \n
        result.insert(result.end(), text_toks.begin(), text_toks.end());
        result.push_back(talker_->eos_token_id());   // <|im_end|>
        result.push_back(nl_tok);                    // \n
        result.push_back(talker_->bos_token_id());   // <|im_start|>
        result.push_back(asst_tok);                  // assistant
        result.push_back(nl_tok);                    // \n
        return result;
    };

    // ── Tokenize synthesis text in chat format ──────────────────────────
    std::vector<int32_t> syn_tokens = encode_for_tts(req.text);

    // ── Tokenize instruct (emotion / style) — kept SEPARATE from text ──
    // The instruct tokens are wrapped in a system-role turn and projected
    // into a separate prefill segment prepended before the role tokens.
    // This keeps syn_tokens[0..2] aligned as `<|im_start|>assistant\n`
    // for role_embed, and gives the talker explicit emotion conditioning.
    //
    // The canonical Qwen3-TTS pattern uses the `system` role for emotion
    // cues (e.g., "Speak in a happy, cheerful tone."). The user role is
    // for content-bearing instructions and tends to confuse the Base
    // variant into interpreting the cue as content to be read aloud.
    std::vector<int32_t> instruct_tokens;  // full system-turn token seq
    if (!effective_instruct.empty()) {
        std::vector<int32_t> inst_text_toks;
        talker_->tokenize(effective_instruct, false, false, &inst_text_toks);
        // <|im_start|>system\n{instruct}<|im_end|>\n
        instruct_tokens.push_back(talker_->bos_token_id());
        instruct_tokens.push_back(sys_tok);  // system role (emotion cue)
        instruct_tokens.push_back(nl_tok);
        instruct_tokens.insert(instruct_tokens.end(), inst_text_toks.begin(), inst_text_toks.end());
        instruct_tokens.push_back(talker_->eos_token_id());
        instruct_tokens.push_back(nl_tok);
    }
    const int32_t n_instruct = static_cast<int32_t>(instruct_tokens.size());

    // Also expose the raw text-to-speak tokens (excluding chat scaffolding)
    // for downstream text_region / trailing_text_hidden assembly.
    std::vector<int32_t> text_ids_raw;
    {
        talker_->tokenize(req.text, false, false, &text_ids_raw);
    }
    const int32_t n_text_raw = static_cast<int32_t>(text_ids_raw.size());
    if (syn_tokens.empty()) {
        if (error) *error = "empty input after tokenization";
        return false;
    }
    std::fprintf(stderr, "qwen3_tts: %zu syn tokens (chat format, mode=%d, variant=%s)\n",
                 syn_tokens.size(), static_cast<int>(mode), variant_name(config_.variant));
    if (std::getenv("QWEN3TTS_DEBUG")) {
        // Decode the content tokens (indices 3..size-9) back to text for sanity.
        std::string decoded;
        std::string ids_str;
        for (size_t i = 0; i < syn_tokens.size(); ++i) {
            std::string piece;
            std::string piece_err;
            const int32_t tid = syn_tokens[i];
            const bool ok = talker_->token_to_piece(tid, &piece, &piece_err);
            ids_str += std::to_string(tid) + (ok ? ":" + piece : ":<?>") + " ";
            if (i >= 3 && i + 9 <= syn_tokens.size() && ok)
                decoded += piece;
        }
        std::fprintf(stderr, "qwen3_tts: syn_tokens size=%zu, ids=[%s]\n",
                     syn_tokens.size(), ids_str.c_str());
        std::fprintf(stderr, "qwen3_tts: syn_text decoded='%s' (n_content=%d)\n",
                     decoded.c_str(),
                     static_cast<int>(syn_tokens.size()) - 9);
    }

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
    //
    // Canonical faster-qwen3-tts uses TTS-specific tokens from the model
    // config: tts_bos=151672, tts_eos=151673, tts_pad=151671 (verified in
    // every Qwen3-TTS config.json: 0.6B-Base, 1.7B-Base, CustomVoice,
    // VoiceDesign all carry these exact IDs).
    //
    // EMPIRICAL STATUS on our GGUFs:
    //   - 1.7B-Base: TTS-specific IDs FIXED the Chinese-gibberish bug
    //     (chat-template IDs put the model out-of-distribution; the model
    //     would emit coherent but wrong-language content).
    //   - 0.6B-Base: TTS-specific IDs COLLAPSE the model to RMS=0.002
    //     (near-silence). Chat-template IDs (151644/151645) work for 0.6B.
    //
    // The 0.6B-specific collapse with canonical tokens is likely a GGUF
    // conversion quirk (token-type metadata or embedding-row initialization
    // for IDs in the 1516xx range). Investigating that is tracked as gap
    // A1.b. For now, pick the IDs variant-aware based on talker hidden.
    //
    // Override via env vars QWEN3TTS_TTS_BOS / _EOS / _PAD for debugging.
    std::vector<float> special_proj(static_cast<size_t>(3) * n_embd);
    {
        // Variant-aware defaults: 1.7B (hidden>=2048) uses canonical
        // TTS-specific IDs; 0.6B (hidden<2048) uses chat-template IDs that
        // have been empirically verified to work.
        const bool is_1_7b = (n_embd >= 2048);
        int32_t tts_bos_id = is_1_7b ? 151672 : talker_->bos_token_id();
        int32_t tts_eos_id = is_1_7b ? 151673 : talker_->eos_token_id();
        int32_t tts_pad_id = is_1_7b ? 151671 : static_cast<int32_t>(pad_idx);
        if (const char* v = std::getenv("QWEN3TTS_TTS_BOS")) tts_bos_id = std::atoi(v);
        if (const char* v = std::getenv("QWEN3TTS_TTS_EOS")) tts_eos_id = std::atoi(v);
        if (const char* v = std::getenv("QWEN3TTS_TTS_PAD")) tts_pad_id = std::atoi(v);
        int32_t special_ids[3] = {tts_bos_id, tts_eos_id, tts_pad_id};
        std::fprintf(stderr, "qwen3_tts: tts_bos/eos/pad ids = %d/%d/%d (variant=%s, hidden=%d)\n",
                     tts_bos_id, tts_eos_id, tts_pad_id,
                     is_1_7b ? "1.7B" : "0.6B", n_embd);
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

        // Speaker slot — codec_embedding (spk_token) OR raw ECAPA vector.
        //
        // Canonical layout (modeling_qwen3_tts.py:~2218) adds a tts_pad
        // text-overlay at this position uniformly for BOTH paths:
        //   speaker_embed_or_codec_token + tts_pad_overlay
        //
        // EMPIRICAL QUIRK (gap A1.b): on our 0.6B-Base GGUFs, adding the
        // tts_pad overlay to a RAW ECAPA vector at this slot shifts the
        // vector out of distribution and collapses output to near-silence
        // (RMS 0.20 → 0.023). So for the raw-ECAPA path we omit the overlay.
        //
        // For CV variants (where the "speaker" is a LEARNED codec token
        // like vivian=3065), the model was TRAINED with the canonical
        // tts_pad overlay at this position. Omitting it puts the model
        // out-of-distribution. So for the learned-token path we DO add
        // the overlay (matching canonical).
        if (has_spk_slot) {
            float* dst = &codec_overlay[pos * n_embd];
            if (has_vc_spk) {
                std::memcpy(dst, vc_spk_emb.data(), static_cast<size_t>(n_embd) * sizeof(float));
                // Raw ECAPA vector: by default NO tts_pad overlay (gap A1.b).
                // Canonical HF Qwen3-TTS does not add tts_pad to a raw
                // speaker_embedding either, so this matches upstream.
                // QWEN3TTS_VC_ADD_PAD=1 enables the overlay for experiments.
                // (Pre-fix BF16-corrupt GGUFs collapsed to near-silence with
                // the overlay; on fresh GGUFs the combo works either way —
                // verified 0.6B-Base 2025-07-05.)
                if (std::getenv("QWEN3TTS_VC_ADD_PAD") != nullptr) {
                    for (int j = 0; j < n_embd; j++)
                        dst[j] += tts_pad_embed[j];
                }
            } else if (speaker_token < codec_vocab) {
                const float* spk_row = codec_row(speaker_token);
                if (spk_row) std::memcpy(dst, spk_row, static_cast<size_t>(ce_dim) * sizeof(float));
                // Learned speaker token (CV variants): ADD canonical tts_pad overlay.
                for (int j = 0; j < n_embd; j++)
                    dst[j] += tts_pad_embed[j];
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

    // ── Synthesis text region ─────────────────────────────────────────────
    // Two modes (selected at runtime via QWEN3TTS_COMPACT_TEXT env var):
    //
    //   STREAMING (default, canonical vllm-omni streaming mode):
    //     - first_text + codec_bos goes into prefill
    //     - remaining text tokens + tts_eos are streamed 1-per-AR-step
    //     - good for long text (temporal alignment between text & audio)
    //     - 0.6B handles short text fine in this mode
    //
    //   COMPACT (canonical vllm-omni non_streaming_mode):
    //     - ALL text tokens + tts_eos go into prefill (each overlaid with
    //       codec_pad), followed by a single (tts_pad + codec_bos)
    //       transition frame
    //     - AR loop feeds tts_pad every step (no per-step text)
    //     - 1.7B requires this for short text — streaming mode leaves the
    //       larger model with too little text context in the prefill and
    //       it hallucinates ("bup bup bup", Korean, Chinese, etc).
    const bool compact_text = std::getenv("QWEN3TTS_COMPACT_TEXT") != nullptr;
    // n_syn_content is the number of raw text-to-speak tokens (excluding
    // chat-template scaffolding). This is independent of whether an
    // instruct user turn is present.
    const int32_t n_syn_content = std::max(1, n_text_raw);

    // text_region holds either [first_text+codec_bos] (streaming) or
    // [text_tok + codec_pad]*N + [tts_eos + codec_pad] + [tts_pad + codec_bos]
    // (compact).
    std::vector<float> text_region;
    int32_t text_region_len = 0;
    if (compact_text) {
        // All raw text tokens, then tts_eos terminator.
        // A2 fix: use the variant-aware tts_eos ID (151673 canonical, or
        // chat-template ID for 0.6B) — NOT talker_->eos_token_id() which
        // always returns the sidecar's chat-template eos (151645). The
        // canonical Qwen3-TTS compact-text prefill expects tts_eos_embed
        // at the terminator slot; passing the wrong ID collapses the
        // model to silence (RMS=0.0024 on 0.6B-Base compact).
        const int32_t n_text = n_syn_content;
        std::vector<int32_t> text_ids(static_cast<size_t>(n_text) + 1);
        for (int32_t i = 0; i < n_text; i++)
            text_ids[static_cast<size_t>(i)] = text_ids_raw[static_cast<size_t>(i)];
        // tts_eos_id resolved earlier via the variant-aware logic. Fall
        // back to sidecar eos if the special-projection path was skipped.
        int32_t tts_eos_id_resolved = talker_->eos_token_id();
        if (const char* v = std::getenv("QWEN3TTS_TTS_EOS")) {
            tts_eos_id_resolved = std::atoi(v);
        } else {
            const bool is_1_7b_eos = (n_embd >= 2048);
            tts_eos_id_resolved = is_1_7b_eos ? 151673 : talker_->eos_token_id();
        }
        text_ids[static_cast<size_t>(n_text)] = tts_eos_id_resolved;
        std::fprintf(stderr,
                     "qwen3_tts: compact text region %d text tokens + tts_eos=%d\n",
                     n_text, tts_eos_id_resolved);

        std::vector<float> text_proj_buf(static_cast<size_t>(n_text + 1) * n_embd);
        talker_->project_text_tokens(text_ids.data(), n_text + 1,
                                      text_proj_buf.data(), error);

        // Each text position overlaid with codec_pad
        const float* pad_row = codec_row(pad_idx);
        text_region.resize(static_cast<size_t>(n_text + 2) * n_embd);
        for (int32_t i = 0; i <= n_text; i++) {
            const float* tp = &text_proj_buf[static_cast<size_t>(i) * n_embd];
            float* dst = &text_region[static_cast<size_t>(i) * n_embd];
            for (int j = 0; j < n_embd; j++)
                dst[j] = tp[j] + (pad_row ? pad_row[j] : 0.0f);
        }
        // Final transition frame: tts_pad + codec_bos
        {
            const float* bos_row = codec_row(bos_idx);
            float* dst = &text_region[static_cast<size_t>(n_text + 1) * n_embd];
            for (int j = 0; j < n_embd; j++)
                dst[j] = tts_pad_embed[j] + (bos_row ? bos_row[j] : 0.0f);
        }
        text_region_len = n_text + 2;
    } else {
        // Streaming: just first_text + codec_bos
        text_region.resize(static_cast<size_t>(n_embd));
        float first_text_proj[8192];
        talker_->project_single_token(text_ids_raw[0], first_text_proj, error);
        const float* codec_bos_row = codec_row(bos_idx);
        for (int j = 0; j < n_embd; j++)
            text_region[static_cast<size_t>(j)] =
                first_text_proj[j] + (codec_bos_row ? codec_bos_row[j] : 0.0f);
        text_region_len = 1;
    }

    // ── Assemble prefill input ───────────────────────────────────────────
    // [instruct_overlay(n_instruct_proj)?] [role(3)] [codec_overlay(overlay_len)] [text_region(text_region_len)]
    //
    // The optional instruct overlay is the user-role turn
    //   <|im_start|>user\n{instruct}<|im_end|>\n
    // projected through text_proj and prepended so the talker sees the
    // emotion cue BEFORE the assistant turn. It is only present when
    // `req.instruct` is non-empty.
    std::vector<float> instruct_proj_overlay;
    if (n_instruct > 0) {
        instruct_proj_overlay.resize(static_cast<size_t>(n_instruct) * n_embd);
        talker_->project_text_tokens(instruct_tokens.data(), n_instruct,
                                      instruct_proj_overlay.data(), error);
        std::fprintf(stderr,
                     "qwen3_tts: instruct overlay %d tokens (emotion conditioning)\n",
                     n_instruct);
    }
    const int32_t prefill_len = n_instruct + 3 + overlay_len + text_region_len;
    std::vector<float> talker_input(static_cast<size_t>(prefill_len) * n_embd);
    {
        size_t off = 0;
        if (n_instruct > 0) {
            std::memcpy(talker_input.data() + off * n_embd,
                        instruct_proj_overlay.data(),
                        static_cast<size_t>(n_instruct) * n_embd * sizeof(float));
            off += static_cast<size_t>(n_instruct);
        }
        std::memcpy(talker_input.data() + off * n_embd, role_embed.data(),
                    static_cast<size_t>(3) * n_embd * sizeof(float));
        off += 3;
        std::memcpy(talker_input.data() + off * n_embd,
                    codec_overlay.data(),
                    static_cast<size_t>(overlay_len) * n_embd * sizeof(float));
        off += static_cast<size_t>(overlay_len);
        std::memcpy(talker_input.data() + off * n_embd,
                    text_region.data(),
                    static_cast<size_t>(text_region_len) * n_embd * sizeof(float));
    }

    // ── Trailing text hidden states (for AR loop overlay) ───────────────
    // Compact mode: nothing to stream — AR loop feeds tts_pad every step.
    // Streaming mode: text_ids_raw[1..] + tts_eos streamed 1-per-step.
    std::vector<float> trailing_text_hidden;
    if (compact_text) {
        // Just tts_pad — AR loop will use tts_pad_embed via fallback.
        trailing_text_hidden.resize(0);
    } else {
        // Stream tokens after the first one (first is in prefill).
        const int32_t n_trail = std::max(0, n_text_raw - 1);
        trailing_text_hidden.resize(static_cast<size_t>(n_trail + 1) * n_embd);
        if (n_trail > 0) {
            talker_->project_text_tokens(&text_ids_raw[1], n_trail,
                                          trailing_text_hidden.data(), error);
        }
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

    // Diagnostic: verify prefill components and output for NaN
    if (std::getenv("QWEN3TTS_DEBUG")) {
        auto vec_stats = [](const float* p, size_t n, const char* label) {
            double mn = 1e30, mx = -1e30, sum = 0, sum_sq = 0;
            int nan_cnt = 0;
            for (size_t i = 0; i < n; i++) {
                float v = p[i];
                if (std::isnan(v)) { nan_cnt++; continue; }
                if (v < mn) mn = v;
                if (v > mx) mx = v;
                sum += v; sum_sq += double(v) * v;
            }
            double mean = nan_cnt ? 0.0 : sum / n;
            double var  = nan_cnt ? 0.0 : sum_sq / n - mean * mean;
            std::fprintf(stderr, "  [%s] n=%zu nan=%d min=%.4f max=%.4f mean=%.4f std=%.4f\n",
                         label, n, nan_cnt, mn, mx, mean, std::sqrt(std::max(0.0, var)));
        };
        vec_stats(role_embed.data(), 3 * n_embd, "role_embed");
        vec_stats(codec_overlay.data(), overlay_len * n_embd, "codec_overlay");
        vec_stats(text_region.data(), text_region_len * n_embd, "text_region");
        vec_stats(talker_input.data(), prefill_len * n_embd, "talker_input");
        vec_stats(talker_hidden.data(), prefill_len * n_embd, "talker_hidden");
        if (has_vc_spk) vec_stats(vc_spk_emb.data(), vc_spk_emb.size(), "vc_spk_emb");
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
    // Qwen3-TTS 12Hz codec: 16 codebooks total = 1 coarse (cb0, sampled by
    // the talker's codec_head) + 15 fine (cb1..cb15, predicted by the
    // predictor's MTP loop). Matches the codec GGUF `rvq=16` metadata and
    // the predictor's codec_embd.{0..14}.weight tensor count. The previous
    // 32/31 values caused predict_one_step's n_fine_books sanity check to
    // fail (31 != 15), forcing a fallback through forward_tokens whose
    // cb_vocab = vocab_size/32 math was wrong, producing random fine codes
    // and ultimately unintelligible audio.
    const int32_t n_total_books = 16;
    const int32_t n_fine_books  = 15;

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

    // ── Stuck-loop detector ────────────────────────────────────────────
    // If the model emits the same cb0 token for `repeat_limit` consecutive
    // steps, it's stuck in a degenerate loop (common on CPU f16, also
    // happens on GPU when the model loses coherence). We break early AND
    // zero out the repeated frames so the post-loop frame scan discards
    // them — otherwise they decode to audible noise that trim_silence
    // can't remove (it has energy, just no information).
    int32_t last_cb0 = -1;
    int32_t repeat_count = 0;
    const int32_t repeat_limit = 12;  // 1 second at 12 Hz

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

            // Min-length guard: don't allow CODEC_EOS until the model has
            // generated at least min_eos_step frames. Without this, the
            // moment the trailing-text overlay feeds tts_eos_embed (at
            // step == trailing_len), the model emits codec_eos and the
            // audio collapses to a single word ("Hello"). Forcing a
            // minimum of ~1 second of audio (12 frames) past the trailing
            // text gives the talker time to actually synthesise the words.
            const int32_t min_eos_step = trailing_len + 12;
            if (step < min_eos_step) {
                c0_logits[(size_t)CODEC_EOS] = -INFINITY;
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

        // Debug: first 5 steps log cb0 + top logits
        if (step < 5 && std::getenv("QWEN3TTS_DEBUG")) {
            std::fprintf(stderr, "  ar[%d]: cb0=%d (unique=%zu)\n",
                         step, code_0, generated_cb0.size());
        }

        // ── 3b. MTP loop: predict 31 fine codebooks ───
        std::vector<int32_t> fine_codes((size_t)n_fine_books, 0);

        // Build prev_codes for the predictor:
        //   [code_0, prev_fine[0], ..., prev_fine[30]]
        std::vector<int32_t> pred_input((size_t)n_total_books);
        pred_input[0] = code_0;
        for (int i = 0; i < n_fine_books; i++) {
            pred_input[(size_t)(i + 1)] = prev_fine[(size_t)i];
        }

        // layer0_embed = talker's codec_embedding[code_0] at talker hidden.
        // The predictor's predict_one_step internally projects this through
        // small_to_mtp (1.7B) or uses it directly (0.6B). This must NOT be
        // the token-id array — that path was a regression that produced
        // unintelligible, high-pitched screeching on 1.7B.
        std::vector<float> layer0_embed_vec;
        const float* layer0_embed_ptr = nullptr;
        if (talker_->codec_embedding() && code_0 >= 0 && code_0 < codec_vocab) {
            layer0_embed_vec.assign(
                talker_->codec_embedding() + static_cast<size_t>(code_0) * ce_dim,
                talker_->codec_embedding() + static_cast<size_t>(code_0) * ce_dim + ce_dim);
            layer0_embed_ptr = layer0_embed_vec.data();
        }

        bool mtp_ok = false;
        // Diagnostic bypass: when QWEN3TTS_NO_MTP=1, skip the predictor
        // entirely and emit zero fine codes. Helps isolate whether the
        // 1.7B distortion comes from the predictor or the talker.
        const bool no_mtp = std::getenv("QWEN3TTS_NO_MTP") != nullptr;
        if (no_mtp) {
            for (int i = 0; i < n_fine_books; i++) fine_codes[i] = 0;
            mtp_ok = true;
        } else if (predictor_->has_mtp() && layer0_embed_ptr) {
            if (predictor_->predict_one_step(cur_hidden.data(),
                                             layer0_embed_ptr,
                                             code_0,
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

        // ── Stuck-loop detector ──────────────────────────────────────────
        // At this point the full frame (cb0 + fine codes) for this step is
        // in code_matrix. Compare with the previous step's frame: if every
        // codebook is identical for repeat_limit consecutive steps, the
        // model's hidden state has frozen (common on f16 CPU, also happens
        // on GPU). We break and zero out the frozen frames so they don't
        // decode to audible noise.
        //
        // Note: checking ONLY cb0 is wrong — cb0=327 is a natural pause
        // token. But if all 16 codebooks repeat identically, it's stuck.
        if (step > 0) {
            bool full_repeat = true;
            for (int cb = 0; cb < n_total_books && full_repeat; cb++) {
                if (code_matrix[(size_t)cb * max_steps + step] !=
                    code_matrix[(size_t)cb * max_steps + step - 1])
                    full_repeat = false;
            }
            if (full_repeat) {
                ++repeat_count;
                if (repeat_count >= repeat_limit && step > 18) {
                    std::fprintf(stderr,
                        "qwen3_tts: stuck-loop break at step %d "
                        "(full 16-book frame repeated %d times, cb0=%d)\n",
                        step, repeat_count, code_0);
                    int32_t keep_until = step - repeat_count + 2;
                    for (int32_t s = keep_until; s <= step; s++) {
                        for (int cb = 0; cb < n_total_books; cb++)
                            code_matrix[(size_t)cb * max_steps + s] = 0;
                    }
                    break;
                }
            } else {
                repeat_count = 0;
            }
            last_cb0 = code_0;
        }

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
        // Canonical vllm-omni Qwen3TTSTalkerForConditionalGeneration.talker_mtp
        // (GPU fast-path in vllm_omni/.../qwen3_tts_talker.py):
        //   last_id_hidden = input_embeds.reshape(bsz, 1, -1)
        //                   # = embed_input_ids(prev_code_0) — talker codec_embedding
        //   audio_codes    = self.code_predictor(
        //                       layer0_code=prev_code_0,
        //                       layer0_embed=last_id_hidden,
        //                       last_talker_hidden=past_hidden, ...)  # past_hidden
        //                                                       # only conditions
        //                                                       # the predictor
        //   residual_ids_t = audio_codes[:, 1:]               # the 15 fine codes
        //   embeds = [last_id_hidden]
        //   for i in range(max_steps):
        //       embeds.append(code_predictor.get_input_embeddings()[i](
        //                         residual_ids_t[:, i:i+1]))
        //   summed = torch.cat(embeds, dim=1).sum(1, keepdim=True)
        //   inputs_embeds_out = (summed + text_step).reshape(bsz, -1)
        //
        // NOTE: past_hidden (the previous talker output) is NOT added to the
        // next input embedding. It only conditions the predictor. The talker
        // transformer itself carries state across steps via its KV cache.
        // Adding cur_hidden here (a previous regression) double-counts the
        // residual stream and was responsible for the heavy clipping and
        // wrong-word outputs ("Hello world" → "Because you're not ill").
        std::vector<float> next_embd(static_cast<size_t>(n_embd), 0.0f);

        // Diagnostic: track component magnitudes for collapse debugging
        double mag_cur_hidden = 0.0, mag_codec = 0.0, mag_fine = 0.0, mag_text = 0.0;
        for (int j = 0; j < n_embd; j++) {
            double ch = cur_hidden[static_cast<size_t>(j)];
            mag_cur_hidden += ch * ch;
        }
        mag_cur_hidden = std::sqrt(mag_cur_hidden);

        if (talker_->codec_embedding()) {
            // (b1) Coarse code (codebook 0) uses talker's codec_embedding
            int32_t c0 = std::max(0, code_0);
            if (c0 < talker_->codec_vocab()) {
                const float* c0_row = talker_->codec_embedding() + static_cast<size_t>(c0) * ce_dim;
                for (int d = 0; d < ce_dim; d++) next_embd[static_cast<size_t>(d)] += c0_row[d];
            }

            // (b2) Fine codes use predictor's fine_embd tables at talker hidden
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

        // (c) trailing text overlay (temporal alignment: one text token per
        // frame). Once trailing_len is exhausted, fall back to tts_pad_embed
        // so the model knows text conditioning has ended.
        const float* trail_row = (step < trailing_len)
            ? &trailing_text_hidden[static_cast<size_t>(step) * n_embd]
            : tts_pad_embed;
        for (int j = 0; j < n_embd; j++)
            next_embd[static_cast<size_t>(j)] += trail_row[j];

        // Diagnostic: log magnitudes for first 5 steps and at multiples of 25
        if (step < 5 || (step % 25) == 0) {
            double mag_next = 0.0;
            for (int j = 0; j < n_embd; j++) {
                double v = next_embd[static_cast<size_t>(j)];
                mag_next += v * v;
            }
            mag_next = std::sqrt(mag_next);
            std::fprintf(stderr, "  ar[%d]: |cur_hidden|=%.4f |next_embd|=%.4f cb0=%d fine0=%d fine14=%d\n",
                         step, mag_cur_hidden, mag_next, code_0,
                         fine_codes[0], fine_codes[n_fine_books - 1]);
        }

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

    // ── Trim trailing silence ────────────────────────────────────────────
    // The AR loop's EOS detection is imperfect — the model often generates
    // hundreds of near-silent frames after the actual speech content. We
    // trim leading/trailing silence with a generous margin so the output
    // sounds tight without clipping natural onset/decay.
    {
        const size_t before = resp.pcm_mono.size();
        auto trimmed = trim_silence(resp.pcm_mono.data(), resp.pcm_mono.size(),
                                    24000, /*threshold_db=*/40.0f,
                                    /*window_ms=*/10.0f, /*margin_ms=*/120.0f);
        if (trimmed.size() < before) {
            std::fprintf(stderr, "qwen3_tts: trimmed %.2fs → %.2fs (removed %.2fs silence)\n",
                         double(before) / 24000.0,
                         double(trimmed.size()) / 24000.0,
                         double(before - trimmed.size()) / 24000.0);
            resp.pcm_mono = std::move(trimmed);
        }
    }

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
