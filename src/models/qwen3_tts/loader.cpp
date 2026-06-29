// loader.cpp — Qwen3-TTS weight loading + family registration.
//
// Loads:
//   1. Talker GGUF    — qwen3tts architecture (llama.cpp); also carries
//                       `speaker.*` tensors for Stage 17b ECAPA-TDNN speaker
//                       encoder (loaded separately via GgufReader).
//   2. Predictor GGUF — qwen3tts_cp architecture (llama.cpp)
//   3. Codec GGUF     — qwen3-tts-tokenizer-12hz (Stage 17: ggml port,
//                       discovered via extras["codec_path"] or auto-probe
//                       for tokenizer-{f16,q8_0}.gguf next to the talker).
//                       Soft-fail: a setup without the codec sidecar or
//                       speaker encoder still loads; unsupported modes fall
//                       back to silence or fail-fast with a GAPS.md pointer.
//
// The model_path in server.json is treated as a directory containing:
//   qwen3_tts_talker.gguf      (or talker_path in extras)
//   qwen3_tts_predictor.gguf   (or predictor_path in extras)
//   tokenizer-f16.gguf         (or codec_path in extras; Stage 17)

#include "audiocore/models/qwen3_tts/family.h"
#include "audiocore/models/qwen3/runner.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include "audiocore/framework/core/backend.h"
#include "audiocore/framework/io/gguf_reader.h"
#include "audiocore/framework/io/weight_loader.h"
#include "audiocore/framework/runtime/registry.h"

namespace fs = std::filesystem;

namespace audiocore::qwen3_tts {

using audiocore::qwen3::Runner;
using audiocore::qwen3::RunnerConfig;

// ── Helpers: discover model files from directory ────────────────────────────

// If `path` is a directory, look for the expected filenames inside.
// If it's a file, use it directly.
static std::string resolve_path(const std::string& base_dir,
                                 const std::string& explicit_path,
                                 const std::string& default_name) {
    if (!explicit_path.empty()) return explicit_path;
    if (base_dir.empty()) return default_name;
    // Check if base_dir is itself a file
    if (fs::is_regular_file(base_dir)) return base_dir;
    // Look for default_name inside directory
    auto candidate = fs::path(base_dir) / default_name;
    if (fs::exists(candidate)) return candidate.string();
    // Fall back to base_dir (may be the file path itself)
    return base_dir;
}

// Stage 17: discover the Qwen3-TTS-Tokenizer-12Hz codec GGUF. Resolution
// order matches the manifest's `extras.codec_path` contract:
//   1. extras["codec_path"] (explicit user override) — returned verbatim.
//   2. tokenizer-f16.gguf   (full-precision; canonical name from
//      cstr/qwen3-tts-tokenizer-12hz-GGUF).
//   3. tokenizer-q8_0.gguf  (quantized fallback for limited-VRAM setups).
//   4. "" — no codec sidecar; the caller keeps the silence fallback.
static std::string resolve_codec_path(const std::string& base_dir,
                                       const std::string& explicit_path) {
    if (!explicit_path.empty()) return explicit_path;
    if (base_dir.empty()) return {};
    // Probe the canonical sidecar names published in models/manifest.json.
    for (const char* name : {"tokenizer-f16.gguf", "tokenizer-q8_0.gguf"}) {
        auto candidate = fs::path(base_dir) / name;
        if (fs::exists(candidate)) return candidate.string();
    }
    return {};
}

// ── Main load() ─────────────────────────────────────────────────────────────

bool Qwen3TtsSession::load(const std::string& model_path,
                            const LoadOptions& opts,
                            const BackendConfig& backend_cfg,
                            std::string* error) {
    // Parse config from extras
    {
        auto it = opts.extras.find("talker_path");
        if (it != opts.extras.end()) config_.talker_path = it->second;
    }
    {
        auto it = opts.extras.find("predictor_path");
        if (it != opts.extras.end()) config_.predictor_path = it->second;
    }
    {
        auto it = opts.extras.find("codec_path");
        if (it != opts.extras.end()) config_.codec_path = it->second;
    }

    // Resolve file paths from model directory
    std::string dir = model_path;
    if (!fs::is_directory(model_path)) {
        // model_path could be the talker GGUF directly; look for siblings
        dir = fs::path(model_path).parent_path().string();
    }

    config_.talker_path    = resolve_path(dir, config_.talker_path,    "qwen3_tts_talker.gguf");
    config_.predictor_path = resolve_path(dir, config_.predictor_path, "qwen3_tts_predictor.gguf");
    config_.codec_path     = resolve_codec_path(dir, config_.codec_path);

    // Parse numeric extras
    {
        auto it = opts.extras.find("n_gpu_layers");
        if (it != opts.extras.end()) config_.n_gpu_layers = std::stoi(it->second);
    }
    {
        auto it = opts.extras.find("temperature");
        if (it != opts.extras.end()) config_.temperature = std::stof(it->second);
    }
    {
        auto it = opts.extras.find("flash_attn");
        if (it != opts.extras.end())
            config_.flash_attn = it->second == "1" || it->second == "true";
    }
    {
        auto it = opts.extras.find("model_size_b");
        if (it != opts.extras.end()) {
            try { config_.model_size_b = std::stof(it->second); }
            catch (...) { /* leave at 0 = unknown */ }
        }
    }

    // Variant detection. extras["variant"] wins. Otherwise we look at the
    // directory name for a hint ("1.7b-customvoice", "voicedesign", …).
    // Without config.json parsing this is best-effort; the safe fallback is
    // Unknown, which behaves like Base (plain TTS works on every variant).
    {
        auto it = opts.extras.find("variant");
        if (it != opts.extras.end()) {
            const std::string& v = it->second;
            if (v == "base" || v == "Base")              config_.variant = Qwen3TtsVariant::Base;
            else if (v == "customvoice" || v == "CustomVoice")
                                                     config_.variant = Qwen3TtsVariant::CustomVoice;
            else if (v == "voicedesign" || v == "VoiceDesign")
                                                     config_.variant = Qwen3TtsVariant::VoiceDesign;
            else                                config_.variant = Qwen3TtsVariant::Unknown;
        }
    }
    if (config_.variant == Qwen3TtsVariant::Unknown) {
        // Infer from directory name. Lowercase substring match — the
        // manifest's variant keys ("qwen3-tts-1.7b-customvoice") are the
        // canonical shape; community paths may differ.
        std::string hay = dir;
        std::transform(hay.begin(), hay.end(), hay.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        if (hay.find("voicedesign") != std::string::npos)
            config_.variant = Qwen3TtsVariant::VoiceDesign;
        else if (hay.find("customvoice") != std::string::npos)
            config_.variant = Qwen3TtsVariant::CustomVoice;
        else if (hay.find("qwen3-tts") != std::string::npos ||
                 hay.find("qwen3_tts") != std::string::npos)
            config_.variant = Qwen3TtsVariant::Base;
    }

    std::fprintf(stderr, "qwen3_tts: loading talker from %s\n", config_.talker_path.c_str());
    std::fprintf(stderr, "qwen3_tts: loading predictor from %s\n", config_.predictor_path.c_str());
    std::fprintf(stderr, "qwen3_tts: variant=%s%s\n",
                 variant_name(config_.variant),
                 config_.model_size_b > 0
                     ? (" (" + std::to_string(config_.model_size_b) + "B)").c_str()
                     : " (size unknown)");

    // ── 1. Load Talker ─────────────────────────────────────────────────────
    RunnerConfig talker_cfg;
    talker_cfg.n_ctx        = config_.n_ctx_talker;
    talker_cfg.n_gpu_layers = config_.n_gpu_layers;
    talker_cfg.flash_attn   = config_.flash_attn;

    talker_ = Runner::load(config_.talker_path, talker_cfg, error);
    if (!talker_) {
        if (error) *error = "talker: " + *error;
        return false;
    }
    // Pull text-embedding + codec embd/head tensors out of the same GGUF.
    // Non-fatal if absent (Lunavox-style GGUFs); the family falls back to
    // the plain-token forward path.
    {
        std::string talker_extra_err;
        if (!talker_->load_extras(config_.talker_path,
                                   Runner::ExtraKind::Talker,
                                   /*n_fine_books=*/31, &talker_extra_err)) {
            std::fprintf(stderr,
                         "qwen3_tts: talker extra-tensor load (optional): %s\n",
                         talker_extra_err.c_str());
        }
    }
    std::fprintf(stderr, "qwen3_tts: talker loaded (hidden=%d, vocab=%d, text_embd=%s)\n",
                 talker_->hidden_size(), talker_->vocab_size(),
                 talker_->has_text_embedding() ? "yes" : "no");

    // ── 2a. Load ECAPA-TDNN speaker encoder (Stage 17b) ───────────────────
    // The speaker.* tensors live in the talker GGUF alongside the talker
    // transformer weights. llama.cpp skips unknown tensor names, so we open
    // the talker file with a separate GgufReader to resolve speaker.* tensors.
    // Soft-fail: Base models without the speaker encoder tensors skip this
    // step and Voice Clone mode returns a clear error.
    {
        std::string spk_err;
        speaker_reader_ = std::make_unique<GgufReader>();
        if (!speaker_reader_->load(config_.talker_path, &spk_err)) {
            std::fprintf(stderr, "qwen3_tts: speaker encoder GGUF load skipped (%s); "
                                 "voice_clone will fail fast\n", spk_err.c_str());
            speaker_reader_.reset();
        } else {
            // Parse HP from GGUF KV
            int32_t i32 = 0;
            if (speaker_reader_->get_kv_i32("qwen3tts.speaker.enc_dim", &i32))
                speaker_encoder_.hp.enc_dim = uint32_t(i32);
            if (speaker_reader_->get_kv_i32("qwen3tts.speaker.sample_rate", &i32))
                speaker_encoder_.hp.sample_rate = uint32_t(i32);

            speaker_backend_ = make_backend(backend_cfg, nullptr);
            ggml_backend_t be = speaker_backend_
                ? speaker_backend_->raw_ggml_backend() : nullptr;
            if (!be) {
                std::fprintf(stderr, "qwen3_tts: speaker encoder backend unavailable; "
                                     "voice_clone will fail fast\n");
                speaker_reader_.reset();
                speaker_backend_.reset();
            } else if (!speaker_encoder_.bind(speaker_reader_->meta_ctx(),
                                               be, &spk_err)) {
                std::fprintf(stderr, "qwen3_tts: speaker encoder bind failed (%s); "
                                     "voice_clone will fail fast\n", spk_err.c_str());
                speaker_reader_.reset();
                speaker_backend_.reset();
            } else {
                config_.speaker_present = true;
                std::fprintf(stderr,
                    "qwen3_tts: speaker encoder loaded (ECAPA-TDNN 128→%u)\n",
                    speaker_encoder_.hp.enc_dim);
            }
        }
    }

    // ── 2. Load Code Predictor ─────────────────────────────────────────────
    RunnerConfig pred_cfg;
    pred_cfg.n_ctx        = config_.n_ctx_predictor;
    pred_cfg.n_gpu_layers = config_.n_gpu_layers;
    pred_cfg.flash_attn   = config_.flash_attn;

    predictor_ = Runner::load(config_.predictor_path, pred_cfg, error);
    if (!predictor_) {
        if (error) *error = "predictor: " + *error;
        return false;
    }
    {
        std::string pred_extra_err;
        if (!predictor_->load_extras(config_.predictor_path,
                                      Runner::ExtraKind::Predictor,
                                      /*n_fine_books=*/31, &pred_extra_err)) {
            std::fprintf(stderr,
                         "qwen3_tts: predictor extra-tensor load (optional): %s\n",
                         pred_extra_err.c_str());
        }
    }
    std::fprintf(stderr, "qwen3_tts: predictor loaded (hidden=%d, vocab=%d, mtp=%s)\n",
                 predictor_->hidden_size(), predictor_->vocab_size(),
                 predictor_->has_mtp() ? "yes" : "no");

    // ── 3. Codec decoder (Stage 17) ──────────────────────────────────────
    // The Qwen3-TTS-Tokenizer-12Hz codec lives in its own GGUF
    // (cstr/qwen3-tts-tokenizer-12hz-GGUF) — it is NOT bundled with the
    // talker (unlike MOSS where the codec weights ride the same sidecar as
    // the backbone). resolve_codec_path() probes extras["codec_path"] then
    // falls back to tokenizer-{f16,q8_0}.gguf next to the talker GGUF.
    //
    // Soft-fail: if no codec GGUF is found, the session still loads; the
    // run_inference Phase 4 path emits 1 s silence (GAPS.md §2.3).
    if (!config_.codec_path.empty() && fs::exists(config_.codec_path)) {
        std::string codec_err;
        // Open via the standard GGUF reader so the codec tensors are
        // mmap'd (zero-copy) into a ggml_context.
        codec_reader_ = std::make_unique<GgufReader>();
        if (!codec_reader_->load(config_.codec_path, &codec_err)) {
            std::fprintf(stderr, "qwen3_tts: codec GGUF load skipped (%s); "
                                 "falling back to silence\n", codec_err.c_str());
            codec_reader_.reset();
        } else {
            // Parse codec hyperparameters from GGUF KV. All optional — the
            // HP struct's defaults already match the CrispASR reference
            // build (the only known-good config). Keys mirror the
            // qwen3tts_codec.dec.* convention CrispASR emits.
            Qwen3TtsCodecGraphs::HP& hp = codec_graphs_.hp;
            int32_t i32 = 0; float f32 = 0.0f;
            if (codec_reader_->get_kv_i32("qwen3tts_codec.dec.n_layers",     &i32)) hp.n_layers       = uint32_t(i32);
            if (codec_reader_->get_kv_i32("qwen3tts_codec.dec.d_model",      &i32)) hp.d_model        = uint32_t(i32);
            if (codec_reader_->get_kv_i32("qwen3tts_codec.dec.n_heads",      &i32)) hp.n_heads        = uint32_t(i32);
            if (codec_reader_->get_kv_i32("qwen3tts_codec.dec.head_dim",     &i32)) hp.head_dim       = uint32_t(i32);
            if (codec_reader_->get_kv_i32("qwen3tts_codec.dec.ff_dim",       &i32)) hp.ff_dim         = uint32_t(i32);
            if (codec_reader_->get_kv_i32("qwen3tts_codec.dec.n_quantizers", &i32)) hp.n_q            = uint32_t(i32);
            if (codec_reader_->get_kv_i32("qwen3tts_codec.dec.codebook_size",&i32)) hp.codebook_size = uint32_t(i32);
            if (codec_reader_->get_kv_i32("qwen3tts_codec.dec.latent_dim",   &i32)) hp.latent_dim    = uint32_t(i32);
            if (codec_reader_->get_kv_i32("qwen3tts_codec.dec.decoder_dim",  &i32)) hp.decoder_dim   = uint32_t(i32);
            if (codec_reader_->get_kv_i32("qwen3tts_codec.dec.sliding_window",&i32)) hp.sliding_window = uint32_t(i32);
            if (codec_reader_->get_kv_i32("qwen3tts_codec.dec.max_pos",      &i32)) hp.max_pos       = uint32_t(i32);
            if (codec_reader_->get_kv_f32("qwen3tts_codec.dec.rope_theta",   &f32)) hp.rope_theta    = f32;
            if (codec_reader_->get_kv_f32("qwen3tts_codec.dec.rms_norm_eps", &f32)) hp.rms_norm_eps  = f32;

            // Build the same ggml Backend the talker uses so codec + talker
            // share VRAM rather than round-tripping through host RAM.
            codec_backend_ = make_backend(backend_cfg, nullptr);
            ggml_backend_t be = codec_backend_ ? codec_backend_->raw_ggml_backend() : nullptr;
            if (!be) {
                std::fprintf(stderr, "qwen3_tts: codec backend unavailable; "
                                     "falling back to silence\n");
                codec_reader_.reset();
                codec_backend_.reset();
            } else if (!codec_graphs_.bind(codec_reader_->meta_ctx(), be, &codec_err)) {
                std::fprintf(stderr, "qwen3_tts: codec bind failed (%s); "
                                     "falling back to silence\n", codec_err.c_str());
                codec_reader_.reset();
                codec_backend_.reset();
            } else {
                config_.codec_present = true;
                std::fprintf(stderr,
                             "qwen3_tts: codec loaded from %s  (%uL d=%u/%u  rvq=%u)\n",
                             config_.codec_path.c_str(),
                             hp.n_layers, hp.d_model, hp.latent_dim, hp.n_q);
            }
        }
    } else {
        std::fprintf(stderr,
                     "qwen3_tts: no codec GGUF discovered (looked for "
                     "extras[codec_path] and tokenizer-{f16,q8_0}.gguf next "
                     "to talker); decode will emit silence\n");
    }

    loaded_ = true;
    return true;
}

// ── Factory registration ────────────────────────────────────────────────────

namespace {
std::unique_ptr<Session> make_qwen3_tts_session() {
    return std::unique_ptr<Session>(new Qwen3TtsSession());
}
}  // namespace

AUDIOCORE_REGISTER_FAMILY(qwen3_tts, make_qwen3_tts_session)
AUDIOCORE_EXTERN_C_GUARD(qwen3_tts, make_qwen3_tts_session)

}  // namespace audiocore::qwen3_tts
