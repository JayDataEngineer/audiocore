// loader.cpp — Qwen3-TTS weight loading + family registration.
//
// Loads:
//   1. Talker GGUF    — qwen3tts architecture (llama.cpp)
//   2. Predictor GGUF — qwen3tts_cp architecture (llama.cpp)
//
// Codec (speech-tokenizer) decode previously ran through ONNX Runtime; that
// path has been removed. A ggml-based codec decoder is the planned
// replacement.
//
// The model_path in server.json is treated as a directory containing:
//   qwen3_tts_talker.gguf      (or talker_path in extras)
//   qwen3_tts_predictor.gguf   (or predictor_path in extras)

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
    // codec_decoder_path used to point at a speech-tokenizer ONNX; the ONNX
    // surface has been removed and the key is now ignored. A ggml codec
    // decoder will read its weights from the talker GGUF directly.

    // Resolve file paths from model directory
    std::string dir = model_path;
    if (!fs::is_directory(model_path)) {
        // model_path could be the talker GGUF directly; look for siblings
        dir = fs::path(model_path).parent_path().string();
    }

    config_.talker_path    = resolve_path(dir, config_.talker_path,    "qwen3_tts_talker.gguf");
    config_.predictor_path = resolve_path(dir, config_.predictor_path, "qwen3_tts_predictor.gguf");

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

    // ── 3. Codec decoder ───────────────────────────────────────────────────
    // The ONNX speech tokenizer has been removed. Codec decode now happens
    // inside run_inference() — currently as a silence stub pending a ggml
    // port of the speech-tokenizer graph.

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
