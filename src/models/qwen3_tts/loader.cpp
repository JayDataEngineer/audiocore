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
#include "audiocore/models/qwen3_tts/talker_runner.h"
#include "audiocore/models/qwen3_tts/predictor_runner.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include "audiocore/framework/core/backend.h"
#include "audiocore/framework/runtime/registry.h"

namespace fs = std::filesystem;

namespace audiocore::qwen3_tts {

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

    std::fprintf(stderr, "qwen3_tts: loading talker from %s\n", config_.talker_path.c_str());
    std::fprintf(stderr, "qwen3_tts: loading predictor from %s\n", config_.predictor_path.c_str());

    // ── 1. Load Talker ─────────────────────────────────────────────────────
    TalkerConfig talker_cfg;
    talker_cfg.n_ctx        = config_.n_ctx_talker;
    talker_cfg.n_gpu_layers = config_.n_gpu_layers;
    talker_cfg.flash_attn   = config_.flash_attn;

    talker_ = TalkerRunner::load(config_.talker_path, talker_cfg, error);
    if (!talker_) {
        if (error) *error = "talker: " + *error;
        return false;
    }
    std::fprintf(stderr, "qwen3_tts: talker loaded (hidden=%d, vocab=%d)\n",
                 talker_->hidden_size(), talker_->vocab_size());

    // ── 2. Load Code Predictor ─────────────────────────────────────────────
    PredictorConfig pred_cfg;
    pred_cfg.n_ctx        = config_.n_ctx_predictor;
    pred_cfg.n_gpu_layers = config_.n_gpu_layers;
    pred_cfg.flash_attn   = config_.flash_attn;
    pred_cfg.n_codebooks  = config_.n_codebooks;

    predictor_ = PredictorRunner::load(config_.predictor_path, pred_cfg, error);
    if (!predictor_) {
        if (error) *error = "predictor: " + *error;
        return false;
    }
    std::fprintf(stderr, "qwen3_tts: predictor loaded (hidden=%d, vocab=%d, codebooks=%d)\n",
                 predictor_->hidden_size(), predictor_->vocab_size(),
                 predictor_->n_codebooks());

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
