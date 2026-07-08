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
    if (!explicit_path.empty()) {
        // If explicit_path is a relative path, join it with base_dir.
        fs::path p(explicit_path);
        if (p.is_relative() && !base_dir.empty() &&
            fs::is_directory(base_dir)) {
            return (fs::path(base_dir) / p).string();
        }
        return explicit_path;
    }
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
    if (!explicit_path.empty()) {
        // If explicit_path is a relative path, join it with base_dir.
        fs::path p(explicit_path);
        if (p.is_relative() && !base_dir.empty() &&
            fs::is_directory(base_dir)) {
            return (fs::path(base_dir) / p).string();
        }
        return explicit_path;
    }
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
    {
        auto it = opts.extras.find("speaker_encoder_path");
        if (it != opts.extras.end()) config_.speaker_encoder_path = it->second;
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
    // Speaker encoder GGUF is optional. Auto-probe sibling
    // `qwen3tts-speaker-encoder.gguf` if not explicitly provided.
    if (config_.speaker_encoder_path.empty()) {
        const fs::path probe = fs::path(dir) / "qwen3tts-speaker-encoder.gguf";
        if (fs::exists(probe)) config_.speaker_encoder_path = probe.string();
    } else if (config_.speaker_encoder_path != "/dev/null") {
        // Resolve relative paths against the model dir.
        config_.speaker_encoder_path =
            resolve_path(dir, config_.speaker_encoder_path, "");
    }

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

    // ── 1a. WDELTA patch (Base → CustomVoice) ────────────────────────────
    // If this is a CustomVoice talker AND a sibling Base talker GGUF is
    // available (or extras["wdelta_base_path"] points at one), overwrite
    // text_proj biases + codec_embedding in the loaded CV talker with Base's
    // versions. After the patch, CV accepts a continuous ECAPA embedding at
    // the speaker slot AND retains its instruct-tuned transformer norms.
    // See Runner::apply_wdelta_patch docstring + QWEN3-TTS-GAPS.md §A4.
    //
    // Enabled by default when CV + sibling Base is detected; opt out via
    // extras["wdelta_disable"]="1" (e.g. for A/B comparison).
    if (config_.variant == Qwen3TtsVariant::CustomVoice) {
        auto it_disable = opts.extras.find("wdelta_disable");
        const bool wdelta_disabled =
            (it_disable != opts.extras.end() && it_disable->second == "1");
        if (!wdelta_disabled) {
            // Resolve the Base GGUF path.
            // 1. Explicit override via extras["wdelta_base_path"].
            // 2. Sibling directory: replace "customvoice" → "base" in the
            //    talker's parent dir (handles 0.6b-customvoice → 0.6b-base
            //    and 1.7b-customvoice → 1.7b-base conventions).
            // 3. Same directory: pick any *-base*.gguf or *0b6*base*.gguf.
            std::string base_path;
            auto it_base = opts.extras.find("wdelta_base_path");
            if (it_base != opts.extras.end() && !it_base->second.empty()) {
                base_path = it_base->second;
            } else {
                fs::path cv_dir = fs::path(config_.talker_path).parent_path();
                std::string cv_dir_s = cv_dir.string();
                // Case-insensitive replace of "customvoice" → "base" in the
                // directory path. Both needles are the same length so the
                // replacement preserves the rest of the string.
                std::string hay = cv_dir_s;
                std::transform(hay.begin(), hay.end(), hay.begin(),
                               [](unsigned char c){ return std::tolower(c); });
                size_t pos = hay.find("customvoice");
                if (pos != std::string::npos) {
                    std::string base_dir_s = cv_dir_s;
                    base_dir_s.replace(pos, std::string("customvoice").size(), "base");
                    fs::path base_dir(base_dir_s);
                    // Look for the same talker filename the CV used, then a
                    // few canonical alternates.
                    const std::string cv_talker_fn =
                        fs::path(config_.talker_path).filename().string();
                    std::vector<fs::path> cands = {
                        base_dir / cv_talker_fn,
                        base_dir / "qwen3_tts_talker.gguf",
                        base_dir / "qwen3tts-talker-0b6-f16.gguf",
                        base_dir / "qwen3tts-talker-1b7-f16.gguf",
                    };
                    for (const auto& c : cands) {
                        if (fs::exists(c)) { base_path = c.string(); break; }
                    }
                }
                // Fallback #3: same directory — any *-base* or *base* GGUF.
                if (base_path.empty()) {
                    std::error_code ec;
                    for (auto& e : fs::directory_iterator(cv_dir, ec)) {
                        if (e.path().extension() != ".gguf") continue;
                        std::string n = e.path().filename().string();
                        std::transform(n.begin(), n.end(), n.begin(),
                                       [](unsigned char c){ return std::tolower(c); });
                        if (n.find("base") != std::string::npos &&
                            n.find("talker") != std::string::npos) {
                            base_path = e.path().string();
                            break;
                        }
                    }
                }
            }

            if (!base_path.empty() && fs::exists(base_path)) {
                std::string wdelta_err;
                if (!talker_->apply_wdelta_patch(base_path, &wdelta_err)) {
                    std::fprintf(stderr,
                        "qwen3_tts: WDELTA patch FAILED — continuing with the "
                        "unpatched CV talker (B1 path remains broken).\n"
                        "  reason: %s\n"
                        "  base GGUF: %s\n",
                        wdelta_err.c_str(), base_path.c_str());
                } else {
                    config_.wdelta_applied = true;
                    config_.wdelta_base_path = base_path;
                }
            } else if (base_path.empty()) {
                std::fprintf(stderr,
                    "qwen3_tts: WDELTA patch SKIPPED — no sibling Base talker "
                    "GGUF found for CV talker at %s.\n"
                    "  To enable the 4-feature pipeline (custom embedding + "
                    "instruct + text + quality), point extras[\"wdelta_base_path\"]\n"
                    "  at the matching Base talker GGUF (e.g. 0.6b-base/"
                    "qwen3_tts_talker.gguf).\n",
                    config_.talker_path.c_str());
            } else {
                std::fprintf(stderr,
                    "qwen3_tts: WDELTA patch SKIPPED — resolved base path '%s' "
                    "does not exist.\n", base_path.c_str());
            }
        }
    }


    // ── 1b. Load tokenizer sidecar (real Qwen3 BPE text tokenizer) ────────
    // The talker GGUF carries a dummy codec-vocab tokenizer (n_vocab == 3072
    // to match token_embd = codec_embedding). Text tokenization needs the
    // real Qwen3 BPE (151 936 tokens, pre=qwen2), shipped as a vocab-only
    // sidecar GGUF. Search the talker's directory for it.
    std::fprintf(stderr, "qwen3_tts: tokenizer sidecar probe — has_tokenizer=%d talker_path=%s\n",
                 static_cast<int>(talker_->has_tokenizer()),
                 config_.talker_path.c_str());
    if (!talker_->has_tokenizer()) {
        fs::path talker_dir = fs::path(config_.talker_path).parent_path();
        std::vector<fs::path> candidates;
        // Explicit override via extras["tokenizer_path"] (handled by caller
        // writing config_.codec_path — but for the tokenizer we scan the dir).
        for (const char* p : {
            "qwen3tts-tokenizer.gguf",
            "tokenizer-text.gguf",
            "text-tokenizer.gguf",
        }) {
            candidates.push_back(talker_dir / p);
        }
        // Glob: any qwen3tts-tokenizer-*.gguf in the directory.
        std::error_code ec;
        for (auto& e : fs::directory_iterator(talker_dir, ec)) {
            auto name = e.path().filename().string();
            if (name.rfind("qwen3tts-tokenizer", 0) == 0 &&
                e.path().extension() == ".gguf")
                candidates.push_back(e.path());
        }
        bool loaded = false;
        for (auto& p : candidates) {
            if (!fs::exists(p)) continue;
            std::string tok_err;
            if (talker_->load_tokenizer(p.string(), &tok_err)) {
                loaded = true;
                break;
            }
            std::fprintf(stderr, "qwen3_tts: tokenizer candidate %s failed: %s\n",
                         p.string().c_str(), tok_err.c_str());
        }
        if (!loaded) {
            std::fprintf(stderr,
                "qwen3_tts: WARNING — no tokenizer sidecar found in %s.\n"
                "  Text tokenization will use the talker's dummy codec-vocab\n"
                "  tokenizer → GARBAGE AUDIO. Re-run convert_qwen3tts without\n"
                "  --skip-tokenizer to produce qwen3tts-tokenizer-*.gguf.\n",
                talker_dir.c_str());
        }
    }

    // ── 2a. Load ECAPA-TDNN speaker encoder (Stage 17b) ───────────────────
    // The speaker encoder is shipped as a STANDALONE ~23 MB GGUF
    // (qwen3tts-speaker-encoder.gguf) produced by tools/convert_ecapa.cpp
    // from marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B. Tensor names are
    // namespaced as `speaker.*` so the existing bind() resolves them.
    //
    // If `speaker_encoder_path` is not set we fall back to the talker GGUF
    // itself — this preserves backwards compatibility with legacy bundled
    // layouts (CrispASR-style), even though no current upstream Qwen3-TTS
    // talker ships ECAPA weights inside the talker GGUF.
    //
    // Soft-fail: Base models without the speaker encoder skip this step and
    // Voice Clone mode returns a clear error.
    {
        const std::string& spk_path = !config_.speaker_encoder_path.empty()
            ? config_.speaker_encoder_path : config_.talker_path;
        std::string spk_err;
        speaker_reader_ = std::make_unique<GgufReader>();
        if (!speaker_reader_->load(spk_path, &spk_err)) {
            std::fprintf(stderr, "qwen3_tts: speaker encoder GGUF load skipped (%s); "
                                 "voice_clone will fail fast\n", spk_err.c_str());
            speaker_reader_.reset();
        } else {
            // Parse HP from GGUF KV. Two key conventions are accepted:
            //   qwen3tts_spk.*   — current standalone GGUF (general.architecture="qwen3tts_spk")
            //   qwen3tts.speaker.* — legacy bundled layout
            int32_t i32 = 0;
            if (speaker_reader_->get_kv_i32("qwen3tts_spk.enc_dim", &i32) ||
                speaker_reader_->get_kv_i32("qwen3tts.speaker.enc_dim", &i32))
                speaker_encoder_.hp.enc_dim = uint32_t(i32);
            if (speaker_reader_->get_kv_i32("qwen3tts_spk.sample_rate", &i32) ||
                speaker_reader_->get_kv_i32("qwen3tts.speaker.sample_rate", &i32))
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
                    "qwen3_tts: speaker encoder loaded from %s (ECAPA-TDNN 128→%u)\n",
                    spk_path.c_str(), speaker_encoder_.hp.enc_dim);

                // Register weight data sources (GGUF mmap → CPU backend
                // buffer). The meta_ctx tensors have data==NULL (no_alloc),
                // so run_on_mel's gallocr allocates fresh zero-init buffers
                // for them; upload_weights_() copies the real mmap data in
                // after allocation. Without this, all ECAPA weights are
                // silently zero and compute_embedding returns an all-zero
                // vector (latent bug — the ICL codec-token path carries
                // voice identity separately, so voice_clone still produced
                // speech, just without the ECAPA contribution).
                {
                    ggml_context* meta = speaker_reader_->meta_ctx();
                    if (meta) {
                        int n_reg = 0;
                        for (ggml_tensor* t = ggml_get_first_tensor(meta);
                             t; t = ggml_get_next_tensor(meta, t)) {
                            const char* name = ggml_get_name(t);
                            if (!name || strncmp(name, "speaker.", 8) != 0)
                                continue;
                            const TensorStorage* ts = speaker_reader_->find(name);
                            if (!ts) continue;
                            const void* mmap_ptr = speaker_reader_->tensor_data_ptr(*ts);
                            if (!mmap_ptr) continue;
                            speaker_encoder_.register_weight(t, mmap_ptr,
                                                             size_t(ggml_nbytes(t)));
                            n_reg++;
                        }
                        std::fprintf(stderr,
                            "qwen3_tts: registered %d speaker weight sources\n",
                            n_reg);
                    }
                }
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

                // Register weight data sources (GGUF mmap → backend device
                // memory). The meta_ctx tensors have data==NULL
                // (no_alloc=true); decode() uploads the actual weight data
                // from these registered sources after galloc allocates.
                {
                    ggml_context* meta = codec_reader_->meta_ctx();
                    if (meta) {
                        int n_reg = 0;
                        for (ggml_tensor* t = ggml_get_first_tensor(meta);
                             t; t = ggml_get_next_tensor(meta, t)) {
                            const char* name = ggml_get_name(t);
                            // Register both decoder and encoder weights.
                            // Decoder weights feed the decode graph (GPU
                            // upload via upload_weights_). Encoder weights
                            // feed both the encode graph (seanet/xfmr/ds →
                            // GPU) and the CPU RVQ quantizer
                            // (cenc_rvq_encode_, which reads the mmap host
                            // pointer directly via weight_host_data_).
                            // Previously only codec.dec.* was registered,
                            // leaving codec.enc.* with tensor->data == NULL.
                            // Fixed 2026-07-03 (Gap K).
                            if (!name || (strncmp(name, "codec.dec.", 10) != 0 &&
                                          strncmp(name, "codec.enc.", 10) != 0))
                                continue;
                            const TensorStorage* ts = codec_reader_->find(name);
                            if (!ts) continue;
                            const void* mmap_ptr = codec_reader_->tensor_data_ptr(*ts);
                            if (!mmap_ptr) continue;
                            codec_graphs_.register_weight(t, mmap_ptr,
                                                         size_t(ggml_nbytes(t)));
                            n_reg++;
                        }
                        std::fprintf(stderr,
                            "qwen3_tts: registered %d codec weight sources\n",
                            n_reg);
                    }
                }
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
