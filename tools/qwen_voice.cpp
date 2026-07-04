// tools/qwen_voice.cpp — Standalone Qwen3 voice-embedding CLI.
//
// Lets you treat Qwen3-TTS voices as serialisable 1024-/2048-d float vectors
// (the marksverdhei/Qwen3-Voice-Embedding workflow). With the vector on disk
// you can: cache a reference voice (encode once, reuse across thousands of
// run_tts calls), average multiple takes of the same speaker for a stabler
// identity, interpolate between two voices (gender/timbre slider), subtract
// voices to isolate attribute directions (gender, emotion, accent), shift a
// voice along such a direction, and then apply the result to a TTS call with
// an emotional `instruct` prompt.
//
// Voice file format (QWEN3VOICE, little-endian):
//   offset 0   16 bytes  magic = "QWEN3VOICE\0\0\0\0\0\0"
//   offset 16   4 bytes  version (u32) = 1
//   offset 20   4 bytes  dim      (u32) — 1024 (0.6B) or 2048 (1.7B)
//   offset 24   4 bytes  flags    (u32) — 0
//   offset 28   4 bytes  reserved (u32) — 0
//   offset 32   dim*4    float32  values
//
// Subcommands:
//   info <voice>
//       Print dim, L2 norm, mean, std, min, max, and first 8 components.
//
//   encode --model-dir <dir> [--encoder <spk.gguf>] --wav <ref.wav>
//           --out <voice> [--extra-wav a.wav [--extra-wav b.wav ...]]
//       Loads a qwen3_tts session (talker+predictor+codec+encoder), encodes
//       one or more reference WAVs, averages them, writes <voice>. The
//       averaging improves robustness — see u/k_means_clusterfuck's rec.
//       Set talker/predictor/codec filenames via QWEN3TTS_TALKER /
//       QWEN3TTS_PREDICTOR / QWEN3TTS_CODEC if non-default.
//
//   average <out.voice> <a.voice> [<b.voice> ...]
//       Element-wise mean of two or more saved voice files (same dim).
//
//   mix <a.voice> <b.voice> <alpha> <out.voice>
//       Linear interpolation: out = (1-α)·a + α·b. α in [0,1] typically.
//
//   add <a.voice> <b.voice> <out.voice>
//       Vector sum a + b. (For combining attribute directions.)
//
//   scale <voice> <scalar> <out.voice>
//       Multiply every component by `scalar`.
//
//   direction <from.voice> <to.voice> <out.direction>
//       Save (to - from) as a `.direction` file (same format as `.voice`).
//       Use to isolate an attribute axis (calm→angry, male↔female, etc.).
//
//   shift <base.voice> <direction.dir> <scale> <out.voice>
//       out = base + scale · direction. Apply an attribute axis to a voice.
//
//   apply --model-dir <dir> --text "..." (--voice <voice> | --reference-audio <wav>)
//         [--instruct "..."] [--language en] [--out out.wav]
//         [--temperature 0.7] [--top-p 0.9] [--speaker-name ""]
//         [--max-new-tokens 0] [--seed 0] [--reference-text "..."]
//       Loads a qwen3_tts session and runs TTS. Two cloning modes:
//         x-vector  : --voice <file.voice>  (pre-computed ECAPA embedding,
//                                             injected as a single spk slot)
//         ICL clone : --reference-audio <wav> [--reference-text "..."]
//                     (codec-encodes the WAV → ref_codes, the STRONG path)
//       Both can be combined (--voice + --reference-audio) for spk_emb +
//       codec tokens, matching the Python voice_clone_prompt API.
//       `--instruct` carries the emotional / conditional prompt
//       (e.g. "With deep sadness, whispering."). Ignored for ICL Base task.
//
// All vector-math subcommands are model-free and run instantly. Only
// `encode` and `apply` load the model.
//
// Build:
//   make qwen_voice
//
// Examples:
//   qwen_voice encode --model-dir ./weights/qwen3_tts/1.7b-base \
//                     --wav ref.wav --out myvoice.voice
//   qwen_voice mix calm.voice angry.voice 0.25 blended.voice
//   qwen_voice apply --model-dir ./weights/qwen3_tts/1.7b-base \
//                    --voice myvoice.voice --text "Hello there." \
//                    --instruct "Whispering, with deep sadness." \
//                    --out hello.wav

#include "audiocore/framework/core/backend.h"
#include "audiocore/framework/core/session.h"
#include "audiocore/framework/runtime/registry.h"
#include "audiocore/models/qwen3_tts/family.h"
#include "audiocore/server/server.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

extern "C" void audiocore_register_qwen3_tts();

namespace {

constexpr char     kMagic[16] = {'Q','W','E','N','3','V','O','I','C','E',
                                  0,0,0,0,0,0};
constexpr uint32_t kVersion   = 1;

#pragma pack(push, 1)
struct VoiceHeader {
    char     magic[16];
    uint32_t version;
    uint32_t dim;
    uint32_t flags;
    uint32_t reserved;
};
#pragma pack(pop)
static_assert(sizeof(VoiceHeader) == 32, "VoiceHeader must be 32 bytes");

// ── File I/O ─────────────────────────────────────────────────────────────

bool write_voice(const std::string& path, const std::vector<float>& v) {
    if (v.empty()) { std::fprintf(stderr, "[err] empty vector\n"); return false; }
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "[err] cannot open %s for write\n", path.c_str()); return false; }
    VoiceHeader h;
    std::memcpy(h.magic, kMagic, 16);
    h.version = kVersion;
    h.dim     = static_cast<uint32_t>(v.size());
    h.flags   = 0;
    h.reserved = 0;
    f.write(reinterpret_cast<const char*>(&h), sizeof(h));
    f.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(float)));
    return f.good();
}

std::vector<float> read_voice(const std::string& path, uint32_t* dim_out = nullptr) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "[err] cannot open %s\n", path.c_str()); return {}; }
    VoiceHeader h;
    f.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!f || std::memcmp(h.magic, kMagic, 16) != 0) {
        std::fprintf(stderr, "[err] %s is not a QWEN3VOICE file\n", path.c_str());
        return {};
    }
    if (h.version != kVersion) {
        std::fprintf(stderr, "[err] %s: unknown version %u\n", path.c_str(), h.version);
        return {};
    }
    if (h.dim == 0 || h.dim > 65536) {
        std::fprintf(stderr, "[err] %s: implausible dim %u\n", path.c_str(), h.dim);
        return {};
    }
    std::vector<float> v(h.dim);
    f.read(reinterpret_cast<char*>(v.data()),
           static_cast<std::streamsize>(h.dim * sizeof(float)));
    if (!f) { std::fprintf(stderr, "[err] %s: truncated body\n", path.c_str()); return {}; }
    if (dim_out) *dim_out = h.dim;
    return v;
}

// ── Vector helpers ───────────────────────────────────────────────────────

void print_stats(const std::string& label, const std::vector<float>& v) {
    if (v.empty()) { std::printf("[err] empty\n"); return; }
    double sum = 0, sum_sq = 0, mn = v[0], mx = v[0];
    for (float x : v) {
        sum += x; sum_sq += double(x) * x;
        if (x < mn) mn = x;
        if (x > mx) mx = x;
    }
    double mean = sum / v.size();
    double var  = sum_sq / v.size() - mean * mean;
    double std_ = std::sqrt(var > 0 ? var : 0);
    double l2   = std::sqrt(sum_sq);
    std::printf("%s: dim=%zu L2=%.4f mean= %.5f std=%.5f min=%.5f max=%.5f\n[",
                label.c_str(), v.size(), l2, mean, std_, mn, mx);
    const size_t kShow = std::min<size_t>(v.size(), 8);
    for (size_t i = 0; i < kShow; ++i) {
        std::printf("%+.4f%s", v[i], (i + 1 == kShow) ? "" : " ");
    }
    if (v.size() > kShow) std::printf(" …");
    std::printf("]\n");
}

bool same_dim_or_die(const std::vector<std::vector<float>>& vs) {
    if (vs.size() < 2) return true;
    size_t d = vs[0].size();
    for (size_t i = 1; i < vs.size(); ++i) {
        if (vs[i].size() != d) {
            std::fprintf(stderr,
                "[err] dimension mismatch: input 0 has %zu, input %zu has %zu\n",
                d, i, vs[i].size());
            return false;
        }
    }
    return true;
}

// ── CLI arg helpers ──────────────────────────────────────────────────────

// Return the value after `--key`, or empty if missing. Advances consumption.
std::string grab_arg(int& argc, char**& argv, const std::string& key) {
    for (int i = 0; i < argc - 1; ++i) {
        if (key == argv[i]) {
            std::string v = argv[i + 1];
            // shift remaining args left by 2
            for (int j = i; j + 2 < argc; ++j) argv[j] = argv[j + 2];
            argc -= 2;
            argv += 0;  // we don't actually re-base; caller rewrites below
            return v;
        }
    }
    return "";
}

// Repeated flag → list of values (e.g. --extra-wav a.wav --extra-wav b.wav).
std::vector<std::string> grab_args(int argc, char** argv, const std::string& key) {
    std::vector<std::string> out;
    for (int i = 0; i < argc - 1; ++i) {
        if (key == argv[i]) {
            out.push_back(argv[i + 1]);
        }
    }
    return out;
}

// ── Session helpers (encode / apply only) ────────────────────────────────

struct SessionCfg {
    std::string model_dir;
    std::string talker_fn   = "qwen3tts-talker-1b7-f16.gguf";
    std::string pred_fn     = "qwen3tts-predictor-1b7-f16.gguf";
    // IMPORTANT: the audio codec (Qwen3-TTS-Tokenizer-12Hz, 398 tensors,
    // ~358 MB) is published as tokenizer-f16.gguf. The similarly-named
    // qwen3tts-tokenizer-1b7-f16.gguf is the much smaller TEXT tokenizer
    // (~5.9 MB, 0 codec tensors) and will soft-fail decode → silence.
    std::string codec_fn    = "tokenizer-f16.gguf";
    std::string encoder_fn;  // optional override
    int         ngpu        = 99;
    std::string backend     = "ggml_cuda";
};

SessionCfg session_cfg_from_env(const std::string& model_dir,
                                const std::string& encoder_override) {
    SessionCfg c;
    c.model_dir = model_dir;
    if (const char* t = std::getenv("QWEN3TTS_TALKER"))    c.talker_fn = t;
    if (const char* p = std::getenv("QWEN3TTS_PREDICTOR"))  c.pred_fn   = p;
    if (const char* k = std::getenv("QWEN3TTS_CODEC"))      c.codec_fn  = k;
    if (const char* n = std::getenv("QWEN3TTS_NGPU"))       c.ngpu      = std::atoi(n);
    if (const char* d = std::getenv("QWEN3TTS_DEVICE"))     c.backend   = d;
    c.encoder_fn = encoder_override;
    return c;
}

std::unique_ptr<audiocore::Session> load_session(const SessionCfg& c,
                                                 std::string* err) {
    using namespace audiocore;
    auto sess = FamilyRegistry::instance().create("qwen3_tts");
    if (!sess) { *err = "FamilyRegistry::create(qwen3_tts) failed"; return nullptr; }

    LoadOptions opts;
    opts.extras["talker_path"]    = c.talker_fn;
    opts.extras["predictor_path"] = c.pred_fn;
    opts.extras["codec_path"]     = c.codec_fn;
    opts.extras["n_gpu_layers"]   = std::to_string(c.ngpu);
    if (!c.encoder_fn.empty())
        opts.extras["speaker_encoder_path"] = c.encoder_fn;

    BackendKind kind = BackendKind::ggml_cpu;
    if (c.backend == "ggml_cuda")   kind = BackendKind::ggml_cuda;
    if (c.backend == "ggml_vulkan") kind = BackendKind::ggml_vulkan;
    BackendConfig bc = { .kind = kind, .device_id = 0, .n_threads = 4 };

    if (!sess->load(c.model_dir, opts, bc, err)) return nullptr;
    return sess;
}

// Downcast to the qwen3_tts session to reach encoder-aware methods.
audiocore::qwen3_tts::Qwen3TtsSession* as_q3tts(audiocore::Session* s) {
    return dynamic_cast<audiocore::qwen3_tts::Qwen3TtsSession*>(s);
}

}  // namespace

// ── Subcommand handlers ──────────────────────────────────────────────────

static int cmd_info(int argc, char** argv) {
    if (argc < 1) { std::fprintf(stderr, "usage: qwen_voice info <voice>\n"); return 2; }
    auto v = read_voice(argv[0]);
    if (v.empty()) return 1;
    print_stats(argv[0], v);
    return 0;
}

static int cmd_average(int argc, char** argv) {
    // usage: average <out.voice> <a.voice> [<b.voice> ...]
    if (argc < 3) {
        std::fprintf(stderr, "usage: qwen_voice average <out.voice> <a.voice> [<b.voice> ...]\n");
        return 2;
    }
    const std::string out_path = argv[0];
    std::vector<std::vector<float>> vs;
    for (int i = 1; i < argc; ++i) {
        auto v = read_voice(argv[i]);
        if (v.empty()) return 1;
        vs.push_back(std::move(v));
    }
    if (!same_dim_or_die(vs)) return 1;
    std::vector<float> out(vs[0].size(), 0.0f);
    for (const auto& v : vs)
        for (size_t i = 0; i < v.size(); ++i) out[i] += v[i];
    const float n = static_cast<float>(vs.size());
    for (float& x : out) x /= n;
    if (!write_voice(out_path, out)) return 1;
    print_stats("wrote " + out_path, out);
    return 0;
}

static int cmd_mix(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: qwen_voice mix <a.voice> <b.voice> <alpha> <out.voice>\n");
        return 2;
    }
    auto a = read_voice(argv[0]);
    auto b = read_voice(argv[1]);
    if (a.empty() || b.empty()) return 1;
    if (a.size() != b.size()) {
        std::fprintf(stderr, "[err] dim mismatch %zu vs %zu\n", a.size(), b.size());
        return 1;
    }
    float alpha = std::strtof(argv[2], nullptr);
    std::vector<float> out(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        out[i] = (1.0f - alpha) * a[i] + alpha * b[i];
    if (!write_voice(argv[3], out)) return 1;
    print_stats(std::string("wrote ") + argv[3], out);
    return 0;
}

static int cmd_add(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: qwen_voice add <a.voice> <b.voice> <out.voice>\n");
        return 2;
    }
    auto a = read_voice(argv[0]);
    auto b = read_voice(argv[1]);
    if (a.empty() || b.empty() || a.size() != b.size()) {
        std::fprintf(stderr, "[err] read/dim mismatch\n"); return 1;
    }
    std::vector<float> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] + b[i];
    if (!write_voice(argv[2], out)) return 1;
    print_stats(std::string("wrote ") + argv[2], out);
    return 0;
}

static int cmd_scale(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: qwen_voice scale <voice> <scalar> <out.voice>\n");
        return 2;
    }
    auto v = read_voice(argv[0]);
    if (v.empty()) return 1;
    float s = std::strtof(argv[1], nullptr);
    for (float& x : v) x *= s;
    if (!write_voice(argv[2], v)) return 1;
    print_stats(std::string("wrote ") + argv[2], v);
    return 0;
}

static int cmd_direction(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: qwen_voice direction <from.voice> <to.voice> <out.direction>\n");
        return 2;
    }
    auto a = read_voice(argv[0]);
    auto b = read_voice(argv[1]);
    if (a.empty() || b.empty() || a.size() != b.size()) {
        std::fprintf(stderr, "[err] read/dim mismatch\n"); return 1;
    }
    std::vector<float> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = b[i] - a[i];
    if (!write_voice(argv[2], out)) return 1;
    print_stats(std::string("wrote direction ") + argv[2], out);
    return 0;
}

static int cmd_shift(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr,
            "usage: qwen_voice shift <base.voice> <direction.dir> <scale> <out.voice>\n");
        return 2;
    }
    auto base = read_voice(argv[0]);
    auto dir  = read_voice(argv[1]);
    if (base.empty() || dir.empty() || base.size() != dir.size()) {
        std::fprintf(stderr, "[err] read/dim mismatch\n"); return 1;
    }
    float scale = std::strtof(argv[2], nullptr);
    std::vector<float> out(base.size());
    for (size_t i = 0; i < base.size(); ++i) out[i] = base[i] + scale * dir[i];
    if (!write_voice(argv[3], out)) return 1;
    print_stats(std::string("wrote ") + argv[3], out);
    return 0;
}

static int cmd_encode(int argc, char** argv) {
    std::string model_dir = grab_arg(argc, argv, "--model-dir");
    std::string encoder   = grab_arg(argc, argv, "--encoder");
    std::string out       = grab_arg(argc, argv, "--out");
    std::string wav       = grab_arg(argc, argv, "--wav");
    auto extra_wavs       = grab_args(argc, argv, "--extra-wav");
    if (model_dir.empty() || out.empty() || wav.empty()) {
        std::fprintf(stderr,
            "usage: qwen_voice encode --model-dir <dir> --wav <ref.wav> \\\n"
            "       --out <voice> [--encoder <spk.gguf>] [--extra-wav a.wav ...]\n");
        return 2;
    }

    audiocore_register_qwen3_tts();
    SessionCfg c = session_cfg_from_env(model_dir, encoder);
    std::string err;
    auto sess = load_session(c, &err);
    if (!sess) { std::fprintf(stderr, "[err] load failed: %s\n", err.c_str()); return 1; }
    auto* q = as_q3tts(sess.get());
    if (!q) { std::fprintf(stderr, "[err] session is not qwen3_tts\n"); return 1; }

    // Encode the primary WAV.
    auto emb = q->extract_speaker_embedding(wav, &err);
    if (emb.empty()) { std::fprintf(stderr, "[err] encode failed: %s\n", err.c_str()); return 1; }
    std::fprintf(stderr, "[info] %s: encoded dim=%zu from %s\n",
                 out.c_str(), emb.size(), wav.c_str());

    // Average any additional takes.
    for (const auto& extra : extra_wavs) {
        auto e2 = q->extract_speaker_embedding(extra, &err);
        if (e2.empty()) { std::fprintf(stderr, "[err] extra encode failed: %s\n", err.c_str()); return 1; }
        if (e2.size() != emb.size()) {
            std::fprintf(stderr, "[err] dim mismatch on %s\n", extra.c_str()); return 1;
        }
        for (size_t i = 0; i < emb.size(); ++i) emb[i] = 0.5f * (emb[i] + e2[i]);
        std::fprintf(stderr, "[info] averaged with %s\n", extra.c_str());
    }

    if (!write_voice(out, emb)) return 1;
    print_stats("wrote " + out, emb);
    return 0;
}

static int cmd_apply(int argc, char** argv) {
    std::string model_dir = grab_arg(argc, argv, "--model-dir");
    std::string voice     = grab_arg(argc, argv, "--voice");
    std::string text      = grab_arg(argc, argv, "--text");
    std::string instruct  = grab_arg(argc, argv, "--instruct");
    std::string language  = grab_arg(argc, argv, "--language");
    std::string out       = grab_arg(argc, argv, "--out");
    std::string spk_name  = grab_arg(argc, argv, "--speaker-name");
    std::string temp_s    = grab_arg(argc, argv, "--temperature");
    std::string topp_s    = grab_arg(argc, argv, "--top-p");
    std::string maxnew_s  = grab_arg(argc, argv, "--max-new-tokens");
    std::string seed_s    = grab_arg(argc, argv, "--seed");
    std::string encoder   = grab_arg(argc, argv, "--encoder");
    // ICL voice-clone inputs. When --reference-audio is supplied the session
    // runs the codec encoder over the WAV and injects the resulting frames
    // as in-context reference codes — the strong cloning path (see
    // session.cpp:279+). --reference-text, if provided, supplies the
    // transcript that gets tokenized and injected alongside, matching the
    // Python `voice_clone_prompt=(ref_code, ref_text, spk_emb)` API.
    std::string ref_audio = grab_arg(argc, argv, "--reference-audio");
    std::string ref_text  = grab_arg(argc, argv, "--reference-text");
    if (model_dir.empty() || text.empty() || (voice.empty() && ref_audio.empty())) {
        std::fprintf(stderr,
            "usage: qwen_voice apply --model-dir <dir> --text \"...\" \\\n"
            "       (--voice <voice> | --reference-audio <wav>) \\\n"
            "       [--instruct \"...\"] [--out out.wav] \\\n"
            "       [--language en] [--speaker-name \"\"] [--temperature 0.7]\n"
            "       [--encoder <spk.gguf>]\n"
            "       [--reference-audio <wav>] [--reference-text \"transcript\"]\n"
            "  x-vector mode:  --voice <file.voice>   (pre-computed ECAPA embedding)\n"
            "  ICL clone mode: --reference-audio <wav> [--reference-text \"...\"]\n"
            "  both:           --voice + --reference-audio (spk_emb + codec tokens)\n");
        return 2;
    }
    if (out.empty()) out = "qwen_voice_apply.wav";

    std::vector<float> emb;
    if (!voice.empty()) {
        emb = read_voice(voice);
        if (emb.empty()) return 1;
        std::fprintf(stderr, "[info] loaded voice: dim=%zu\n", emb.size());
    }

    audiocore_register_qwen3_tts();
    SessionCfg c = session_cfg_from_env(model_dir, encoder);
    std::string err;
    auto sess = load_session(c, &err);
    if (!sess) { std::fprintf(stderr, "[err] load failed: %s\n", err.c_str()); return 1; }
    auto* q = as_q3tts(sess.get());
    if (!q) { std::fprintf(stderr, "[err] session is not qwen3_tts\n"); return 1; }

    audiocore::TtsRequest req;
    req.text        = text;
    req.instruct    = instruct;
    req.language    = language.empty() ? "en" : language;
    req.speaker_name= spk_name;
    req.temperature = temp_s.empty() ? 0.7f : std::strtof(temp_s.c_str(), nullptr);
    req.top_p       = topp_s.empty() ? 0.9f : std::strtof(topp_s.c_str(), nullptr);
    req.max_new_tokens = maxnew_s.empty() ? 100 : std::atoi(maxnew_s.c_str());
    req.seed        = seed_s.empty() ? 0 : std::atoi(seed_s.c_str());
    req.reference_audio = ref_audio;
    req.reference_text  = ref_text;

    if (!instruct.empty())
        std::fprintf(stderr, "[info] instruct: \"%s\"\n", instruct.c_str());

    audiocore::TtsResponse resp;
    // x-vector fast path: pre-computed embedding only. ICL path: drive
    // run_tts directly so the session runs the codec encoder on
    // reference_audio and injects the resulting frames as in-context
    // reference codes (the strong cloning path). Combining --voice with
    // --reference-audio gives spk_emb + codec tokens, matching the Python
    // voice_clone_prompt=(ref_code, ref_spk_embedding, ref_text) API.
    bool ok = false;
    if (!ref_audio.empty()) {
        req.mode = "voice_clone";
        if (!emb.empty()) req.speaker_embedding = emb;
        std::fprintf(stderr, "[info] ICL voice clone: ref_audio=%s ref_text=%s\n",
                     ref_audio.c_str(), ref_text.empty() ? "(none)" : "(provided)");
        ok = q->run_tts(&req, &resp, &err);
        if (!ok)
            std::fprintf(stderr, "[err] run_tts (ICL) failed: %s\n", err.c_str());
    } else {
        ok = q->synthesize_with_embedding(req, emb.data(), emb.size(), resp, &err);
        if (!ok)
            std::fprintf(stderr, "[err] synthesize_with_embedding failed: %s\n", err.c_str());
    }
    if (!ok) return 1;
    if (resp.pcm_mono.empty()) {
        std::fprintf(stderr, "[err] empty PCM\n"); return 1;
    }

    double sum_sq = 0.0;
    for (float s : resp.pcm_mono) sum_sq += double(s) * s;
    double rms = std::sqrt(sum_sq / resp.pcm_mono.size());
    std::fprintf(stderr, "[info] out: %zu samples @ %d Hz (%.2fs), RMS=%.6f\n",
                 resp.pcm_mono.size(), resp.sampling_rate,
                 double(resp.pcm_mono.size()) / resp.sampling_rate, rms);

    std::vector<float> pcm = std::move(resp.pcm_mono);
    std::string wav = audiocore::pcm_mono_to_wav(pcm, resp.sampling_rate);
    std::ofstream f(out, std::ios::binary);
    if (!f) { std::fprintf(stderr, "[err] cannot write %s\n", out.c_str()); return 1; }
    f.write(wav.data(), static_cast<std::streamsize>(wav.size()));
    std::fprintf(stderr, "[ok] wrote %s\n", out.c_str());
    return 0;
}

static void usage() {
    std::fprintf(stderr,
        "qwen_voice — Qwen3 voice-embedding CLI\n\n"
        "Vector-math subcommands (instant, model-free):\n"
        "  info <voice>\n"
        "  average <out.voice> <a.voice> [<b.voice> ...]\n"
        "  mix <a.voice> <b.voice> <alpha> <out.voice>\n"
        "  add <a.voice> <b.voice> <out.voice>\n"
        "  scale <voice> <scalar> <out.voice>\n"
        "  direction <from.voice> <to.voice> <out.direction>\n"
        "  shift <base.voice> <direction.dir> <scale> <out.voice>\n\n"
        "Model-backed subcommands (load qwen3_tts session):\n"
        "  encode --model-dir <dir> --wav <ref.wav> --out <voice>\n"
        "         [--encoder <spk.gguf>] [--extra-wav a.wav ...]\n"
        "  apply  --model-dir <dir> --voice <voice> --text \"...\"\n"
        "         [--instruct \"...\"] [--out out.wav] [--language en]\n\n"
        "Voice file format: magic 'QWEN3VOICE' + u32 version + u32 dim +\n"
        "                   u32 flags + u32 reserved + dim*float32 values.\n");
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 2; }
    std::string sub = argv[1];
    int sub_argc    = argc - 2;
    char** sub_argv = argv + 2;

    if      (sub == "info")      return cmd_info(sub_argc, sub_argv);
    else if (sub == "average")   return cmd_average(sub_argc, sub_argv);
    else if (sub == "mix")       return cmd_mix(sub_argc, sub_argv);
    else if (sub == "add")       return cmd_add(sub_argc, sub_argv);
    else if (sub == "scale")     return cmd_scale(sub_argc, sub_argv);
    else if (sub == "direction") return cmd_direction(sub_argc, sub_argv);
    else if (sub == "shift")     return cmd_shift(sub_argc, sub_argv);
    else if (sub == "encode")    return cmd_encode(argc, argv);
    else if (sub == "apply")     return cmd_apply(argc, argv);
    else if (sub == "-h" || sub == "--help" || sub == "help") { usage(); return 0; }

    std::fprintf(stderr, "[err] unknown subcommand '%s'\n", sub.c_str());
    usage();
    return 2;
}
