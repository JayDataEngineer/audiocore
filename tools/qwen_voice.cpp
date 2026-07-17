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
//       averaging improves robustness — see a community recommendation.
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

#include "ggml.h"
#include "ggml-backend.h"

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
    std::string talker_fn;   // auto-resolved below
    std::string pred_fn;     // auto-resolved below
    // IMPORTANT: the audio codec (Qwen3-TTS-Tokenizer-12Hz, 398 tensors,
    // ~358 MB) is published as tokenizer-f16.gguf. The similarly-named
    // qwen3tts-tokenizer-{0b6,1b7}-f16.gguf is the much smaller TEXT
    // tokenizer (~5.9 MB, 0 codec tensors) and will soft-fail decode → silence.
    std::string codec_fn    = "tokenizer-f16.gguf";
    std::string encoder_fn;  // optional override
    int         ngpu        = 99;
    std::string backend     = "ggml_cuda";
};

// D3 fix: pick the first existing candidate filename inside `dir`.
// Falls back to the legacy hardcoded default if none of the candidates
// are present (so the original error message stays meaningful).
static std::string resolve_first_existing(const std::string& dir,
                                          std::initializer_list<const char*> cands,
                                          const char* fallback) {
    for (const char* c : cands) {
        std::string p = dir.empty() ? std::string(c) : (dir + "/" + c);
        std::ifstream f(p, std::ios::binary);
        if (f.good()) return c;
    }
    return fallback;
}

SessionCfg session_cfg_from_env(const std::string& model_dir,
                                const std::string& encoder_override) {
    SessionCfg c;
    c.model_dir = model_dir;

    // D3: variant-aware defaults. The published 1.7B files use the
    // `qwen3tts-talker-1b7-f16.gguf` naming scheme, but locally converted
    // copies use `qwen3_tts_talker.gguf`. Pick whichever exists on disk.
    // Order matters: prefer the size-suffixed name when present (it pins
    // the variant unambiguously), then fall back to the generic name.
    c.talker_fn = resolve_first_existing(
        model_dir,
        { "qwen3tts-talker-1b7-f16.gguf",
          "qwen3tts-talker-0b6-f16.gguf",
          "qwen3_tts_talker.gguf" },
        "qwen3tts-talker-1b7-f16.gguf");
    c.pred_fn = resolve_first_existing(
        model_dir,
        { "qwen3tts-predictor-1b7-f16.gguf",
          "qwen3tts-predictor-0b6-f16.gguf",
          "qwen3_tts_predictor.gguf" },
        "qwen3tts-predictor-1b7-f16.gguf");
    c.codec_fn = resolve_first_existing(
        model_dir,
        { "tokenizer-f16.gguf" },   // only valid codec name; text tok sidecar is excluded
        "tokenizer-f16.gguf");

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

    // Backend resolution: respect the user's choice when the backend is
    // actually linked in. If they asked for CUDA/Vulkan but no GPU device
    // is registered (e.g. CPU-only build, no driver), transparently fall
    // back to CPU instead of letting the speaker/codec backend creation
    // silently fail and leave the session half-loaded.
    BackendKind kind = BackendKind::ggml_cpu;
    if (c.backend == "ggml_cuda")   kind = BackendKind::ggml_cuda;
    if (c.backend == "ggml_vulkan") kind = BackendKind::ggml_vulkan;

    int n_devs = ggml_backend_dev_count();
    bool gpu_ok = false;
    for (int i = 0; i < n_devs; i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (d && ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_ok = true;
            break;
        }
    }
    if ((kind == BackendKind::ggml_cuda || kind == BackendKind::ggml_vulkan) && !gpu_ok) {
        std::fprintf(stderr, "[info] no GPU backend registered; falling back to CPU\n");
        kind = BackendKind::ggml_cpu;
        // ngpu_layers > 0 is meaningless on CPU and would just log noise.
        opts.extras["n_gpu_layers"] = "0";
    }

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

// ── SLERP: Spherical Linear Interpolation ───────────────────────────────
//
// The ECAPA-TDNN embeddings are NOT unit-norm (typical L2 ≈ 17.7), but
// they all live near a thin shell of similar radius. SLERP generalizes
// linear interpolation to great-circle arcs on the unit hypersphere:
//
//   SLERP(p, q, t) = (sin((1-t)·Ω) / sin Ω) · p + (sin(t·Ω) / sin Ω) · q
//
// where Ω = arccos( cos_sim(p, q) ) and t ∈ [0, 1].
//
// Because the magnitudes are nearly identical across voices, we normalize
// both inputs first, SLERP, then re-scale the result to the average L2 to
// match what the talker expects. This gives a perceptually smoother blend
// than linear `mix` when |t - 0.5| is far from 0 or 1.
static int cmd_slerp(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr,
            "usage: qwen_voice slerp <a.voice> <b.voice> <t> <out.voice>\n"
            "  t ∈ [0, 1]: 0 = pure a, 1 = pure b, 0.5 = equal mix.\n");
        return 2;
    }
    auto a = read_voice(argv[0]);
    auto b = read_voice(argv[1]);
    if (a.empty() || b.empty() || a.size() != b.size()) {
        std::fprintf(stderr, "[err] read/dim mismatch\n"); return 1;
    }
    float t = std::strtof(argv[2], nullptr);
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;

    // Magnitudes (we'll restore to the average of the two).
    double na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    na = std::sqrt(na);
    nb = std::sqrt(nb);
    if (na < 1e-9 || nb < 1e-9) {
        std::fprintf(stderr, "[err] zero-norm voice\n"); return 1;
    }
    const double avg_norm = 0.5 * (na + nb);

    // Unit-normalized copies.
    std::vector<double> ua(a.size()), ub(b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        ua[i] = static_cast<double>(a[i]) / na;
        ub[i] = static_cast<double>(b[i]) / nb;
    }

    // Cosine similarity clamped to valid arccos domain.
    double dot = 0.0;
    for (size_t i = 0; i < ua.size(); ++i) dot += ua[i] * ub[i];
    if (dot >  1.0) dot =  1.0;
    if (dot < -1.0) dot = -1.0;
    const double omega = std::acos(dot);

    std::vector<float> out(a.size());
    // If Ω is tiny, the two voices are nearly identical — fall back to LERP
    // to avoid division by zero in sin(Ω).
    if (omega < 1e-4) {
        for (size_t i = 0; i < a.size(); ++i)
            out[i] = static_cast<float>(((1.0 - t) * ua[i] + t * ub[i]) * avg_norm);
    } else {
        const double sin_omega = std::sin(omega);
        const double w_a = std::sin((1.0 - t) * omega) / sin_omega;
        const double w_b = std::sin(t * omega) / sin_omega;
        for (size_t i = 0; i < a.size(); ++i)
            out[i] = static_cast<float>((w_a * ua[i] + w_b * ub[i]) * avg_norm);
    }
    if (!write_voice(argv[3], out)) return 1;
    print_stats(std::string("wrote ") + argv[3], out);
    std::fprintf(stderr, "[info] SLERP: omega=%.4f rad (%.1f deg), t=%.3f\n",
                 omega, omega * 180.0 / M_PI, t);
    return 0;
}

// ── discover_direction: build a steering vector from labeled groups ─────
//
// Knob 2 in the user's voice-modification taxonomy. Reads every .voice
// in <positive_dir> and <negative_dir>, computes the group means, and
// emits `mean_pos - mean_neg` as a direction file. Apply with `shift`.
//
//   qwen_voice discover_direction voices/high/ voices/low/ pitch.dir
//   qwen_voice shift base.voice pitch.dir 0.5 out.voice
//
// Convention: positive group = the "more X" end (e.g., higher pitch),
// negative group = "less X". Positive `scale` in `shift` then = more X.
static int cmd_discover_direction(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr,
            "usage: qwen_voice discover_direction <positive_dir> <negative_dir> <out.dir>\n"
            "  Reads every *.voice in each dir, computes group means, emits\n"
            "  mean_pos - mean_neg as a steering vector. Apply with `shift`.\n"
            "  Example: discover_direction voices/high_pitch/ voices/low_pitch/ pitch.dir\n");
        return 2;
    }
    const std::string pos_dir  = argv[0];
    const std::string neg_dir  = argv[1];
    const std::string out_path = argv[2];

    auto load_dir_means = [](const std::string& dir, std::vector<float>& mean,
                              size_t& dim, size_t& count) -> bool {
        namespace fs = std::filesystem;
        if (!fs::is_directory(dir)) {
            std::fprintf(stderr, "[err] not a directory: %s\n", dir.c_str());
            return false;
        }
        std::vector<float> acc;
        dim = 0; count = 0;
        for (auto& e : fs::directory_iterator(dir)) {
            if (!e.is_regular_file()) continue;
            if (e.path().extension() != ".voice") continue;
            auto v = read_voice(e.path().string());
            if (v.empty()) continue;
            if (acc.empty()) {
                acc.assign(v.size(), 0.0f);
                dim = v.size();
            } else if (v.size() != dim) {
                std::fprintf(stderr, "[warn] dim mismatch in %s (%zu vs %zu), skipping\n",
                             e.path().string().c_str(), v.size(), dim);
                continue;
            }
            for (size_t i = 0; i < dim; ++i) acc[i] += v[i];
            ++count;
        }
        if (count == 0) {
            std::fprintf(stderr, "[err] no .voice files in %s\n", dir.c_str());
            return false;
        }
        mean.assign(dim, 0.0f);
        for (size_t i = 0; i < dim; ++i) mean[i] = acc[i] / static_cast<float>(count);
        return true;
    };

    std::vector<float> mean_pos, mean_neg;
    size_t dim_pos = 0, dim_neg = 0, cnt_pos = 0, cnt_neg = 0;
    if (!load_dir_means(pos_dir, mean_pos, dim_pos, cnt_pos)) return 1;
    if (!load_dir_means(neg_dir, mean_neg, dim_neg, cnt_neg)) return 1;
    if (dim_pos != dim_neg) {
        std::fprintf(stderr, "[err] positive dim %zu vs negative dim %zu\n", dim_pos, dim_neg);
        return 1;
    }
    std::vector<float> direction(dim_pos);
    for (size_t i = 0; i < dim_pos; ++i)
        direction[i] = mean_pos[i] - mean_neg[i];
    if (!write_voice(out_path, direction)) return 1;
    double norm = 0.0;
    for (float x : direction) norm += static_cast<double>(x) * x;
    norm = std::sqrt(norm);
    std::fprintf(stderr,
        "[info] direction from %zu positive + %zu negative voices (dim=%zu)\n"
        "       ||direction||_2 = %.4f\n",
        cnt_pos, cnt_neg, dim_pos, norm);
    print_stats(std::string("wrote direction ") + out_path, direction);
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
    std::string topk_s    = grab_arg(argc, argv, "--top-k");
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
    // --mode voice_design: dedicated VoiceDesign variant only. The instruct
    // slot carries the natural-language voice description ("Deep male voice
    // with British accent"). No reference / voice / speaker-name needed.
    std::string mode      = grab_arg(argc, argv, "--mode");
    if (mode == "voice_design") {
        if (model_dir.empty() || text.empty() || instruct.empty()) {
            std::fprintf(stderr,
                "usage: qwen_voice apply --model-dir <voicedesign-dir> \\\n"
                "       --text \"...\" --instruct \"<voice description>\" \\\n"
                "       --mode voice_design [--out out.wav] [--temperature 0.8]\n"
                "  VoiceDesign: instruct is the natural-language voice description.\n");
            return 2;
        }
    } else if (model_dir.empty() || text.empty() ||
        (voice.empty() && ref_audio.empty() && spk_name.empty())) {
        std::fprintf(stderr,
            "usage: qwen_voice apply --model-dir <dir> --text \"...\" \\\n"
            "       (--voice <voice> | --reference-audio <wav> | --speaker-name <name>) \\\n"
            "       [--instruct \"...\"] [--out out.wav] \\\n"
            "       [--language en] [--speaker-name \"\"] [--temperature 0.7]\n"
            "       [--encoder <spk.gguf>]\n"
            "       [--reference-audio <wav>] [--reference-text \"transcript\"]\n"
            "       [--mode voice_design]  (requires VoiceDesign variant + --instruct)\n"
            "  x-vector mode:    --voice <file.voice>           (Base/Instruct variants)\n"
            "  named speaker:    --speaker-name Vivian          (CustomVoice variants)\n"
            "  ICL clone mode:   --reference-audio <wav> [--reference-text \"...\"]\n"
            "  both:             --voice + --reference-audio    (spk_emb + codec tokens)\n"
            "  voice_design:     --mode voice_design + --instruct \"description\"\n");
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
    // text_top_k controls the cb0 (coarse codec) sampler's top-k filter.
    // 0 disables top-k (sample from the full distribution after top-p).
    // The canonical Qwen3-TTS default is top_k=50 with temperature=0.8-1.0.
    req.text_top_k  = topk_s.empty() ? 50 : std::atoi(topk_s.c_str());
    req.max_new_tokens = maxnew_s.empty() ? 100 : std::atoi(maxnew_s.c_str());
    req.seed        = seed_s.empty() ? 0 : std::atoi(seed_s.c_str());
    req.reference_audio = ref_audio;
    req.reference_text  = ref_text;

    if (!instruct.empty())
        std::fprintf(stderr, "[info] instruct: \"%s\"\n", instruct.c_str());

    audiocore::TtsResponse resp;
    // Dispatch matrix:
    //   • --reference-audio         → ICL voice_clone via run_tts (codec encoder
    //                                 injects ref frames as in-context examples)
    //   • --voice                   → x-vector fast path via synthesize_with_embedding
    //   • --speaker-name only       → CustomVoice variant: token-based speaker
    //                                 resolution, no embedding needed. Drive via
    //                                 run_tts with empty speaker_embedding.
    // Combining --voice with --reference-audio gives spk_emb + codec tokens,
    // matching the Python voice_clone_prompt=(ref_code, ref_spk_embedding,
    // ref_text) API.
    bool ok = false;
    if (mode == "voice_design") {
        req.mode = "voice_design";
        std::fprintf(stderr, "[info] voice_design: \"%s\"\n", instruct.c_str());
        ok = q->run_tts(&req, &resp, &err);
        if (!ok)
            std::fprintf(stderr, "[err] run_tts (voice_design) failed: %s\n", err.c_str());
    } else if (!ref_audio.empty()) {
        req.mode = "voice_clone";
        if (!emb.empty()) req.speaker_embedding = emb;
        std::fprintf(stderr, "[info] ICL voice clone: ref_audio=%s ref_text=%s\n",
                     ref_audio.c_str(), ref_text.empty() ? "(none)" : "(provided)");
        ok = q->run_tts(&req, &resp, &err);
        if (!ok)
            std::fprintf(stderr, "[err] run_tts (ICL) failed: %s\n", err.c_str());
    } else if (!spk_name.empty() && emb.empty()) {
        // CustomVoice named-speaker path: no embedding needed.
        req.mode = "voice_clone";  // any mode works; speaker_name drives the slot
        std::fprintf(stderr, "[info] named speaker: %s (token resolved in session)\n",
                     spk_name.c_str());
        ok = q->run_tts(&req, &resp, &err);
        if (!ok)
            std::fprintf(stderr, "[err] run_tts (named speaker) failed: %s\n", err.c_str());
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
        "  shift <base.voice> <direction.dir> <scale> <out.voice>\n"
        "  slerp <a.voice> <b.voice> <t> <out.voice>\n"
        "        (Spherical Linear Interpolation; t∈[0,1]; smoother than mix)\n"
        "  discover_direction <positive_dir> <negative_dir> <out.direction>\n"
        "        (steering-vector from labeled groups; e.g. high/ vs low/)\n\n"
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
    else if (sub == "slerp")     return cmd_slerp(sub_argc, sub_argv);
    else if (sub == "discover_direction") return cmd_discover_direction(sub_argc, sub_argv);
    else if (sub == "encode")    return cmd_encode(argc, argv);
    else if (sub == "apply")     return cmd_apply(argc, argv);
    else if (sub == "-h" || sub == "--help" || sub == "help") { usage(); return 0; }

    std::fprintf(stderr, "[err] unknown subcommand '%s'\n", sub.c_str());
    usage();
    return 2;
}
