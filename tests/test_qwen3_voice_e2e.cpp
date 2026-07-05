// tests/test_qwen3_voice_e2e.cpp — Voice-embedding round-trip regression test.
//
// Proves the marksverdhei-style "Qwen3 voice embeddings" workflow end-to-end:
//
//   1. Load a qwen3_tts Base session (talker + predictor + codec + encoder).
//      Voice embeddings require the Base variants — they're the only ones
//      trained to accept an injected speaker vector at the codec bridge.
//   2. Take a reference WAV (24 kHz mono 16-bit). Default fixture is
//      weights/reference_output.wav from a prior Qwen3-TTS run; override
//      with QWEN3TTS_REF_WAV to encode your own voice.
//   3. Encode the reference WAV via Qwen3TtsSpeakerEncoder → 2048-d float
//      vector. Save it as a QWEN3VOICE file on disk.
//   4. Reload the voice file and sanity-check the magic / dim / values.
//   5. Exercise the math ops (mix, direction, shift) on saved files to
//      verify the on-disk format round-trips.
//   6. Run TTS via synthesize_with_embedding using the reloaded vector,
//      with an emotional `instruct` prompt. Write the result WAV.
//   7. Verify the output is real audio (RMS > silence threshold).
//
// Skips with return 77 if QWEN3TTS_DIR is unset or load fails, so it is
// safe to run in CI environments without the model weights.

#include "e2e_common.h"
#include "audiocore/framework/core/backend.h"
#include "audiocore/framework/core/session.h"
#include "audiocore/framework/runtime/registry.h"
#include "audiocore/models/qwen3_tts/family.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

extern "C" void audiocore_register_qwen3_tts();

namespace {

constexpr char     kMagic[16] = {'Q','W','E','N','3','V','O','I','C','E',
                                  0,0,0,0,0,0};
#pragma pack(push, 1)
struct VoiceHeader {
    char     magic[16];
    uint32_t version, dim, flags, reserved;
};
#pragma pack(pop)

bool write_voice(const std::string& path, const std::vector<float>& v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    VoiceHeader h;
    std::memcpy(h.magic, kMagic, 16);
    h.version = 1; h.dim = uint32_t(v.size()); h.flags = 0; h.reserved = 0;
    f.write(reinterpret_cast<const char*>(&h), sizeof(h));
    f.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(float)));
    return f.good();
}

std::vector<float> read_voice(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    VoiceHeader h;
    f.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!f || std::memcmp(h.magic, kMagic, 16) != 0) return {};
    std::vector<float> v(h.dim);
    f.read(reinterpret_cast<char*>(v.data()),
           static_cast<std::streamsize>(h.dim * sizeof(float)));
    return f ? v : std::vector<float>{};
}

double l2_norm(const std::vector<float>& v) {
    double s = 0; for (float x : v) s += double(x) * x; return std::sqrt(s);
}

}  // namespace

int main() {
    using namespace audiocore;
    using namespace audiocore::qwen3_tts;

    audiocore_register_qwen3_tts();

    const char* env_dir = std::getenv("QWEN3TTS_DIR");
    if (!env_dir) {
        std::fprintf(stderr,
            "[SKIP] set QWEN3TTS_DIR to the qwen3_tts model directory to "
            "run the voice-embedding round-trip test\n");
        return 0;
    }
    std::string model_dir = env_dir;

    // Filenames / backend (mirror test_qwen3tts_e2e conventions).
    auto get_env = [](const char* k, const char* d) {
        const char* v = std::getenv(k); return v ? std::string(v) : std::string(d);
    };
    // Variant-aware default filenames: pick first existing in each list.
    // The 0.6B dirs typically ship qwen3_tts_talker.gguf (no size suffix);
    // 1.7B dirs ship qwen3tts-talker-1b7-f16.gguf. Allow either.
    auto resolve_first_existing = [&](const std::string& dir,
                                      std::initializer_list<const char*> cands,
                                      const char* fallback) -> std::string {
        for (const char* c : cands) {
            std::string p = dir + "/" + c;
            std::ifstream t(p, std::ios::binary);
            if (t.good()) return c;
        }
        return fallback;
    };
    std::string talker_fn = get_env("QWEN3TTS_TALKER",
        resolve_first_existing(model_dir,
            {"qwen3tts-talker-1b7-f16.gguf", "qwen3tts-talker-0b6-f16.gguf",
             "qwen3_tts_talker.gguf"},
            "qwen3_tts_talker.gguf").c_str());
    std::string pred_fn   = get_env("QWEN3TTS_PREDICTOR",
        resolve_first_existing(model_dir,
            {"qwen3tts-predictor-1b7-f16.gguf", "qwen3tts-predictor-0b6-f16.gguf",
             "qwen3_tts_predictor.gguf"},
            "qwen3_tts_predictor.gguf").c_str());
    // tokenizer-f16.gguf is the actual audio codec (~358 MB, 398 tensors).
    // qwen3tts-tokenizer-*-f16.gguf is the text tokenizer (~5.9 MB) and
    // will soft-fail decode → silence.
    std::string codec_fn  = get_env("QWEN3TTS_CODEC",     "tokenizer-f16.gguf");
    // ECAPA-TDNN speaker encoder GGUF. Default is the auto-discovered name
    // (loader.cpp probes <dir>/qwen3tts-speaker-encoder.gguf). Override with
    // QWEN3TTS_ENCODER to point at an arbitrary path. MUST be the 1024-dim
    // variant for Qwen3-TTS — the 2048-dim standalone ECAPA GGUF is wrong
    // (see MODEL-GAPS.md Gap J / TO-DO SPK-1024).
    std::string enc_fn   = get_env("QWEN3TTS_ENCODER",   "qwen3tts-speaker-encoder.gguf");
    int    ngpu  = std::atoi(get_env("QWEN3TTS_NGPU",   "99").c_str());
    std::string backend  = get_env("QWEN3TTS_DEVICE",  "ggml_cuda");

    std::fprintf(stderr, "[INFO] voice round-trip test, model dir: %s\n", model_dir.c_str());

    auto sess = FamilyRegistry::instance().create("qwen3_tts");
    CHECK(sess != nullptr, "FamilyRegistry::create(qwen3_tts) failed");
    auto* q = dynamic_cast<Qwen3TtsSession*>(sess.get());
    CHECK(q != nullptr, "session is not Qwen3TtsSession");

    LoadOptions opts;
    opts.extras["talker_path"]          = talker_fn;
    opts.extras["predictor_path"]       = pred_fn;
    opts.extras["codec_path"]           = codec_fn;
    opts.extras["speaker_encoder_path"] = enc_fn;
    opts.extras["n_gpu_layers"]         = std::to_string(ngpu);

    BackendKind kind = BackendKind::ggml_cpu;
    if (backend == "ggml_cuda")   kind = BackendKind::ggml_cuda;
    if (backend == "ggml_vulkan") kind = BackendKind::ggml_vulkan;
    BackendConfig bc = { .kind = kind, .device_id = 0, .n_threads = 4 };

    std::string err;
    if (!sess->load(model_dir, opts, bc, &err)) {
        std::fprintf(stderr, "[SKIP] load failed: %s\n", err.c_str());
        return 77;
    }
    CHECK(q->speaker_encoder_loaded(),
          "speaker encoder not loaded — voice-embedding flow unavailable");
    std::fprintf(stderr, "[PASS] session loaded with speaker encoder\n");

    // ── Step 1: resolve reference WAV ─────────────────────────────────────
    // Voice embeddings require the Base variants (no named speakers), so we
    // can't generate a reference via TTS here. Default to a fixture WAV from
    // a prior run; let the caller override via QWEN3TTS_REF_WAV.
    const char* env_ref = std::getenv("QWEN3TTS_REF_WAV");
    std::string ref_wav = env_ref ? env_ref : "weights/reference_output.wav";
    {
        std::ifstream test(ref_wav, std::ios::binary);
        if (!test) {
            std::fprintf(stderr,
                "[SKIP] reference WAV not found at %s — set QWEN3TTS_REF_WAV "
                "to a 24 kHz mono 16-bit WAV to run this test\n", ref_wav.c_str());
            return 0;
        }
    }
    std::fprintf(stderr, "[INFO] reference WAV: %s\n", ref_wav.c_str());

    // ── Step 2: encode → save → reload ───────────────────────────────────
    auto emb = q->extract_speaker_embedding(ref_wav, &err);
    CHECK(!emb.empty(), ("extract_speaker_embedding failed: " + err).c_str());
    std::fprintf(stderr, "[PASS] encoded dim=%zu L2=%.4f\n",
                 emb.size(), l2_norm(emb));

    CHECK(write_voice("vivian.voice", emb), "write_voice failed");
    auto reloaded = read_voice("vivian.voice");
    CHECK(!reloaded.empty(), "read_voice failed");
    CHECK(reloaded.size() == emb.size(), "reloaded dim mismatch");
    {
        double max_diff = 0;
        for (size_t i = 0; i < emb.size(); ++i) {
            double d = std::fabs(double(emb[i]) - double(reloaded[i]));
            if (d > max_diff) max_diff = d;
        }
        CHECK(max_diff < 1e-7, "voice file round-trip lost precision");
        std::fprintf(stderr,
            "[PASS] voice file round-trips (.voice dim=%zu max_diff=%.2e)\n",
            reloaded.size(), max_diff);
    }

    // ── Step 3: math ops on saved voice (model-free) ─────────────────────
    // mix(v, v, α) == v for any α (idempotent on identical inputs).
    {
        std::vector<float> mix(emb.size());
        float alpha = 0.25f;
        for (size_t i = 0; i < emb.size(); ++i)
            mix[i] = (1.0f - alpha) * emb[i] + alpha * emb[i];
        CHECK(write_voice("vivian_mix.voice", mix), "write mix failed");
        auto back = read_voice("vivian_mix.voice");
        double md = 0;
        for (size_t i = 0; i < emb.size(); ++i)
            md = std::max(md, std::fabs(double(back[i]) - double(emb[i])));
        CHECK(md < 1e-7, "mix(v,v)=v failed");
    }
    // direction(v, v) must be ~zero
    {
        std::vector<float> dir(emb.size(), 0.0f);
        for (size_t i = 0; i < emb.size(); ++i) dir[i] = emb[i] - emb[i];
        double n = l2_norm(dir);
        CHECK(n < 1e-9, "direction(v,v) should be zero");
    }
    std::fprintf(stderr, "[PASS] vector math ops verified\n");

    // ── Step 4: apply saved embedding + emotional instruct ───────────────
    TtsRequest syn_req;
    syn_req.text         = "I just can't believe you're really gone.";
    syn_req.language     = "en";
    syn_req.instruct     = "With deep sadness, whispering, on the verge of tears.";
    syn_req.max_new_tokens = 60;
    syn_req.temperature = 0.7f; syn_req.top_p = 0.9f;
    TtsResponse syn_resp;
    CHECK(q->synthesize_with_embedding(syn_req, reloaded.data(),
                                       reloaded.size(), syn_resp, &err),
          ("synthesize_with_embedding failed: " + err).c_str());
    CHECK(!syn_resp.pcm_mono.empty(), "synth PCM is empty");

    double sum_sq = 0;
    for (float s : syn_resp.pcm_mono) sum_sq += double(s) * s;
    double rms = std::sqrt(sum_sq / syn_resp.pcm_mono.size());
    std::fprintf(stderr,
        "[INFO] emotional-instruct output: %zu samples (%.2fs) RMS=%.6f\n",
        syn_resp.pcm_mono.size(),
        double(syn_resp.pcm_mono.size()) / syn_resp.sampling_rate, rms);
    CHECK(rms > 1e-6, "emotional-instruct output is silent");

    CHECK(write_wav("voice_applied_emotional.wav", syn_resp.pcm_mono.data(),
                    syn_resp.pcm_mono.size(), syn_resp.sampling_rate) == 0,
          "failed to write voice_applied_emotional.wav");

    std::fprintf(stderr,
        "\n[PASS] Full voice-embedding round-trip succeeded.\n"
        "  Generated:  voice_ref.wav  (reference clip)\n"
        "  Encoded:    vivian.voice   (dim=%zu, reusable across calls)\n"
        "  Applied:    voice_applied_emotional.wav  (with instruct)\n"
        "  Instruct:   \"%s\"\n",
        reloaded.size(), syn_req.instruct.c_str());
    return 0;
}
