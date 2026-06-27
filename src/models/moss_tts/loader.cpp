// loader.cpp — MOSS-TTS weight loading + family registration.
//
// One GGUF file in → backbone (Qwen3-8B, delegated to libllama via
// qwen3::Runner) + extension tensors (moss.*, bound into our own
// ggml_context). The community MOSS GGUF ("moss-tts-q8_0.gguf") ships both
// in one file with all moss.* keys in the KV+tensor sections, so a single
// GgufReader pass is enough.
//
// DRY note: there is exactly ONE place in audiocore that loads Qwen3
// tensors (qwen3::Runner::load, which calls libllama). This file does not
// touch token_embd / blk.* / output.* at all — libllama owns them. We only
// bind the moss.* extensions that libllama doesn't know about.

#include "audiocore/models/moss_tts/family.h"

#include "ggml.h"

#include "audiocore/framework/io/gguf_reader.h"
#include "audiocore/framework/io/weight_loader.h"
#include "audiocore/framework/runtime/registry.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

namespace audiocore::moss {

bool MossSession::bind_extension_tensors(const GgufReader& r,
                                         std::string* err) {
    // First pass: count moss.* tensors and compute their total byte footprint.
    // We allocate one ggml_context that holds ONLY the tensor structs (the
    // weight bytes themselves stay in the GgufReader's mmap — zero-copy).
    size_t max_n_tensors = 0;
    for (const TensorStorage& t : r.tensors()) {
        if (t.name.rfind("moss.", 0) == 0) ++max_n_tensors;
    }
    if (max_n_tensors == 0) {
        if (err) *err = "no moss.* tensors found in GGUF — wrong file?";
        return false;
    }

    struct ggml_init_params gip {
        /*.mem_size   =*/ ggml_tensor_overhead() * (max_n_tensors + 8),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,   // weight bytes stay in the GgufReader mmap
    };
    ext_ctx_ = ggml_init(gip);
    if (!ext_ctx_) {
        if (err) *err = "ggml_init failed for moss.* context";
        return false;
    }
    owns_ext_ctx_ = true;

    // Second pass: create ggml_tensor views into the mmap'd GGUF data.
    // The view structs live in ext_ctx_; their ->data pointers alias the
    // mmap'd weight bytes via tensor_data_ptr().
    for (const TensorStorage& t : r.tensors()) {
        if (t.name.rfind("moss.", 0) != 0) continue;

        // ggml ne[] is row-major (ne[0] innermost); TensorStorage ne[] is
        // already in that order. ggml_new_tensor copies ne[0..n_dims-1].
        ggml_tensor* gt = ggml_new_tensor(ext_ctx_, t.type, t.n_dims, t.ne);
        if (!gt) {
            if (err) *err = "ggml_new_tensor failed for " + t.name;
            return false;
        }
        ggml_set_name(gt, t.name.c_str());

        // Point the tensor's data pointer into the GgufReader mmap. Stays
        // valid because loader_ (owned by Session base) keeps the
        // GgufReader — and its mmap — alive for the lifetime of MossSession.
        const void* data_ptr = r.tensor_data_ptr(t);
        if (!data_ptr) {
            if (err) *err = "tensor_data_ptr(" + t.name + ") returned null "
                            "(GgufReader has no mmap — Path 2 unsupported)";
            return false;
        }
        gt->data = const_cast<void*>(data_ptr);

        // Bind well-known tensors to slots on MossSession for fast access.
        // The codec tensors stay addressable by name via ggml_get_tensor()
        // for codec.cpp — we don't anchor every one here.
        //
        // Parse "{prefix}{digits}.weight" exactly — won't match ".bias" or
        // any other suffix, won't fall through to atoi on non-digits.
        constexpr static const char kEmbedPrefix[]  = "moss.audio_embed.";
        constexpr static const char kHeadPrefix[]   = "moss.audio_head.";
        constexpr static const char kWeightSuffix[] = ".weight";
        constexpr static size_t kEmbedLen  = sizeof(kEmbedPrefix)  - 1;
        constexpr static size_t kHeadLen   = sizeof(kHeadPrefix)   - 1;
        constexpr static size_t kSuffixLen = sizeof(kWeightSuffix) - 1;

        auto parse_indexed = [&](const char* prefix, size_t plen,
                                 int max_n) -> int {
            const size_t nlen = t.name.size();
            if (nlen <= plen + kSuffixLen) return -1;
            if (t.name.compare(0, plen, prefix) != 0) return -1;
            if (t.name.compare(nlen - kSuffixLen, kSuffixLen, kWeightSuffix) != 0)
                return -1;
            for (size_t i = plen; i < nlen - kSuffixLen; ++i) {
                if (t.name[i] < '0' || t.name[i] > '9') return -1;
            }
            const int idx = std::atoi(t.name.c_str() + plen);
            return (idx >= 0 && idx < max_n) ? idx : -1;
        };

        int idx = -1;
        if ((idx = parse_indexed(kEmbedPrefix, kEmbedLen, 32)) >= 0) {
            audio_embed_[idx] = gt;
        } else if ((idx = parse_indexed(kHeadPrefix, kHeadLen, 32)) >= 0) {
            audio_head_[idx] = gt;
        } else if (t.name == "moss.codec.dec.0.weight" ||
                   t.name == "moss.codec.decoder.0.weight") {
            codec_dec_root_ = gt;   // anchor; full codec wiring in codec.cpp
        }
    }

    // Sanity: we need at least n_vq embeds + heads.
    for (int i = 0; i < cfg_.n_vq; ++i) {
        if (!audio_embed_[i]) {
            if (err) *err = "missing moss.audio_embed." + std::to_string(i) + ".weight";
            return false;
        }
        if (!audio_head_[i]) {
            if (err) *err = "missing moss.audio_head." + std::to_string(i) + ".weight";
            return false;
        }
    }
    return true;
}

bool MossSession::load(const std::string& model_path,
                       const LoadOptions& opts,
                       const BackendConfig& backend_cfg,
                       std::string* error) {
    // (1) Open the GGUF via the format-neutral factory. Dispatches by file
    //     magic; returns a GgufReader for .gguf paths.
    loader_ = make_weight_loader(model_path, error);
    if (!loader_) return false;
    auto* gguf = dynamic_cast<GgufReader*>(loader_.get());
    if (!gguf) {
        if (error) *error = "moss_tts requires a .gguf file";
        return false;
    }

    // (2) Parse MOSS KV metadata. Required keys first; missing → fail loudly.
    if (!gguf->get_kv_i32("moss.n_vq",             &cfg_.n_vq)             ||
        !gguf->get_kv_i32("moss.audio_vocab_size", &cfg_.audio_vocab_size) ||
        !gguf->get_kv_i32("moss.sampling_rate",    &cfg_.sampling_rate)) {
        if (error) *error = "missing required moss.* KV metadata";
        return false;
    }
    // Optional keys — defaults baked into MossConfig struct.
    int32_t tmp = 0;
    if (gguf->get_kv_i32("moss.audio_pad_code",  &tmp)) cfg_.audio_pad_code  = tmp;
    if (gguf->get_kv_i32("moss.downsample_rate", &tmp)) cfg_.downsample_rate = tmp;
    if (gguf->get_kv_i32("moss.token.audio_start",      &tmp)) cfg_.tok_audio_start = tmp;
    if (gguf->get_kv_i32("moss.token.audio_end",        &tmp)) cfg_.tok_audio_end   = tmp;
    if (gguf->get_kv_i32("moss.token.user_slot",        &tmp)) cfg_.tok_user_slot   = tmp;
    if (gguf->get_kv_i32("moss.token.audio_gen_slot",   &tmp)) cfg_.tok_audio_gen   = tmp;
    if (gguf->get_kv_i32("moss.token.audio_delay_slot", &tmp)) cfg_.tok_audio_delay = tmp;
    if (gguf->get_kv_i32("moss.token.im_start",         &tmp)) cfg_.tok_im_start    = tmp;
    if (gguf->get_kv_i32("moss.token.im_end",           &tmp)) cfg_.tok_im_end      = tmp;
    if (gguf->get_kv_i32("moss.token.pad",              &tmp)) cfg_.tok_pad         = tmp;
    if (gguf->get_kv_i32("moss.codec.present",          &tmp)) cfg_.codec_present   = (tmp != 0);

    // (3) Bind moss.* extension tensors into our ggml_context.
    if (!bind_extension_tensors(*gguf, error)) return false;

    // (4) Spin up the Qwen3 backbone via the unified runner. libllama reads
    //     its own tensors (token_embd.*, blk.*, output.*) from the same file
    //     — we just hand it the path. There is no other Qwen3 path in audiocore.
    qwen3::RunnerConfig rc;
    rc.n_ctx        = 8192;
    rc.n_threads    = backend_cfg.n_threads;
    rc.n_gpu_layers = (backend_cfg.kind == BackendKind::ggml_cuda) ? -1 : 0;
    rc.flash_attn   = true;
    backbone_ = qwen3::Runner::load(model_path, rc, error);
    if (!backbone_) return false;

    loaded_ = true;
    (void)opts;   // voice_path / language used at run_tts time
    return true;
}

// ---------------------------------------------------------------------------
// Factory registered with FamilyRegistry at static-init time. The server's
// FamilyRegistry::create("moss_tts") call returns a fresh MossSession.
// ---------------------------------------------------------------------------

namespace {
std::unique_ptr<Session> make_moss_session() {
    return std::unique_ptr<Session>(new MossSession());
}
}  // namespace

AUDIOCORE_REGISTER_FAMILY(moss_tts, make_moss_session)

// Explicit registration anchor called from main(). Defeats the linker
// stripping this TU out of static archives — without it, the registrar above
// gets dropped and FamilyRegistry::list() comes back empty at runtime.
// (Static initializers in archive members are only retained if at least one
// symbol from the same TU is reachable from a live link unit.)
extern "C" void audiocore_register_moss_tts() {
    static bool done = false;
    if (!done) {
        FamilyRegistry::instance().register_family("moss_tts", make_moss_session);
        done = true;
    }
}

}  // namespace audiocore::moss
