// loader.cpp — MOSS-TTS weight loading + family registration.
//
// Two loading paths:
//   Path A (GGUF-embedded):  moss.* tensors live inside the same GGUF as the
//                             Qwen3 backbone. The full_community.gguf from
//                             pwilkin/openmoss works this way.
//   Path B (npy-separate):    The GGUF has only the backbone; audio embeddings
//                             and heads come from .npy files in embeddings/ and
//                             lm_heads/ subdirectories. This is what the
//                             OpenMOSS-Team/MOSS-TTS-GGUF repo ships.
//
// In both paths the Qwen3 backbone is always loaded via llama.cpp (qwen3::Runner).

#include "audiocore/models/moss_tts/family.h"

#include "ggml.h"
#include "ggml-backend.h"

#include "audiocore/framework/io/gguf_reader.h"
#include "audiocore/framework/io/weight_loader.h"
#include "audiocore/framework/runtime/registry.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace audiocore::moss {

// ===========================================================================
// .npy header parser (minimal — float16 only)
// ===========================================================================
// Parses enough of the NPY format to get shape + data-offset.
// Returns data_offset (after header), or 0 on error.
static int64_t parse_npy_header(const std::string& path,
                                 int64_t& n_rows, int64_t& n_cols,
                                 std::string* error) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        if (error) *error = "cannot open " + path;
        return 0;
    }
    const int64_t file_size = static_cast<int64_t>(f.tellg());
    f.seekg(0);

    // Read magic + version
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        if (error) *error = path + ": not a .npy file";
        return 0;
    }
    const int ver_major = magic[6];
    const int ver_minor = magic[7];

    // Read header length
    int64_t header_len = 0;
    if (ver_major == 1) {
        uint16_t hlen = 0;
        f.read(reinterpret_cast<char*>(&hlen), 2);
        header_len = hlen;
    } else if (ver_major == 2 || ver_major == 3) {
        uint32_t hlen = 0;
        f.read(reinterpret_cast<char*>(&hlen), 4);
        header_len = hlen;
    } else {
        if (error) *error = path + ": unsupported NPY version " +
                             std::to_string(ver_major) + "." +
                             std::to_string(ver_minor);
        return 0;
    }

    // Read header dict as string
    std::string header(static_cast<size_t>(header_len), '\0');
    f.read(header.data(), header_len);
    if (!f) {
        if (error) *error = path + ": truncated header";
        return 0;
    }

    const int64_t data_offset = static_cast<int64_t>(f.tellg());

    // Minimal shape parser: find "shape": (...)
    auto shape_pos = header.find("shape");
    if (shape_pos == std::string::npos) {
        if (error) *error = path + ": no shape in npy header";
        return 0;
    }
    auto paren_open = header.find('(', shape_pos);
    auto paren_close = header.find(')', paren_open);
    if (paren_open == std::string::npos || paren_close == std::string::npos) {
        if (error) *error = path + ": malformed shape in npy header";
        return 0;
    }

    std::string shape_str = header.substr(paren_open + 1,
                                           paren_close - paren_open - 1);
    // strip spaces
    shape_str.erase(std::remove_if(shape_str.begin(), shape_str.end(),
                                    [](char c) { return std::isspace(c); }),
                    shape_str.end());

    std::vector<int64_t> dims;
    size_t start = 0;
    while (start < shape_str.size()) {
        auto comma = shape_str.find(',', start);
        std::string tok;
        if (comma == std::string::npos) {
            tok = shape_str.substr(start);
            start = shape_str.size();
        } else {
            tok = shape_str.substr(start, comma - start);
            start = comma + 1;
        }
        if (tok.empty() && dims.size() >= 1) break;  // trailing comma
        if (!tok.empty()) {
            dims.push_back(std::stoll(tok));
        }
    }

    if (dims.empty()) {
        if (error) *error = path + ": empty shape in npy header";
        return 0;
    }

    // Verify the dtype is float16 (<f2 or |f2)
    auto descr_key = header.find("descr");
    bool is_f16 = false;
    if (descr_key != std::string::npos) {
        // Find the colon after "descr", then the first quote after the colon.
        auto colon = header.find(':', descr_key);
        auto q1 = header.find('\'', colon);
        auto q2 = header.find('\'', q1 + 1);
        if (q1 != std::string::npos && q2 != std::string::npos) {
            std::string descr = header.substr(q1 + 1, q2 - q1 - 1);
            is_f16 = (descr == "<f2" || descr == "|f2");
        }
    }
    if (!is_f16) {
        if (error) *error = path + ": expected float16 npy, got non-f16 descr";
        return 0;
    }

    if (dims.size() == 1) {
        n_rows = dims[0];
        n_cols = 1;
    } else if (dims.size() == 2) {
        n_rows = dims[0];
        n_cols = dims[1];
    } else {
        if (error) *error = path + ": unexpected ndim=" + std::to_string(dims.size());
        return 0;
    }

    // Verify the data fits.
    const int64_t expected_bytes = n_rows * n_cols * 2;  // float16 = 2 bytes
    if (data_offset + expected_bytes > file_size) {
        if (error) *error = path + ": file truncated (expected " +
                             std::to_string(expected_bytes) + " data bytes)";
        return 0;
    }
    return data_offset;
}

// Load a .npy file into a ggml float16 tensor. Returns the tensor and keeps
// the raw data alive in `buf` (owned by the caller).
static ggml_tensor* load_npy_tensor(const std::string& path,
                                     ggml_context* ctx,
                                     std::vector<uint8_t>* buf,
                                     int32_t n_rows, int32_t n_cols,
                                     int64_t data_offset,
                                     std::string* error) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        if (error) *error = "cannot open " + path + " for reading";
        return nullptr;
    }

    // Read the raw float16 data
    const size_t n_bytes = static_cast<size_t>(n_rows * n_cols) * 2;
    buf->resize(n_bytes);
    f.seekg(data_offset);
    f.read(reinterpret_cast<char*>(buf->data()),
           static_cast<std::streamsize>(n_bytes));
    if (!f) {
        if (error) *error = path + ": short read (" +
                             std::to_string(f.gcount()) + " vs " +
                             std::to_string(n_bytes) + ")";
        buf->clear();
        return nullptr;
    }

    // Create ggml tensor view. ne[0] = cols (innermost), ne[1] = rows.
    int64_t ne[2] = {n_cols, n_rows};
    ggml_tensor* t = ggml_new_tensor(ctx, GGML_TYPE_F16, 2, ne);
    if (!t) {
        if (error) *error = path + ": ggml_new_tensor failed";
        buf->clear();
        return nullptr;
    }
    t->data = buf->data();
    ggml_set_name(t, path.c_str());
    return t;
}

// ===========================================================================
// Path B: load from .npy files
// ===========================================================================
bool MossSession::load_npy_extras(const std::string& emb_dir,
                                   const std::string& lm_dir,
                                   std::string* error) {
    const int32_t n_vq = cfg_.n_vq;
    const int32_t hs = static_cast<int32_t>(backbone_->hidden_size());

    // We'll create ggml tensors in the ext_ctx_. Estimate overhead.
    // 1 embed_tokens + 32 emb_ext + 32 lm_head = 65 tensors.
    const size_t n_tensors = 1 + static_cast<size_t>(n_vq) * 2;
    const size_t overhead = ggml_tensor_overhead() * (n_tensors + 8);
    if (!ext_ctx_) {
        struct ggml_init_params gip = {
            /*.mem_size   =*/ overhead,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,   // data lives in npy_buffers_
        };
        ext_ctx_ = ggml_init(gip);
        if (!ext_ctx_) {
            if (error) *error = "ggml_init failed for npy tensor context";
            return false;
        }
        owns_ext_ctx_ = true;
    }

    // (1) embed_tokens.npy
    std::string et_path = emb_dir + "/embed_tokens.npy";
    int64_t et_rows = 0, et_cols = 0;
    int64_t et_off = parse_npy_header(et_path, et_rows, et_cols, error);
    if (!et_off) return false;
    if (et_cols != hs) {
        if (error) *error = "embed_tokens.npy: cols=" + std::to_string(et_cols) +
                             " but hidden_size=" + std::to_string(hs);
        return false;
    }
    npy_buffers_.emplace_back();
    token_embd_ = load_npy_tensor(et_path, ext_ctx_, &npy_buffers_.back(),
                                   static_cast<int32_t>(et_rows),
                                   static_cast<int32_t>(et_cols),
                                   et_off, error);
    if (!token_embd_) return false;
    ggml_set_name(token_embd_, "token_embd.weight");

    // (2) emb_ext_*.npy (audio embedding tables)
    for (int i = 0; i < n_vq; ++i) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%s/emb_ext_%02d.npy",
                      emb_dir.c_str(), i);
        std::string epath(buf);
        int64_t erows = 0, ecols = 0;
        int64_t eoff = parse_npy_header(epath, erows, ecols, error);
        if (!eoff) return false;
        if (ecols != hs) {
            if (error) *error = epath + ": cols=" + std::to_string(ecols) +
                                 " but hidden_size=" + std::to_string(hs);
            return false;
        }
        npy_buffers_.emplace_back();
        auto* t = load_npy_tensor(epath, ext_ctx_, &npy_buffers_.back(),
                                   static_cast<int32_t>(erows),
                                   static_cast<int32_t>(ecols),
                                   eoff, error);
        if (!t) return false;
        audio_embed_[i] = t;
    }

    // (3) lm_head_audio_*.npy
    for (int i = 0; i < n_vq; ++i) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%s/lm_head_audio_%02d.npy",
                      lm_dir.c_str(), i);
        std::string hpath(buf);
        int64_t hrows = 0, hcols = 0;
        int64_t hoff = parse_npy_header(hpath, hrows, hcols, error);
        if (!hoff) return false;
        if (hcols != hs) {
            if (error) *error = hpath + ": cols=" + std::to_string(hcols) +
                                 " but hidden_size=" + std::to_string(hs);
            return false;
        }
        npy_buffers_.emplace_back();
        auto* t = load_npy_tensor(hpath, ext_ctx_, &npy_buffers_.back(),
                                   static_cast<int32_t>(hrows),
                                   static_cast<int32_t>(hcols),
                                   hoff, error);
        if (!t) return false;
        audio_head_[i] = t;
    }

    return true;
}

// ===========================================================================
// Path A: bind moss.* tensors from GGUF
// ===========================================================================
bool MossSession::bind_extension_tensors(const GgufReader& r,
                                         std::string* err) {
    // First pass: count the tensors we'll bind. We bind everything prefixed
    // `moss.` (codec, audio heads/embeds) PLUS the Qwen3 `token_embd.weight`
    // (so we can do raw embedding gathers without a forward pass — libllama
    // has its own copy internally for inference; ours is read-only).
    auto wants = [](const std::string& n) {
        return n.rfind("moss.", 0) == 0 || n == "token_embd.weight";
    };
    size_t max_n_tensors = 0;
    for (const TensorStorage& t : r.tensors()) {
        if (wants(t.name)) ++max_n_tensors;
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
    for (const TensorStorage& t : r.tensors()) {
        if (!wants(t.name)) continue;

        ggml_tensor* gt = ggml_new_tensor(ext_ctx_, t.type, t.n_dims, t.ne);
        if (!gt) {
            if (err) *err = "ggml_new_tensor failed for " + t.name;
            return false;
        }

        const void* data_ptr = r.tensor_data_ptr(t);
        if (data_ptr) {
            gt->data = const_cast<void*>(data_ptr);
        } else {
            // tensor_data_ptr may return null when the GGUF's data offsets
            // don't map into the mmap'd region (e.g. extras GGUFs with
            // incorrect alignment). Fall back to materialize() into a buffer.
            gguf_buffers_.emplace_back(t.nbytes_to_read());
            if (!r.materialize(t, gguf_buffers_.back().data(), err)) {
                // Individual tensor materialization failure is not fatal.
                // Some extras GGUFs have corrupted offset metadata for
                // tensors we don't strictly need (e.g., the codec encoder
                // which is not ported). Log a warning and skip the tensor.
                gguf_buffers_.pop_back();
                // Tensor stays allocated in ext_ctx_ but won't be looked up
                // by name since skip_unreadable_ will remain unset.
                std::fprintf(stderr, "moss_tts: warning: skipping unreadable "
                                     "tensor \"%s\" (%s)\n",
                             t.name.c_str(), err->c_str());
                err->clear();
                continue;
            }
            gt->data = gguf_buffers_.back().data();
        }

        // Only name the tensor after successfully assigning data. Skipped
        // tensors (materialize failure) remain nameless so ggml_get_tensor
        // won't find them, preventing downstream crashes.
        ggml_set_name(gt, t.name.c_str());

        // Bind well-known tensors to slots for fast access.
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
            codec_dec_root_ = gt;
        } else if (t.name == "token_embd.weight") {
            token_embd_ = gt;
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

// ===========================================================================
// Main load()
// ===========================================================================
bool MossSession::load(const std::string& model_path,
                       const LoadOptions& opts,
                       const BackendConfig& backend_cfg,
                       std::string* error) {
    // Resolve model path: if it's a directory, find .gguf files inside.
    // The per-family loader is responsible for file discovery within its
    // own directory (see discovery.cpp docstring).
    std::string resolved_path = model_path;
    LoadOptions resolved_opts = opts;
    {
        std::error_code ec;
        namespace fs = std::filesystem;
        if (fs::is_directory(model_path, ec)) {
            std::string primary, extras;
            for (const auto& entry : fs::directory_iterator(model_path, ec)) {
                const auto& p = entry.path();
                if (p.extension() != ".gguf") continue;
                const std::string fn = p.filename().string();
                if (fn.find("extras") != std::string::npos) {
                    if (extras.empty()) extras = p.string();
                } else if (primary.empty()) {
                    primary = p.string();
                }
            }
            if (!primary.empty()) {
                resolved_path = primary;
            }
            if (!extras.empty()) {
                resolved_opts.extras["extras_gguf_path"] = extras;
            }
        }
    }

    // (1) Open the GGUF via the format-neutral factory.
    loader_ = make_weight_loader(resolved_path, error);
    if (!loader_) return false;
    auto* gguf = dynamic_cast<GgufReader*>(loader_.get());
    if (!gguf) {
        if (error) *error = "moss_tts requires a .gguf file";
        return false;
    }

    // (2) Parse MOSS KV metadata. All optional — defaults baked into MossConfig
    //     headers (see family.h). Community GGUFs (backbone-only) omit these;
    //     the extras sidecar GGUF carries them.
    auto parse_moss_metadata = [&](const GgufReader* g) {
        if (!g) return;
        int32_t tmp = 0;
        if (g->get_kv_i32("moss.n_vq",             &tmp)) cfg_.n_vq             = tmp;
        if (g->get_kv_i32("moss.audio_vocab_size", &tmp)) cfg_.audio_vocab_size = tmp;
        if (g->get_kv_i32("moss.sampling_rate",    &tmp)) cfg_.sampling_rate    = tmp;
        if (g->get_kv_i32("moss.audio_pad_code",   &tmp)) cfg_.audio_pad_code   = tmp;
        if (g->get_kv_i32("moss.downsample_rate",  &tmp)) cfg_.downsample_rate  = tmp;
        {   float ftmp = 0.0f;
            if (g->get_kv_f32("moss.frame_rate", &ftmp)) cfg_.frame_rate = ftmp; }
        if (g->get_kv_i32("moss.token.audio_start",      &tmp)) cfg_.tok_audio_start = tmp;
        if (g->get_kv_i32("moss.token.audio_end",        &tmp)) cfg_.tok_audio_end   = tmp;
        if (g->get_kv_i32("moss.token.user_slot",        &tmp)) cfg_.tok_user_slot   = tmp;
        if (g->get_kv_i32("moss.token.audio_gen_slot",   &tmp)) cfg_.tok_audio_gen   = tmp;
        if (g->get_kv_i32("moss.token.audio_delay_slot", &tmp)) cfg_.tok_audio_delay = tmp;
        if (g->get_kv_i32("moss.token.im_start",         &tmp)) cfg_.tok_im_start    = tmp;
        if (g->get_kv_i32("moss.token.im_end",           &tmp)) cfg_.tok_im_end      = tmp;
        if (g->get_kv_i32("moss.token.pad",              &tmp)) cfg_.tok_pad         = tmp;
        if (g->get_kv_i32("moss.codec.present",          &tmp)) cfg_.codec_present   = (tmp != 0);
        if (g->get_kv_i32("moss.n_quantized_embd",       &tmp)) cfg_.n_quantized_embd = tmp;
    };
    parse_moss_metadata(gguf);

    // (3) Spin up the Qwen3 backbone via the unified runner.
    qwen3::RunnerConfig rc;
    rc.n_ctx        = 8192;
    rc.n_threads    = backend_cfg.n_threads;
    rc.n_gpu_layers = (backend_cfg.kind == BackendKind::ggml_cuda) ? -1 : 0;
    rc.flash_attn   = true;
    backbone_ = qwen3::Runner::load(resolved_path, rc, error);
    if (!backbone_) return false;

    // (3b) Construct the same ggml Backend the Qwen3 backbone chose, so the
    //      Stage 16 codec port can submit graphs through the same device.
    //      Soft-fail: if backend construction fails, the codec just stays
    //      unbound and decode_codec() falls back to silence (GAPS.md §1.3).
    backend_ = make_backend(backend_cfg, nullptr);
    // (Silently tolerating a null backend here is deliberate — every path
    // other than the codec still runs through libllama, which carries its
    // own backend selection.)

    // (4) Load extension tensors (audio embeds + heads + token_embd).
    //     Try Path A (GGUF-embedded moss.*) first; if none found, try Path B
    //     (npy files from embeddings_dir / lm_heads_dir in extras).
    bool has_moss_tensors = false;
    for (const auto& t : gguf->tensors()) {
        if (t.name.rfind("moss.", 0) == 0) { has_moss_tensors = true; break; }
    }

    if (has_moss_tensors) {
        if (!bind_extension_tensors(*gguf, error)) return false;
    } else {
        // Path C: separate extras GGUF (e.g. moss-tts-v1.5-q8_0.extras.gguf).
        auto it_extras_gguf = resolved_opts.extras.find("extras_gguf_path");
        if (it_extras_gguf != resolved_opts.extras.end()) {
            auto extras_loader = make_weight_loader(it_extras_gguf->second, error);
            if (!extras_loader) return false;
            auto* extras_gguf = dynamic_cast<GgufReader*>(extras_loader.get());
            if (!extras_gguf) {
                if (error) *error = "extras_gguf_path must be a .gguf file";
                return false;
            }
            bool extras_has_moss = false;
            for (const auto& t : extras_gguf->tensors()) {
                if (t.name.rfind("moss.", 0) == 0) { extras_has_moss = true; break; }
            }
            if (!extras_has_moss) {
                if (error) *error = "extras GGUF \"" + it_extras_gguf->second +
                                    "\" has no moss.* tensors";
                return false;
            }
            parse_moss_metadata(extras_gguf);
            extra_loaders_.push_back(std::move(extras_loader));
            if (!bind_extension_tensors(*extras_gguf, error)) return false;
        } else {
            // Path B: .npy files from embeddings_dir / lm_heads_dir.
            auto it_emb = resolved_opts.extras.find("embeddings_dir");
            auto it_lm  = resolved_opts.extras.find("lm_heads_dir");
            if (it_emb == resolved_opts.extras.end() || it_lm == resolved_opts.extras.end()) {
                if (error) *error = "GGUF has no moss.* tensors — pass extras_gguf_path,"
                                    " embeddings_dir, or lm_heads_dir in LoadOptions::extras";
                return false;
            }
            if (!load_npy_extras(it_emb->second, it_lm->second, error))
                return false;
        }
    }

    // (4b) Bind the codec graphs (Stage 16) if the GGUF carries the
    //      moss.codec.* tensors. Soft-fail: a GGUF without the codec (the
    //      common community backbone-only pattern) skips this and the
    //      decode_codec() path falls back to silence. Detection is by
    //      checking for the first codebook tensor, which every codec-bearing
    //      sidecar must have (openmoss codec.cpp resolve_decoder_).
    cfg_.codec_present = false;
    ggml_backend_t be = backend_ ? backend_->raw_ggml_backend() : nullptr;
    if (be && (codec_dec_root_ ||
               ggml_get_tensor(ext_ctx_, "moss.codec.quantizer.q.0.codebook.weight"))) {
        std::string codec_err;
        if (codec_graphs_.bind(ext_ctx_, be, &codec_err)) {
            cfg_.codec_present = true;
        } else {
            std::fprintf(stderr, "moss_tts: codec tensors present but bind failed "
                                 "(%s)\n", codec_err.c_str());
        }
    }

    // (4c) If codec still not bound, try a fallback GGUF with intact tensors.
    //      The v1.5 extras GGUF (moss-tts-v1.5-q8_0.extras.gguf) is often
    //      truncated, but the v1.4 moss-tts.extras.gguf (4 GB) in the same
    //      directory has complete codec data. The codec architecture is shared
    //      across all MOSS-TTS variants so this is safe.
    if (!cfg_.codec_present && be) {
        std::error_code ec;
        namespace fs = std::filesystem;
        fs::path fallback_path = fs::path(resolved_path).parent_path() / "moss-tts.extras.gguf";
        if (fs::exists(fallback_path, ec)) {
            std::fprintf(stderr, "moss_tts: trying codec fallback %s\n",
                         fallback_path.c_str());
            auto fallback_loader = make_weight_loader(fallback_path.string(), error);
            if (fallback_loader) {
                auto* fallback_gguf = dynamic_cast<GgufReader*>(fallback_loader.get());
                if (fallback_gguf && fallback_gguf->meta_ctx() &&
                    ggml_get_tensor(fallback_gguf->meta_ctx(),
                        "moss.codec.quantizer.q.0.codebook.weight")) {
                    std::string fb_err;
                    if (codec_graphs_.bind(fallback_gguf->meta_ctx(), be, &fb_err)) {
                        cfg_.codec_present = true;
                        extra_loaders_.push_back(std::move(fallback_loader));
                        std::fprintf(stderr, "moss_tts: codec fallback OK\n");
                    } else {
                        std::fprintf(stderr, "moss_tts: codec fallback also failed (%s)\n",
                                     fb_err.c_str());
                    }
                }
            }
        }
        if (!cfg_.codec_present) {
            std::fprintf(stderr, "moss_tts: no codec tensors bound — "
                                 "decode_codec() will return silence\n");
        }
    }

    // (5) Stash codec-related extras for the future ggml codec path. The
    //     old ONNX decoder / encoder paths are no longer honored; codec
    //     decoding will read moss.codec.dec.* tensors from the GGUF directly.
    (void)opts;
    (void)resolved_opts;

    loaded_ = true;
    return true;
}

// ===========================================================================
// Factory
// ===========================================================================
namespace {
std::unique_ptr<Session> make_moss_session() {
    return std::unique_ptr<Session>(new MossSession());
}
}  // namespace

AUDIOCORE_REGISTER_FAMILY(moss_tts, make_moss_session)
AUDIOCORE_EXTERN_C_GUARD(moss_tts, make_moss_session)

}  // namespace audiocore::moss
