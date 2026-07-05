// convert_qwen3tts.cpp — Convert Qwen3-TTS HuggingFace safetensors to GGUF.
//
// Reads a HF Qwen3-TTS model directory (config.json + *.safetensors) and
// writes two GGUFs: a talker GGUF and a predictor GGUF, with tensor names
// matching what src/models/qwen3_tts/ expects.
//
// Usage:
//   convert_qwen3tts /path/to/model --outdir ./models
//   convert_qwen3tts /path/to/model --outdir ./models --quant f16
//   convert_qwen3tts /path/to/model --outdir ./models --skip-predictor

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "ggml.h"
#include "gguf.h"
#include "nlohmann/json.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

// ── Helpers ────────────────────────────────────────────────────────────────

[[noreturn]] static void die(const char* msg) {
    std::fprintf(stderr, "error: %s\n", msg);
    std::exit(1);
}

template <typename... Args>
static void die_f(const char* fmt, const Args&... args) {
    std::fprintf(stderr, "error: ");
    std::fprintf(stderr, fmt, args...);
    std::fprintf(stderr, "\n");
    std::exit(1);
}

static json load_json(const fs::path& path) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) die_f("cannot open %s", path.c_str());
    std::fseek(fp, 0, SEEK_END);
    long len = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    std::string buf(static_cast<size_t>(len), '\0');
    std::fread(buf.data(), 1, static_cast<size_t>(len), fp);
    std::fclose(fp);
    return json::parse(buf);
}

// ── Safetensors reader (mmap-based, no RAM copy of weights) ──────────────

struct SSTensor {
    std::string  name;
    std::string  dtype;     // "F32", "F16", "BF16"
    std::vector<int64_t> shape;
    size_t       file_idx;  // index into TensorBag::files
    size_t       offset;    // byte offset within the file
    size_t       nbytes;
};

// Memory-mapped file handle (movable, not copyable)
struct MappedFile {
    char*  data = nullptr;
    size_t size = 0;
    int    fd   = -1;

    MappedFile() = default;
    ~MappedFile() { unmap(); }
    MappedFile(MappedFile&& o) noexcept : data(o.data), size(o.size), fd(o.fd) { o.data = nullptr; o.size = 0; o.fd = -1; }
    MappedFile& operator=(MappedFile&& o) noexcept {
        if (this != &o) { unmap(); data = o.data; size = o.size; fd = o.fd; o.data = nullptr; o.size = 0; o.fd = -1; }
        return *this;
    }
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    void unmap() {
        if (data && data != MAP_FAILED) munmap(data, size);
        if (fd >= 0) close(fd);
        data = nullptr; size = 0; fd = -1;
    }
};

// Load all tensors from a directory of .safetensors files.
struct TensorBag {
    std::vector<MappedFile> files;   // one mmap per safetensors file
    std::vector<SSTensor> tensors;

    // Owned conversion buffers — kept alive so ggml tensor data pointers
    // remain valid until the GGUF file is written.  Used when the source
    // dtype (e.g. BF16) differs from the desired output type (e.g. F32).
    // Mutable: bf16_to_f32 is logically const (lazy cache, doesn't change
    // the bag's observable state).
    mutable std::vector<std::vector<uint8_t>> converted_buffers;

    // Look up tensor by name.
    const SSTensor* find(const std::string& name) const {
        for (auto& t : tensors)
            if (t.name == name) return &t;
        return nullptr;
    }

    // Convenience: try name and alt_name; returns first found.
    const SSTensor* find2(const std::string& name, const std::string& alt) const {
        auto* p = find(name);
        return p ? p : find(alt);
    }

    // Return a pointer to the tensor's data within the correct mmap region.
    const void* data_ptr(const SSTensor& t) const {
        if (t.file_idx >= files.size()) return nullptr;
        auto& f = files[t.file_idx];
        if (t.offset + t.nbytes > f.size) return nullptr;
        return f.data + t.offset;
    }

    // Convert BF16 source data → F32 buffer.  Returns pointer owned by
    // *this (valid until bag destruction).  Needed because the safetensors
    // stores all weights in BF16 but many 1-D tensors (norms, biases) are
    // written to the GGUF as F32 — without conversion the data pointer
    // would be half the expected size and gguf would read past it.
    const void* bf16_to_f32(const SSTensor& st) const {
        const uint16_t* src = static_cast<const uint16_t*>(data_ptr(st));
        size_t n = st.nbytes / 2;  // BF16 = 2 bytes each
        auto& buf = converted_buffers.emplace_back(n * 4);
        // Use memcpy to avoid aliasing / strict-aliasing UB.
        for (size_t i = 0; i < n; i++) {
            uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
            memcpy(&buf[i * 4], &bits, 4);
        }
        return buf.data();
    }
};

static TensorBag load_safetensors_dir(const fs::path& dir, std::string* err) {
    TensorBag bag;
    std::vector<fs::path> st_files;
    for (auto& e : fs::directory_iterator(dir)) {
        if (e.path().extension() == ".safetensors")
            st_files.push_back(e.path());
    }
    std::sort(st_files.begin(), st_files.end());
    if (st_files.empty()) {
        *err = "no .safetensors files found";
        return {};
    }

    for (auto& p : st_files) {
        // Read only the JSON header (tiny, <100KB) — never load weights into RAM.
        FILE* fp = std::fopen(p.c_str(), "rb");
        if (!fp) { *err = "cannot open"; return {}; }
        uint64_t header_len = 0;
        if (std::fread(&header_len, sizeof(header_len), 1, fp) != 1) { *err = "short read on header length"; fclose(fp); return {}; }
        std::string header_str(static_cast<size_t>(header_len), '\0');
        if (std::fread(header_str.data(), 1, static_cast<size_t>(header_len), fp) != static_cast<size_t>(header_len)) { *err = "short read on header"; fclose(fp); return {}; }
        fclose(fp);

        json hdr;
        try { hdr = json::parse(header_str); }
        catch (...) { *err = "JSON parse failed in safetensors header"; return {}; }

        // mmap the entire file — no RAM allocation for weights.
        int fd = ::open(p.c_str(), O_RDONLY);
        if (fd < 0) { *err = "cannot open for mmap"; return {}; }
        size_t file_size = static_cast<size_t>(::lseek(fd, 0, SEEK_END));
        char* map = static_cast<char*>(::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0));
        if (map == MAP_FAILED) { *err = "mmap failed"; ::close(fd); return {}; }
        size_t file_idx = bag.files.size();
        bag.files.push_back(MappedFile{});
        bag.files.back().data = map;
        bag.files.back().size = file_size;
        bag.files.back().fd   = fd;

        size_t data_offset = sizeof(uint64_t) + static_cast<size_t>(header_len);

        for (auto& [key, val] : hdr.items()) {
            if (key == "__metadata__") continue;
            if (!val.is_object() || !val.contains("dtype") ||
                !val.contains("shape") || !val.contains("data_offsets"))
                continue;

            SSTensor st;
            st.name     = key;
            st.dtype    = val["dtype"].get<std::string>();
            st.shape    = val["shape"].get<std::vector<int64_t>>();
            auto offs   = val["data_offsets"].get<std::vector<size_t>>();
            st.file_idx = file_idx;
            st.offset   = data_offset + offs[0];
            st.nbytes   = offs[1] - offs[0];

            int64_t n_el = 1;
            for (auto d : st.shape) n_el *= d;
            int el_size = 0;
            if (st.dtype == "F32")       el_size = 4;
            else if (st.dtype == "F16")  el_size = 2;
            else if (st.dtype == "BF16") el_size = 2;
            else die_f("unsupported dtype %s in tensor %s", st.dtype.c_str(), st.name.c_str());
            if (static_cast<int64_t>(st.nbytes) != n_el * el_size)
                die_f("size mismatch for %s: %zu bytes but %ld elements * %d = %ld",
                      st.name.c_str(), st.nbytes, n_el, el_size, n_el * el_size);

            bag.tensors.push_back(std::move(st));
        }
    }

    return bag;
}

// ── GGUF writer helpers ───────────────────────────────────────────────────

static ggml_type ggml_type_from_dtype(const std::string& dtype) {
    if (dtype == "F32")  return GGML_TYPE_F32;
    if (dtype == "F16")  return GGML_TYPE_F16;
    if (dtype == "BF16") return GGML_TYPE_BF16;
    return GGML_TYPE_F32;
}

// Create a 1-D ggml_tensor from an SSTensor (view into the bag's data).
// If override_type differs from the source dtype and conversion is needed
// (currently BF16→F32), a persistent buffer is allocated in the bag.
static ggml_tensor* make_tensor_1d(ggml_context* ctx, const TensorBag& bag,
                                   const SSTensor& st, ggml_type override_type) {
    ggml_type src = ggml_type_from_dtype(st.dtype);
    ggml_type t   = (override_type != GGML_TYPE_COUNT) ? override_type : src;
    int64_t ne[1] = {st.shape.empty() ? 1 : st.shape[0]};
    ggml_tensor* gt = ggml_new_tensor(ctx, t, 1, ne);
    if (!gt) return nullptr;
    if (src == t) {
        gt->data = const_cast<void*>(bag.data_ptr(st));
    } else if (src == GGML_TYPE_BF16 && t == GGML_TYPE_F32) {
        gt->data = const_cast<void*>(bag.bf16_to_f32(st));
    } else {
        die_f("unsupported 1-D conversion %s→%s for tensor %s",
              st.dtype.c_str(), ggml_type_name(t), st.name.c_str());
    }
    return gt;
}

// Create a 2-D ggml_tensor from an SSTensor.
static ggml_tensor* make_tensor_2d(ggml_context* ctx, const TensorBag& bag,
                                   const SSTensor& st, ggml_type override_type) {
    ggml_type src = ggml_type_from_dtype(st.dtype);
    ggml_type t   = (override_type != GGML_TYPE_COUNT) ? override_type : src;
    int64_t ne[2] = {st.shape[1], st.shape[0]};  // ne[0]=innermost
    ggml_tensor* gt = ggml_new_tensor(ctx, t, 2, ne);
    if (!gt) return nullptr;
    if (src == t) {
        gt->data = const_cast<void*>(bag.data_ptr(st));
    } else if (src == GGML_TYPE_BF16 && t == GGML_TYPE_F32) {
        gt->data = const_cast<void*>(bag.bf16_to_f32(st));
    } else {
        die_f("unsupported 2-D conversion %s→%s for tensor %s",
              st.dtype.c_str(), ggml_type_name(t), st.name.c_str());
    }
    return gt;
}

static void add_tensor_2d(gguf_context* gctx, ggml_context* mctx,
                          const char* name, const TensorBag& bag,
                          const SSTensor& st, ggml_type override_type) {
    ggml_tensor* gt = make_tensor_2d(mctx, bag, st, override_type);
    if (!gt) die_f("make_tensor_2d failed for %s", name);
    ggml_set_name(gt, name);
    gguf_add_tensor(gctx, gt);
}

// Add a tensor to GGUF, auto-detecting 1D vs 2D from shape.
static void add_tensor_auto(gguf_context* gctx, ggml_context* mctx,
                            const char* name, const TensorBag& bag,
                            const SSTensor& st, ggml_type override_type) {
    if (st.shape.size() == 1) {
        ggml_tensor* gt = make_tensor_1d(mctx, bag, st, override_type);
        if (!gt) die_f("make_tensor_1d failed for %s", name);
        ggml_set_name(gt, name);
        gguf_add_tensor(gctx, gt);
    } else {
        add_tensor_2d(gctx, mctx, name, bag, st, override_type);
    }
}

// ── Write talker GGUF ─────────────────────────────────────────────────────

static void write_talker(const TensorBag& bag, const json& config,
                         const fs::path& output, const std::string& quant) {
    auto tc = config["talker_config"];
    int n_layer   = tc.value("num_hidden_layers", 20);
    int n_embd    = tc.value("hidden_size", 1024);
    int n_head    = tc.value("num_attention_heads", 16);
    int n_kv_head = tc.value("num_key_value_heads", 2);
    int n_ff      = tc.value("intermediate_size", 2048);
    int head_dim  = tc.value("head_dim", n_embd / n_head);
    int n_rot     = head_dim;  // full rotation width (= head_dim, not head_dim/2)
    int text_vocab = tc.value("text_vocab_size", 152064);
    int codec_vocab = tc.value("vocab_size", 3072);
    float rms_eps = tc.value("rms_norm_eps", 1e-6);

    // Quantization type
    ggml_type qtype = (quant == "f32") ? GGML_TYPE_F32 : GGML_TYPE_Q8_0;
    ggml_type stype = (quant == "f32") ? GGML_TYPE_COUNT : GGML_TYPE_F16;
    // For F16 quant, use GGML_TYPE_F16 as storage; for Q8_0, keep Q8_0
    if (quant == "f16") { qtype = GGML_TYPE_F16; stype = GGML_TYPE_COUNT; }

    std::printf("Talker: %d layers, %d hidden, %d heads/%d kv, %d ff, quant=%s\n",
                n_layer, n_embd, n_head, n_kv_head, n_ff, quant.c_str());

    gguf_context* gctx = gguf_init_empty();
    if (!gctx) die("gguf_init_empty failed");

    // KV metadata
    gguf_set_val_u32(gctx, "general.architecture", GGUF_TYPE_UINT32);
    // We can't use the gguf convenience writers for architecture since the
    // GGUF C API expects specific key names. Write them manually.
    gguf_set_val_str(gctx, "general.architecture", "qwen3tts");
    gguf_set_val_u32(gctx, "general.alignment", 32);
    gguf_set_val_u32(gctx, "qwen3tts.context_length", 32768);
    gguf_set_val_u32(gctx, "qwen3tts.embedding_length", n_embd);
    gguf_set_val_u32(gctx, "qwen3tts.feed_forward_length", n_ff);
    gguf_set_val_u32(gctx, "qwen3tts.block_count", n_layer);
    gguf_set_val_u32(gctx, "qwen3tts.attention.head_count", n_head);
    gguf_set_val_u32(gctx, "qwen3tts.attention.head_count_kv", n_kv_head);
    gguf_set_val_u32(gctx, "qwen3tts.attention.key_length", head_dim);
    gguf_set_val_u32(gctx, "qwen3tts.attention.value_length", head_dim);
    gguf_set_val_f32(gctx, "qwen3tts.attention.layer_norm_rms_epsilon", rms_eps);
    gguf_set_val_u32(gctx, "qwen3tts.rope.dimension_count", n_rot);
    // RoPE frequency base: talker uses rope_theta=1000000 (from HF config).
    // Without this, llama.cpp defaults to 10000, producing garbage attention.
    {
        float rope_theta = tc.value("rope_theta", 1000000.0f);
        gguf_set_val_f32(gctx, "qwen3tts.rope.freq_base", rope_theta);
    }
    // IMROPE sections (3D position encoding)
    {
        int sections[4] = {24, 20, 20, 0};
        gguf_set_arr_data(gctx, "qwen3tts.rope.dimension_sections", GGUF_TYPE_INT32, sections, 4);
    }
    gguf_set_val_u32(gctx, "qwen3tts.text_vocab_size", text_vocab);
    gguf_set_val_u32(gctx, "qwen3tts.codec_vocab_size", codec_vocab);

    // Dummy GPT-2 tokenizer so llama.cpp sets n_vocab == codec_vocab_size
    // (matching token_embd.weight's second dim). Not actually used for
    // inference — the session tokenizes through the separate codec GGUF.
    {
        gguf_set_val_str(gctx, "tokenizer.ggml.model", "gpt2");
        const int nt = static_cast<int>(codec_vocab);
        // Build token names on the heap so pointers survive until gguf write.
        std::vector<std::string> token_pool;
        std::vector<const char*> token_ptrs;
        token_pool.reserve(static_cast<size_t>(nt));
        token_ptrs.reserve(static_cast<size_t>(nt));
        for (int i = 0; i < nt; i++) {
            if (i < 3) {
                static const char* names[] = {"a", "b", "ab"};
                token_pool.emplace_back(names[i]);
            } else {
                token_pool.emplace_back("<t_" + std::to_string(i) + ">");
            }
            token_ptrs.push_back(token_pool.back().c_str());
        }
        gguf_set_arr_str(gctx, "tokenizer.ggml.tokens", token_ptrs.data(), nt);
        // All type 1 (normal tokens)
        std::vector<int32_t> ttype(static_cast<size_t>(nt), 1);
        gguf_set_arr_data(gctx, "tokenizer.ggml.token_type", GGUF_TYPE_INT32,
                          ttype.data(), nt);
        // Single dummy merge (required by gpt2 tokenizer loading path)
        const char* merges[] = {"a b"};
        gguf_set_arr_str(gctx, "tokenizer.ggml.merges", merges, 1);
        gguf_set_val_u32(gctx, "tokenizer.ggml.bos_token_id", 1);
        gguf_set_val_u32(gctx, "tokenizer.ggml.eos_token_id", 2);
    }

    // Allocate ggml_context for tensor metadata
    size_t n_tensors = 9 + static_cast<size_t>(n_layer) * 11;
    size_t mem = ggml_tensor_overhead() * (n_tensors + 256);
    struct ggml_init_params gip = {mem, nullptr, true};
    ggml_context* mctx = ggml_init(gip);
    if (!mctx) die("ggml_init failed for talker");

    // token_embd.weight = codec_embedding [codec_vocab, n_embd]
    // Required by llama.cpp's model graph (shape check: ne[0] == embedding_length).
    // The session feeds embeddings via forward_embeddings() and never looks up
    // token_embd for values — it exists purely to satisfy the graph allocator.
    auto* te = bag.find2("talker.model.codec_embedding.weight", "model.codec_embedding.weight");
    if (te) add_tensor_2d(gctx, mctx, "token_embd.weight", bag, *te, stype);

    // ── Text embedding + text projection (Stage: TEXT-FIX) ────────────────
    // These are the tensors the runner needs for project_text_tokens(). Without
    // them has_text_embd_ is false and ALL text input becomes zeros → garbage
    // audio. The HF names map as:
    //   talker.model.text_embedding.weight           → text_embd.weight
    //   talker.text_projection.linear_fc1.{w,b}      → text_proj.0.{weight,bias}
    //   talker.text_projection.linear_fc2.{w,b}      → text_proj.1.{weight,bias}
    {
        auto* t_embd = bag.find2("talker.model.text_embedding.weight",
                                 "model.text_embedding.weight");
        if (t_embd) add_tensor_2d(gctx, mctx, "text_embd.weight", bag, *t_embd, stype);

        auto* tp0w = bag.find2("talker.text_projection.linear_fc1.weight",
                               "text_projection.linear_fc1.weight");
        if (tp0w) add_tensor_2d(gctx, mctx, "text_proj.0.weight", bag, *tp0w, stype);

        auto* tp0b = bag.find2("talker.text_projection.linear_fc1.bias",
                               "text_projection.linear_fc1.bias");
        if (tp0b) add_tensor_auto(gctx, mctx, "text_proj.0.bias", bag, *tp0b, GGML_TYPE_F32);

        auto* tp1w = bag.find2("talker.text_projection.linear_fc2.weight",
                               "text_projection.linear_fc2.weight");
        if (tp1w) add_tensor_2d(gctx, mctx, "text_proj.1.weight", bag, *tp1w, stype);

        auto* tp1b = bag.find2("talker.text_projection.linear_fc2.bias",
                               "text_projection.linear_fc2.bias");
        if (tp1b) add_tensor_auto(gctx, mctx, "text_proj.1.bias", bag, *tp1b, GGML_TYPE_F32);

        std::printf("  text_embd: %s, text_proj.0: %s, text_proj.1: %s\n",
                    t_embd ? "yes" : "MISSING",
                    tp0w ? "yes" : "MISSING",
                    tp1w ? "yes" : "MISSING");
    }

    // output_norm.weight [n_embd]
    auto* on = bag.find2("talker.model.norm.weight", "model.norm.weight");
    if (on) {
        ggml_tensor* gt = make_tensor_1d(mctx, bag, *on, GGML_TYPE_F32);
        ggml_set_name(gt, "output_norm.weight");
        if (gt) gguf_add_tensor(gctx, gt);
    }

    // output.weight = codec_head [codec_vocab, n_embd]
    auto* ch = bag.find2("talker.codec_head.weight", "codec_head.weight");
    if (ch) add_tensor_2d(gctx, mctx, "output.weight", bag, *ch, stype);

    // Transformer layers
    for (int i = 0; i < n_layer; i++) {
        std::string p = "talker.model.layers." + std::to_string(i) + ".";
        std::string ap = "model.layers." + std::to_string(i) + ".";

        auto get = [&](const std::string& name) -> const SSTensor* {
            auto* r = bag.find(p + name);
            return r ? r : bag.find(ap + name);
        };

        auto add = [&](const char* gname, const std::string& sname, ggml_type ot) {
            auto* t = get(sname);
            if (t) add_tensor_auto(gctx, mctx, gname, bag, *t, ot);
        };

        char bn[64];
        auto blk = [&](int i, const char* sfx) {
            std::snprintf(bn, sizeof(bn), "blk.%d.%s", i, sfx);
            return bn;
        };

        add(blk(i, "attn_norm.weight"),   "input_layernorm.weight",             GGML_TYPE_F32);
        add(blk(i, "attn_q.weight"),      "self_attn.q_proj.weight",           stype);
        add(blk(i, "attn_k.weight"),      "self_attn.k_proj.weight",           stype);
        add(blk(i, "attn_v.weight"),      "self_attn.v_proj.weight",           stype);
        add(blk(i, "attn_q_norm.weight"), "self_attn.q_norm.weight",           GGML_TYPE_F32);
        add(blk(i, "attn_k_norm.weight"), "self_attn.k_norm.weight",           GGML_TYPE_F32);
        add(blk(i, "attn_output.weight"),"self_attn.o_proj.weight",           stype);
        add(blk(i, "ffn_norm.weight"),    "post_attention_layernorm.weight",   GGML_TYPE_F32);
        add(blk(i, "ffn_gate.weight"),    "mlp.gate_proj.weight",              stype);
        add(blk(i, "ffn_down.weight"),    "mlp.down_proj.weight",              stype);
        add(blk(i, "ffn_up.weight"),      "mlp.up_proj.weight",                stype);
    }

    std::printf("  writing %s ... ", output.c_str());
    if (!gguf_write_to_file(gctx, output.c_str(), false))
        die("gguf_write_to_file failed");
    std::printf("done (%zu MB)\n", fs::file_size(output) / 1048576);

    gguf_free(gctx);
    ggml_free(mctx);
}

// ── Write predictor GGUF ───────────────────────────────────────────────────

static void write_predictor(const TensorBag& bag, const json& config,
                            const fs::path& output, const std::string& quant) {
    auto tc = config["talker_config"];
    auto cp = tc["code_predictor_config"];
    int n_layer     = cp.value("num_hidden_layers", 5);
    int n_embd      = cp.value("hidden_size", 1024);
    int n_head      = cp.value("num_attention_heads", 16);
    int n_kv_head   = cp.value("num_key_value_heads", 8);
    int n_ff        = cp.value("intermediate_size", 3072);
    int head_dim    = cp.value("head_dim", 128);
    int n_codebooks = cp.value("num_code_groups", 32);
    int fine_vocab  = cp.value("vocab_size", 2048);
    int codebook0_vocab = tc.value("vocab_size", 3072);
    float rms_eps   = cp.value("rms_norm_eps", 1e-6);

    ggml_type qtype = (quant == "f32") ? GGML_TYPE_F32 : GGML_TYPE_Q8_0;
    ggml_type stype = (quant == "f32") ? GGML_TYPE_COUNT : GGML_TYPE_F16;
    if (quant == "f16") { qtype = GGML_TYPE_F16; stype = GGML_TYPE_COUNT; }

    std::printf("Predictor: %d layers, %d hidden, %d heads/%d kv, %d ff, "
                "%d codebooks, quant=%s\n",
                n_layer, n_embd, n_head, n_kv_head, n_ff, n_codebooks, quant.c_str());

    gguf_context* gctx = gguf_init_empty();
    if (!gctx) die("gguf_init_empty failed");

    gguf_set_val_str(gctx, "general.architecture", "qwen3tts_cp");
    gguf_set_val_u32(gctx, "general.alignment", 32);
    gguf_set_val_u32(gctx, "qwen3tts_cp.context_length", 32768);
    gguf_set_val_u32(gctx, "qwen3tts_cp.embedding_length", n_embd);
    gguf_set_val_u32(gctx, "qwen3tts_cp.feed_forward_length", n_ff);
    gguf_set_val_u32(gctx, "qwen3tts_cp.block_count", n_layer);
    gguf_set_val_u32(gctx, "qwen3tts_cp.attention.head_count", n_head);
    gguf_set_val_u32(gctx, "qwen3tts_cp.attention.head_count_kv", n_kv_head);
    gguf_set_val_u32(gctx, "qwen3tts_cp.attention.key_length", head_dim);
    gguf_set_val_u32(gctx, "qwen3tts_cp.attention.value_length", head_dim);
    gguf_set_val_f32(gctx, "qwen3tts_cp.attention.layer_norm_rms_epsilon", rms_eps);
    gguf_set_val_u32(gctx, "qwen3tts_cp.rope.dimension_count", head_dim);
    // Predictor also uses rope_theta=1000000
    {
        float rope_theta = tc.value("rope_theta", 1000000.0f);
        gguf_set_val_f32(gctx, "qwen3tts_cp.rope.freq_base", rope_theta);
    }
    gguf_set_val_u32(gctx, "qwen3tts_cp.num_code_groups", n_codebooks);
    gguf_set_val_u32(gctx, "qwen3tts_cp.codebook_0_vocab", codebook0_vocab);
    gguf_set_val_u32(gctx, "qwen3tts_cp.fine_vocab", fine_vocab);

    // Dummy GPT-2 tokenizer so llama.cpp sets n_vocab correctly
    {
        gguf_set_val_str(gctx, "tokenizer.ggml.model", "gpt2");
        int nt = codebook0_vocab + fine_vocab;
        std::vector<std::string> tokens;
        tokens.reserve(static_cast<size_t>(nt));
        for (int i = 0; i < nt; i++) {
            if (i < 3) {
                static const char* names[] = {"a", "b", "ab"};
                tokens.emplace_back(names[i]);
            } else {
                tokens.emplace_back("<t_" + std::to_string(i) + ">");
            }
        }
        std::vector<const char*> cstrs;
        cstrs.reserve(tokens.size());
        for (auto& s : tokens) cstrs.push_back(s.c_str());
        gguf_set_arr_str(gctx, "tokenizer.ggml.tokens", cstrs.data(), cstrs.size());
        std::vector<int> types(static_cast<size_t>(nt), 1);
        gguf_set_arr_data(gctx, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
        const char* merges[] = {"a b"};
        gguf_set_arr_str(gctx, "tokenizer.ggml.merges", merges, 1);
        gguf_set_val_u32(gctx, "tokenizer.ggml.bos_token_id", 1);
        gguf_set_val_u32(gctx, "tokenizer.ggml.eos_token_id", 2);
    }

    size_t n_tensors = 3 + static_cast<size_t>(n_codebooks - 1) * 2 + static_cast<size_t>(n_layer) * 11;
    size_t mem = ggml_tensor_overhead() * (n_tensors + 256);
    struct ggml_init_params gip = {mem, nullptr, true};
    ggml_context* mctx = ggml_init(gip);
    if (!mctx) die("ggml_init failed for predictor");

    // Dummy token_embd.weight required for llama.cpp graph (never used for values)
    {
        int n_vocab = codebook0_vocab + fine_vocab;
        auto* gt = ggml_new_tensor_2d(mctx, GGML_TYPE_F32, n_embd, n_vocab);
        if (!gt) die("ggml_new_tensor_2d failed for dummy token_embd.weight");
        void* zeros = std::calloc(1, ggml_nbytes(gt));
        if (!zeros) die("calloc failed for dummy token_embd.weight");
        gt->data = zeros;
        ggml_set_name(gt, "token_embd.weight");
        gguf_add_tensor(gctx, gt);
    }

    // small_to_mtp.weight + bias [n_embd, n_embd]
    auto* smw = bag.find2("talker.code_predictor.small_to_mtp_projection.weight",
                          "code_predictor.small_to_mtp_projection.weight");
    if (smw) add_tensor_2d(gctx, mctx, "small_to_mtp.weight", bag, *smw, stype);
    auto* smb = bag.find2("talker.code_predictor.small_to_mtp_projection.bias",
                          "code_predictor.small_to_mtp_projection.bias");
    if (smb) add_tensor_auto(gctx, mctx, "small_to_mtp.bias", bag, *smb, GGML_TYPE_F32);

    // output_norm.weight [n_embd]
    auto* on = bag.find2("talker.code_predictor.model.norm.weight",
                         "code_predictor.model.norm.weight");
    if (on) {
        ggml_tensor* gt = make_tensor_1d(mctx, bag, *on, GGML_TYPE_F32);
        ggml_set_name(gt, "output_norm.weight");
        if (gt) gguf_add_tensor(gctx, gt);
    }

    // codec_embd.{i}.weight [fine_vocab, n_embd] for i=0..30
    int fine_books = n_codebooks - 1;
    for (int i = 0; i < fine_books; i++) {
        std::string key = "talker.code_predictor.model.codec_embedding." +
                          std::to_string(i) + ".weight";
        std::string alt = "code_predictor.model.codec_embedding." +
                          std::to_string(i) + ".weight";
        auto* t = bag.find2(key, alt);
        if (!t) continue;
        char name[64];
        std::snprintf(name, sizeof(name), "codec_embd.%d.weight", i);
        add_tensor_2d(gctx, mctx, name, bag, *t, stype);
    }

    // lm_head.{i}.weight [fine_vocab, n_embd] for i=0..30
    for (int i = 0; i < fine_books; i++) {
        std::string key = "talker.code_predictor.lm_head." +
                          std::to_string(i) + ".weight";
        std::string alt = "code_predictor.lm_head." +
                          std::to_string(i) + ".weight";
        auto* t = bag.find2(key, alt);
        if (!t) continue;
        char name[64];
        std::snprintf(name, sizeof(name), "lm_head.%d.weight", i);
        add_tensor_2d(gctx, mctx, name, bag, *t, stype);
    }

    // Transformer layers
    for (int i = 0; i < n_layer; i++) {
        std::string p = "talker.code_predictor.model.layers." + std::to_string(i) + ".";
        std::string ap = "code_predictor.model.layers." + std::to_string(i) + ".";

        auto get = [&](const std::string& name) -> const SSTensor* {
            auto* r = bag.find(p + name);
            return r ? r : bag.find(ap + name);
        };

        auto add = [&](const char* gname, const std::string& sname, ggml_type ot) {
            auto* t = get(sname);
            if (t) add_tensor_auto(gctx, mctx, gname, bag, *t, ot);
        };

        char bn[64];
        auto blk = [&](int i, const char* sfx) {
            std::snprintf(bn, sizeof(bn), "blk.%d.%s", i, sfx);
            return bn;
        };

        add(blk(i, "attn_norm.weight"),   "input_layernorm.weight",             GGML_TYPE_F32);
        add(blk(i, "attn_q.weight"),      "self_attn.q_proj.weight",           stype);
        add(blk(i, "attn_k.weight"),      "self_attn.k_proj.weight",           stype);
        add(blk(i, "attn_v.weight"),      "self_attn.v_proj.weight",           stype);
        add(blk(i, "attn_q_norm.weight"), "self_attn.q_norm.weight",           GGML_TYPE_F32);
        add(blk(i, "attn_k_norm.weight"), "self_attn.k_norm.weight",           GGML_TYPE_F32);
        add(blk(i, "attn_output.weight"),"self_attn.o_proj.weight",           stype);
        add(blk(i, "ffn_norm.weight"),    "post_attention_layernorm.weight",   GGML_TYPE_F32);
        add(blk(i, "ffn_gate.weight"),    "mlp.gate_proj.weight",              stype);
        add(blk(i, "ffn_down.weight"),    "mlp.down_proj.weight",              stype);
        add(blk(i, "ffn_up.weight"),      "mlp.up_proj.weight",                stype);
    }

    std::printf("  writing %s ... ", output.c_str());
    if (!gguf_write_to_file(gctx, output.c_str(), false))
        die("gguf_write_to_file failed");
    std::printf("done (%zu MB)\n", fs::file_size(output) / 1048576);

    gguf_free(gctx);
    ggml_free(mctx);
}

// ── Write tokenizer GGUF (text BPE sidecar) ────────────────────────────────
//
// The talker GGUF carries a dummy codec-vocab tokenizer (required so n_vocab
// matches token_embd = codec_embedding). The REAL Qwen3 BPE text tokenizer
// (151 936 tokens, 151 291 merges) lives in this separate vocab-only GGUF,
// loaded by Runner::load_tokenizer() and used by Runner::tokenize().

static void write_tokenizer(const fs::path& model_dir, const json& config,
                            const fs::path& output) {
    // 1. Load vocab.json → invert to id-ordered array
    auto vocab_json = load_json(model_dir / "vocab.json");
    // vocab.json maps token_string → id. Invert it.
    int text_vocab = config.value("talker_config", json::object())
                         .value("text_vocab_size", 151936);
    // Each unused slot gets a unique name to avoid duplicate-token assertion
    // in llama.cpp (id_to_token.size() == token_to_id.size()).
    std::vector<std::string> tokens;
    tokens.reserve(static_cast<size_t>(text_vocab));
    for (int i = 0; i < text_vocab; i++)
        tokens.emplace_back("<unused_" + std::to_string(i) + ">");
    std::vector<int32_t>     token_types(static_cast<size_t>(text_vocab), 1); // 1=normal

    int max_id = -1;
    for (auto& [key, val] : vocab_json.items()) {
        int id = val.get<int>();
        if (id >= 0 && id < text_vocab) {
            tokens[static_cast<size_t>(id)] = key;
            token_types[static_cast<size_t>(id)] = 1; // normal
            if (id > max_id) max_id = id;
        }
    }

    // 2. Merge added/special tokens from tokenizer_config.json
    auto tok_config = load_json(model_dir / "tokenizer_config.json");
    auto added = tok_config.value("added_tokens_decoder", json::object());
    int n_special = 0;
    for (auto& [id_str, info] : added.items()) {
        int id = std::stoi(id_str);
        if (id >= 0 && id < text_vocab) {
            tokens[static_cast<size_t>(id)] = info.value("content", "<unk>");
            // 3 = user-defined control token (llama-normal control = 2,
            // user-defined = 3). Special tokens get type 3.
            token_types[static_cast<size_t>(id)] = info.value("special", false) ? 3 : 1;
            ++n_special;
        }
    }

    std::printf("Tokenizer: %d base tokens (max id %d), %d special tokens, "
                "vocab size %d\n",
                max_id + 1, max_id, n_special, text_vocab);

    // 3. Load merges.txt
    std::vector<std::string> merges;
    {
        FILE* fp = std::fopen((model_dir / "merges.txt").c_str(), "rb");
        if (!fp) die("cannot open merges.txt — is this a Qwen3-TTS HF model?");
        char line[512];
        bool first = true;
        while (std::fgets(line, sizeof(line), fp)) {
            // Strip trailing newline
            size_t len = std::strlen(line);
            while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
                line[--len] = '\0';
            if (len == 0) continue;
            // Skip header (#version: 0.2)
            if (first && line[0] == '#') { first = false; continue; }
            first = false;
            merges.emplace_back(line);
        }
        std::fclose(fp);
    }
    std::printf("  loaded %zu BPE merges\n", merges.size());

    // 4. Write the vocab-only GGUF
    gguf_context* gctx = gguf_init_empty();
    if (!gctx) die("gguf_init_empty failed for tokenizer");

    // Minimal architecture metadata so llama.cpp loads it in vocab_only mode.
    // The arch itself is irrelevant — only tokenizer.ggml.* keys matter.
    gguf_set_val_str(gctx, "general.architecture", "llama");
    gguf_set_val_u32(gctx, "general.alignment", 32);
    gguf_set_val_u32(gctx, "llama.context_length", 32768);
    gguf_set_val_u32(gctx, "llama.embedding_length", text_vocab);
    gguf_set_val_u32(gctx, "llama.block_count", 1);
    gguf_set_val_u32(gctx, "llama.attention.head_count", 1);
    gguf_set_val_u32(gctx, "llama.attention.head_count_kv", 1);
    gguf_set_val_u32(gctx, "llama.attention.key_length", 64);
    gguf_set_val_u32(gctx, "llama.attention.value_length", 64);
    gguf_set_val_f32(gctx, "llama.attention.layer_norm_rms_epsilon", 1e-6f);
    gguf_set_val_u32(gctx, "llama.rope.dimension_count", 64);
    gguf_set_val_u32(gctx, "llama.feed_forward_length", 1);

    // Tokenizer metadata — this is what actually matters.
    gguf_set_val_str(gctx, "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(gctx, "tokenizer.ggml.pre", "qwen2");

    // Token strings + types
    std::vector<const char*> token_ptrs;
    token_ptrs.reserve(tokens.size());
    for (auto& s : tokens) token_ptrs.push_back(s.c_str());
    gguf_set_arr_str(gctx, "tokenizer.ggml.tokens",
                     token_ptrs.data(), static_cast<int64_t>(tokens.size()));
    gguf_set_arr_data(gctx, "tokenizer.ggml.token_type", GGUF_TYPE_INT32,
                      token_types.data(), static_cast<int64_t>(token_types.size()));

    // Merges
    std::vector<const char*> merge_ptrs;
    merge_ptrs.reserve(merges.size());
    for (auto& s : merges) merge_ptrs.push_back(s.c_str());
    gguf_set_arr_str(gctx, "tokenizer.ggml.merges",
                     merge_ptrs.data(), static_cast<int64_t>(merges.size()));

    // Special token IDs (from config.json)
    int32_t bos_id = config.value("im_start_token_id", 151644);
    int32_t eos_id = config.value("im_end_token_id", 151645);
    int32_t unk_id = 151643; // <|endoftext|>
    gguf_set_val_u32(gctx, "tokenizer.ggml.bos_token_id", bos_id);
    gguf_set_val_u32(gctx, "tokenizer.ggml.eos_token_id", eos_id);
    gguf_set_val_u32(gctx, "tokenizer.ggml.unknown_token_id", unk_id);

    std::printf("  writing %s ... ", output.c_str());
    if (!gguf_write_to_file(gctx, output.c_str(), false))
        die("gguf_write_to_file failed for tokenizer");
    std::printf("done (%zu KB)\n", fs::file_size(output) / 1024);

    gguf_free(gctx);
}

// ── Main ───────────────────────────────────────────────────────────────────

static int usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s <model_dir> --outdir <dir> [--quant f32|f16|q8_0] "
        "[--skip-talker] [--skip-predictor]\n"
        "  Convert Qwen3-TTS HuggingFace safetensors to GGUF.\n",
        argv0);
    return 2;
}

int main(int argc, char** argv) {
    if (argc < 3) return usage(argv[0]);

    fs::path model_dir;
    fs::path outdir = ".";
    std::string quant = "q8_0";
    bool skip_talker = false, skip_predictor = false, skip_tokenizer = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--outdir" && i + 1 < argc) outdir = argv[++i];
        else if (a == "--quant" && i + 1 < argc) quant = argv[++i];
        else if (a == "--skip-talker")    skip_talker = true;
        else if (a == "--skip-predictor") skip_predictor = true;
        else if (a == "--skip-tokenizer") skip_tokenizer = true;
        else if (a[0] != '-')             model_dir = argv[i];
    }

    if (model_dir.empty()) return usage(argv[0]);
    if (quant != "f32" && quant != "f16" && quant != "q8_0")
        die("--quant must be f32, f16, or q8_0");

    // Read config
    auto config = load_json(model_dir / "config.json");
    auto tc = config["talker_config"];
    int n_layer = tc.value("num_hidden_layers", 0);
    int n_embd  = tc.value("hidden_size", 0);
    int cp_n_layer = tc.value("code_predictor_config", json::object())
                         .value("num_hidden_layers", 0);
    std::string variant;
    if (n_layer == 28 && n_embd == 2048) variant = "1b7";
    else if (n_layer == 28 && n_embd == 1024) variant = "0b6";
    else if (n_layer == 20)              variant = "0b6";
    else                                 variant = std::to_string(n_layer) + "l";

    std::printf("Variant: talker=%d layers, predictor=%d layers (%s)\n",
                n_layer, cp_n_layer, variant.c_str());

    // Load tensors
    std::string err;
    auto bag = load_safetensors_dir(model_dir, &err);
    if (bag.tensors.empty()) die_f("load_safetensors_dir: %s", err.c_str());
    std::printf("Loaded %zu tensors from %s\n", bag.tensors.size(),
                model_dir.c_str());

    fs::create_directories(outdir);

    if (!skip_talker) {
        auto path = outdir / ("qwen3tts-talker-" + variant + "-" + quant + ".gguf");
        write_talker(bag, config, path, quant);
    }
    if (!skip_predictor) {
        auto path = outdir / ("qwen3tts-predictor-" + variant + "-" + quant + ".gguf");
        write_predictor(bag, config, path, quant);
    }
    if (!skip_tokenizer) {
        // The tokenizer GGUF is arch-independent (same BPE for all variants)
        // but is placed in the output dir alongside the talker/predictor.
        auto path = outdir / ("qwen3tts-tokenizer-" + variant + "-" + quant + ".gguf");
        write_tokenizer(model_dir, config, path);
    }

    return 0;
}
