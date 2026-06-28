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

static void die_f(const char* fmt, const auto&... args) {
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

// ── Safetensors reader ─────────────────────────────────────────────────────

struct SSTensor {
    std::string  name;
    std::string  dtype;     // "F32", "F16", "BF16"
    std::vector<int64_t> shape;
    size_t       offset;    // byte offset in the concatenated data blob
    size_t       nbytes;
};

// Read a single .safetensors file. Returns the JSON header + concatenated data.
// The data blob contains all tensors at their respective offsets.
static bool read_safetensors(const fs::path& path,
                             json* out_header,
                             std::vector<char>* out_data,
                             std::string* err) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) { *err = "cannot open"; return false; }

    uint64_t header_len = 0;
    if (std::fread(&header_len, sizeof(header_len), 1, fp) != 1) {
        *err = "short read on header length"; fclose(fp); return false;
    }

    std::string header_str(static_cast<size_t>(header_len), '\0');
    if (std::fread(header_str.data(), 1, static_cast<size_t>(header_len), fp) != header_len) {
        *err = "short read on header"; fclose(fp); return false;
    }

    auto pos = std::ftell(fp);
    std::fseek(fp, 0, SEEK_END);
    long file_end = std::ftell(fp);
    size_t data_size = static_cast<size_t>(file_end - pos);
    std::fseek(fp, pos, SEEK_SET);

    out_data->resize(data_size);
    if (data_size > 0 && std::fread(out_data->data(), 1, data_size, fp) != data_size) {
        *err = "short read on tensor data"; fclose(fp); return false;
    }
    fclose(fp);

    try {
        *out_header = json::parse(header_str);
    } catch (...) {
        *err = "JSON parse failed in safetensors header";
        return false;
    }
    return true;
}

// Load all tensors from a directory of .safetensors files.
struct TensorBag {
    std::vector<char> data;     // concatenated data from all files
    std::vector<SSTensor> tensors;

    // Look up tensor by name. Returns nullptr if not found.
    const SSTensor* find(const std::string& name) const {
        for (auto& t : tensors)
            if (t.name == name) return &t;
        return nullptr;
    }

    // Convenience: try name and alt_name; returns first found
    const SSTensor* find2(const std::string& name, const std::string& alt) const {
        auto* p = find(name);
        return p ? p : find(alt);
    }

    // Return a pointer to the tensor's data within the concatenated blob.
    const void* data_ptr(const SSTensor& t) const {
        if (t.offset + t.nbytes > data.size()) return nullptr;
        return data.data() + t.offset;
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
        json hdr;
        std::vector<char> file_data;
        if (!read_safetensors(p, &hdr, &file_data, err)) return {};

        // Each key-value in the JSON is either "__metadata__" or a tensor entry.
        for (auto& [key, val] : hdr.items()) {
            if (key == "__metadata__") continue;
            if (!val.is_object() || !val.contains("dtype") ||
                !val.contains("shape") || !val.contains("data_offsets"))
                continue;

            SSTensor st;
            st.name  = key;
            st.dtype = val["dtype"].get<std::string>();
            st.shape = val["shape"].get<std::vector<int64_t>>();
            auto offs = val["data_offsets"].get<std::vector<size_t>>();
            st.offset = offs[0];
            st.nbytes = offs[1] - offs[0];

            // Compute element count from shape
            int64_t n_el = 1;
            for (auto d : st.shape) n_el *= d;

            // Verify the dtype size matches the byte count
            int el_size = 0;
            if (st.dtype == "F32")      el_size = 4;
            else if (st.dtype == "F16") el_size = 2;
            else if (st.dtype == "BF16") el_size = 2;
            else die_f("unsupported dtype %s in tensor %s", st.dtype.c_str(), st.name.c_str());

            if (static_cast<int64_t>(st.nbytes) != n_el * el_size) {
                die_f("size mismatch for %s: %zu bytes but %ld elements * %d = %ld",
                      st.name.c_str(), st.nbytes, n_el, el_size, n_el * el_size);
            }

            // Append the file's data into the bag's concatenated blob.
            // We adjust the offset to point into the concatenated blob.
            st.offset += bag.data.size();
            bag.data.insert(bag.data.end(), file_data.begin(), file_data.end());
            bag.tensors.push_back(std::move(st));
        }

        // The file_data is now part of bag.data; we must NOT keep the local copy.
    }

    return bag;
}

// ── GGUF writer helpers ───────────────────────────────────────────────────

static ggml_type ggml_type_from_dtype(const std::string& dtype) {
    if (dtype == "F32")  return GGML_TYPE_F32;
    if (dtype == "F16")  return GGML_TYPE_F16;
    if (dtype == "BF16") return GGML_TYPE_F16;  // stored as F16 in GGUF
    return GGML_TYPE_F32;
}

// Create a 1-D ggml_tensor from an SSTensor (view into the bag's data).
static ggml_tensor* make_tensor_1d(ggml_context* ctx, const TensorBag& bag,
                                   const SSTensor& st, ggml_type override_type) {
    ggml_type t = (override_type != GGML_TYPE_COUNT) ? override_type : ggml_type_from_dtype(st.dtype);
    int64_t ne[1] = {st.shape.empty() ? 1 : st.shape[0]};
    ggml_tensor* gt = ggml_new_tensor(ctx, t, 1, ne);
    if (!gt) return nullptr;
    gt->data = const_cast<void*>(bag.data_ptr(st));
    return gt;
}

// Create a 2-D ggml_tensor from an SSTensor.
static ggml_tensor* make_tensor_2d(ggml_context* ctx, const TensorBag& bag,
                                   const SSTensor& st, ggml_type override_type) {
    ggml_type t = (override_type != GGML_TYPE_COUNT) ? override_type : ggml_type_from_dtype(st.dtype);
    int64_t ne[2] = {st.shape[1], st.shape[0]};  // ne[0]=innermost
    ggml_tensor* gt = ggml_new_tensor(ctx, t, 2, ne);
    if (!gt) return nullptr;
    gt->data = const_cast<void*>(bag.data_ptr(st));
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

// ── Write talker GGUF ─────────────────────────────────────────────────────

static void write_talker(const TensorBag& bag, const json& config,
                         const fs::path& output, const std::string& quant) {
    auto tc = config["talker_config"];
    int n_layer   = tc.value("num_hidden_layers", 20);
    int n_embd    = tc.value("hidden_size", 1024);
    int n_head    = tc.value("num_attention_heads", 16);
    int n_kv_head = tc.value("num_key_value_heads", 2);
    int n_ff      = tc.value("intermediate_size", 2048);
    int n_rot     = 64;
    int text_vocab = tc.value("text_vocab_size", 152064);
    int codec_vocab = tc.value("vocab_size", 3072);
    int head_dim  = tc.value("head_dim", 64);
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
    gguf_set_val_f32(gctx, "qwen3tts.attention.layer_norm_rms_epsilon", rms_eps);
    gguf_set_val_u32(gctx, "qwen3tts.rope.dimension_count", n_rot);
    // IMROPE sections (3D position encoding)
    {
        int sections[4] = {24, 20, 20, 0};
        gguf_set_arr_data(gctx, "qwen3tts.rope.sections", GGUF_TYPE_INT32, sections, 4);
    }
    gguf_set_val_u32(gctx, "qwen3tts.text_vocab_size", text_vocab);
    gguf_set_val_u32(gctx, "qwen3tts.codec_vocab_size", codec_vocab);

    // Allocate ggml_context for tensor metadata
    size_t n_tensors = 4 + 3 + 2 + 2 + static_cast<size_t>(n_layer) * 10;
    size_t mem = ggml_tensor_overhead() * (n_tensors + 8);
    struct ggml_init_params gip = {mem, nullptr, true};
    ggml_context* mctx = ggml_init(gip);
    if (!mctx) die("ggml_init failed for talker");

    // token_embd.weight = codec_embedding [codec_vocab, n_embd]
    auto* te = bag.find2("talker.model.codec_embedding.weight", "model.codec_embedding.weight");
    if (te) add_tensor_2d(gctx, mctx, "token_embd.weight", bag, *te, stype);

    // text_embd.weight [text_vocab, 2048]
    auto* tx = bag.find2("talker.model.text_embedding.weight", "model.text_embedding.weight");
    if (tx) add_tensor_2d(gctx, mctx, "text_embd.weight", bag, *tx, stype);

    // text_proj.0.weight + bias [2048, 2048]
    auto* p0w = bag.find2("talker.text_projection.linear_fc1.weight", "text_projection.linear_fc1.weight");
    if (p0w) add_tensor_2d(gctx, mctx, "text_proj.0.weight", bag, *p0w, stype);
    auto* p0b = bag.find2("talker.text_projection.linear_fc1.bias", "text_projection.linear_fc1.bias");
    if (p0b) add_tensor_2d(gctx, mctx, "text_proj.0.bias", bag, *p0b, GGML_TYPE_F32);

    // text_proj.1.weight + bias [1024, 2048]
    auto* p1w = bag.find2("talker.text_projection.linear_fc2.weight", "text_projection.linear_fc2.weight");
    if (p1w) add_tensor_2d(gctx, mctx, "text_proj.1.weight", bag, *p1w, stype);
    auto* p1b = bag.find2("talker.text_projection.linear_fc2.bias", "text_projection.linear_fc2.bias");
    if (p1b) add_tensor_2d(gctx, mctx, "text_proj.1.bias", bag, *p1b, GGML_TYPE_F32);

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

    // rope_freqs.weight
    auto* rf = bag.find2("talker.model.rotary_emb.inv_freq", "model.rotary_emb.inv_freq");
    if (rf) {
        ggml_tensor* gt = make_tensor_1d(mctx, bag, *rf, GGML_TYPE_F32);
        ggml_set_name(gt, "rope_freqs.weight");
        if (gt) gguf_add_tensor(gctx, gt);
    }

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
            if (t) add_tensor_2d(gctx, mctx, gname, bag, *t, ot);
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
        add(blk(i, "attn_out.weight"),    "self_attn.o_proj.weight",           stype);
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
    gguf_set_val_f32(gctx, "qwen3tts_cp.attention.layer_norm_rms_epsilon", rms_eps);
    gguf_set_val_u32(gctx, "qwen3tts_cp.rope.dimension_count", head_dim);
    gguf_set_val_u32(gctx, "qwen3tts_cp.num_code_groups", n_codebooks);
    gguf_set_val_u32(gctx, "qwen3tts_cp.codebook_0_vocab", codebook0_vocab);
    gguf_set_val_u32(gctx, "qwen3tts_cp.fine_vocab", fine_vocab);

    size_t n_tensors = 2 + 2 + 1 + static_cast<size_t>(n_codebooks) * 2 + static_cast<size_t>(n_layer) * 10;
    size_t mem = ggml_tensor_overhead() * (n_tensors + 8);
    struct ggml_init_params gip = {mem, nullptr, true};
    ggml_context* mctx = ggml_init(gip);
    if (!mctx) die("ggml_init failed for predictor");

    // small_to_mtp.weight + bias [n_embd, n_embd]
    auto* smw = bag.find2("talker.code_predictor.small_to_mtp_projection.weight",
                          "code_predictor.small_to_mtp_projection.weight");
    if (smw) add_tensor_2d(gctx, mctx, "small_to_mtp.weight", bag, *smw, stype);
    auto* smb = bag.find2("talker.code_predictor.small_to_mtp_projection.bias",
                          "code_predictor.small_to_mtp_projection.bias");
    if (smb) add_tensor_2d(gctx, mctx, "small_to_mtp.bias", bag, *smb, GGML_TYPE_F32);

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
            if (t) add_tensor_2d(gctx, mctx, gname, bag, *t, ot);
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
        add(blk(i, "attn_out.weight"),    "self_attn.o_proj.weight",           stype);
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
    bool skip_talker = false, skip_predictor = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--outdir" && i + 1 < argc) outdir = argv[++i];
        else if (a == "--quant" && i + 1 < argc) quant = argv[++i];
        else if (a == "--skip-talker")    skip_talker = true;
        else if (a == "--skip-predictor") skip_predictor = true;
        else if (a[0] != '-')             model_dir = argv[i];
    }

    if (model_dir.empty()) return usage(argv[0]);
    if (quant != "f32" && quant != "f16" && quant != "q8_0")
        die("--quant must be f32, f16, or q8_0");

    // Read config
    auto config = load_json(model_dir / "config.json");
    auto tc = config["talker_config"];
    int n_layer = tc.value("num_hidden_layers", 0);
    int cp_n_layer = tc.value("code_predictor_config", json::object())
                         .value("num_hidden_layers", 0);
    std::string variant;
    if (n_layer == 28)      variant = "1b7";
    else if (n_layer == 20) variant = "0b6";
    else                    variant = std::to_string(n_layer) + "l";

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

    return 0;
}
