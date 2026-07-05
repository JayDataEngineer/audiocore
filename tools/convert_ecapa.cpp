// convert_ecapa.cpp — Convert ECAPA-TDNN speaker-encoder HuggingFace
// safetensors to a standalone GGUF for audiocore's Qwen3TtsSpeakerEncoder.
//
// Source model class: EcapaTdnnSpeakerEncoder (marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B
// and equivalent checkpoints — 76 tensors, ~23 MB BF16).
//
// The converter adds a `speaker.` prefix to every tensor name so that the
// existing `bind()` in src/models/qwen3_tts/speaker_encoder.cpp resolves them
// unchanged. Output GGUF is self-describing: KV metadata captures enc_dim,
// mel_dim, res2net_scale, se_channels, attention_channels, sample_rate so the
// runtime can sanity-check the architecture before binding.
//
// Usage:
//   convert_ecapa <model_dir> <output.gguf>
//   convert_ecapa <model_dir> <output.gguf> --filter-prefix speaker_encoder.
//       # Only emit tensors whose name starts with the prefix; strip the
//       # prefix before adding `speaker.` Used when the source safetensors
//       # bundles the ECAPA weights inside a larger talker checkpoint
//       # (e.g. Qwen3-TTS-1.7B-Base/model.safetensors, 3.6 GB — the
//       # canonical extraction path for the 2048-dim encoder).
//   convert_ecapa <model_dir> <output.gguf> --strip-prefix
//       # Source tensors are ALREADY named `speaker.*` — emit them verbatim
//       # WITHOUT adding another `speaker.` prefix. (Rare; only useful for
//       # standalone ECAPA checkpoints like marksverdhei's that already use
//       # the runtime-expected namespacing.)
//
// `model_dir` must contain config.json and model.safetensors.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <map>
#include <memory>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#include "ggml.h"
#include "gguf.h"
#include "nlohmann/json.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

// ── BF16 → F16 conversion ──────────────────────────────────────────────────
//
// The marksverdhei checkpoint ships BF16 weights, but ggml's CPU conv_1d
// kernel (used by the ECAPA encoder's same_conv1d_) requires F16 src0.
// Qwen3-TTS talker GGUFs in audiocore are also F16, so we normalize to F16
// here. Conversion goes BF16 → FP32 (lossless: BF16 is the upper 16 bits of
// FP32) → F16 (standard half-precision rounding, round-to-nearest-even).
static uint16_t f32_bits_to_f16_bits(uint32_t f) {
    uint32_t sign = (f >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((f >> 23) & 0xff) - 127 + 15;
    uint32_t mant = f & 0x7fffff;

    if (exp >= 0x1f) {                       // overflow → Inf (or NaN if mant)
        return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
    }
    if (exp <= 0) {                          // zero or subnormal
        if (exp < -10) return (uint16_t)sign;  // underflow → signed zero
        mant |= 0x800000u;                     // implicit leading 1
        const uint32_t shift = (uint32_t)(14 - exp);
        const uint32_t half  = 1u << (shift - 1);
        uint32_t m = mant >> shift;
        uint32_t rem = mant & ((half << 1) - 1);
        if (rem > half || (rem == half && (m & 1u))) m++;  // RTE
        return (uint16_t)(sign | m);
    }
    // Normal: round 23-bit mantissa to 10 bits (RTE).
    const uint32_t shift = 13;
    const uint32_t half  = 1u << (shift - 1);
    uint32_t m = mant >> shift;
    uint32_t rem = mant & ((half << 1) - 1);
    if (rem > half || (rem == half && (m & 1u))) m++;
    if (m > 0x3ffu) { m = 0; exp++; }
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | (uint32_t(exp) << 10) | m);
}

static void bf16_buf_to_f16(const void* src, size_t n_elements,
                            std::vector<uint16_t>* out) {
    out->resize(n_elements);
    const uint16_t* s = static_cast<const uint16_t*>(src);
    for (size_t i = 0; i < n_elements; i++) {
        uint32_t f32 = uint32_t(s[i]) << 16;              // BF16 → FP32 (lossless)
        (*out)[i] = f32_bits_to_f16_bits(f32);            // FP32 → F16 (RTE)
    }
}

[[noreturn]] static void die(const char* msg) {
    std::fprintf(stderr, "error: %s\n", msg);
    std::exit(1);
}

template <typename... Args>
[[noreturn]] static void die_f(const char* fmt, const Args&... args) {
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

// ── Safetensors reader (single-file, mmap) ──────────────────────────────────

struct SSTensor {
    std::string name;
    std::string dtype;            // "F32", "F16", "BF16"
    std::vector<int64_t> shape;
    size_t offset = 0;            // byte offset within the mmap
    size_t nbytes = 0;
};

struct SafeTensorFile {
    char* data = nullptr;
    size_t size = 0;
    int fd = -1;
    std::vector<SSTensor> tensors;

    ~SafeTensorFile() {
        if (data && data != MAP_FAILED) munmap(data, size);
        if (fd >= 0) close(fd);
    }

    const SSTensor* find(const std::string& name) const {
        for (auto& t : tensors) if (t.name == name) return &t;
        return nullptr;
    }
};

static std::unique_ptr<SafeTensorFile> load_safetensors(const fs::path& path) {
    auto out = std::make_unique<SafeTensorFile>();
    out->fd = ::open(path.c_str(), O_RDONLY);
    if (out->fd < 0) die_f("cannot open %s", path.c_str());
    out->size = static_cast<size_t>(::lseek(out->fd, 0, SEEK_END));
    out->data = static_cast<char*>(
        ::mmap(nullptr, out->size, PROT_READ, MAP_PRIVATE, out->fd, 0));
    if (out->data == MAP_FAILED) die("mmap failed");

    uint64_t header_len = 0;
    std::memcpy(&header_len, out->data, sizeof(header_len));
    size_t data_offset = sizeof(uint64_t) + static_cast<size_t>(header_len);
    std::string hdr_str(out->data + sizeof(uint64_t), static_cast<size_t>(header_len));
    json hdr = json::parse(hdr_str);

    for (auto& [key, val] : hdr.items()) {
        if (key == "__metadata__") continue;
        if (!val.is_object() || !val.contains("dtype") ||
            !val.contains("shape") || !val.contains("data_offsets"))
            continue;
        SSTensor t;
        t.name = key;
        t.dtype = val["dtype"].get<std::string>();
        t.shape = val["shape"].get<std::vector<int64_t>>();
        auto offs = val["data_offsets"].get<std::vector<size_t>>();
        t.offset = data_offset + offs[0];
        t.nbytes = offs[1] - offs[0];

        // Sanity check
        int64_t n_el = 1;
        for (auto d : t.shape) n_el *= d;
        int el_size = (t.dtype == "F32") ? 4 : 2;  // F16 or BF16
        if (t.dtype != "F32" && t.dtype != "F16" && t.dtype != "BF16")
            die_f("unsupported dtype %s in tensor %s", t.dtype.c_str(), t.name.c_str());
        if (static_cast<int64_t>(t.nbytes) != n_el * el_size)
            die_f("size mismatch for %s: %zu bytes vs %ld elements * %d",
                  t.name.c_str(), t.nbytes, n_el, el_size);

        out->tensors.push_back(std::move(t));
    }
    return out;
}

static ggml_type dtype_to_ggml(const std::string& dt) {
    if (dt == "F32")  return GGML_TYPE_F32;
    if (dt == "F16")  return GGML_TYPE_F16;
    if (dt == "BF16") return GGML_TYPE_BF16;
    return GGML_TYPE_COUNT;
}

// Add a tensor to the GGUF, preserving shape and dtype from the safetensors.
// ggml dims are [innermost, ..., outermost] (reverse of numpy/PyTorch).
//
// BF16 source tensors are converted to F16 on the fly: ggml's CPU conv_1d
// requires F16 src0, and the ECAPA encoder's same_conv1d_() goes through
// that path. We allocate a scratch buffer owned by `convert_bufs` so the
// pointer stays valid through gguf_write_to_file.
static void add_tensor(gguf_context* gctx, ggml_context* mctx,
                       const char* gguf_name, const SafeTensorFile& stf,
                       const SSTensor& st,
                       std::vector<std::vector<uint16_t>>* convert_bufs) {
    int n_dim = static_cast<int>(st.shape.size());
    if (n_dim < 1 || n_dim > 3)
        die_f("tensor %s has unsupported rank %d", gguf_name, n_dim);

    // ggml ne[]: ne[0] is the innermost (fastest-varying) dimension.
    // PyTorch conv1d weight [O, I, K] should map to ggml ne = [K, I, O].
    // (matches what CrispASR writes and what same_conv1d_() reads.)
    int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
    for (int i = 0; i < n_dim; i++) {
        ne[i] = st.shape[n_dim - 1 - i];
    }
    int64_t n_el = 1;
    for (int i = 0; i < n_dim; i++) n_el *= ne[i];

    const void* data_ptr = const_cast<char*>(stf.data) + st.offset;
    ggml_type gt;
    if (st.dtype == "BF16") {
        if (n_dim == 1) {
            // Rank-1 tensors are biases — added to F32 conv outputs, so the
            // CPU backend needs them as F32 (no mixed F32+F16 binary ops).
            // Expand BF16 → F32 by padding each u16 to a u32 (lossless).
            std::vector<uint32_t> f32_buf(static_cast<size_t>(n_el));
            const uint16_t* s16 = static_cast<const uint16_t*>(data_ptr);
            for (int64_t i = 0; i < n_el; i++) {
                f32_buf[size_t(i)] = uint32_t(s16[size_t(i)]) << 16;
            }
            // Stash as uint16 vector of doubled size, then alias back. We
            // keep the buffer alive in convert_bufs via a reinterpret trick:
            // store the bytes in a uint16_t vector of 2x length.
            std::vector<uint16_t> stash(static_cast<size_t>(n_el) * 2);
            std::memcpy(stash.data(), f32_buf.data(), stash.size() * 2);
            convert_bufs->push_back(std::move(stash));
            data_ptr = convert_bufs->back().data();
            gt = GGML_TYPE_F32;
        } else {
            // Rank-2/3 tensors are conv/mul-mat weights. ggml's CPU conv_1d
            // requires F16 src0 — convert BF16 → F16 (RTE).
            convert_bufs->emplace_back();
            bf16_buf_to_f16(data_ptr, static_cast<size_t>(n_el), &convert_bufs->back());
            data_ptr = convert_bufs->back().data();
            gt = GGML_TYPE_F16;
        }
    } else {
        gt = dtype_to_ggml(st.dtype);
    }

    ggml_tensor* t = ggml_new_tensor(mctx, gt, n_dim, ne);
    if (!t) die_f("ggml_new_tensor failed for %s", gguf_name);
    t->data = const_cast<void*>(data_ptr);
    ggml_set_name(t, gguf_name);
    gguf_add_tensor(gctx, t);
}

// ── Main ────────────────────────────────────────────────────────────────────

static int usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s <model_dir> <output.gguf> [--strip-prefix] [--filter-prefix PREFIX]\n"
        "  Convert ECAPA-TDNN speaker-encoder HuggingFace safetensors to a\n"
        "  standalone audiocore GGUF. The runtime expects tensors named\n"
        "  `speaker.*` (see Qwen3TtsSpeakerEncoder::bind). By default this\n"
        "  converter PREPENDS `speaker.` to every source tensor name.\n"
        "  --filter-prefix PREFIX: only emit tensors whose name starts with\n"
        "      PREFIX; strip PREFIX, THEN add `speaker.`. Use to extract\n"
        "      the ECAPA weights from a bundled talker checkpoint, e.g.\n"
        "        convert_ecapa Qwen3-TTS-1.7B-Base out.gguf \\\n"
        "            --filter-prefix speaker_encoder.\n"
        "      turns `speaker_encoder.asp.conv.bias` → `speaker.asp.conv.bias`.\n"
        "  --strip-prefix: emit source tensor names VERBATIM (no `speaker.`\n"
        "      added). Only use when the source already names tensors\n"
        "      `speaker.*` (rare; standalone ECAPA checkpoints).\n",
        argv0);
    return 2;
}

int main(int argc, char** argv) {
    bool strip_prefix = false;
    std::string filter_prefix;  // e.g. "speaker_encoder." — only include matching tensors
    std::vector<std::string> positional;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--strip-prefix") strip_prefix = true;
        else if (a == "--filter-prefix") {
            if (++i >= argc) die("--filter-prefix needs an argument");
            filter_prefix = argv[i];
        }
        else if (a == "-h" || a == "--help") return usage(argv[0]);
        else positional.push_back(a);
    }
    if (positional.size() < 2) return usage(argv[0]);

    fs::path model_dir = positional[0];
    fs::path output    = positional[1];

    auto config = load_json(model_dir / "config.json");

    // Architecture constants from the config. The standalone marksverdhei
    // checkpoint has these at the top level; the bundled 0.6B-Base config
    // nests them under `speaker_encoder_config` and only carries enc_dim +
    // sample_rate (the ECAPA-TDNN architecture is fixed across Qwen3-TTS
    // variants). Fall back to defaults that match the 1.7B reference for any
    // field the source config doesn't surface.
    json spk_cfg = config.value("speaker_encoder_config", json::object());
    uint32_t enc_dim           = spk_cfg.value("enc_dim", config.value("enc_dim", 2048));
    uint32_t mel_dim           = spk_cfg.value("mel_dim", config.value("mel_dim", 128));
    uint32_t sample_rate       = spk_cfg.value("sample_rate", config.value("sample_rate", 24000));
    uint32_t res2net_scale     = spk_cfg.value("enc_res2net_scale",
                                               config.value("enc_res2net_scale", 8));
    uint32_t se_channels      = spk_cfg.value("enc_se_channels",
                                              config.value("enc_se_channels", 128));
    uint32_t attention_channels = spk_cfg.value("enc_attention_channels",
                                                config.value("enc_attention_channels", 128));
    auto enc_channels  = spk_cfg.value("enc_channels",  config.value("enc_channels",
                                       std::vector<uint32_t>{512,512,512,512,1536}));
    auto enc_kernels   = spk_cfg.value("enc_kernel_sizes", config.value("enc_kernel_sizes",
                                       std::vector<uint32_t>{5,3,3,3,1}));
    auto enc_dilations = spk_cfg.value("enc_dilations", config.value("enc_dilations",
                                       std::vector<uint32_t>{1,2,3,4,1}));

    std::printf("ECAPA-TDNN config: enc_dim=%u mel_dim=%u sr=%u res2net=%u se=%u attn=%u\n",
                enc_dim, mel_dim, sample_rate, res2net_scale, se_channels, attention_channels);
    std::printf("  enc_channels : [%u, %u, %u, %u, %u]\n",
                enc_channels[0], enc_channels[1], enc_channels[2], enc_channels[3], enc_channels[4]);
    std::printf("  enc_kernels  : [%u, %u, %u, %u, %u]\n",
                enc_kernels[0], enc_kernels[1], enc_kernels[2], enc_kernels[3], enc_kernels[4]);
    std::printf("  enc_dilations: [%u, %u, %u, %u, %u]\n",
                enc_dilations[0], enc_dilations[1], enc_dilations[2], enc_dilations[3], enc_dilations[4]);

    // Find safetensors file (single-file convention for this model)
    fs::path st_path;
    for (auto& e : fs::directory_iterator(model_dir)) {
        if (e.path().extension() == ".safetensors") {
            st_path = e.path();
            break;
        }
    }
    if (st_path.empty()) die("no .safetensors file in model_dir");
    std::printf("Loading %s ...\n", st_path.c_str());
    auto stf = load_safetensors(st_path);
    std::printf("  %zu tensors loaded\n", stf->tensors.size());

    // Set up GGUF writer
    gguf_context* gctx = gguf_init_empty();
    if (!gctx) die("gguf_init_empty failed");

    gguf_set_val_str(gctx, "general.architecture", "qwen3tts_spk");
    gguf_set_val_u32(gctx, "general.alignment", 32);
    gguf_set_val_u32(gctx, "qwen3tts_spk.enc_dim", enc_dim);
    gguf_set_val_u32(gctx, "qwen3tts_spk.mel_dim", mel_dim);
    gguf_set_val_u32(gctx, "qwen3tts_spk.sample_rate", sample_rate);
    gguf_set_val_u32(gctx, "qwen3tts_spk.res2net_scale", res2net_scale);
    gguf_set_val_u32(gctx, "qwen3tts_spk.se_channels", se_channels);
    gguf_set_val_u32(gctx, "qwen3tts_spk.attention_channels", attention_channels);
    if (enc_channels.size() == 5) {
        gguf_set_arr_data(gctx, "qwen3tts_spk.enc_channels",
                          GGUF_TYPE_UINT32, enc_channels.data(), 5);
        gguf_set_arr_data(gctx, "qwen3tts_spk.enc_kernel_sizes",
                          GGUF_TYPE_UINT32, enc_kernels.data(), 5);
        gguf_set_arr_data(gctx, "qwen3tts_spk.enc_dilations",
                          GGUF_TYPE_UINT32, enc_dilations.data(), 5);
    }

    // ggml context for tensor metadata — generous overhead, model is tiny
    size_t n_tensors = stf->tensors.size();
    size_t mem = ggml_tensor_overhead() * (n_tensors + 16);
    struct ggml_init_params gip = {mem, nullptr, true};
    ggml_context* mctx = ggml_init(gip);
    if (!mctx) die("ggml_init failed");

    // Write every tensor with `speaker.` prefix (unless --strip-prefix).
    // BF16 source tensors are converted to F16 in scratch buffers that live
    // in `convert_bufs` until gguf_write_to_file completes.
    std::vector<std::vector<uint16_t>> convert_bufs;
    convert_bufs.reserve(n_tensors);

    std::vector<const SSTensor*> sorted;
    sorted.reserve(n_tensors);
    for (auto& t : stf->tensors) sorted.push_back(&t);
    std::sort(sorted.begin(), sorted.end(),
              [](const SSTensor* a, const SSTensor* b) { return a->name < b->name; });

    size_t added = 0, skipped = 0, filtered = 0;
    for (auto* t : sorted) {
        if (t->name == "__metadata__") continue;
        std::string base_name = t->name;
        // Apply --filter-prefix: only emit tensors whose name starts with the
        // prefix; strip it before renaming so the output namespacing matches
        // what Qwen3TtsSpeakerEncoder::bind() expects (speaker.*).
        if (!filter_prefix.empty()) {
            if (base_name.size() < filter_prefix.size() ||
                base_name.compare(0, filter_prefix.size(), filter_prefix) != 0) {
                ++filtered;
                continue;
            }
            base_name = base_name.substr(filter_prefix.size());
        }
        std::string out_name = strip_prefix ? base_name : "speaker." + base_name;
        add_tensor(gctx, mctx, out_name.c_str(), *stf, *t, &convert_bufs);
        ++added;
        if (added <= 5 || added == n_tensors) {
            std::printf("  [%4zu] %-50s  shape=[", added, out_name.c_str());
            for (size_t i = 0; i < t->shape.size(); i++)
                std::printf("%s%ld", (i ? "," : ""), t->shape[i]);
            std::printf("] dtype=%s→%s\n", t->dtype.c_str(),
                        (t->dtype == "BF16") ? "F16" : t->dtype.c_str());
        } else if (added == 6) {
            std::printf("  ...\n");
        }
    }

    std::printf("Writing %s (%zu tensors, %zu filtered, %zu skipped) ...\n",
                output.c_str(), added, filtered, skipped);
    if (!gguf_write_to_file(gctx, output.c_str(), /*only_meta=*/false))
        die("gguf_write_to_file failed");
    std::printf("done (%zu MB)\n", fs::file_size(output) / 1048576);

    gguf_free(gctx);
    ggml_free(mctx);
    return 0;
}
