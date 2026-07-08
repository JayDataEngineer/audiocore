// convert_mse2.cpp — Convert MOSS-SoundEffect-v2 HuggingFace safetensors to
// audiocore GGUFs.
//
// Produces two GGUFs:
//   1. <output>.gguf                   — DiT weights (WanAudioModel)
//   2. <output>.vae.gguf               — DAC decoder weights
//
// Usage:
//   convert_mse2 <model_dir> <output_prefix>
//
// model_dir layout:
//   transformer/config.json + diffusion_pytorch_model.safetensors
//   vae/vae_128d_48k.pth
//   scheduler/scheduler_config.json
//
// The DiT GGUF carries KV metadata for architecture + scheduler config so the
// runtime can initialise without separate config files.
//
// The VAE GGUF stores weight-norm pairs (weight_g / weight_v) and bias —
// the loader reconstitutes the effective WNConv1d / WNConvTranspose1d weights
// at bind time.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#include "ggml.h"
#include "gguf.h"
#include "nlohmann/json.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

// ── Helpers ─────────────────────────────────────────────────────────────────

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

// ── BF16 <-> F16 conversion ─────────────────────────────────────────────────

static uint16_t f32_bits_to_f16_bits(uint32_t f) {
    uint32_t sign = (f >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((f >> 23) & 0xff) - 127 + 15;
    uint32_t mant = f & 0x7fffff;
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        const uint32_t half  = 1u << (shift - 1);
        uint32_t m = mant >> shift;
        uint32_t rem = mant & ((half << 1) - 1);
        if (rem > half || (rem == half && (m & 1u))) m++;
        return (uint16_t)(sign | m);
    }
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
        uint32_t f32 = uint32_t(s[i]) << 16;
        (*out)[i] = f32_bits_to_f16_bits(f32);
    }
}

// ── Safetensors reader ──────────────────────────────────────────────────────

struct SSTensor {
    std::string name;
    std::string dtype;  // "F32", "F16", "BF16"
    std::vector<int64_t> shape;
    size_t offset = 0;
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
        SSTensor t;
        t.name = key;
        t.dtype = val["dtype"].get<std::string>();
        t.shape = val["shape"].get<std::vector<int64_t>>();
        auto offs = val["data_offsets"].get<std::vector<size_t>>();
        t.offset = data_offset + offs[0];
        t.nbytes = offs[1] - offs[0];
        int64_t n_el = 1;
        for (auto d : t.shape) n_el *= d;
        int el_size = (t.dtype == "F32") ? 4 : 2;
        if (t.dtype != "F32" && t.dtype != "F16" && t.dtype != "BF16")
            die_f("unsupported dtype %s in tensor %s", t.dtype.c_str(), t.name.c_str());
        if (static_cast<int64_t>(t.nbytes) != n_el * el_size)
            die_f("size mismatch for %s", t.name.c_str());
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

struct AddTensorBufs {
    std::vector<std::vector<uint16_t>> bufs;
};

static void add_tensor(gguf_context* gctx, ggml_context* mctx,
                       const char* gguf_name, const SafeTensorFile& stf,
                       const SSTensor& st, AddTensorBufs* bufs) {
    int n_dim = static_cast<int>(st.shape.size());
    if (n_dim < 1 || n_dim > 3)
        die_f("tensor %s has unsupported rank %d", gguf_name, n_dim);

    int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
    for (int i = 0; i < n_dim; i++)
        ne[i] = st.shape[n_dim - 1 - i];
    int64_t n_el = 1;
    for (int i = 0; i < n_dim; i++) n_el *= ne[i];

    const void* data_ptr = const_cast<char*>(stf.data) + st.offset;
    ggml_type gt;
    if (st.dtype == "BF16") {
        if (n_dim == 1) {
            // bias → F32
            std::vector<uint16_t> stash(static_cast<size_t>(n_el) * 2);
            const uint16_t* s16 = static_cast<const uint16_t*>(data_ptr);
            for (int64_t i = 0; i < n_el; i++)
                reinterpret_cast<uint32_t*>(stash.data())[i] = uint32_t(s16[i]) << 16;
            bufs->bufs.push_back(std::move(stash));
            data_ptr = bufs->bufs.back().data();
            gt = GGML_TYPE_F32;
        } else {
            bufs->bufs.emplace_back();
            bf16_buf_to_f16(data_ptr, static_cast<size_t>(n_el), &bufs->bufs.back());
            data_ptr = bufs->bufs.back().data();
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

// ── HF → Native name conversion ────────────────────────────────────────────
//
// The diffusers-format checkpoint uses Wan-AI's naming convention:
//   blocks.{i}.attn1.to_q.weight       → blocks.{i}.self_attn.q.weight
//   blocks.{i}.attn1.to_out.0.weight   → blocks.{i}.self_attn.o.weight
//   blocks.{i}.attn2.norm_k.weight     → blocks.{i}.cross_attn.norm_k.weight
//   blocks.{i}.ffn.net.0.proj.weight   → blocks.{i}.ffn.0.weight
//   blocks.{i}.norm2.weight            → blocks.{i}.norm3.weight
//   blocks.{i}.scale_shift_table       → blocks.{i}.modulation
//   condition_embedder.*               → (various)
//   proj_out.*                         → head.head.*
//   patch_embedding.weight             → patch_embedding.weight (unchanged)

struct RenameRule {
    const char* hf_prefix;
    const char* native_prefix;
};

static const RenameRule BLOCK_RULES[] = {
    {"attn1.to_q.",         "self_attn.q."},
    {"attn1.to_k.",         "self_attn.k."},
    {"attn1.to_v.",         "self_attn.v."},
    {"attn1.to_out.0.",     "self_attn.o."},
    {"attn1.norm_q.",       "self_attn.norm_q."},
    {"attn1.norm_k.",       "self_attn.norm_k."},
    {"attn2.to_q.",         "cross_attn.q."},
    {"attn2.to_k.",         "cross_attn.k."},
    {"attn2.to_v.",         "cross_attn.v."},
    {"attn2.to_out.0.",     "cross_attn.o."},
    {"attn2.norm_k.",       "cross_attn.norm_k."},
    {"attn2.norm_q.",       "cross_attn.norm_q."},
    {"ffn.net.0.proj.",     "ffn.0."},
    {"ffn.net.2.",          "ffn.2."},
    {"norm2.",              "norm3."},
};

static const RenameRule GLOBAL_RULES[] = {
    {"condition_embedder.text_embedder.linear_1.", "text_embedding.0."},
    {"condition_embedder.text_embedder.linear_2.", "text_embedding.2."},
    {"condition_embedder.time_embedder.linear_1.", "time_embedding.0."},
    {"condition_embedder.time_embedder.linear_2.", "time_embedding.2."},
    {"condition_embedder.time_proj.",              "time_projection.1."},
};

// strip "blocks.{i}." prefix, apply BLOCK_RULES, then put prefix back
static std::string convert_block_name(const std::string& hf_name, int block_idx) {
    std::string prefix = "blocks." + std::to_string(block_idx) + ".";
    std::string suffix = hf_name.substr(prefix.size());
    for (auto& r : BLOCK_RULES) {
        if (suffix.compare(0, strlen(r.hf_prefix), r.hf_prefix) == 0) {
            return prefix + r.native_prefix + suffix.substr(strlen(r.hf_prefix));
        }
    }
    // Special cases
    if (suffix == "scale_shift_table")
        return prefix + "modulation";
    return hf_name;  // fallback — pass through unchanged
}

static std::string convert_global_name(const std::string& hf_name) {
    // condition_embedder.* rules
    for (auto& r : GLOBAL_RULES) {
        if (hf_name.compare(0, strlen(r.hf_prefix), r.hf_prefix) == 0) {
            return r.native_prefix + hf_name.substr(strlen(r.hf_prefix));
        }
    }
    // Other renames
    if (hf_name == "patch_embedding.weight") return hf_name;
    if (hf_name == "patch_embedding.bias")   return hf_name;
    if (hf_name == "scale_shift_table")      return "head.modulation";
    if (hf_name.rfind("proj_out.", 0) == 0) {
        return "head.head." + hf_name.substr(strlen("proj_out."));
    }
    return hf_name;
}

static std::string hf_to_native_name(const std::string& hf_name) {
    // Check if it starts with "blocks.{i}."
    for (int i = 0; i < 30; i++) {
        std::string prefix = "blocks." + std::to_string(i) + ".";
        if (hf_name.compare(0, prefix.size(), prefix) == 0) {
            return convert_block_name(hf_name, i);
        }
    }
    return convert_global_name(hf_name);
}

// ── Write GGUF for DiT ─────────────────────────────────────────────────────

static void write_dit_gguf(const fs::path& model_dir,
                           const fs::path& output,
                           int n_layers, int dim, int ffn_dim, int n_heads,
                           int in_dim, int out_dim, int text_dim,
                           int freq_dim, double eps,
                           const std::vector<int>& patch_size,
                           bool has_image_input, const std::string& vae_type,
                           int shift, double sigma_min,
                           bool extra_one_step, int num_train_timesteps) {
    // Find safetensors
    fs::path st_path = model_dir / "transformer" / "diffusion_pytorch_model.safetensors";
    if (!fs::exists(st_path)) die_f("DiT safetensors not found at %s", st_path.c_str());
    std::printf("Loading DiT from %s ...\n", st_path.c_str());
    auto stf = load_safetensors(st_path);
    std::printf("  %zu tensors loaded\n", stf->tensors.size());

    // Setup GGUF
    gguf_context* gctx = gguf_init_empty();
    if (!gctx) die("gguf_init_empty failed");

    gguf_set_val_str(gctx, "general.architecture", "moss_sfx_v2");
    gguf_set_val_u32(gctx, "general.alignment", 32);
    gguf_set_val_u32(gctx, "moss_sfx_v2.n_layers", n_layers);
    gguf_set_val_u32(gctx, "moss_sfx_v2.dim", dim);
    gguf_set_val_u32(gctx, "moss_sfx_v2.ffn_dim", ffn_dim);
    gguf_set_val_u32(gctx, "moss_sfx_v2.n_heads", n_heads);
    gguf_set_val_u32(gctx, "moss_sfx_v2.in_dim", in_dim);
    gguf_set_val_u32(gctx, "moss_sfx_v2.out_dim", out_dim);
    gguf_set_val_u32(gctx, "moss_sfx_v2.text_dim", text_dim);
    gguf_set_val_u32(gctx, "moss_sfx_v2.freq_dim", freq_dim);
    gguf_set_val_f32(gctx, "moss_sfx_v2.eps", eps);
    gguf_set_val_u32(gctx, "moss_sfx_v2.patch_size", patch_size[0]);
    gguf_set_val_bool(gctx, "moss_sfx_v2.has_image_input", has_image_input);
    gguf_set_val_str(gctx, "moss_sfx_v2.vae_type", vae_type.c_str());
    // Scheduler config
    gguf_set_val_u32(gctx, "moss_sfx_v2.scheduler.shift", shift);
    gguf_set_val_f32(gctx, "moss_sfx_v2.scheduler.sigma_min", sigma_min);
    gguf_set_val_bool(gctx, "moss_sfx_v2.scheduler.extra_one_step", extra_one_step);
    gguf_set_val_u32(gctx, "moss_sfx_v2.scheduler.num_train_timesteps", num_train_timesteps);

    size_t n_tensors = stf->tensors.size();
    size_t mem = ggml_tensor_overhead() * (n_tensors + 16);
    ggml_context* mctx = ggml_init({mem, nullptr, true});
    if (!mctx) die("ggml_init failed");

    std::vector<const SSTensor*> sorted;
    sorted.reserve(n_tensors);
    for (auto& t : stf->tensors) sorted.push_back(&t);
    std::sort(sorted.begin(), sorted.end(),
              [](const SSTensor* a, const SSTensor* b) { return a->name < b->name; });

    AddTensorBufs bufs;
    size_t added = 0, errors = 0;
    for (auto* t : sorted) {
        std::string native_name = hf_to_native_name(t->name);
        if (native_name.empty()) { errors++; continue; }
        // Prefix with "moss_sfx_v2."
        std::string out_name = "moss_sfx_v2." + native_name;
        add_tensor(gctx, mctx, out_name.c_str(), *stf, *t, &bufs);
        added++;
        if (added <= 5 || added == n_tensors || added == n_tensors - 1) {
            std::printf("  [%4zu] %-60s  shape=[", added, out_name.c_str());
            for (size_t i = 0; i < t->shape.size(); i++)
                std::printf("%s%ld", (i ? "," : ""), t->shape[i]);
            std::printf("]\n");
        } else if (added == 6) {
            std::printf("  ...\n");
        }
    }

    std::printf("Writing %s (%zu tensors) ...\n", output.c_str(), added);
    if (!gguf_write_to_file(gctx, output.c_str(), false))
        die("gguf_write_to_file failed");
    std::printf("  done (%zu MB)\n", fs::file_size(output) / 1048576);

    gguf_free(gctx);
    ggml_free(mctx);
}

// ── Read PTH checkpoint ────────────────────────────────────────────────────
//
// The DAC VAE checkpoint is a torch .pth file. We have a minimal parser
// that reads the raw Pickle format using Python interop. For now we just
// print the keys and error out suggesting the user extract them separately.
//
// TODO: port a tiny pickle reader or output the VAE weights via a companion
// Python script.

static void dump_vae_state_dict_keys(const fs::path& model_dir) {
    // The VAE checkpoint is a torch .pth. We can't read it from C++ easily.
    // Instead, we rely on the Python tensor-dump script which already saved
    // vae_state_dict_keys.json.
    fs::path json_path = model_dir / ".." / "dump_mse2_tensors" / "vae_state_dict_keys.json";
    if (fs::exists(json_path)) {
        auto j = load_json(json_path);
        std::printf("VAE state dict keys (%zu entries):\n", j.size());
        int count = 0;
        for (auto& [k, v] : j.items()) {
            if (count < 10) std::printf("  %s\n", k.c_str());
            count++;
            if (count == 10) { std::printf("  ... (%d more)\n", (int)j.size() - 10); break; }
        }
    }
}

// ── Main ────────────────────────────────────────────────────────────────────

static int usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s <model_dir> <output_prefix>\n"
        "  Converts MOSS-SoundEffect-v2 HuggingFace checkpoint to audiocore GGUFs.\n"
        "  Produces:\n"
        "    <output_prefix>.gguf       — DiT (WanAudioModel)\n"
        "    <output_prefix>.vae.gguf   — DAC VAE decoder\n",
        argv0);
    return 2;
}

int main(int argc, char** argv) {
    if (argc < 3) return usage(argv[0]);
    fs::path model_dir = argv[1];
    fs::path output_prefix = argv[2];

    // Load configs
    auto dit_cfg = load_json(model_dir / "transformer" / "config.json");
    auto sched_cfg = load_json(model_dir / "scheduler" / "scheduler_config.json");

    int n_layers     = dit_cfg.value("num_layers", 30);
    int dim          = dit_cfg.value("dim", 1536);
    int ffn_dim      = dit_cfg.value("ffn_dim", 8960);
    int n_heads      = dit_cfg.value("num_heads", 12);
    int in_dim       = dit_cfg.value("in_dim", 128);
    int out_dim      = dit_cfg.value("out_dim", 128);
    int text_dim     = dit_cfg.value("text_dim", 2048);
    int freq_dim     = dit_cfg.value("freq_dim", 256);
    double eps       = dit_cfg.value("eps", 1e-6);
    auto patch_size  = dit_cfg.value("patch_size", std::vector<int>{1});
    bool has_img     = dit_cfg.value("has_image_input", false);
    std::string vae_type = dit_cfg.value("vae_type", "dac");

    int shift             = sched_cfg.value("shift", 5);
    double sigma_min      = sched_cfg.value("sigma_min", 0.0);
    bool extra_one_step   = sched_cfg.value("extra_one_step", true);
    int num_train_ts      = sched_cfg.value("num_train_timesteps", 1000);

    std::printf("MOSS-SoundEffect-v2 DiT config:\n");
    std::printf("  layers=%d dim=%d ffn_dim=%d heads=%d in=%d out=%d text=%d\n",
                n_layers, dim, ffn_dim, n_heads, in_dim, out_dim, text_dim);
    std::printf("  patch_size=%d vae_type=%s\n", patch_size[0], vae_type.c_str());

    // Write DiT GGUF
    fs::path dit_output = output_prefix;
    dit_output += ".gguf";
    write_dit_gguf(model_dir, dit_output,
                   n_layers, dim, ffn_dim, n_heads,
                   in_dim, out_dim, text_dim, freq_dim, eps,
                   patch_size, has_img, vae_type,
                   shift, sigma_min, extra_one_step, num_train_ts);

    // Write VAE GGUF (TODO)
    fs::path vae_output = output_prefix;
    vae_output += ".vae.gguf";
    std::printf("\nVAE GGUF not yet written — use the companion Python script.\n");
    dump_vae_state_dict_keys(model_dir);

    return 0;
}
