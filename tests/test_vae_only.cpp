// Focused VAE parity test: compares C++ VAE output with Python reference.
// Uses both decode (sub-graph) and decode_traced (per-op) paths.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <filesystem>

#include "ggml.h"
#include "gguf.h"
#include "audiocore/models/moss_sfx_v2/vae_runner.h"

namespace fs = std::filesystem;

static float bf16_to_f32(uint16_t b) {
    uint32_t f = (uint32_t)b << 16;
    float r; memcpy(&r, &f, 4); return r;
}

static ggml_context* load_gguf(const char* path) {
    ggml_context* ctx = nullptr;
    gguf_init_params p = { false, &ctx };
    gguf_context* g = gguf_init_from_file(path, p);
    if (!g) return nullptr;
    gguf_free(g);
    return ctx;
}

struct Shape { int64_t ne[4] = {1,1,1,1}; int ndim = 0; };
static Shape read_shape(const std::string& path) {
    Shape s; FILE* f = fopen(path.c_str(), "r"); if (!f) return s;
    int64_t v; while (fscanf(f, "%ld", &v) == 1) s.ne[s.ndim++] = v;
    fclose(f); for (int i = s.ndim; i < 4; i++) s.ne[i] = 1;
    return s;
}

struct RefTensor {
    std::vector<float> data; Shape shape;
    int64_t n_el() const { int64_t n=1; for(int i=0;i<shape.ndim;i++) n*=shape.ne[i]; return n; }
};

static RefTensor load_ref(const char* dir, const char* name) {
    RefTensor t;
    std::string base = std::string(dir) + "/" + name;
    t.shape = read_shape(base + ".shape");
    int64_t n = t.n_el(); if (n == 0) return t;
    FILE* f = fopen((base + ".bin").c_str(), "rb"); if (!f) return t;
    t.data.resize(n);
    fread(t.data.data(), 4, n, f); fclose(f);
    return t;
}

// Convert PyTorch [B,C,T] to time-major [T,C]
static std::vector<float> to_time_major(const RefTensor& ref, int64_t& T, int64_t& C) {
    if (ref.shape.ndim >= 3) { C = ref.shape.ne[1]; T = ref.shape.ne[2]; }
    else if (ref.shape.ndim == 2) { C = ref.shape.ne[0]; T = ref.shape.ne[1]; }
    else { C = 1; T = ref.shape.ne[0]; }
    std::vector<float> flat((size_t)T * C);
    for (int64_t t = 0; t < T; t++)
        for (int64_t c = 0; c < C; c++)
            flat[(size_t)t * C + c] = ref.data[(size_t)c * T + t];
    return flat;
}

int main(int argc, char** argv) {
    using namespace audiocore::moss_sfx_v2;
    
    const char* vae_gguf = argv[1];
    const char* dump_dir = argv[2];
    
    fprintf(stderr, "Loading VAE GGUF: %s\n", vae_gguf);
    ggml_context* vae_ctx = load_gguf(vae_gguf);
    if (!vae_ctx) { fprintf(stderr, "Failed to load VAE GGUF\n"); return 1; }
    
    // Load Python reference VAE input (post_quant_conv output = pre-conv_in)
    auto ref_pqc = load_ref(dump_dir, "post_unpatchify");
    if (ref_pqc.data.empty()) {
        // Fallback: use post_unpatchify (pre-post_quant_conv)
        ref_pqc = load_ref(dump_dir, "post_unpatchify");
        if (ref_pqc.data.empty()) {
            fprintf(stderr, "No VAE input reference found\n"); return 1;
        }
    }
    
    int64_t T_latent, C_latent;
    auto z = to_time_major(ref_pqc, T_latent, C_latent);
    fprintf(stderr, "VAE input: T=%ld C=%ld\n", (long)T_latent, (long)C_latent);
    
    VAEConfig vcfg;
    vcfg.latent_dim = 128; vcfg.decoder_dim = 2048;
    vcfg.hop_length = 960; vcfg.sample_rate = 48000;
    vcfg.continuous = true;
    
    VAERunner runner(vae_ctx, vcfg);
    
    // ── Test 1: decode_traced (per-op path with ggml_conv_transpose_1d) ──
    fprintf(stderr, "\n=== decode_traced (per-op path) ===\n");
    {
        std::vector<float> out((size_t)T_latent * vcfg.hop_length * 2);
        VAERunner::Trace trace;
        std::string err;
        bool ok = runner.decode_traced(z.data(), 1, (int32_t)T_latent,
                                        out.data(), &trace, &err);
        if (!ok) { fprintf(stderr, "decode_traced failed: %s\n", err.c_str()); }
        else {
            fprintf(stderr, "decode_traced output: %zu samples\n", trace.vae_final.size());
            // Compare with vae_dec_7
            auto ref = load_ref(dump_dir, "vae_dec_7");
            if (!ref.data.empty()) {
                int64_t rT, rC;
                auto rflat = to_time_major(ref, rT, rC);
                if (trace.vae_final.size() == rflat.size()) {
                    double mae = 0, rms_g = 0, rms_r = 0;
                    for (size_t i = 0; i < rflat.size(); i++) {
                        mae = std::max(mae, (double)std::fabs(trace.vae_final[i] - rflat[i]));
                        rms_g += (double)trace.vae_final[i]*trace.vae_final[i];
                        rms_r += (double)rflat[i]*rflat[i];
                    }
                    rms_g = std::sqrt(rms_g / rflat.size());
                    rms_r = std::sqrt(rms_r / rflat.size());
                    fprintf(stderr, "  final: MAE=%.4f RMS_cpp=%.4f RMS_py=%.4f\n", mae, rms_g, rms_r);
                    fprintf(stderr, "  first 5 cpp: "); 
                    for(int i=0;i<5;i++) fprintf(stderr,"%.4f ", trace.vae_final[i]);
                    fprintf(stderr, "\n  first 5 py:  ");
                    for(int i=0;i<5;i++) fprintf(stderr,"%.4f ", rflat[i]);
                    fprintf(stderr, "\n");
                    
                    // Dump for spectral analysis
                    FILE* f = fopen("/tmp/vae_traced_cpp.f32", "wb");
                    if (f) { fwrite(trace.vae_final.data(), 4, trace.vae_final.size(), f); fclose(f); }
                    f = fopen("/tmp/vae_traced_py.f32", "wb");
                    if (f) { fwrite(rflat.data(), 4, rflat.size(), f); fclose(f); }
                } else {
                    fprintf(stderr, "  size mismatch: cpp=%zu py=%zu\n",
                            trace.vae_final.size(), rflat.size());
                }
            }
            
            // Compare intermediate layers
            for (int i = 0; i <= 7; i++) {
                char name[32]; snprintf(name, 32, "vae_dec_%d", i);
                auto ref = load_ref(dump_dir, name);
                if (ref.data.empty()) continue;
                int64_t rT, rC;
                auto rflat = to_time_major(ref, rT, rC);
                const auto& got = trace.vae_dec[i];
                if (got.empty()) continue;
                if (got.size() != rflat.size()) {
                    fprintf(stderr, "  [SIZE] %s: cpp=%zu py=%zu\n", name, got.size(), rflat.size());
                    continue;
                }
                double mae = 0;
                for (size_t j = 0; j < rflat.size(); j++)
                    mae = std::max(mae, (double)std::fabs(got[j] - rflat[j]));
                fprintf(stderr, "  %s: MAE=%.4f\n", name, mae);
            }
        }
    }
    
    // ── Test 2: decode (sub-graph path with col2im_1d) ──
    fprintf(stderr, "\n=== decode (sub-graph path) ===\n");
    {
        std::vector<float> out((size_t)T_latent * vcfg.hop_length * 2);
        std::string err;
        bool ok = runner.decode(z.data(), 1, (int32_t)T_latent, out.data(), &err);
        if (!ok) { fprintf(stderr, "decode failed: %s\n", err.c_str()); }
        else {
            int64_t T_audio = T_latent * vcfg.hop_length;
            fprintf(stderr, "decode output: %ld samples\n", (long)T_audio);
            auto ref = load_ref(dump_dir, "vae_dec_7");
            if (!ref.data.empty()) {
                int64_t rT, rC;
                auto rflat = to_time_major(ref, rT, rC);
                fprintf(stderr, "  ref: T=%ld C=%ld (n=%zu)\n", (long)rT, (long)rC, rflat.size());
                fprintf(stderr, "  cpp: T=%ld (n=%zu)\n", (long)T_audio, (size_t)T_audio);
                
                // Compare up to min length
                size_t n = std::min((size_t)T_audio, rflat.size());
                double mae = 0, rms_g = 0, rms_r = 0;
                for (size_t i = 0; i < n; i++) {
                    mae = std::max(mae, (double)std::fabs(out[i] - rflat[i]));
                    rms_g += (double)out[i]*out[i];
                    rms_r += (double)rflat[i]*rflat[i];
                }
                rms_g = std::sqrt(rms_g / n);
                rms_r = std::sqrt(rms_r / n);
                fprintf(stderr, "  final: MAE=%.4f RMS_cpp=%.4f RMS_py=%.4f\n", mae, rms_g, rms_r);
                fprintf(stderr, "  first 5 cpp: ");
                for(size_t i=0;i<5;i++) fprintf(stderr,"%.4f ", out[i]);
                fprintf(stderr, "\n  first 5 py:  ");
                for(size_t i=0;i<5;i++) fprintf(stderr,"%.4f ", rflat[i]);
                fprintf(stderr, "\n");
                
                FILE* f = fopen("/tmp/vae_subgraph_cpp.f32", "wb");
                if (f) { fwrite(out.data(), 4, n, f); fclose(f); }
            }
        }
    }
    
    ggml_free(vae_ctx);
    return 0;
}
