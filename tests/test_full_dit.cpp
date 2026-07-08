// Full DiT parity test: feeds reference inputs to DiTRunner, compares with post_head.bin
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#include "ggml.h"
#include "gguf.h"

#include "audiocore/models/moss_sfx_v2/dit_runner.h"

static ggml_context* load_gguf(const char* path) {
    ggml_context* ctx = nullptr;
    gguf_init_params params = { false, &ctx };
    gguf_context* gctx = gguf_init_from_file(path, params);
    if (!gctx) {
        fprintf(stderr, "Failed to load GGUF: %s\n", path);
        return nullptr;
    }
    gguf_free(gctx);
    return ctx;
}

static std::vector<float> load_f32(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<float> data(sz / sizeof(float));
    if (fread(data.data(), sizeof(float), data.size(), f) != data.size()) {}
    fclose(f);
    return data;
}

int main() {
    using namespace audiocore::moss_sfx_v2;
    
    const char* gguf_path = "weights/moss_sfx_v2/mse2-dit.gguf";
    const char* dump_dir = "dump_mse2_tensors";
    
    fprintf(stderr, "Loading GGUF...\n");
    ggml_context* ext_ctx = load_gguf(gguf_path);
    if (!ext_ctx) return 1;
    fprintf(stderr, "Loaded GGUF\n");
    
    // Load reference inputs
    auto dummy_latents = load_f32((std::string(dump_dir) + "/dummy_latents.bin").c_str());
    auto dummy_context = load_f32((std::string(dump_dir) + "/dummy_context.bin").c_str());
    auto ref_post_head = load_f32((std::string(dump_dir) + "/post_head.bin").c_str());
    auto ref_pre_head = load_f32((std::string(dump_dir) + "/pre_head.bin").c_str());
    
    fprintf(stderr, "dummy_latents: %zu elems\n", dummy_latents.size());
    fprintf(stderr, "dummy_context: %zu elems\n", dummy_context.size());
    fprintf(stderr, "ref_post_head: %zu elems\n", ref_post_head.size());
    fprintf(stderr, "ref_pre_head: %zu elems\n", ref_pre_head.size());
    
    // Config from dump
    DitConfig cfg;
    cfg.dim = 1536;
    cfg.ffn_dim = 8960;
    cfg.n_heads = 12;
    cfg.head_dim = 128;
    cfg.n_layers = 30;
    cfg.in_dim = 128;
    cfg.out_dim = 128;
    cfg.text_dim = 2048;
    cfg.freq_dim = 256;
    cfg.eps = 1e-6f;
    cfg.patch_size = 1;
    
    // dummy_latents shape: [1, 128, 150] = [B, in_dim, T_latent]
    // → B=1, T_latent=150, in_dim=128
    int B = 1;
    int T_latent = 150;
    int in_dim = 128;
    int T_text = 512;  // dummy_context: [1, 512, 2048]
    int text_dim = 2048;
    
    // Create runner
    DiTRunner runner(ext_ctx, cfg);
    
    // Run forward with dummy_timestep = 500
    // DiTRunner expects t as a float (sigma)
    // For the reference, timestep=500, sigma = 500/1000 = 0.5
    float t = 0.5f;
    
    std::vector<float> output(dummy_latents.size());
    std::string error;
    
    fprintf(stderr, "\nRunning DiTRunner.forward()...\n");
    bool ok = runner.forward(dummy_latents.data(), &t,
                              dummy_context.data(), T_text,
                              B, T_latent,
                              output.data(), &error);
    if (!ok) {
        fprintf(stderr, "forward failed: %s\n", error.c_str());
        return 1;
    }
    
    // Compare with reference post_head
    fprintf(stderr, "\n=== Comparing with reference post_head ===\n");
    double sum_abs_err = 0, sum_sq = 0, sum_ref_sq = 0;
    float max_err = 0;
    for (size_t i = 0; i < output.size(); i++) {
        float err = std::abs(output[i] - ref_post_head[i]);
        sum_abs_err += err;
        sum_sq += (double)err * err;
        sum_ref_sq += (double)ref_post_head[i] * ref_post_head[i];
        if (err > max_err) max_err = err;
    }
    double mae = sum_abs_err / output.size();
    double rmse = std::sqrt(sum_sq / output.size());
    double ref_rms = std::sqrt(sum_ref_sq / output.size());
    fprintf(stderr, "Output vs post_head: MAE=%.4f RMSE=%.4f max_err=%.4f ref_RMS=%.4f\n",
            mae, rmse, max_err, ref_rms);
    fprintf(stderr, "rel_err (RMSE/ref_RMS) = %.4f\n", rmse / ref_rms);
    
    // Stats
    double o_sum = 0, o_sum_sq = 0, r_sum = 0, r_sum_sq = 0;
    for (size_t i = 0; i < output.size(); i++) {
        o_sum += output[i]; o_sum_sq += (double)output[i] * output[i];
        r_sum += ref_post_head[i]; r_sum_sq += (double)ref_post_head[i] * ref_post_head[i];
    }
    fprintf(stderr, "Output: mean=%.4f rms=%.4f\n", o_sum / output.size(), std::sqrt(o_sum_sq / output.size()));
    fprintf(stderr, "Ref:    mean=%.4f rms=%.4f\n", r_sum / output.size(), std::sqrt(r_sum_sq / output.size()));
    
    // First 10 values
    fprintf(stderr, "\nOutput[0:10]: ");
    for (int i = 0; i < 10; i++) fprintf(stderr, "%.4f ", output[i]);
    fprintf(stderr, "\nRef   [0:10]: ");
    for (int i = 0; i < 10; i++) fprintf(stderr, "%.4f ", ref_post_head[i]);
    fprintf(stderr, "\n");
    
    ggml_free(ext_ctx);
    return 0;
}
