#include "audiocore/models/zonos2/family.h"

#include <cstdio>
#include <cstring>
#include <string>

#include "audiocore/framework/runtime/registry.h"
#include "ggml.h"
#include "gguf.h"

namespace audiocore::zonos2 {

// ── Tensor name helpers ────────────────────────────────────────────────────
static std::string tn_emb(int i)  { return "multi_embedder.embedders." + std::to_string(i) + ".weight"; }
static std::string tn_emb_norm()  { return "emb_norm.weight"; }
static std::string tn_out_norm()  { return "out_norm.weight"; }
static std::string tn_lda_w()     { return "speaker_lda_projection.weight"; }
static std::string tn_lda_b()     { return "speaker_lda_projection.bias"; }
static std::string tn_sp_w()      { return "speaker_projection.weight"; }
static std::string tn_sp_b()      { return "speaker_projection.bias"; }
static std::string tn_attn_norm(int l) { return "layers." + std::to_string(l) + ".attention_norm.weight"; }
static std::string tn_ffn_norm(int l)  { return "layers." + std::to_string(l) + ".ffn_norm.weight"; }
static std::string tn_wq(int l)   { return "layers." + std::to_string(l) + ".attention.wq.weight"; }
static std::string tn_wkv(int l)  { return "layers." + std::to_string(l) + ".attention.wkv.weight"; }
static std::string tn_wo(int l)   { return "layers." + std::to_string(l) + ".attention.wo.weight"; }
static std::string tn_temp(int l) { return "layers." + std::to_string(l) + ".attention.temp"; }
static std::string tn_gater(int l){ return "layers." + std::to_string(l) + ".attention.gater.weight"; }
static std::string tn_win(int l)  { return "layers." + std::to_string(l) + ".feed_forward.w_in.weight"; }
static std::string tn_wout(int l) { return "layers." + std::to_string(l) + ".feed_forward.w_out.weight"; }

static std::string tn_rdw(int l)  { return "layers." + std::to_string(l) + ".feed_forward.router.down_proj.weight"; }
static std::string tn_rdb(int l)  { return "layers." + std::to_string(l) + ".feed_forward.router.down_proj.bias"; }
static std::string tn_rmw(int l, int n) { return "layers." + std::to_string(l) + ".feed_forward.router.router_mlp." + std::to_string(n) + ".weight"; }
static std::string tn_rmb(int l, int n) { return "layers." + std::to_string(l) + ".feed_forward.router.router_mlp." + std::to_string(n) + ".bias"; }
static std::string tn_rne(int l)  { return "layers." + std::to_string(l) + ".feed_forward.router.rmsnorm_eda.weight"; }
static std::string tn_rsc(int l)  { return "layers." + std::to_string(l) + ".feed_forward.router.router_states_scale"; }
static std::string tn_rbi(int l)  { return "layers." + std::to_string(l) + ".feed_forward.router.balancing_biases"; }
static std::string tn_gup(int l)  { return "layers." + std::to_string(l) + ".feed_forward.experts.gate_up_proj"; }
static std::string tn_dpr(int l)  { return "layers." + std::to_string(l) + ".feed_forward.experts.down_proj"; }
static std::string tn_mout()      { return "multi_output.weight"; }

#define GET_TENSOR(ctx, name, ptr) do { \
    *(ptr) = ggml_get_tensor(ctx, (name).c_str()); \
    if (!*(ptr)) { \
        std::fprintf(stderr, "zonos2: missing tensor: %s\n", (name).c_str()); \
        return false; \
    } \
} while(0)

// ── Zonos2Session::load ────────────────────────────────────────────────────

bool Zonos2Session::load(const std::string& model_path,
                          const LoadOptions& opts,
                          const BackendConfig& backend_cfg,
                          std::string* error) {
    (void)opts;
    (void)backend_cfg;

    std::fprintf(stderr, "zonos2: loading C++ native model from '%s'\n", model_path.c_str());

    // Let gguf_init_from_file create the ggml context with tensors + data.
    struct ggml_context* g_ctx = nullptr;
    struct gguf_init_params ufparams = {false, &g_ctx};
    struct gguf_context* gguf = gguf_init_from_file(model_path.c_str(), ufparams);
    if (!gguf) {
        if (error) *error = "zonos2: failed to open model file";
        return false;
    }
    ctx_ = g_ctx;

    // ── Extract tensors ────────────────────────────────────────────────

    // Config from KV metadata (if available)
    // Zonos2 uses hardcoded defaults from params.json

    int n_embedders = cfg_.embedderCount;
    std::string tn;
    for (int i = 0; i < n_embedders; i++) {
        tn = tn_emb(i);
        GET_TENSOR(ctx_, tn, &embedders_[i]);
    }

    tn = tn_emb_norm(); GET_TENSOR(ctx_, tn, &embNorm_);
    tn = tn_out_norm(); GET_TENSOR(ctx_, tn, &outNorm_);

    tn = tn_lda_w(); speakerLdaW_ = ggml_get_tensor(ctx_, tn.c_str());
    tn = tn_lda_b(); speakerLdaB_ = ggml_get_tensor(ctx_, tn.c_str());
    tn = tn_sp_w();  speakerProjW_ = ggml_get_tensor(ctx_, tn.c_str());
    tn = tn_sp_b();  speakerProjB_ = ggml_get_tensor(ctx_, tn.c_str());

    for (int l = 0; l < cfg_.nLayers; l++) {
        tn = tn_attn_norm(l); GET_TENSOR(ctx_, tn, &attnNorm_[l]);
        tn = tn_ffn_norm(l);  GET_TENSOR(ctx_, tn, &ffnNorm_[l]);
        tn = tn_wq(l);        GET_TENSOR(ctx_, tn, &wq_[l]);
        tn = tn_wkv(l);       GET_TENSOR(ctx_, tn, &wkv_[l]);
        tn = tn_wo(l);        GET_TENSOR(ctx_, tn, &wo_[l]);
        tn = tn_temp(l);      GET_TENSOR(ctx_, tn, &temp_[l]);
        tn = tn_gater(l);     GET_TENSOR(ctx_, tn, &gater_[l]);

        bool moe = (cfg_.nExperts > 1 &&
                    l >= cfg_.moeStartLayer &&
                    (cfg_.nLayers - l) > cfg_.moeEndLayer);
        if (moe) {
            tn = tn_rdw(l);  GET_TENSOR(ctx_, tn, &rdW_[l]);
            tn = tn_rdb(l);  GET_TENSOR(ctx_, tn, &rdB_[l]);
            tn = tn_rmw(l, 0); GET_TENSOR(ctx_, tn, &rm0W_[l]);
            tn = tn_rmb(l, 0); GET_TENSOR(ctx_, tn, &rm0B_[l]);
            tn = tn_rmw(l, 2); GET_TENSOR(ctx_, tn, &rm2W_[l]);
            tn = tn_rmb(l, 2); GET_TENSOR(ctx_, tn, &rm2B_[l]);
            tn = tn_rmw(l, 4); GET_TENSOR(ctx_, tn, &rm4W_[l]);
            tn = tn_rne(l);  GET_TENSOR(ctx_, tn, &rnE_[l]);
            tn = tn_rsc(l);  GET_TENSOR(ctx_, tn, &rSc_[l]);
            tn = tn_rbi(l);  GET_TENSOR(ctx_, tn, &rBi_[l]);
            tn = tn_gup(l);  GET_TENSOR(ctx_, tn, &gUp_[l]);
            tn = tn_dpr(l);  GET_TENSOR(ctx_, tn, &dPr_[l]);
        } else {
            tn = tn_win(l);  GET_TENSOR(ctx_, tn, &wIn_[l]);
            tn = tn_wout(l); GET_TENSOR(ctx_, tn, &wOut_[l]);
        }
    }

    tn = tn_mout(); GET_TENSOR(ctx_, tn, &multiOutput_);

    gguf_free(gguf);
    loaded_ = true;

    std::fprintf(stderr, "zonos2: model loaded successfully (%d layers)\n", cfg_.nLayers);
    return true;
}

bool Zonos2Session::run_tts(const void* request, void* response,
                             std::string* error) {
    (void)request;
    (void)response;
    if (error) *error = "zonos2: C++ native forward pass not yet implemented";
    return false;
}

// ── Factory registration ───────────────────────────────────────────────────

namespace {
std::unique_ptr<Session> make_zonos2_session() {
    return std::unique_ptr<Session>(new Zonos2Session());
}
}  // namespace

AUDIOCORE_REGISTER_FAMILY(zonos2, make_zonos2_session)
AUDIOCORE_EXTERN_C_GUARD(zonos2, make_zonos2_session)

}  // namespace audiocore::zonos2
