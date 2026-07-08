#ifndef AUDIOCORE_MODELS_MOSS_SFX_V2_FAMILY_H
#define AUDIOCORE_MODELS_MOSS_SFX_V2_FAMILY_H

#include <memory>
#include <string>
#include <vector>

#include "audiocore/framework/core/session.h"
#include "audiocore/framework/io/gguf_reader.h"
#include "audiocore/framework/runtime/tasks.h"
#include "audiocore/models/moss_sfx_v2/dit_runner.h"
#include "audiocore/models/moss_sfx_v2/vae_runner.h"
#include "audiocore/models/qwen3/runner.h"

struct ggml_context;
struct ggml_tensor;

namespace audiocore::moss_sfx_v2 {

// ── Config (parsed from DiT GGUF metadata) ──────────────────────────────
struct SfxConfig {
    DitConfig  dit;
    int32_t    sample_rate = 48000;
    // Scheduler
    int32_t    scheduler_shift = 5;
    float      scheduler_sigma_min = 0.0f;
    bool       scheduler_extra_one_step = true;
    int32_t    scheduler_num_train_timesteps = 1000;
};

class SfxSession : public Session {
public:
    ~SfxSession() override;
    std::string family_name() const override { return "moss_sfx_v2"; }
    bool load(const std::string& model_path,
              const LoadOptions& opts,
              const BackendConfig& backend_cfg,
              std::string* error = nullptr) override;
    bool run_tts(const void* request, void* response,
                 std::string* error = nullptr) override;

    const SfxConfig& config() const { return cfg_; }

private:
    bool bind_dit(const GgufReader& r, std::string* error);
    bool bind_vae(const GgufReader& r, std::string* error);

    // Denoising loop
    bool run_sfx(const TtsRequest& req, std::vector<float>* pcm_out,
                 std::string* error);

    std::unique_ptr<DiTRunner> dit_runner_;
    std::unique_ptr<VAERunner> vae_runner_;
    std::unique_ptr<qwen3::Runner> te_;   // Qwen3 text encoder

    ggml_context* ext_ctx_ = nullptr;
    SfxConfig cfg_;
    bool owns_ext_ctx_ = false;
    std::vector<std::vector<uint8_t>> dit_buffers_;   // fallback materialize storage
    std::vector<std::vector<uint8_t>> vae_buffers_;   // fallback materialize storage
};

}  // namespace audiocore::moss_sfx_v2

#endif
