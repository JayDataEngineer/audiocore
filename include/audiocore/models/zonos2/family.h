#ifndef AUDIOCORE_MODELS_ZONOS2_FAMILY_H
#define AUDIOCORE_MODELS_ZONOS2_FAMILY_H

#include <memory>
#include <string>
#include <vector>
#include <cstdint>

#include "audiocore/framework/core/session.h"

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

namespace audiocore::zonos2 {

// Architecture constants derived from params.json:
//   n_layers=28, dim=2048, head_dim=128, n_heads=16 (dim/head_dim),
//   n_kv_heads=4, intermediate_size=3072 (dim*1.5, rounded to 256),
//   moe: 16 experts, top_k=1 (layer 26: top_k=2), router_dim=128,
//   moe from layer 3 to layer 26, layers 0-2 and 27 are dense.
//   n_codebooks=9, codebook_size=1024, audio_vocab=1026
//   text_vocab=519 (192 legacy + 256 bytes + 8 rate + 60 quality + 2 bg + 1 acc)
//   speaker_embedding_dim=2048, speaker_lda_dim=1024

struct Zonos2ModelConfig {
    int32_t nLayers = 28;
    int32_t hiddenSize = 2048;
    int32_t headDim = 128;
    int32_t nHeads = 16;
    int32_t nKVHeads = 4;
    int32_t intermediateSize = 3072;
    int32_t nCodebooks = 9;
    int32_t codebookSize = 1024;
    int32_t audioVocab = 1026;     // codebookSize + 2 (eoa + pad)
    int32_t textVocab = 519;
    int32_t nExperts = 16;
    int32_t nExpertsPerTok = 1;
    int32_t moeRouterDim = 128;
    int32_t moeStartLayer = 3;
    int32_t moeEndLayer = 1;       // last N layers that are dense
    int32_t eoaId = 1024;
    int32_t audioPadId = 1025;
    float   lossSoftcap = 15.0f;
    bool    speakerEnabled = true;
    int32_t speakerEmbeddingDim = 2048;
    int32_t speakerLdaDim = 1024;
    int32_t speakingRateNumBuckets = 8;
    int32_t qualityNumBuckets = 60;
    // derived
    int32_t kvDim;
    int32_t qoDim;
    int32_t audioEmbedCount;
    int32_t embedderCount;

    Zonos2ModelConfig() {
        kvDim = nKVHeads * headDim;
        qoDim = nHeads * headDim;
        audioEmbedCount = nCodebooks;
        embedderCount = audioEmbedCount + 1;
    }
};

struct TtsRequest {
    std::string text;
    std::string language = "en";
    std::vector<float> speakerEmbedding;
    float temperature = 1.15f;
    float topP = 0.0f;
    float topK = 106.0f;
    float minP = 0.18f;
    int32_t maxNewTokens = 1024;
    float repetitionPenalty = 1.2f;
    int32_t repetitionWindow = 50;
    int32_t repetitionCodebooks = 8;
    int32_t seed = 0;
    int32_t speakingRateBucket = -1;
};

struct TtsResponse {
    std::vector<float> pcmMono;
    int32_t samplingRate = 44100;
};

class Zonos2Session : public Session {
public:
    Zonos2Session();
    ~Zonos2Session() override;

    Zonos2Session(const Zonos2Session&) = delete;
    Zonos2Session& operator=(const Zonos2Session&) = delete;

    std::string family_name() const override { return "zonos2"; }

    bool load(const std::string& model_path,
              const LoadOptions& opts,
              const BackendConfig& backend_cfg,
              std::string* error = nullptr) override;

    bool run_tts(const void* request, void* response,
                 std::string* error = nullptr) override;

    const Zonos2ModelConfig& config() const { return cfg_; }

private:
    // ── Model weights (all point into ctx_) ───────────────────────────
    struct ggml_context* ctx_ = nullptr;

    struct ggml_tensor* embedders_[10]{};
    struct ggml_tensor* embNorm_ = nullptr;
    struct ggml_tensor* outNorm_ = nullptr;

    struct ggml_tensor* attnNorm_[28]{};
    struct ggml_tensor* ffnNorm_[28]{};

    struct ggml_tensor* speakerLdaW_ = nullptr;
    struct ggml_tensor* speakerLdaB_ = nullptr;
    struct ggml_tensor* speakerProjW_ = nullptr;
    struct ggml_tensor* speakerProjB_ = nullptr;

    struct ggml_tensor* wq_[28]{};
    struct ggml_tensor* wkv_[28]{};
    struct ggml_tensor* wo_[28]{};
    struct ggml_tensor* temp_[28]{};
    struct ggml_tensor* gater_[28]{};
    struct ggml_tensor* wIn_[28]{};
    struct ggml_tensor* wOut_[28]{};
    struct ggml_tensor* rdW_[28]{};
    struct ggml_tensor* rdB_[28]{};
    struct ggml_tensor* rm0W_[28]{};
    struct ggml_tensor* rm0B_[28]{};
    struct ggml_tensor* rm2W_[28]{};
    struct ggml_tensor* rm2B_[28]{};
    struct ggml_tensor* rm4W_[28]{};
    struct ggml_tensor* rnE_[28]{};
    struct ggml_tensor* rSc_[28]{};
    struct ggml_tensor* rBi_[28]{};
    struct ggml_tensor* gUp_[28]{};
    struct ggml_tensor* dPr_[28]{};
    struct ggml_tensor* multiOutput_ = nullptr;

    Zonos2ModelConfig cfg_;
    bool loaded_ = false;
};

}  // namespace audiocore::zonos2

#endif  // AUDIOCORE_MODELS_ZONOS2_FAMILY_H
