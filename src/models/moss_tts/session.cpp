// session.cpp — MOSS-TTS run_tts flow.
//
// Pipeline (see src/models/moss_tts/README.md for the spec):
//   1. Tokenize text → Qwen3 text token IDs (chat-templated).
//   2. Forward through Qwen3-8B backbone via the unified qwen3::Runner.
//   3. Project hidden states → n_vq logit streams via audio_head.{i}.weight.
//   4. Sample one codec token per stream, autoregressively.
//   5. Decode codec tokens → PCM via moss.codec.dec.* graphs.
//   6. Write PCM into the response.
//
// Everything transformer-shaped goes through qwen3::Runner (libllama). This
// file is everything that ISN'T the transformer: the audio-head projection,
// the sampler, and the codec decode.
//
// TODO markers name the upstream file each stub needs to be ported from.
// Reference: https://github.com/pwilkin/openmoss

#include "audiocore/models/moss_tts/family.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

#include "ggml.h"

namespace audiocore::moss {

namespace {

// Tiny deterministic argmax sampler — used as the default. The real MOSS
// sampler (top-p / temperature / top-k) lives in openmoss/src/sampler.cpp;
// we port it when we wire the temperature/top_p fields in TtsRequest.
int32_t argmax(const float* logits, int32_t n) {
    int32_t best = 0;
    float   bestv = logits[0];
    for (int32_t i = 1; i < n; ++i) {
        if (logits[i] > bestv) { bestv = logits[i]; best = i; }
    }
    return best;
}

}  // namespace

bool MossSession::build_input_embeddings(const int32_t* text_tokens,
                                         int32_t n_text,
                                         const int32_t* audio_tokens,
                                         int32_t n_audio,
                                         std::vector<float>* embd_out,
                                         std::string* error) {
    // (n_text + n_audio) rows × hidden_size cols, row-major float32.
    const int32_t hs = backbone_->hidden_size();
    embd_out->assign(static_cast<size_t>(n_text + n_audio) * hs, 0.0f);

    // Text-token rows: delegate to libllama. The runner exposes the model's
    // embedding matrix through forward_tokens with a single token; for an
    // n-token batch we read embeddings_ith per row.
    if (n_text > 0) {
        // TODO(port from openmoss/src/model.cpp:embed_text_tokens):
        //   The cleanest libllama-side path is a single forward_tokens call
        //   on the text batch, then walk llama_get_embeddings_ith for each
        //   row. The runner already does this internally; expose a helper.
        // For now we mirror the runner's forward_tokens path:
        std::vector<float> embd_buf(static_cast<size_t>(n_text) * hs);
        // NOTE: forward_tokens returns logits, not embeddings. We need a
        // sibling forward_text_embeddings() on the runner that exposes
        // llama_get_embeddings_ith after a token-id forward pass.
        if (error) *error = "build_input_embeddings: text-token embedding "
                            "lookup not yet exposed on qwen3::Runner "
                            "(TODO: add Runner::embed_tokens)";
        return false;
    }

    // Audio-token rows: sum over streams of (one_hot(token) @ audio_embed[i]).
    // Each codec time-step has n_vq streams; their embeddings sum into one
    // hidden_size vector that goes into the Qwen3 context at that position.
    //
    // Reference path supports F32 and F16 only — quantized weights (Q8_0 etc.)
    // need a dequantize step (or the ggml_cgraph production path). Refuse
    // cleanly so callers see the limitation, not garbage floats.
    auto dtype_supported = [](ggml_tensor* W) {
        return W->type == GGML_TYPE_F32 || W->type == GGML_TYPE_F16;
    };
    if (n_audio > 0) {
        for (int32_t pos = 0; pos < n_audio; ++pos) {
            const int32_t vq_base = pos * cfg_.n_vq;
            float* row = embd_out->data() + (n_text + pos) * hs;
            for (int s = 0; s < cfg_.n_vq; ++s) {
                const int32_t code = audio_tokens[vq_base + s];
                if (code < 0) continue;   // pad slot
                ggml_tensor* W = audio_embed_[s];
                if (!dtype_supported(W)) {
                    if (error) *error = "audio_embed weights are quantized; "
                        "reference path supports F32/F16 only "
                        "(TODO: ggml_cgraph production path with native dequant)";
                    return false;
                }
                // W shape: (hidden_size, audio_vocab_size+1), row-major.
                // Row `code` is the embedding for codebook value `code`.
                if (W->ne[0] != hs) {
                    if (error) *error = "audio_embed dim mismatch";
                    return false;
                }
                const float* wrow_f32 = nullptr;
                std::vector<float>    f32_buf;   // only used for F16 dequant
                const size_t row_off = static_cast<size_t>(code) * hs;
                if (W->type == GGML_TYPE_F32) {
                    wrow_f32 = static_cast<const float*>(W->data) + row_off;
                } else {   // F16
                    f32_buf.resize(hs);
                    const ggml_fp16_t* wrow_f16 =
                        static_cast<const ggml_fp16_t*>(W->data) + row_off;
                    for (int j = 0; j < hs; ++j)
                        f32_buf[j] = ggml_fp16_to_fp32(wrow_f16[j]);
                    wrow_f32 = f32_buf.data();
                }
                for (int j = 0; j < hs; ++j) row[j] += wrow_f32[j];
            }
        }
    }
    return true;
}

bool MossSession::project_to_audio_logits(const float* hidden,
                                          int32_t n_tokens,
                                          std::vector<float>* logits_out,
                                          std::string* error) {
    // For each token row, multiply by each audio_head.{i}.weight to get
    // n_vq logit streams. Output: (n_tokens, n_vq, audio_vocab_size+1).
    const int32_t hs    = backbone_->hidden_size();
    const int32_t vocab = cfg_.audio_vocab_size + 1;   // +1 for pad
    logits_out->assign(static_cast<size_t>(n_tokens) * cfg_.n_vq * vocab, 0.0f);

    // Reference C++ path: O(n_tokens × n_vq × hs × vocab). Slow but correct;
    // the production path builds one ggml_cgraph per stream (TODO below).
    // Supports F32 and F16 weights only — quantized weights (Q8_0 etc.) need
    // the production path which dequantizes inside the cgraph.
    for (int32_t t = 0; t < n_tokens; ++t) {
        const float* hrow = hidden + static_cast<size_t>(t) * hs;
        for (int s = 0; s < cfg_.n_vq; ++s) {
            ggml_tensor* H = audio_head_[s];
            if (H->ne[0] != hs || H->ne[1] != vocab) {
                if (error) *error = "audio_head dim mismatch";
                return false;
            }
            if (H->type != GGML_TYPE_F32 && H->type != GGML_TYPE_F16) {
                if (error) *error = "audio_head weights are quantized; "
                    "reference path supports F32/F16 only "
                    "(TODO: ggml_cgraph production path with native dequant)";
                return false;
            }
            float* lrow = logits_out->data() +
                          (static_cast<size_t>(t) * cfg_.n_vq + s) * vocab;
            // W is (vocab, hs) row-major; lrow[v] = dot(W[v,*], hrow[*]).
            if (H->type == GGML_TYPE_F32) {
                const float* W = static_cast<const float*>(H->data);
                for (int v = 0; v < vocab; ++v) {
                    const float* wrow = W + static_cast<size_t>(v) * hs;
                    float acc = 0.0f;
                    for (int j = 0; j < hs; ++j) acc += wrow[j] * hrow[j];
                    lrow[v] = acc;
                }
            } else {   // F16
                const ggml_fp16_t* W =
                    static_cast<const ggml_fp16_t*>(H->data);
                for (int v = 0; v < vocab; ++v) {
                    const ggml_fp16_t* wrow = W + static_cast<size_t>(v) * hs;
                    float acc = 0.0f;
                    for (int j = 0; j < hs; ++j)
                        acc += ggml_fp16_to_fp32(wrow[j]) * hrow[j];
                    lrow[v] = acc;
                }
            }
        }
    }
    // TODO(perf): replace the inner two loops with a single ggml_cgraph that
    // ggml_cuda backend executes as a batched matmul. Anchor: build the graph
    // once at load(), reuse per forward. Reference: openmoss/src/model.cpp
    // builds exactly this graph in build_audio_head_graph().
    return true;
}

bool MossSession::decode_codec(const int32_t* codec_tokens,
                               int32_t n_tokens,
                               std::vector<float>* pcm_out,
                               std::string* error) {
    // The codec converts (n_tokens, n_vq) codebook indices into mono PCM at
    // cfg_.sampling_rate. Implementation ported from openmoss/src/codec.cpp:
    //
    //   1. For each stream s and each time-step t: look up
    //      moss.codec.quantizer.q.{s}.codebook.weight row codec_tokens[t,s]
    //      → continuous vector. Sum across streams.
    //   2. Feed the resulting (1, downsample_rate * n_tokens) signal through
    //      moss.codec.dec.<layer>.* (Snake-activated conv stack) to upsample
    //      to sampling_rate Hz.
    //
    // That's ~600 lines of ggml graph building. Until the port lands, refuse
    // with a clear error so callers see a clean signal instead of silence.
    (void)codec_tokens; (void)n_tokens; (void)pcm_out;
    if (error) *error = "MOSS codec decode not yet ported "
                        "(TODO: port openmoss/src/codec.cpp into "
                        "src/models/moss_tts/codec.cpp)";
    return false;
}

bool MossSession::run_tts(const void* request, void* response,
                          std::string* error) {
    if (!loaded_) {
        if (error) *error = "MossSession not loaded";
        return false;
    }
    const auto* req = static_cast<const TtsRequest*>(request);
    auto*       res = static_cast<TtsResponse*>(response);
    if (!req || !res) {
        if (error) *error = "null request/response";
        return false;
    }
    res->sampling_rate = cfg_.sampling_rate;

    // (1) Tokenize. TODO: SentencePiece wrapper around the Qwen3 tokenizer
    //     vocab embedded in the GGUF (libllama exposes llama_token_get_piece
    //     and the chat template as metadata). For now: fail cleanly.
    std::vector<int32_t> text_tokens;   // placeholder until tokenizer lands
    if (req->text.empty()) {
        if (error) *error = "empty text";
        return false;
    }
    // TODO(tokenizer): port openmoss/src/tokenizer.cpp. Until then we can't
    // turn req->text into Qwen3 IDs, so we can't proceed.
    if (error) *error = "tokenizer not yet wired "
                        "(TODO: vendor sentencepiece + Qwen3 chat template)";
    (void)text_tokens;
    return false;

    // (2)–(5): see pipeline comment at top of file. These run once (1)/(2)
    // above land — the orchestration code is mechanical on top of the helpers
    // already defined (build_input_embeddings, project_to_audio_logits,
    // decode_codec). Sketch:
    //
    //   std::vector<int32_t> codec_tokens;
    //   for (step = 0; step < max_steps; ++step) {
    //       build_input_embeddings(text_tokens, n_text,
    //                              codec_tokens.data(), codec_tokens.size()/n_vq,
    //                              &embd_buf, error);
    //       backbone_->forward_embeddings(embd_buf, n_rows, n_pos, &hidden, error);
    //       project_to_audio_logits(hidden, n_rows, &logits, error);
    //       for (s = 0; s < n_vq; ++s)
    //           codec_tokens.push_back(argmax(logits_row(s)));
    //       if (all streams emitted tok_audio_end) break;
    //   }
    //   decode_codec(codec_tokens, codec_tokens.size()/n_vq, &res->pcm_mono, error);
}

// Family destructor: free the moss.* ggml_context that loader.cpp allocated.
// (Defined in this TU so ggml.h stays out of family.h — which only forward-
// declares ggml_context.)
MossSession::~MossSession() {
    if (owns_ext_ctx_ && ext_ctx_) {
        ggml_free(ext_ctx_);
        ext_ctx_ = nullptr;
    }
}

}  // namespace audiocore::moss
