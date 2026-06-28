// projection.cpp — see projection.h.
//
// Reference and production paths share the ProjectionRefs shape so a test can
// run both on the same inputs and compare outputs byte-for-byte. The
// production path (project_logits_cgraph) builds one cgraph per stream that
// does:
//
//     logits_s = hidden @ audio_head_s^T
//
// in ggml row-major convention, where hidden is (hs, n_tokens) [ne[0]=hs],
// audio_head_s is (hs, vocab), and the result is (vocab, n_tokens).
//
// ggml_mul_mat(A, B) computes A @ B^T with the standard ggml ne[] convention:
//   • A.ne = [k, m]  → A is m rows of length k
//   • B.ne = [k, n]  → B is n rows of length k
//   • C = A @ B^T    → C.ne = [n, m]
// so:
//   • A = hidden: ne = [hidden_size, n_tokens]
//   • B = audio_head_s: ne = [hidden_size, vocab]
//   • C = logits_s: ne = [vocab, n_tokens]
// which matches our output layout (n_tokens outer, vocab innermost). ✓

#include "audiocore/models/moss_tts/projection.h"

#include <cstring>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

namespace audiocore::moss {

namespace {

bool supported_dtype(const ggml_tensor* W) {
    return W->type == GGML_TYPE_F32 || W->type == GGML_TYPE_F16;
}

}  // namespace

// ── Reference path ──────────────────────────────────────────────────────────
bool project_logits_reference(const ProjectionRefs& refs,
                              float* logits_out,
                              std::string* error) {
    if (!refs.hidden || !refs.heads || !logits_out) {
        if (error) *error = "null pointer in ProjectionRefs";
        return false;
    }
    const int32_t hs = refs.hidden_size;
    const int32_t V  = refs.vocab;

    for (int s = 0; s < refs.n_vq; ++s) {
        const ggml_tensor* H = refs.heads[s];
        if (!H) {
            if (error) *error = "heads[" + std::to_string(s) + "] is null";
            return false;
        }
        if (H->ne[0] != hs || H->ne[1] != V) {
            if (error) *error = "audio_head[" + std::to_string(s) +
                                "] dim mismatch";
            return false;
        }
        if (!supported_dtype(H)) {
            if (error) *error = "audio_head[" + std::to_string(s) +
                                "] is quantized; reference path supports "
                                "F32/F16 only (use project_logits_cgraph)";
            return false;
        }
    }

    for (int32_t t = 0; t < refs.n_tokens; ++t) {
        const float* hrow = refs.hidden + static_cast<size_t>(t) * hs;
        for (int s = 0; s < refs.n_vq; ++s) {
            const ggml_tensor* H = refs.heads[s];
            float* lrow = logits_out +
                          (static_cast<size_t>(t) * refs.n_vq + s) * V;
            if (H->type == GGML_TYPE_F32) {
                const float* W = static_cast<const float*>(H->data);
                for (int v = 0; v < V; ++v) {
                    const float* wrow = W + static_cast<size_t>(v) * hs;
                    float acc = 0.0f;
                    for (int j = 0; j < hs; ++j) acc += wrow[j] * hrow[j];
                    lrow[v] = acc;
                }
            } else {   // F16
                const ggml_fp16_t* W =
                    static_cast<const ggml_fp16_t*>(H->data);
                for (int v = 0; v < V; ++v) {
                    const ggml_fp16_t* wrow =
                        W + static_cast<size_t>(v) * hs;
                    float acc = 0.0f;
                    for (int j = 0; j < hs; ++j)
                        acc += ggml_fp16_to_fp32(wrow[j]) * hrow[j];
                    lrow[v] = acc;
                }
            }
        }
    }
    return true;
}

// ── ggml_cgraph production path ────────────────────────────────────────────
bool project_logits_cgraph(const ProjectionRefs& refs,
                           float* logits_out,
                           std::string* error) {
    if (!refs.hidden || !refs.heads || !logits_out) {
        if (error) *error = "null pointer in ProjectionRefs";
        return false;
    }
    const int32_t hs = refs.hidden_size;
    const int32_t V  = refs.vocab;
    const int32_t T  = refs.n_tokens;

    // Validate head shapes up front so the graph build doesn't leave a
    // half-allocated backend buffer on a bad call.
    for (int s = 0; s < refs.n_vq; ++s) {
        const ggml_tensor* H = refs.heads[s];
        if (!H) {
            if (error) *error = "heads[" + std::to_string(s) + "] is null";
            return false;
        }
        if (H->ne[0] != hs || H->ne[1] != V) {
            if (error) *error = "audio_head[" + std::to_string(s) +
                                "] dim mismatch";
            return false;
        }
    }

    // One CPU backend for the whole call. The backend outlives the buffer
    // and cgraph; all three are freed before return.
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        if (error) *error = "ggml_backend_cpu_init failed";
        return false;
    }

    const bool ok = [&]() -> bool {
        for (int s = 0; s < refs.n_vq; ++s) {
            // Per-stream ctx holds: input hidden view, weight view, output
            // logits, and the cgraph itself (which alone needs
            // ggml_graph_overhead() bytes for the default 2048-node capacity).
            // Fine to free per-iteration — the backend buffer keeps the
            // post-compute bytes alive until we copy them out below.
            struct ggml_init_params gip {
                /*.mem_size   =*/ ggml_tensor_overhead() * 8 +
                                 ggml_graph_overhead() + 64,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context* ctx = ggml_init(gip);
            if (!ctx) return false;

            // A = hidden viewed as (hs, n_tokens). Leave data null — the
            // backend alloc will assign a real buffer; we copy the caller's
            // bytes in afterwards via ggml_backend_tensor_set. (Setting
            // A->data here would make ggml_backend_alloc_ctx_tensors skip A,
            // leaving its buffer null and breaking the set call below.)
            ggml_tensor* A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hs, T);

            // B = audio_head[s] reinterpreted in ggml's ne[] convention.
            // The on-disk layout is already (hs, vocab) row-major — which is
            // exactly what ggml_new_tensor_2d(..., hs, vocab) expects — so
            // the backend-allocated B will hold a contiguous copy regardless
            // of dtype.
            ggml_tensor* B = ggml_new_tensor_2d(ctx,
                                                refs.heads[s]->type,
                                                hs, V);

            // C = A @ B^T → (vocab, n_tokens).
            ggml_tensor* C = ggml_mul_mat(ctx, B, A);
            ggml_set_name(C, "logits");

            ggml_cgraph* gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, C);

            // Allocate graph tensors on the backend. The returned buffer
            // owns the storage for A, B, C for as long as compute needs.
            ggml_backend_buffer_t buf =
                ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!buf) {
                ggml_free(ctx);
                return false;
            }
            // Copy the caller's hidden + weight bytes into the backend-managed
            // tensor storage. Reads through A/B/C->data now hit the buffer.
            ggml_backend_tensor_set(A, refs.hidden, 0,
                                    static_cast<size_t>(hs) * T * sizeof(float));
            ggml_backend_tensor_set(B, refs.heads[s]->data, 0,
                                    ggml_nbytes(refs.heads[s]));

            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                ggml_backend_buffer_free(buf);
                ggml_free(ctx);
                return false;
            }

            // Stream the result out into the caller's interleaved layout:
            //   logits_out[t, s, v] = C[t, v]   (C is (vocab, n_tokens))
            // i.e. for each t, copy V contiguous floats starting at C + t*V.
            const float* cdata = static_cast<const float*>(C->data);
            for (int t = 0; t < T; ++t) {
                float* dst = logits_out +
                             (static_cast<size_t>(t) * refs.n_vq + s) * V;
                std::memcpy(dst, cdata + static_cast<size_t>(t) * V,
                            static_cast<size_t>(V) * sizeof(float));
            }

            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
        return true;
    }();

    ggml_backend_free(backend);
    if (!ok && error) *error = "cgraph projection failed (backend error)";
    return ok;
}

}  // namespace audiocore::moss
