#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <ggml.h>

int main() {
    // Tiny ConvTranspose1d: IC=2, OC=2, K=3, stride=2, no padding
    // weight PyTorch layout [IC, OC, K]: unique values for easy debugging
    const int IC = 2, OC = 2, K = 3, stride = 2;
    float weight_py[IC * OC * K] = {
        1, 2, 3,  4, 5, 6,    // ic=0: oc=0(k=0,1,2), oc=1(k=0,1,2)
        7, 8, 9,  10, 11, 12  // ic=1: oc=0(k=0,1,2), oc=1(k=0,1,2)
    };
    
    // Input [T, IC]: T=3
    int T = 3;
    float input_py[IC * T] = {
        0.5f, -0.3f, 0.1f,  // ic=0
        0.2f,  0.8f, -0.5f   // ic=1
    };
    
    // Expected output (manual computation)
    int T_out = (T - 1) * stride + K;  // = 7
    float expected[OC * T_out] = {0};  // row-major [OC, T_out]
    for (int oc = 0; oc < OC; oc++)
        for (int ic = 0; ic < IC; ic++)
            for (int k = 0; k < K; k++)
                for (int t = 0; t < T; t++)
                    expected[oc * T_out + t * stride + k] +=
                        input_py[ic * T + t] * weight_py[ic * OC * K + oc * K + k];
    
    printf("Expected output [OC=%d, T_out=%d]:\n", OC, T_out);
    for (int oc = 0; oc < OC; oc++) {
        printf("  oc=%d:", oc);
        for (int p = 0; p < T_out; p++)
            printf(" %6.2f", expected[oc * T_out + p]);
        printf("\n");
    }
    
    // ── ggml computation ──
    size_t mem = 1024 * 1024;
    auto ctx_buf = std::unique_ptr<char[]>(new char[mem]);
    ggml_init_params p = {mem, ctx_buf.get(), false};
    ggml_context* ctx = ggml_init(p);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 128, false);
    
    // Build weight: ggml expects [K, OC, IC] with element (k,oc,ic) at k+oc*K+ic*K*OC
    // = weight_py[ic][oc][k] at offset ic*OC*K + oc*K + k
    std::vector<uint16_t> w_f16(K * OC * IC);
    for (int ic = 0; ic < IC; ic++)
        for (int oc = 0; oc < OC; oc++)
            for (int k = 0; k < K; k++)
                w_f16[static_cast<size_t>(k) + oc*K + ic*K*OC] =
                    ggml_fp32_to_fp16(weight_py[ic * OC * K + oc * K + k]);
    
    ggml_tensor* w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, K, OC, IC);
    memcpy(w->data, w_f16.data(), w_f16.size() * sizeof(uint16_t));
    
    // Input [T, IC] ggml layout: ne=[T, IC]
    // row t, col c at t + c*T
    ggml_tensor* in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, IC);
    for (int c = 0; c < IC; c++)
        for (int t = 0; t < T; t++)
            ((float*)in->data)[t + static_cast<size_t>(c) * T] =
                input_py[c * T + t];
    
    auto r = ggml_conv_transpose_1d(ctx, w, in, stride, 0 /*pad*/, 1);
    ggml_build_forward_expand(gf, r);
    int st = ggml_graph_compute_with_ctx(ctx, gf, 4);
    printf("\nggml compute: st=%d\n", st);
    printf("ggml output: ne=[%lld,%lld]\n", (long long)r->ne[0], (long long)r->ne[1]);
    
    int ggml_T = (int)r->ne[0];
    bool match = true;
    for (int oc = 0; oc < OC && match; oc++) {
        printf("  oc=%d:", oc);
        for (int p = 0; p < T_out; p++) {
            float v = ((float*)r->data)[p + oc * ggml_T];
            printf(" %6.2f", v);
            if (std::abs(v - expected[oc * T_out + p]) > 1e-3f) {
                printf(" MISMATCH(expected=%.2f)", expected[oc * T_out + p]);
                match = false;
            }
        }
        printf("\n");
    }
    printf("RESULT: %s\n", match ? "PASS" : "FAIL");
    
    ggml_free(ctx);
    return match ? 0 : 1;
}
