// session.cpp — Flow-matching denoising pipeline.
//
//   DiT flow matching with CFG → DAC VAE decode → PCM mono.
//   Text encoder (Qwen3 TE) is loaded from sidecar GGUF via --extras te_path.

#include "audiocore/models/moss_sfx_v2/family.h"
#include "audiocore/models/qwen3/runner.h"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace audiocore::moss_sfx_v2 {

// ── VAE upsampling factor (hardcoded to match vae_runner.cpp strides) ───
static int vae_upsampling_factor() {
    // 5 DecoderBlocks with strides [8, 5, 4, 3, 2] → 960
    return 8 * 5 * 4 * 3 * 2;  // 960
}

// ═══════════════════════════════════════════════════════════════════════════
//  Flow-matching Euler schedule
// ═══════════════════════════════════════════════════════════════════════════
//
// Ported from FlowMatchScheduler in
// moss_soundeffect_v2/diffsynth/schedulers/flow_match.py.
//
// Returns sigmas in descending order (σ ∈ [1.0, sigma_min]) used as the
// flow-matching timestep t = σ. The caller integrates
//
//   x_{t+dt} = x_t + dt · v_θ(x_t, σ)     dt = σ_{i+1} − σ_i (negative)

struct FlowSchedule {
    std::vector<float> sigmas;
    int n_steps;
};

static FlowSchedule build_schedule(int num_steps, float shift,
                                    float sigma_min,
                                    bool extra_one_step) {
    FlowSchedule s;
    s.n_steps = num_steps;
    s.sigmas.resize(num_steps);

    // sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
    // denoising_strength=1.0, sigma_max=1.0 (default) → 1.0
    const float sigma_start = 1.0f;

    if (extra_one_step) {
        // torch.linspace(sigma_start, sigma_min, n+1)[:-1]
        for (int i = 0; i < num_steps; i++) {
            s.sigmas[i] = sigma_start +
                (sigma_min - sigma_start) * static_cast<float>(i) /
                    static_cast<float>(num_steps);
        }
    } else {
        // torch.linspace(sigma_start, sigma_min, n)
        for (int i = 0; i < num_steps; i++) {
            float f = (num_steps > 1)
                          ? static_cast<float>(i) / (num_steps - 1)
                          : 0.0f;
            s.sigmas[i] = sigma_start + (sigma_min - sigma_start) * f;
        }
    }

    // Apply shift: σ = shift · σ / (1 + (shift − 1) · σ)
    if (std::abs(shift - 1.0f) > 1e-6f) {
        for (int i = 0; i < num_steps; i++) {
            float sig = s.sigmas[i];
            s.sigmas[i] = shift * sig / (1.0f + (shift - 1.0f) * sig);
        }
    }

    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
//  run_sfx — main denoising pipeline
// ═══════════════════════════════════════════════════════════════════════════
//
// 1. Tokenize prompt + instruct → Qwen3 TE embeddings
// 2. Initialize Gaussian noise latent
// 3. Euler flow-matching loop with CFG
// 4. VAE decode → PCM mono
// 5. Post-process (DC offset, gentle normalization)

bool SfxSession::run_sfx(const TtsRequest& req, std::vector<float>* pcm_out,
                          std::string* error) {
    auto err = [&](const char* msg) {
        if (error) *error = msg;
        std::fprintf(stderr, "[moss_sfx_v2] %s\n", msg);
    };

    if (!loaded_) { err("not loaded"); return false; }

    const DitConfig& dc = cfg_.dit;

    // ════════════════════════════════════════════════════════════════════
    // 1. Text encoder → context embeddings
    //
    // CRITICAL: The Python pipeline (WanPrompter) always pads the tokenized
    // prompt to text_len=512 positions. The TE processes all 512 positions
    // (with attention masking), then encode_prompt zeros positions beyond
    // the real sequence length.
    //
    // The DiT cross-attention expects 512 KV pairs. With only the real token
    // count (e.g. 12), the velocity is ~200x too large (v_rms≈209 vs ≈1.05),
    // causing trajectory divergence and static noise.
    //
    // We replicate the padding: run TE on real tokens, then zero-pad the
    // embedding to 512 positions.
    // ════════════════════════════════════════════════════════════════════
    constexpr int32_t kTextLen = 512;  // matches WanPrompter.text_len

    int32_t T_cond = 0;
    std::vector<float> cond_emb;
    int32_t T_uncond = 0;
    std::vector<float> uncond_emb;

    if (te_) {
        // Conditional prompt — append " duration: X.Xs" suffix to match
        // MossSoundEffectPipeline.__call__ which does this by default
        // (append_duration_suffix=True).
        {
            float dur_sec_prompt = 10.0f;
            if (req.duration_tokens > 0) {
                dur_sec_prompt = static_cast<float>(req.duration_tokens) * 0.08f;
            }
            char dur_suffix[64];
            std::snprintf(dur_suffix, sizeof(dur_suffix),
                          " duration: %.1fs", dur_sec_prompt);
            std::string prompt_text = req.text + dur_suffix;
            std::vector<int32_t> toks;
            if (!te_->tokenize(prompt_text, /*add_special=*/true,
                               /*parse_special=*/true, &toks, nullptr, error))
                return false;
            int32_t n_cond = static_cast<int32_t>(toks.size());
            if (n_cond > 0) {
                std::vector<float> raw_emb(
                    static_cast<size_t>(n_cond) * dc.text_dim);
                if (!te_->forward_get_embeddings(toks.data(), n_cond, 0,
                                                 raw_emb.data(), error))
                    return false;
                // Zero-pad to kTextLen positions (matching WanPrompter)
                cond_emb.assign(
                    static_cast<size_t>(kTextLen) * dc.text_dim, 0.0f);
                std::copy(raw_emb.begin(), raw_emb.end(), cond_emb.begin());
                T_cond = kTextLen;
            }
        }

        // Unconditional (negative) prompt — use req.instruct or empty
        {
            std::string neg = req.instruct.empty() ? "" : req.instruct;
            std::vector<int32_t> toks;
            if (!te_->tokenize(neg, true, true, &toks, nullptr, error))
                return false;
            int32_t n_uncond = static_cast<int32_t>(toks.size());
            // Always create a 512-position context (zero-padded if empty)
            uncond_emb.assign(
                static_cast<size_t>(kTextLen) * dc.text_dim, 0.0f);
            if (n_uncond > 0) {
                std::vector<float> raw_emb(
                    static_cast<size_t>(n_uncond) * dc.text_dim);
                if (!te_->forward_get_embeddings(toks.data(), n_uncond, 0,
                                                 raw_emb.data(), error))
                    return false;
                std::copy(raw_emb.begin(), raw_emb.end(),
                          uncond_emb.begin());
            }
            T_uncond = kTextLen;
        }

        std::fprintf(stderr, "[moss_sfx_v2] TE: T_cond=%d T_uncond=%d prompt=\"%s\"\n",
                     T_cond, T_uncond, req.text.c_str());
        // Debug: dump TE embeddings (enable with AUDIOCORE_DEBUG_DUMP=1)
        if (std::getenv("AUDIOCORE_DEBUG_DUMP")) {
            if (T_cond > 0) {
                FILE* f = std::fopen("/tmp/cpp_cond_emb.f32", "wb");
                if (f) {
                    std::fwrite(cond_emb.data(), sizeof(float),
                                cond_emb.size(), f);
                    std::fclose(f);
                    std::fprintf(stderr, "[moss_sfx_v2] dumped cond_emb (%zu elems)\n",
                                 cond_emb.size());
                }
            }
            if (T_uncond > 0) {
                FILE* f = std::fopen("/tmp/cpp_uncond_emb.f32", "wb");
                if (f) {
                    std::fwrite(uncond_emb.data(), sizeof(float),
                                uncond_emb.size(), f);
                    std::fclose(f);
                }
            }
        }
    } else {
        // No TE — dummy zero context padded to kTextLen
        T_cond = kTextLen;
        cond_emb.assign(static_cast<size_t>(kTextLen) * dc.text_dim, 0.0f);
        T_uncond = kTextLen;
        uncond_emb.assign(static_cast<size_t>(kTextLen) * dc.text_dim, 0.0f);
        std::fprintf(stderr,
                     "[moss_sfx_v2] WARNING: no TE loaded; zero context\n");
    }

    // ════════════════════════════════════════════════════════════════════
    // 2. Output duration → T_latent
    //
    // The Python pipeline (MossSoundEffectPipeline) always denoises a
    // fixed-size latent of max_inference_seconds=30 seconds, then crops
    // the output to the requested duration. We must match this because
    // the DiT was trained on 30-second segments. The VAE decode is then
    // chunked along time to avoid VRAM OOM.
    // ════════════════════════════════════════════════════════════════════
    float dur_sec = 10.0f;
    if (req.duration_tokens > 0) {
        dur_sec = static_cast<float>(req.duration_tokens) * 0.08f;
    }
    const int32_t num_samples =
        static_cast<int32_t>(dur_sec * cfg_.sample_rate);
    const int up = vae_upsampling_factor();

    // Always denoise at max_inference_seconds=30 (matching Python pipeline)
    constexpr int32_t kMaxInferenceSeconds = 30;
    const int32_t T_latent =
        (cfg_.sample_rate * kMaxInferenceSeconds) / up;

    std::fprintf(stderr,
                 "[moss_sfx_v2] dur=%.1fs samples=%d T_latent=%d (full=%ds)\n",
                 dur_sec, num_samples, T_latent, kMaxInferenceSeconds);

    // ════════════════════════════════════════════════════════════════════
    // 3. Noise latent ∼ N(0, I)  [T_latent, in_dim]
    // ════════════════════════════════════════════════════════════════════
    std::mt19937_64 rng(static_cast<uint64_t>(req.seed != 0 ? req.seed : 42));
    std::normal_distribution<float> ndist(0.0f, 1.0f);
    std::vector<float> x_t(static_cast<size_t>(T_latent) * dc.in_dim);
    for (auto& v : x_t) v = ndist(rng);

    // Debug: dump initial noise for cross-validation with Python
    if (std::getenv("AUDIOCORE_DEBUG_DUMP")) {
        FILE* f = std::fopen("/tmp/cpp_noise.bin", "wb");
        if (f) {
            std::fwrite(x_t.data(), sizeof(float), x_t.size(), f);
            std::fclose(f);
            std::fprintf(stderr,
                "[moss_sfx_v2] dumped noise (%zu elems, T_latent=%d in_dim=%d)\n",
                x_t.size(), T_latent, dc.in_dim);
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // 4. Flow-matching schedule
    // ════════════════════════════════════════════════════════════════════
    const int num_steps = (req.n_diffusion_steps > 0) ? req.n_diffusion_steps : 50;
    const float guidance_scale = req.guidance_scale;

    FlowSchedule sched = build_schedule(
        num_steps,
        static_cast<float>(cfg_.scheduler_shift),
        cfg_.scheduler_sigma_min,
        cfg_.scheduler_extra_one_step);

    // ════════════════════════════════════════════════════════════════════
    // 5. Velocity buffer
    // ════════════════════════════════════════════════════════════════════
    std::vector<float> v_out(static_cast<size_t>(T_latent) * dc.out_dim);

    // ════════════════════════════════════════════════════════════════════
    // 6. Denoising loop
    // ════════════════════════════════════════════════════════════════════
    auto t0 = ggml_time_us();
    double t_fwd_us = 0;
    for (int step = 0; step < sched.n_steps; step++) {
        const float sigma = sched.sigmas[step];
        const float sigma_next = (step + 1 < sched.n_steps)
                                     ? sched.sigmas[step + 1]
                                     : 0.0f;
        const float dt = sigma_next - sigma;

        auto f0 = ggml_time_us();
        if (!dit_runner_->forward_cfg(
                x_t.data(), &sigma,
                cond_emb.data(), T_cond,
                uncond_emb.data(), T_uncond,
                guidance_scale,
                /*B=*/1, T_latent,
                v_out.data(), error))
            return false;
        double fwd_us = static_cast<double>(ggml_time_us() - f0);
        t_fwd_us += fwd_us;

        for (size_t i = 0; i < x_t.size(); i++)
            x_t[i] += dt * v_out[i];

        // Debug: dump velocity at step 0 for cross-validation
        if (step == 0 && std::getenv("AUDIOCORE_DEBUG_DUMP")) {
            FILE* f = std::fopen("/tmp/cpp_velocity_step0.bin", "wb");
            if (f) {
                std::fwrite(v_out.data(), sizeof(float), v_out.size(), f);
                std::fclose(f);
                std::fprintf(stderr,
                    "[moss_sfx_v2] dumped velocity step 0 (%zu elems)\n",
                    v_out.size());
            }
            // Also dump x_t after step 0
            f = std::fopen("/tmp/cpp_xt_step0.bin", "wb");
            if (f) {
                std::fwrite(x_t.data(), sizeof(float), x_t.size(), f);
                std::fclose(f);
            }
        }

        // Debug: log velocity AND latent stats every 10 steps + first/last
        if (step % 10 == 0 || step == sched.n_steps - 1) {
            double v_sum = 0.0, v_sum_sq = 0.0;
            double x_sum = 0.0, x_sum_sq = 0.0;
            for (size_t i = 0; i < x_t.size(); i++) {
                v_sum += v_out[i];
                v_sum_sq += static_cast<double>(v_out[i]) * v_out[i];
                x_sum += x_t[i];
                x_sum_sq += static_cast<double>(x_t[i]) * x_t[i];
            }
            double v_rms = std::sqrt(v_sum_sq / x_t.size());
            double x_rms = std::sqrt(x_sum_sq / x_t.size());
            std::fprintf(stderr,
                "[moss_sfx_v2] step %d: σ=%.4f v_rms=%.2f x_rms=%.2f\n",
                step, sigma, v_rms, x_rms);
        }

        double step_us = static_cast<double>(ggml_time_us() - f0);
        std::fprintf(stderr,
            "[moss_sfx_v2] step %d/%d σ=%.4f dt=%.4f fwd=%.2fs step=%.2fs\n",
            step, sched.n_steps, sigma, dt,
            fwd_us / 1e6, step_us / 1e6);
    }
    double t_tot = static_cast<double>(ggml_time_us() - t0) / 1e6;
    std::fprintf(stderr,
        "[moss_sfx_v2] loop: %.2fs total, %.2fs in forward (%.1f%%)"
        " avg %.3fs/step\n",
        t_tot, t_fwd_us / 1e6, 100.0 * t_fwd_us / (t_tot * 1e6),
        t_tot / sched.n_steps);
    std::fflush(stderr);

    // ════════════════════════════════════════════════════════════════════
    // 7. VAE decode → PCM mono
    // ════════════════════════════════════════════════════════════════════
    {
        // Debug: dump final latent for cross-validation
        if (std::getenv("AUDIOCORE_DEBUG_DUMP")) {
            FILE* f = std::fopen("/tmp/cpp_final_latent.f32", "wb");
            if (f) {
                std::fwrite(x_t.data(), sizeof(float), x_t.size(), f);
                std::fclose(f);
                std::fprintf(stderr,
                    "[moss_sfx_v2] dumped final latent (%zu elems, T=%d)\n",
                    x_t.size(), T_latent);
            }
        }

        double x_sum = 0.0, x_sum_sq = 0.0;
        float x_min = x_t[0], x_max = x_t[0];
        for (size_t i = 0; i < x_t.size(); i++) {
            x_sum += x_t[i];
            x_sum_sq += static_cast<double>(x_t[i]) * x_t[i];
            if (x_t[i] < x_min) x_min = x_t[i];
            if (x_t[i] > x_max) x_max = x_t[i];
        }
        double x_mean = x_sum / x_t.size();
        double x_rms = std::sqrt(x_sum_sq / x_t.size());
        std::fprintf(stderr,
            "[moss_sfx_v2] x_t pre-vae: mean=%.4f rms=%.4f"
            " min=%.4f max=%.4f shape=[1,%d,%d]\n",
            x_mean, x_rms, x_min, x_max, cfg_.dit.in_dim, T_latent);
    }
    const int32_t T_audio = T_latent * up;
    pcm_out->resize(static_cast<size_t>(T_audio));

    // Chunked VAE decode: decode the latent in time segments to avoid VRAM OOM.
    // The DAC VAE is fully convolutional, so each time segment decodes
    // independently. We use chunks of ≤500 latent frames (= 480000 audio
    // samples = 10 seconds at 48 kHz), which fits comfortably in 24 GB VRAM.
    constexpr int32_t kVaeChunkLatent = 500;
    {
        auto t0 = std::chrono::steady_clock::now();
        const int n_chunks =
            (T_latent + kVaeChunkLatent - 1) / kVaeChunkLatent;
        std::fprintf(stderr,
            "[moss_sfx_v2] VAE decode: %d chunk(s) of ≤%d latent frames\n",
            n_chunks, kVaeChunkLatent);

        for (int c = 0; c < n_chunks; c++) {
            const int t_start = c * kVaeChunkLatent;
            const int t_end = std::min(t_start + kVaeChunkLatent,
                                       static_cast<int>(T_latent));
            const int chunk_T = t_end - t_start;

            // Extract chunk: [chunk_T, in_dim] (time-major)
            std::vector<float> chunk_in(
                static_cast<size_t>(chunk_T) * dc.in_dim);
            for (int t = 0; t < chunk_T; t++) {
                std::memcpy(
                    chunk_in.data() +
                        static_cast<size_t>(t) * dc.in_dim,
                    x_t.data() +
                        static_cast<size_t>(t_start + t) * dc.in_dim,
                    dc.in_dim * sizeof(float));
            }

            // Decode chunk
            std::vector<float> chunk_out(
                static_cast<size_t>(chunk_T) * up);
            if (!vae_runner_->decode(chunk_in.data(), /*B=*/1, chunk_T,
                                      chunk_out.data(), error)) {
                return false;
            }

            // Copy to output PCM
            std::memcpy(
                pcm_out->data() + static_cast<size_t>(t_start) * up,
                chunk_out.data(),
                static_cast<size_t>(chunk_T) * up * sizeof(float));

            std::fprintf(stderr,
                "[moss_sfx_v2]   chunk %d/%d: T=%d → %d audio\n",
                c + 1, n_chunks, chunk_T, chunk_T * up);
        }

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "[moss_sfx_v2] VAE decode total: %.0fms (%d latent → %d audio)\n",
                     ms, T_latent, T_audio);
    }

    if (static_cast<int32_t>(pcm_out->size()) > num_samples)
        pcm_out->resize(static_cast<size_t>(num_samples));

    // ════════════════════════════════════════════════════════════════════
    // 8. Post-processing
    // ════════════════════════════════════════════════════════════════════
    {
        auto& pcm = *pcm_out;
        const size_t n = pcm.size();
        if (n > 0) {
            double sum = 0.0;
            float peak = 0.0f;
            for (size_t i = 0; i < n; i++)
                sum += static_cast<double>(pcm[i]);
            const float dc =
                static_cast<float>(sum / static_cast<double>(n));
            for (size_t i = 0; i < n; i++) {
                pcm[i] -= dc;
                float av = std::fabs(pcm[i]);
                if (av > peak) peak = av;
            }
            if (peak > 1e-6f && peak < 0.5f) {
                const float gain = 0.9f / peak;
                for (size_t i = 0; i < n; i++) pcm[i] *= gain;
            }
            std::fprintf(stderr,
                "[moss_sfx_v2] post: DC=%.5f peak=%.5f %s\n",
                dc, peak,
                (peak > 1e-6f && peak < 0.5f) ? "normalized" : "");
        }
    }

    std::fprintf(stderr,
        "[moss_sfx_v2] done: %zu samples (%.2fs @ %dHz)\n",
        pcm_out->size(),
        static_cast<double>(pcm_out->size()) / cfg_.sample_rate,
        cfg_.sample_rate);

    return true;
}

}  // namespace audiocore::moss_sfx_v2
