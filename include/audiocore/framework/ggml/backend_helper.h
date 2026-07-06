// backend_helper.h — shared GGML backend initialization (GPU + CPU scheduler).
//
// Ports the ServeurpersoCom/acestep.cpp backend.h pattern: load all backends,
// pick best GPU, keep CPU as fallback, create a scheduler that places ops on
// GPU when possible and falls back to CPU otherwise.
//
// Without this, model runners fall back to ggml_graph_compute_with_ctx which
// is CPU-only — making inference 50x slower than necessary.

#ifndef AUDIOCORE_FRAMEWORK_GGML_BACKEND_HELPER_H
#define AUDIOCORE_FRAMEWORK_GGML_BACKEND_HELPER_H

#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unordered_map>
#include <vector>

namespace audiocore::ggml_utils {

struct BackendPair {
    ggml_backend_t backend      = nullptr;
    ggml_backend_t cpu_backend  = nullptr;
    bool           has_gpu      = false;
};

// Physical core count heuristic (logical / 2 for HT/SMT).
static int backend_cpu_n_threads() {
    int n = (int) std::thread::hardware_concurrency() / 2;
    return n > 0 ? n : 1;
}

// Standalone CPU backend via Registry API (DL-safe, no ggml-cpu.h needed).
static ggml_backend_t cpu_backend_new(int n_threads) {
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t     cpu     = nullptr;
    if (cpu_dev) {
        cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    }
    if (!cpu) {
        cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    if (!cpu) return nullptr;

    ggml_backend_dev_t dev = cpu ? ggml_backend_get_device(cpu) : nullptr;
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg) {
        auto set_fn = (ggml_backend_set_n_threads_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (set_fn) set_fn(cpu, n_threads);
    }
    return cpu;
}

// Initialize backends: load all available (CUDA, Metal, Vulkan...),
// pick the best one, keep CPU as fallback. Idempotent.
// Each call returns a fresh BackendPair; the caller owns the lifetimes.
inline BackendPair backend_init(const char* label) {
    ggml_backend_load_all();

    BackendPair bp = {};
    const char* force_backend = std::getenv("GGML_BACKEND");
    if (force_backend) {
        bp.backend = ggml_backend_init_by_name(force_backend, nullptr);
        if (!bp.backend) {
            fprintf(stderr, "[Backend] FATAL: GGML_BACKEND=%s not found. Available:",
                    force_backend);
            for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                fprintf(stderr, " %s", ggml_backend_dev_name(ggml_backend_dev_get(i)));
            }
            fprintf(stderr, "\n");
        }
    } else {
        bp.backend = ggml_backend_init_best();
    }

    if (!bp.backend) {
        fprintf(stderr, "[Backend] %s: no GPU backend available, using CPU only\n", label);
        bp.backend     = cpu_backend_new(backend_cpu_n_threads());
        bp.cpu_backend = bp.backend;
        bp.has_gpu     = false;
    } else {
        bool best_is_cpu = (strcmp(ggml_backend_name(bp.backend), "CPU") == 0);
        int  n_threads   = backend_cpu_n_threads();
        if (best_is_cpu) {
            bp.cpu_backend = bp.backend;
            bp.has_gpu     = false;
        } else {
            bp.cpu_backend = cpu_backend_new(n_threads);
            bp.has_gpu     = true;
        }
    }

    if (!bp.cpu_backend) {
        fprintf(stderr, "[Backend] FATAL: %s: failed to init CPU backend\n", label);
    }

    fprintf(stderr, "[Backend] %s: primary=%s, cpu=%s, has_gpu=%d\n",
            label,
            bp.backend     ? ggml_backend_name(bp.backend)     : "NULL",
            bp.cpu_backend ? ggml_backend_name(bp.cpu_backend) : "NULL",
            (int) bp.has_gpu);
    return bp;
}

// Create a scheduler from a backend pair.
// max_nodes: graph size hint (8192 for DiT/VAE).
// When a GPU is present, use its host buffer type for the CPU backend.
inline ggml_backend_sched_t backend_sched_new(BackendPair bp, int max_nodes) {
    ggml_backend_t             backends[2] = { bp.backend, bp.cpu_backend };
    ggml_backend_buffer_type_t bufts[2]    = { nullptr, nullptr };
    int                        n           = (bp.backend == bp.cpu_backend) ? 1 : 2;

    bufts[0] = ggml_backend_get_default_buffer_type(bp.backend);
    if (n == 2) {
        ggml_backend_dev_t         gpu_dev   = ggml_backend_get_device(bp.backend);
        ggml_backend_buffer_type_t host_buft = gpu_dev ? ggml_backend_dev_host_buffer_type(gpu_dev) : nullptr;
        bufts[1] = host_buft ? host_buft : ggml_backend_get_default_buffer_type(bp.cpu_backend);
    }

    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, n, max_nodes, false, true);
    if (!sched) {
        fprintf(stderr, "[Backend] FATAL: failed to create scheduler\n");
    }
    return sched;
}

// Move all tensors in `ctx` from their current data pointers (e.g. GGUF mmap)
// to the backend's buffer. Steps:
//   1. Snapshot each tensor's data pointer
//   2. Clear data pointers so alloc_ctx_tensors will allocate them
//   3. Allocate on backend buffer
//   4. Copy original data into the backend buffer
//
// Returns the backend buffer (caller frees when ctx is freed) or nullptr on
// failure. After this call, tensor->data points into the backend buffer and
// tensor->buffer is set; the scheduler will route ops to the backend.
inline ggml_backend_buffer_t migrate_ctx_to_backend(ggml_context* ctx,
                                                     ggml_backend_t backend,
                                                     const char* label) {
    if (!ctx || !backend) return nullptr;

    // Step 1+2: snapshot data pointers, then clear.
    struct Snapshot { ggml_tensor* t; void* data; };
    std::vector<Snapshot> snaps;
    for (ggml_tensor* t = ggml_get_first_tensor(ctx); t != nullptr;
         t = ggml_get_next_tensor(ctx, t)) {
        if (t->data != nullptr && t->view_src == nullptr) {
            snaps.push_back({t, t->data});
            t->data = nullptr;
        }
    }
    if (snaps.empty()) {
        fprintf(stderr, "[Backend] %s: no tensors to migrate\n", label);
        return nullptr;
    }

    // Step 3: allocate on backend buffer.
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        // Restore data pointers on failure so callers still work CPU-side.
        for (auto& s : snaps) s.t->data = s.data;
        fprintf(stderr, "[Backend] %s: alloc_ctx_tensors failed, weights stay on CPU\n", label);
        return nullptr;
    }

    // Step 4: copy original data into the backend buffer.
    size_t copied = 0;
    for (auto& s : snaps) {
        size_t nbytes = ggml_nbytes(s.t);
        ggml_backend_tensor_set(s.t, s.data, 0, nbytes);
        copied += nbytes;
    }

    fprintf(stderr, "[Backend] %s: migrated %zu tensors (%.1f MB) to %s\n",
            label, snaps.size(), (float) copied / (1024.0f * 1024.0f),
            ggml_backend_name(backend));
    return buf;
}

}  // namespace audiocore::ggml_utils

#endif  // AUDIOCORE_FRAMEWORK_GGML_BACKEND_HELPER_H
