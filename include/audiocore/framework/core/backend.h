// backend.h — execution backend abstraction.
//
// This is the seam between "weight format" and "execution runtime".
// Only the ggml backend is supported (CUDA/CPU/Vulkan/Metal). All weight
// formats are GGUF; the deprecated ONNX Runtime peer backend has been
// removed.
//
// Model code never sees this directly. It works through Session, which
// owns a Backend instance and exposes task-specific calls (run_tts,
// run_music, …) that map onto the backend's run() primitive.

#ifndef AUDIOCORE_FRAMEWORK_CORE_BACKEND_H
#define AUDIOCORE_FRAMEWORK_CORE_BACKEND_H

#include <memory>
#include <string>

namespace audiocore {

enum class BackendKind {
    ggml_cuda,
    ggml_cpu,
    ggml_vulkan,
    ggml_metal,
};

struct BackendConfig {
    BackendKind kind     = BackendKind::ggml_cuda;
    int device_id        = 0;
    int n_threads        = 1;
    bool cuda_graphs     = true;
    // …backend-specific tuning flags live here.
};

// Owned by a Session. Maps task-level compute graphs onto the active
// backend. The ggml backend builds ggml_cgraphs; the ONNX backend will
// build ORT sessions. The interface is identical from the model's POV.
class Backend {
public:
    virtual ~Backend() = default;
    virtual BackendKind kind() const = 0;

    // Compute graph submission. The opaque `graph_handle` is created by
    // the model code via a backend-specific builder (ggml_cgraph for ggml,
    // Ort::Session for ONNX). Materializing the implementation behind one
    // interface lets us swap backends without rewriting model code.
    virtual bool execute(void* graph_handle, void* io_bindings,
                         std::string* error = nullptr) = 0;
};

std::unique_ptr<Backend> make_backend(const BackendConfig& cfg,
                                      std::string* error = nullptr);

}  // namespace audiocore

#endif  // AUDIOCORE_FRAMEWORK_CORE_BACKEND_H
