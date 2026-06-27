// backend.cpp — ggml backend factory.
//
// Today only the ggml family (CUDA/CPU/Vulkan/Metal). The Backend abstract
// base is the seam ONNX Runtime will plug into in Phase 2; the factory
// here is the only place that knows how to construct each kind.

#include "audiocore/framework/core/backend.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <stdexcept>

namespace audiocore {

namespace {

class GgmlBackend : public Backend {
public:
    explicit GgmlBackend(const BackendConfig& cfg) : cfg_(cfg), kind_(cfg.kind) {
        // ggml picks its device based on cfg.kind; we just initialize the
        // backend registry lazily on first use.
        ggml_backend_dev_t dev = nullptr;
        switch (cfg.kind) {
            case BackendKind::ggml_cuda:
                dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
                break;
            case BackendKind::ggml_cpu:
                dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
                break;
            case BackendKind::ggml_vulkan:
            case BackendKind::ggml_metal:
                // Selected by name once those backends are linked in.
                dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
                break;
            case BackendKind::onnxruntime:
                throw std::runtime_error("ONNX Runtime backend not yet implemented");
        }
        if (!dev) throw std::runtime_error("no matching ggml backend device");
        backend_ = ggml_backend_dev_init(dev, nullptr);
        if (!backend_) throw std::runtime_error("ggml_backend_dev_init failed");
    }
    ~GgmlBackend() override {
        if (backend_) ggml_backend_free(backend_);
    }

    BackendKind kind() const override { return kind_; }
    ggml_backend_t raw() const { return backend_; }

    bool execute(void* graph_handle, void* /*io_bindings*/,
                 std::string* error) override {
        auto* gf = static_cast<ggml_cgraph*>(graph_handle);
        if (!ggml_backend_graph_compute(backend_, gf)) {
            if (error) *error = "ggml_backend_graph_compute failed";
            return false;
        }
        return true;
    }

private:
    BackendConfig cfg_;
    BackendKind kind_;
    ggml_backend_t backend_ = nullptr;
};

}  // namespace

std::unique_ptr<Backend> make_backend(const BackendConfig& cfg,
                                      std::string* error) {
    try {
        return std::make_unique<GgmlBackend>(cfg);
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return nullptr;
    }
}

}  // namespace audiocore
