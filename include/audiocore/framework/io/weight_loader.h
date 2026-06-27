// weight_loader.h — format-agnostic weight loading interface.
//
// Each backend (GGUF, safetensors, ONNX) implements WeightLoader.
// Model code only ever sees TensorStorage descriptors, never the file
// format directly. This is the seam that lets us add ONNX without
// touching model code: ONNX weights just produce TensorStorage entries
// like every other format.

#ifndef AUDIOCORE_FRAMEWORK_IO_WEIGHT_LOADER_H
#define AUDIOCORE_FRAMEWORK_IO_WEIGHT_LOADER_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor_storage.h"

namespace audiocore {

// Abstract weight loader. Reads a model directory (or single file),
// produces TensorStorage metadata for every tensor. Materialization
// into a ggml_context happens lazily, on demand, via materialize().
class WeightLoader {
public:
    virtual ~WeightLoader() = default;

    // Populate `tensors_` from the model path. Returns false on failure
    // (with error filled in if non-null).
    virtual bool load(const std::string& path, std::string* error = nullptr) = 0;

    // All tensor descriptors discovered by load(). Indexed by name in the
    // map for O(1) lookup by model code.
    const std::vector<TensorStorage>& tensors() const { return tensors_; }
    const TensorStorage* find(const std::string& name) const;

    // Materialize one tensor into a destination buffer. The buffer must be
    // sized to TensorStorage::nbytes() and aligned per the backend's rules.
    // Implementations may convert f8/f64/i64 → f16/f32 here.
    virtual bool materialize(const TensorStorage& t, void* dst,
                             std::string* error = nullptr) const = 0;

protected:
    std::vector<TensorStorage> tensors_;
};

// Format dispatch by file magic. Returns a loader appropriate for the
// path (single .gguf file, .safetensors directory, .onnx file, …) or
// nullptr if no loader matches.
std::unique_ptr<WeightLoader> make_weight_loader(const std::string& path,
                                                 std::string* error = nullptr);

}  // namespace audiocore

#endif  // AUDIOCORE_FRAMEWORK_IO_WEIGHT_LOADER_H
