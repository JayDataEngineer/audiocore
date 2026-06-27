// weight_loader.cpp — format-agnostic weight loader dispatch.
//
// Picks a WeightLoader subclass by inspecting the path's file magic.
// Today only GGUF; Phase 2 adds safetensors and ONNX readers behind the
// same interface. Model code calls make_weight_loader() once and asks
// for tensors by name — it never sees the format.

#include "audiocore/framework/io/weight_loader.h"

#include <cstdint>
#include <fstream>
#include "audiocore/framework/io/gguf_reader.h"

namespace audiocore {

const TensorStorage* WeightLoader::find(const std::string& name) const {
    for (const auto& t : tensors_) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

std::unique_ptr<WeightLoader> make_weight_loader(const std::string& path,
                                                 std::string* error) {
    // GGUF: 4-byte magic "GGUF" at file start.
    if (GgufReader::is_gguf_file(path)) {
        auto reader = std::make_unique<GgufReader>();
        if (!reader->load(path, error)) return nullptr;
        return reader;
    }
    // TODO(Phase 2): safetensors (8-byte header length prefix, JSON header).
    // TODO(Phase 2): ONNX initializer enumeration via onnx::ModelProto.

    if (error) *error = "no weight loader matched path: " + path;
    return nullptr;
}

}  // namespace audiocore
