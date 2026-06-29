// discovery.h — llama.cpp-style model resolution.
//
// Two entry points, both used by main.cpp at server boot:
//
//   discover_models(root)       — --model-dir mode. Walks the root one
//                                 level deep, emits one DiscoveredModel
//                                 per subdir whose name matches a
//                                 registered family.
//
//   from_single_path(path)      — --model mode (recommended). Resolves a
//                                 single file or directory to exactly one
//                                 DiscoveredModel. Family is inferred from
//                                 the dirname (directory case) or sniffed
//                                 from GGUF metadata (file case). This is
//                                 the llama.cpp server pattern: one model
//                                 per container, swap by restarting.
//
// Per-family loaders already do their own file resolution within a
// directory, so discovery only picks the right FAMILY — not the files.

#ifndef AUDIOCORE_FRAMEWORK_RUNTIME_DISCOVERY_H
#define AUDIOCORE_FRAMEWORK_RUNTIME_DISCOVERY_H

#include <optional>
#include <string>
#include <vector>

#include "audiocore/framework/core/session.h"   // LoadOptions

namespace audiocore {

struct DiscoveredModel {
    std::string id;          // user-facing model id (preserves the dirname's casing)
    std::string family;      // registered family name (snake_case, e.g. "moss_tts")
    std::string path;        // absolute path to the model directory
    std::string backend = "ggml_cuda";
    LoadOptions load_options;
};

// Walk root_dir non-recursively. For each subdirectory whose name (after
// kebab→snake normalization) matches a registered family, emit a
// DiscoveredModel pointing at that subdirectory. Subdirs that don't match
// any registered family are warned about on stderr and skipped (mirrors
// llama.cpp's --model-dir tolerance of unrelated files / READMEs / etc.).
//
// Returns the discovered list, sorted by id for stable /v1/models output.
// Sets *error only on hard failures (root doesn't exist, isn't a
// directory, permission denied). Empty result without error means
// "scanned fine, nothing recognized".
std::vector<DiscoveredModel> discover_models(const std::string& root_dir,
                                             std::string* error = nullptr);

// Resolve a single model path to one DiscoveredModel — the llama.cpp-style
// "--model /path" entry point. This is the recommended way to run the
// server: one model per container, swap by restarting with a different
// path. Bypasses the JSON models array entirely when set.
//
//   model_path is a directory  → id = dirname, family inferred from
//                                dirname (kebab→snake normalized)
//   model_path is a .gguf file → id = filename stem, family sniffed from
//                                GGUF metadata (general.architecture,
//                                then tensor/KV prefixes)
//
// family_override (if non-empty) bypasses both inference paths and is
// validated against the registry verbatim. Use it for layouts the sniffer
// can't resolve, or to override a wrong guess.
//
// Returns std::nullopt and sets *error on: missing path, undeterminable
// family, family not in the registry.
std::optional<DiscoveredModel> from_single_path(
    const std::string& model_path,
    const std::string& family_override = "",
    std::string* error = nullptr);

}  // namespace audiocore

#endif  // AUDIOCORE_FRAMEWORK_RUNTIME_DISCOVERY_H
