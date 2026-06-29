// discovery.h — llama.cpp-style model directory auto-discovery.
//
// Given a root directory (typically /models in the container, or wherever
// the user pointed --model-dir), walk it one level deep and emit one
// DiscoveredModel per subdir whose name matches a registered family.
//
// Resolution is intentionally name-based for v1 — the per-family loaders
// already do their own file resolution within a directory, so discovery
// only needs to pick the right FAMILY for each subdir, not the right
// files. See discovery.cpp for the kebab→snake normalization rules.
//
// Phase 2 will add loose-*.gguf sniffing (family inferred from GGUF
// metadata) so a flat /models/foo.gguf also auto-registers.

#ifndef AUDIOCORE_FRAMEWORK_RUNTIME_DISCOVERY_H
#define AUDIOCORE_FRAMEWORK_RUNTIME_DISCOVERY_H

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

}  // namespace audiocore

#endif  // AUDIOCORE_FRAMEWORK_RUNTIME_DISCOVERY_H
