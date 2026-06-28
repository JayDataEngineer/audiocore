// manifest.h — machine-readable model catalog reader.
//
// `models/manifest.json` is the source of truth for what audiocore can
// load. This header is the C++ view of it. Two consumers:
//
//   • audiocore_cli --list-supported     prints the family/variant/mode matrix
//   • audiocore_server (future)          validates server.json entries against
//                                        the known set so a typo in a model
//                                        id fails fast
//
// The manifest lives outside the binary so a doc update doesn't force a
// rebuild — find_manifest() walks up from the executable + checks the CWD
// for a `models/manifest.json` file. If neither exists (e.g. an installed
// binary without the repo checked out next to it), the caller gets an
// empty Manifest and can fall back to listing the families it actually
// registered at static-init time via FamilyRegistry::list().

#ifndef AUDIOCORE_FRAMEWORK_RUNTIME_MANIFEST_H
#define AUDIOCORE_FRAMEWORK_RUNTIME_MANIFEST_H

#include <cstdint>
#include <string>
#include <vector>

namespace audiocore {

// One file in a variant. Mirrors models/manifest.json → families.<f>.variants.<v>.files[].
struct ManifestFile {
    std::string name;            // logical role: "backbone_gguf", "dit", "lm", ...
    std::string filename;        // on-disk filename within the variant dir
    std::string provider;        // "huggingface" today; room for "url", "git", ...
    std::string repo;            // e.g. "OpenMOSS-Team/MOSS-TTS-GGUF"
    std::string revision;        // HF revision / git ref — usually "main"
    std::string source_filename; // path inside the repo (may differ from filename)
    std::string source_subpath;  // for repos where files live under a subdir
    std::string convert_with;    // post-processing tool: convert_acestep / convert_qwen3tts
    std::string sha256;          // optional; empty = skip verification
};

// One variant of a family.
struct ManifestVariant {
    std::string key;             // e.g. "qwen3-tts-1.7b-base"
    std::string display;         // human-readable name
    double      params_b   = 0;  // billions
    int         min_vram_gb = 0;
    std::vector<ManifestFile> files;
};

// One mode a family exposes.
struct ManifestMode {
    std::string key;             // "tts", "sfx", "voice_clone", "text_to_music", ...
    std::string status;          // "wired", "partial", "not_impl", "blocked"
    std::string notes;
};

// One family.
struct ManifestFamily {
    std::string key;             // "moss_tts", "qwen3_tts", "ace_step"
    std::string display;
    std::string vendor;
    std::string license;
    std::string homepage;
    std::vector<ManifestMode>    modes;
    std::vector<ManifestVariant> variants;
};

// Parsed view of models/manifest.json.
struct Manifest {
    int32_t manifest_version = 0;
    std::vector<ManifestFamily> families;

    bool empty() const { return families.empty(); }

    // Look up a family by key. Returns nullptr if absent.
    const ManifestFamily* find_family(const std::string& key) const;
};

// Locate and parse models/manifest.json. Returns an empty Manifest on
// failure (with `*error` filled in). Search order:
//   1. $AUDIOCORE_MANIFEST
//   2. <exe_dir>/models/manifest.json
//   3. <exe_dir>/../models/manifest.json
//   4. ./models/manifest.json
//   5. ./manifest.json
Manifest load_manifest(std::string* error = nullptr);

// Render the family/mode matrix as the same text `scripts/fetch_models.sh
// --list` prints. Used by `audiocore_cli --list-supported`. If `families`
// is empty, prints "(no manifest found — falling back to registry)" and
// then the names from FamilyRegistry::list().
std::string render_mode_matrix(const Manifest& m);

}  // namespace audiocore

#endif  // AUDIOCORE_FRAMEWORK_RUNTIME_MANIFEST_H
