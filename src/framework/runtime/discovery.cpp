// discovery.cpp — llama.cpp-style model directory auto-discovery.
//
// Walks a root directory one level deep and emits one DiscoveredModel per
// subdir whose name matches a registered family (after normalization).
//
// Layout audiocore expects (matching the docker/server.example.json
// conventions and the per-family loaders' directory resolution):
//
//   /models/                   ← --model-dir points here
//     moss-tts/                → DiscoveredModel{id="moss-tts", family="moss_tts"}
//       moss-tts-q8_0.gguf       (the moss_tts loader picks its file itself)
//     qwen3-tts/               → DiscoveredModel{id="qwen3-tts", family="qwen3_tts"}
//       talker.q5_k.gguf          (qwen3_tts loader resolves siblings by name)
//       predictor.q8_0.gguf
//       tokenizer-f16.gguf
//     ace-step/                → DiscoveredModel{id="ace-step", family="ace_step"}
//       acestep-v15-turbo-*.gguf  (ace_step loader finds its 4 GGUFs by pattern)
//       5Hz-lm-1.7B-*.gguf
//       Qwen3-Embedding-*.gguf
//       vae-*.gguf
//
// Why directory-only for v1: each per-family loader already knows how to
// resolve its own files within a directory (see moss_tts/loader.cpp,
// qwen3_tts/loader.cpp, ace_step/loader.cpp). Discovery's job is just
// "which family does this subdir belong to?" — name matching is enough.
// Loose .gguf files at the root need metadata sniffing and arrive in
// Phase 2 (see discovery.h docstring).

#include "audiocore/framework/runtime/discovery.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <set>
#include <system_error>

#include "audiocore/framework/io/gguf_reader.h"
#include "audiocore/framework/io/weight_loader.h"
#include "audiocore/framework/runtime/registry.h"

namespace audiocore {

namespace fs = std::filesystem;

namespace {

// Normalize a directory name to the snake_case family key FamilyRegistry
// uses. Handles the three conventions seen in the wild:
//   "moss-tts"     → "moss_tts"   (kebab, docker/server.example.json)
//   "MOSS_TTS"     → "moss_tts"   (uppercase snake, HF repo names)
//   "moss tts"     → "moss_tts"   (defensive — spaces in dirnames)
// Existing underscores are preserved verbatim. Dots are converted too so
// "qwen3.tts" would also resolve, though the canonical form is kebab.
std::string normalize_family_name(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        if (c == '-' || c == ' ' || c == '.') out.push_back('_');
        else out.push_back(static_cast<char>(std::tolower(c)));
    }
    return out;
}

// Sniff the family of a .gguf file from its metadata. Used by
// from_single_path() when the user points at a file rather than a
// directory. Returns "" if no registered family can be identified.
//
// Probe order (most specific → most ambiguous):
//   1. general.architecture KV — unambiguous for qwen3tts / qwen3tts_cp /
//      acestep-dit. "qwen3" is shared between moss_tts backbone and
//      ace_step LM/TE, so we fall through.
//   2. Tensor-name prefixes — family-unique even when arch is ambiguous.
//      moss.*, acestep*, speaker.* are all family-specific.
//   3. Family-specific KV prefixes as a last resort.
std::string sniff_family_from_gguf(const std::string& path) {
    std::string err;
    auto loader = make_weight_loader(path, &err);
    if (!loader) return "";
    auto* gguf = dynamic_cast<GgufReader*>(loader.get());
    if (!gguf) return "";

    std::string arch;
    if (gguf->get_kv_str("general.architecture", &arch)) {
        if (arch == "qwen3tts" || arch == "qwen3tts_cp") return "qwen3_tts";
        if (arch == "acestep-dit" || arch == "acestep") return "ace_step";
        // "qwen3" falls through — it's the backbone arch for both
        // moss_tts and ace_step LM/TE. Tensor probing disambiguates.
    }

    for (const TensorStorage& t : loader->tensors()) {
        if (t.name.rfind("moss.", 0) == 0)     return "moss_tts";
        if (t.name.rfind("acestep", 0) == 0)   return "ace_step";
        if (t.name.rfind("speaker.", 0) == 0)  return "qwen3_tts";
    }

    int32_t dummy = 0;
    if (gguf->get_kv_i32("moss.n_vq", &dummy)) return "moss_tts";
    std::string cfg_json;
    if (gguf->get_kv_str("acestep.config_json", &cfg_json)) return "ace_step";

    return "";
}

}  // namespace

std::vector<DiscoveredModel> discover_models(const std::string& root_dir,
                                             std::string* error) {
    std::vector<DiscoveredModel> out;

    std::error_code ec;
    fs::path root(root_dir);
    if (!fs::exists(root, ec)) {
        if (error) *error = "model-dir '" + root.string() + "' does not exist";
        return out;
    }
    if (!fs::is_directory(root, ec)) {
        if (error) *error = "model-dir '" + root.string() + "' is not a directory";
        return out;
    }

    // FamilyRegistry::list() returns the snake_case keys each family
    // registered via AUDIOCORE_REGISTER_FAMILY. Build a set for O(log n)
    // lookup; the registry is small today but this scales without thought.
    const auto registered = FamilyRegistry::instance().list();
    const std::set<std::string> known(registered.begin(), registered.end());

    for (const auto& entry : fs::directory_iterator(root, ec)) {
        if (ec) {
            if (error) *error = "error iterating '" + root.string() + "': " + ec.message();
            break;
        }
        const fs::path& p = entry.path();

        // Phase 1: directories only. Loose .gguf files at the root need
        // metadata sniffing (Phase 2); silently skip everything else so a
        // stray README or tarball doesn't break boot.
        if (!fs::is_directory(p, ec)) continue;

        const std::string dirname = p.filename().string();
        // Skip hidden directories (.git, .cache, etc.) — same rule as
        // llama.cpp's --model-dir skip list.
        if (!dirname.empty() && dirname[0] == '.') continue;

        const std::string family = normalize_family_name(dirname);
        if (known.find(family) == known.end()) {
            std::fprintf(stderr,
                "discovery: skipping '%s' — no registered family matches '%s'\n",
                dirname.c_str(), family.c_str());
            continue;
        }

        DiscoveredModel m;
        m.id      = dirname;             // preserve user's casing for the model id
        m.family  = family;              // normalized for FamilyRegistry dispatch
        m.path    = p.string();
        m.backend = "ggml_cuda";
        out.push_back(std::move(m));
    }

    // Deterministic order: sort by id so /v1/models output is stable across
    // boots and filesystems (directory_iterator order is unspecified).
    std::sort(out.begin(), out.end(),
              [](const DiscoveredModel& a, const DiscoveredModel& b) {
                  return a.id < b.id;
              });
    return out;
}

std::optional<DiscoveredModel> from_single_path(
    const std::string& model_path,
    const std::string& family_override,
    std::string* error) {

    std::error_code ec;
    fs::path p(model_path);
    if (!fs::exists(p, ec)) {
        if (error) *error = "model path '" + model_path + "' does not exist";
        return std::nullopt;
    }

    DiscoveredModel m;
    m.path    = p.string();
    m.backend = "ggml_cuda";

    if (fs::is_directory(p, ec)) {
        // Directory case: id from dirname, family from normalized dirname.
        // Matches discover_models() so `--model /models/moss-tts` and
        // `--model-dir /models` (with only moss-tts/ inside) land on the
        // same id/family.
        const std::string dirname = p.filename().string();
        m.id     = dirname;
        m.family = family_override.empty()
                       ? normalize_family_name(dirname)
                       : family_override;
    } else {
        // File case: id from filename stem (e.g. "moss-tts-q8_0.gguf" →
        // "moss-tts-q8_0"). Family sniffed from GGUF metadata unless the
        // user overrides via --family.
        m.id     = p.stem().string();
        m.family = family_override.empty()
                       ? sniff_family_from_gguf(model_path)
                       : family_override;
    }

    if (m.family.empty()) {
        if (error) {
            *error = family_override.empty()
                ? "could not infer family for '" + model_path +
                  "' — open the GGUF with inspect_gguf to see what's there, "
                  "or pass --family <name> explicitly"
                : "family override is empty";
        }
        return std::nullopt;
    }

    // Validate against the registry — catches typos in --family and
    // reveals sniff misses (e.g. a community GGUF with no architecture
    // key and unfamiliar tensor names).
    const auto registered = FamilyRegistry::instance().list();
    if (std::find(registered.begin(), registered.end(), m.family) == registered.end()) {
        if (error) {
            *error = "family '" + m.family + "' is not registered. Known: ";
            for (size_t i = 0; i < registered.size(); ++i) {
                *error += (i ? ", " : "") + registered[i];
            }
        }
        return std::nullopt;
    }

    return m;
}

}  // namespace audiocore
