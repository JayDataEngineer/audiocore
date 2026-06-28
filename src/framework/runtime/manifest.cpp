// manifest.cpp — read models/manifest.json into Manifest structs and render
// the mode matrix.

#include "audiocore/framework/runtime/manifest.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "audiocore/framework/runtime/registry.h"

namespace audiocore {

namespace {

using nlohmann::json;

// Read /proc/self/exe (Linux) or _NSGetExecutablePath (macOS). Falls back
// to argv[0]-ish behaviour (returns CWD) on platforms without either.
std::string executable_dir() {
#if defined(__linux__)
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        std::string s(buf);
        const auto slash = s.find_last_of('/');
        return (slash != std::string::npos) ? s.substr(0, slash) : s;
    }
#elif defined(__APPLE__)
    char buf[4096];
    uint32_t sz = sizeof(buf);
    if (_NSGetExecutablePath(buf, &sz) == 0) {
        std::string s(buf);
        const auto slash = s.find_last_of('/');
        return (slash != std::string::npos) ? s.substr(0, slash) : s;
    }
#endif
    char cwd[4096];
    if (getcwd(cwd, sizeof(cwd))) return cwd;
    return ".";
}

bool file_exists(const std::string& path) {
    struct stat st{};
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool read_text_file(const std::string& path, std::string* out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    *out = ss.str();
    return true;
}

ManifestFile parse_file(const json& j) {
    ManifestFile f;
    f.name     = j.value("name", "");
    f.filename = j.value("filename", "");
    if (j.contains("source") && j["source"].is_object()) {
        const auto& s = j["source"];
        f.provider        = s.value("provider", "");
        f.repo            = s.value("repo", "");
        f.revision        = s.value("revision", "main");
        f.source_filename = s.value("filename", "");
        f.source_subpath  = s.value("subpath", "");
        f.convert_with    = s.value("convert_with", s.value("post_process", ""));
        f.sha256          = s.value("sha256", "");
    }
    return f;
}

}  // namespace

const ManifestFamily* Manifest::find_family(const std::string& key) const {
    for (const auto& f : families) {
        if (f.key == key) return &f;
    }
    return nullptr;
}

Manifest load_manifest(std::string* error) {
    Manifest m;
    std::string path;

    if (const char* env = std::getenv("AUDIOCORE_MANIFEST")) {
        if (file_exists(env)) path = env;
    }
    if (path.empty()) {
        const std::string exe = executable_dir();
        const char* candidates[] = {
            (exe + "/models/manifest.json").c_str(),
            (exe + "/../models/manifest.json").c_str(),
            "models/manifest.json",
            "manifest.json",
        };
        for (const char* c : candidates) {
            if (file_exists(c)) { path = c; break; }
        }
    }
    if (path.empty()) {
        if (error) *error = "models/manifest.json not found";
        return m;
    }

    std::string text;
    if (!read_text_file(path, &text)) {
        if (error) *error = "could not read " + path;
        return m;
    }

    json root;
    try { root = json::parse(text); }
    catch (const std::exception& e) {
        if (error) *error = std::string("manifest parse error: ") + e.what();
        return m;
    }

    m.manifest_version = root.value("manifest_version", 0);
    if (!root.contains("families") || !root["families"].is_object()) {
        if (error) *error = "manifest missing 'families' object";
        return m;
    }
    for (const auto& [fkey, fobj] : root["families"].items()) {
        ManifestFamily f;
        f.key      = fkey;
        f.display  = fobj.value("display", fkey);
        f.vendor   = fobj.value("vendor", "");
        f.license  = fobj.value("license", "");
        f.homepage = fobj.value("homepage", "");
        if (fobj.contains("modes") && fobj["modes"].is_object()) {
            for (const auto& [mkey, mobj] : fobj["modes"].items()) {
                ManifestMode mode;
                mode.key    = mkey;
                mode.status = mobj.value("status", "");
                mode.notes  = mobj.value("notes", "");
                f.modes.push_back(std::move(mode));
            }
        }
        if (fobj.contains("variants") && fobj["variants"].is_object()) {
            for (const auto& [vkey, vobj] : fobj["variants"].items()) {
                ManifestVariant v;
                v.key         = vkey;
                v.display     = vobj.value("display", vkey);
                v.params_b    = vobj.value("params_b", 0.0);
                v.min_vram_gb = vobj.value("min_vram_gb", 0);
                if (vobj.contains("files") && vobj["files"].is_array()) {
                    for (const auto& fj : vobj["files"]) v.files.push_back(parse_file(fj));
                }
                f.variants.push_back(std::move(v));
            }
        }
        m.families.push_back(std::move(f));
    }
    return m;
}

std::string render_mode_matrix(const Manifest& m) {
    std::ostringstream o;
    if (m.empty()) {
        o << "(no manifest found — falling back to FamilyRegistry)\n\n";
        for (const auto& name : FamilyRegistry::instance().list()) {
            o << "  " << name << "\n";
        }
        return o.str();
    }
    // Status → glyph, matching GAPS.md.
    auto glyph = [](const std::string& s) -> const char* {
        if (s == "wired")    return "[+]";   // implemented end-to-end
        if (s == "partial")  return "[~]";   // runs but emits silence / stubbed stage
        if (s == "not_impl") return "[ ]";   // not implemented, achievable
        if (s == "blocked")  return "[!]";   // blocked on new model / major port
        return "[?]";
    };
    for (const auto& f : m.families) {
        o << "=== " << f.key << " — " << f.display << " ===\n";
        if (!f.vendor.empty())   o << "  vendor:   " << f.vendor   << "\n";
        if (!f.license.empty())  o << "  license:  " << f.license  << "\n";
        if (!f.homepage.empty()) o << "  homepage: " << f.homepage << "\n";
        o << "\n  Modes:\n";
        for (const auto& mode : f.modes) {
            o << "    " << glyph(mode.status) << " " << mode.key;
            // pad to align notes column at 24 chars
            int pad = 22 - static_cast<int>(mode.key.size());
            if (pad < 1) pad = 1;
            o << std::string(pad, ' ') << mode.notes << "\n";
        }
        o << "\n  Variants:\n";
        for (const auto& v : f.variants) {
            char head[128];
            std::snprintf(head, sizeof(head), "%.1fB params, ≥%dGB VRAM",
                          v.params_b, v.min_vram_gb);
            o << "    • " << v.key << " — " << v.display
              << " (" << head << ")\n";
        }
        o << "\n";
    }
    o << "Legend:  [+] wired   [~] partial (stubbed stage)   "
      << "[ ] not implemented   [!] blocked on new model/port\n"
      << "         see GAPS.md for the full audit.\n";
    return o.str();
}

}  // namespace audiocore
