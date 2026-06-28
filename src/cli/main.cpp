// audiocore_cli — single-shot offline inference + self-describe.
//
// Today this is a thin surface; the heavy lifting (load family, build
// session, submit task) is queued for Phase 2. What IS wired right now:
//
//   --list-supported          print the family/variant/mode matrix read
//                             from models/manifest.json (the same data
//                             `scripts/fetch_models.sh --list` emits)
//   --list-families           print just the families FamilyRegistry has
//                             registered at static-init time
//   --help                    usage
//
// Coming in Phase 2:
//   audiocore_cli --task tts --family moss_tts --model /path --backend cuda \
//                 --text "Hello." --out out.wav
//   audiocore_cli --task music --family ace_step --model /path/to/dir \
//                 --caption "ambient" --duration 30 --out out.wav

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "audiocore/framework/runtime/manifest.h"
#include "audiocore/framework/runtime/registry.h"

// Force the family loader TUs to be linked into the binary. Without these
// calls the linker drops the static registrars (they live in archive
// members nothing else references), and FamilyRegistry is empty at runtime.
// Same trick audiocore_server/main.cpp uses.
extern "C" void audiocore_register_moss_tts();
extern "C" void audiocore_register_ace_step();
extern "C" void audiocore_register_qwen3_tts();

namespace {

void register_all_families() {
    audiocore_register_moss_tts();
    audiocore_register_ace_step();
    audiocore_register_qwen3_tts();
}

int print_help() {
    std::fprintf(stderr,
        "audiocore_cli — offline inference + model catalog\n\n"
        "Usage:\n"
        "  audiocore_cli --list-supported     Print the family/variant/mode matrix\n"
        "                                     from models/manifest.json\n"
        "  audiocore_cli --list-families      Print the registered family names\n"
        "                                     (static-init time)\n"
        "  audiocore_cli --help               This message\n\n"
        "Offline generation (Phase 2):\n"
        "  audiocore_cli --task tts --family moss_tts --model /path \\\n"
        "                 --backend cuda --text \"Hello.\" --out out.wav\n"
        "  audiocore_cli --task music --family ace_step --model /dir \\\n"
        "                 --caption \"ambient\" --out out.wav\n");
    return 0;
}

int print_list_supported() {
    std::string err;
    audiocore::Manifest m = audiocore::load_manifest(&err);
    if (m.empty()) {
        std::fprintf(stderr, "audiocore_cli: %s\n", err.c_str());
        // Still print the registry-only fallback so `--list-supported` is
        // always useful even on an installed binary without the repo next
        // to it.
    }
    std::fputs(audiocore::render_mode_matrix(m).c_str(), stdout);
    return 0;
}

int print_list_families() {
    for (const auto& name : audiocore::FamilyRegistry::instance().list()) {
        std::printf("%s\n", name.c_str());
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    register_all_families();
    bool saw_known = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            return print_help();
        }
        if (std::strcmp(argv[i], "--list-supported") == 0) {
            saw_known = true;
            int rc = print_list_supported();
            if (rc != 0) return rc;
        } else if (std::strcmp(argv[i], "--list-families") == 0) {
            saw_known = true;
            int rc = print_list_families();
            if (rc != 0) return rc;
        } else {
            std::fprintf(stderr, "audiocore_cli: unknown argument '%s' (try --help)\n", argv[i]);
        }
    }
    if (!saw_known) {
        return print_help();
    }
    return 0;
}
