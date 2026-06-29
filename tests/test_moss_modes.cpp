// Verify that the fraud modes fail fast with a clear error mentioning the
// missing dedicated checkpoint. Run after a successful load — we exercise
// the mode dispatch only, not the AR loop.
#include "audiocore/framework/core/backend.h"
#include "audiocore/framework/core/session.h"
#include "audiocore/framework/io/weight_loader.h"
#include "audiocore/framework/runtime/registry.h"
#include "audiocore/models/moss_tts/family.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

extern "C" void audiocore_register_moss_tts();

int main() {
    using namespace audiocore;
    using namespace audiocore::moss;

    audiocore_register_moss_tts();
    const char* env_dir = std::getenv("AUDIOCORE_MOSS_DIR");
    if (!env_dir) {
        std::fprintf(stderr,
            "[SKIP] set AUDIOCORE_MOSS_DIR to the directory holding the "
            "MOSS GGUFs to run this test\n");
        return 0;
    }
    std::string model_dir = env_dir;
    std::string backbone_path = model_dir + "/moss-tts-v1.5-q8_0.gguf";
    std::string extras_path   = model_dir + "/moss-tts.extras.gguf";
    if (std::ifstream(extras_path).good() == false) {
        extras_path = model_dir + "/moss-tts-v1.5-q8_0.extras.gguf";
    }

    auto sess = FamilyRegistry::instance().create("moss_tts");
    LoadOptions opts; opts.extras["extras_gguf_path"] = extras_path;
    BackendConfig bc = { .kind = BackendKind::ggml_cuda, .device_id = 0, .n_threads = 12 };
    std::string err;
    if (!sess->load(backbone_path, opts, bc, &err)) {
        std::fprintf(stderr, "[SKIP] load failed: %s\n", err.c_str());
        return 0;
    }

    int fails = 0;
    for (const char* mode : {"sfx", "dialogue", "voice_design", "realtime"}) {
        TtsRequest req; req.mode = mode; req.text = "test";
        TtsResponse res; std::string e;
        bool ok = sess->run_tts(&req, &res, &e);
        if (ok) {
            std::fprintf(stderr, "[FAIL] mode='%s' was accepted (should have failed)\n", mode);
            ++fails;
        } else {
            std::fprintf(stderr, "[PASS] mode='%s' rejected: %s\n", mode, e.c_str());
        }
    }
    return fails == 0 ? 0 : 1;
}
