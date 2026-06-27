// audiocore_cli — single-shot offline inference.
//
// Mirrors audio.cpp's CLI shape:
//   audiocore_cli --task tts --family moss_tts --model /path --backend cuda \
//                 --text "Hello." --voice-ref ref.wav --out out.wav
//   audiocore_cli --task music --family ace_step --model /path/to/dir \
//                 --dit turbo-Q8_0.gguf --lm 5Hz-lm-1.7B-Q8_0.gguf \
//                 --caption "ambient" --duration 30 --out out.mp3
//
// Used for:
//   - one-off generation
//   - parity testing against reference C++ implementations
//   - regression testing in CI

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    // TODO(Phase 2): wire to FamilyRegistry + Session + audio I/O.
    std::fprintf(stderr, "audiocore_cli: scaffold only\n");
    for (int i = 1; i < argc; ++i) std::fprintf(stderr, "  arg: %s\n", argv[i]);
    return 0;
}
