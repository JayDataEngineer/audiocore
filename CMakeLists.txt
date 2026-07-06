# ════════════════════════════════════════════════════════════════════════════
# audiocore — unified audio GGUF inference server.
#
# Build flags mirror audio.cpp's modular pattern so future contributors
# don't have to relearn the conventions:
#
#   ENGINE_ENABLE_CUDA       OFF   ggml CUDA backend
#   ENGINE_ENABLE_VULKAN     OFF   ggml Vulkan backend
#   ENGINE_ENABLE_METAL      OFF   ggml Metal backend (Apple)
#   ENGINE_ENABLE_CUDA_GRAPHS ON   ggml CUDA graphs (CUDA only)
#   ENGINE_ENABLE_OPENMP     ON    host-side parallel work
#   ENGINE_ENABLE_LLAMAFILE  ON    llamafile SGEMM (CPU)
#   ENGINE_BUILD_CLI         ON    audiocore_cli
#   ENGINE_BUILD_SERVER      ON    audiocore_server
#   ENGINE_BUILD_TESTS       OFF   unit tests
# ════════════════════════════════════════════════════════════════════════════
cmake_minimum_required(VERSION 3.18)
project(audiocore LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

option(ENGINE_ENABLE_CUDA        "ggml CUDA backend"            OFF)
option(ENGINE_ENABLE_VULKAN      "ggml Vulkan backend"          OFF)
option(ENGINE_ENABLE_METAL       "ggml Metal backend"           OFF)
option(ENGINE_ENABLE_CUDA_GRAPHS "Use CUDA graphs (CUDA only)"  ON)
option(ENGINE_ENABLE_OPENMP      "Enable OpenMP"                ON)
option(ENGINE_ENABLE_LLAMAFILE   "Enable llamafile SGEMM"       ON)
option(ENGINE_BUILD_CLI          "Build audiocore_cli"          ON)
option(ENGINE_BUILD_SERVER       "Build audiocore_server"       ON)
option(ENGINE_ENABLE_MP3        "Enable MP3 output via libmp3lame" ON)
option(ENGINE_BUILD_WEBAPP      "Embed web UI in server binary"    ON)
option(ENGINE_BUILD_TESTS        "Build unit tests"             OFF)
option(ENGINE_BUILD_PYTHON       "Build Python bindings"        OFF)

# ── llama.cpp (subtree in third_party/llama.cpp) ─────────────────────────
# Pinned via git submodule. Provides libllama (Qwen3 inference) AND vendors
# its own ggml under llama.cpp/ggml/. We use the llama.cpp copy and drop our
# standalone third_party/ggml submodule to avoid symbol clashes.
#
# ACE-Step DiT GGUFs have tensor names up to 67 chars (e.g.
# "double_b.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm1.norm.weight"
# = 73 chars). Stock GGML_MAX_NAME=64 rejects them. Bump to 128.
add_compile_definitions(GGML_MAX_NAME=128)

# Build flags passed through to llama.cpp's build:
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
if(ENGINE_ENABLE_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()
add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)

# ── audiocore framework library ──────────────────────────────────────────
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(audiocore_framework STATIC
    src/framework/io/gguf_reader.cpp
    src/framework/io/gguf_reader_ext.cpp
    src/framework/io/weight_loader.cpp
    src/framework/io/wav.cpp
    # Time-domain WSOLA pitch-shift + speed-change post-processing.
    src/framework/audio/dsp.cpp
    # Unified sampler shared by every family + the qwen3::Runner MTP step.
    src/framework/sampling/sampler.cpp
    src/framework/core/backend.cpp
    src/framework/core/session.cpp
    src/framework/runtime/registry.cpp
    # llama.cpp-style --model-dir auto-discovery. Walks a directory and
    # emits one DiscoveredModel per subdir whose name matches a registered
    # family. See discovery.h for the layout contract.
    src/framework/runtime/discovery.cpp
    # Manifest reader — powers `audiocore_cli --list-supported` and the
    # server's future server.json validator. Reads models/manifest.json,
    # the same file scripts/fetch_models.sh consumes.
    src/framework/runtime/manifest.cpp
    # The ONE Qwen3 transformer path in audiocore. Used by every family
    # whose model includes a Qwen3 backbone (MOSS 8B) or Qwen3 LM / TE
    # (ACE-Step 5Hz LM, ACE-Step Qwen3-Embedding).
    src/framework/models/qwen3/runner.cpp
    # Per-family code:
    src/models/moss_tts/loader.cpp
    src/models/moss_tts/session.cpp
    src/models/moss_tts/codec.cpp
    src/models/moss_tts/projection.cpp
    src/models/moss_tts/delay_state.cpp
    src/models/moss_tts/sampler.cpp
    src/models/ace_step/loader.cpp
    src/models/ace_step/session.cpp
    src/models/ace_step/dit_runner.cpp
    src/models/ace_step/vae_runner.cpp
    src/models/ace_step/detokenizer_runner.cpp
    # MOSS-SoundEffect-v2 (diffusion transformer + DAC VAE decoder):
    src/models/moss_sfx_v2/loader.cpp
    src/models/moss_sfx_v2/dit_runner.cpp
    src/models/moss_sfx_v2/session.cpp
    src/models/moss_sfx_v2/vae_runner.cpp
    # Qwen3-TTS (C++ via llama.cpp; Stage 17 codec + Stage 17b speaker encoder from CrispASR/MIT):
    src/models/qwen3_tts/loader.cpp
    src/models/qwen3_tts/session.cpp
    src/models/qwen3_tts/codec.cpp
    src/models/qwen3_tts/speaker_encoder.cpp
    # Kokoro-82M (mmwillet/TTS.cpp port — self-contained internal/ dir):
    src/models/kokoro_tts/internal/src/models/kokoro/model.cpp
    src/models/kokoro_tts/internal/src/models/kokoro/phonemizer.cpp
    src/models/kokoro_tts/internal/src/models/kokoro/loader.cpp
    src/models/kokoro_tts/internal/src/tts_model.cpp
    src/models/kokoro_tts/internal/src/tokenizer.cpp
    src/models/kokoro_tts/internal/ggml-patches/llama-mmap.cpp
    src/models/kokoro_tts/internal/src/util.cpp
    src/models/kokoro_tts/internal/src/models/loaders.cpp
    src/models/kokoro_tts/session.cpp
    # HTTP server factory (routes + WAV encoders). Linked into the
    # audiocore_server binary AND into the e2e HTTP test, so both share the
    # same routing code path.
    src/server/server.cpp
)
if(ENGINE_ENABLE_MP3)
    target_sources(audiocore_framework PRIVATE src/framework/io/mp3_encoder.cpp)
    target_link_libraries(audiocore_framework PUBLIC mp3lame)
    target_compile_definitions(audiocore_framework PRIVATE AUDIOCORE_ENABLE_MP3=1)
endif()

# ── Embedded webapp (xxd-style byte arrays from webapp/public/) ────────────
if(ENGINE_BUILD_WEBAPP AND ENGINE_BUILD_SERVER)
    set(WEBAPP_DIR "${CMAKE_SOURCE_DIR}/webapp/public")
    set(WEBAPP_OUT "${CMAKE_CURRENT_BINARY_DIR}/webapp")
    file(MAKE_DIRECTORY "${WEBAPP_OUT}")

    set(WEBAPP_ASSETS index.html style.css app.js)
    set(WEBAPP_GENERATED_SRCS "")

    foreach(asset ${WEBAPP_ASSETS})
        set(input "${WEBAPP_DIR}/${asset}")
        set(output "${WEBAPP_OUT}/${asset}.hpp")
        add_custom_command(
            DEPENDS "${input}"
            OUTPUT "${output}"
            COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}"
                    -P "${CMAKE_SOURCE_DIR}/scripts/xxd.cmake"
            COMMENT "Embedding ${asset} → ${asset}.hpp"
        )
        list(APPEND WEBAPP_GENERATED_SRCS "${output}")
    endforeach()

    add_custom_target(webapp_embed ALL DEPENDS ${WEBAPP_GENERATED_SRCS})
    add_dependencies(audiocore_framework webapp_embed)
    target_include_directories(audiocore_framework PRIVATE "${WEBAPP_OUT}")
    target_compile_definitions(audiocore_framework PRIVATE AUDIOCORE_HAS_WEBAPP=1)
endif()
target_include_directories(audiocore_framework
    PUBLIC
        include
        third_party
        third_party/llama.cpp/include
        third_party/llama.cpp/ggml/include
        third_party/httplib
        third_party/nlohmann
    PRIVATE
        src/models/kokoro_tts/internal/src
        src/models/kokoro_tts/internal/include
        src/models/kokoro_tts/internal/ggml-patches
)
target_link_libraries(audiocore_framework PUBLIC llama)
target_compile_definitions(audiocore_framework PRIVATE ESPEAK_INSTALL)
target_include_directories(audiocore_framework PRIVATE /usr/include/espeak-ng)
target_link_libraries(audiocore_framework PUBLIC espeak-ng)

# ── Binaries ─────────────────────────────────────────────────────────────
if(ENGINE_BUILD_CLI)
    add_executable(audiocore_cli src/cli/main.cpp)
    target_link_libraries(audiocore_cli PRIVATE audiocore_framework)
endif()

if(ENGINE_BUILD_SERVER)
    add_executable(audiocore_server src/server/main.cpp)
    # cpp-httplib needs Threads on linux.
    target_link_libraries(audiocore_server PRIVATE audiocore_framework pthread)
endif()

# ACE-Step HF → llama.cpp tensor-name converter. Builds unconditionally so the
# release pipeline can run it without flipping CMake flags. See
# tools/convert_acestep.cpp for why we need this.
add_executable(convert_acestep tools/convert_acestep.cpp)
target_link_libraries(convert_acestep PRIVATE audiocore_framework)

# Qwen3-TTS safetensors → GGUF converter. Builds unconditionally so the
# release pipeline can produce GGUFs from HuggingFace safetensors without
# a functioning Python environment.
add_executable(convert_qwen3tts tools/convert_qwen3tts.cpp)
target_link_libraries(convert_qwen3tts PRIVATE audiocore_framework)

# ECAPA-TDNN speaker-encoder safetensors → standalone GGUF converter.
# Source: marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B (or any equivalent
# EcapaTdnnSpeakerEncoder checkpoint). Produces ~23 MB GGUF with `speaker.*`
# namespaced tensors that audiocore's Qwen3TtsSpeakerEncoder::bind() resolves
# unchanged. See docs/MODEL-GAPS.md Gap J.
add_executable(convert_ecapa tools/convert_ecapa.cpp)
target_link_libraries(convert_ecapa PRIVATE audiocore_framework)

# MOSS-SoundEffect-v2 safetensors → GGUF converter.
add_executable(convert_mse2 tools/convert_mse2.cpp)
target_link_libraries(convert_mse2 PRIVATE audiocore_framework)

# GGUF inspector — dump tensor names, shapes, and sizes. Useful for debugging
# GGUF layout issues (e.g. truncated files, mismatched tensor counts).
add_executable(inspect_gguf tools/inspect_gguf.cpp)
target_link_libraries(inspect_gguf PRIVATE audiocore_framework)

# qwen_voice — Qwen3 voice-embedding CLI: encode reference WAVs to saveable
# vectors, do arithmetic (mix/average/add/scale/direction/shift), and apply
# a saved voice + emotional `instruct` prompt to TTS. The vector-math
# subcommands are model-free; only encode/apply load a qwen3_tts session.
add_executable(qwen_voice tools/qwen_voice.cpp)
target_link_libraries(qwen_voice PRIVATE audiocore_framework)

if(ENGINE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ── Python bindings ────────────────────────────────────────────────────────
if(ENGINE_BUILD_PYTHON)
    include(FetchContent)
    FetchContent_Declare(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.12.0)
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(audiocore
        src/python/audiocore_bindings.cpp)
    target_link_libraries(audiocore PRIVATE audiocore_framework)
    target_include_directories(audiocore PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party
        ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include
        ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include)
    set_target_properties(audiocore PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python")
endif()

# ── Kokoro TTS test ──────────────────────────────────────────────────────
add_executable(test_kokoro_e2e tests/test_kokoro_e2e.cpp)
target_link_libraries(test_kokoro_e2e PRIVATE audiocore_framework)

# Voice-embedding round-trip regression: encode a generated reference WAV
# into a saved .voice file, reload it, apply it via synthesize_with_embedding
# with an emotional `instruct` prompt, write the result WAV for listening.
add_executable(test_qwen3_voice_e2e tests/test_qwen3_voice_e2e.cpp)
target_link_libraries(test_qwen3_voice_e2e PRIVATE audiocore_framework)
