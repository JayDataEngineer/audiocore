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
option(ENGINE_BUILD_TESTS        "Build unit tests"             OFF)

# ── llama.cpp (subtree in third_party/llama.cpp) ─────────────────────────
# Pinned via git submodule. Provides libllama (Qwen3 inference) AND vendors
# its own ggml under llama.cpp/ggml/. We use the llama.cpp copy and drop our
# standalone third_party/ggml submodule to avoid symbol clashes.
#
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
add_library(audiocore_framework STATIC
    src/framework/io/gguf_reader.cpp
    src/framework/io/gguf_reader_ext.cpp
    src/framework/io/weight_loader.cpp
    src/framework/core/backend.cpp
    src/framework/core/session.cpp
    src/framework/runtime/registry.cpp
    # The ONE Qwen3 transformer path in audiocore. Used by every family
    # whose model includes a Qwen3 backbone (MOSS 8B) or Qwen3 LM / TE
    # (ACE-Step 5Hz LM, ACE-Step Qwen3-Embedding).
    src/framework/models/qwen3/runner.cpp
    # Per-family code:
    src/models/moss_tts/loader.cpp
    src/models/moss_tts/session.cpp
    src/models/moss_tts/projection.cpp
    src/models/moss_tts/delay_state.cpp
    src/models/moss_tts/sampler.cpp
    src/models/ace_step/loader.cpp
    src/models/ace_step/session.cpp
    src/models/ace_step/dit_runner.cpp
    src/models/ace_step/vae_runner.cpp
    # ZONOS2 native-ggml TTS (transformer load only; forward pass pending):
    src/models/zonos2/loader.cpp
    src/models/zonos2/session.cpp
    # Qwen3-TTS (C++ via llama.cpp; codec decode pending ggml port):
    src/models/qwen3_tts/loader.cpp
    src/models/qwen3_tts/session.cpp
    src/models/qwen3_tts/talker_runner.cpp
    src/models/qwen3_tts/predictor_runner.cpp
    # HTTP server factory (routes + WAV encoders). Linked into the
    # audiocore_server binary AND into the e2e HTTP test, so both share the
    # same routing code path.
    src/server/server.cpp
)
target_include_directories(audiocore_framework
    PUBLIC
        include
        third_party
        third_party/llama.cpp/include
        third_party/llama.cpp/ggml/include
        third_party/httplib
        third_party/nlohmann)
target_link_libraries(audiocore_framework PUBLIC llama)

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

if(ENGINE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
