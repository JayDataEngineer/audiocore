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
#   ENGINE_ENABLE_ONNXRUNTIME OFF  ONNX Runtime backend (Phase 2)
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
option(ENGINE_ENABLE_ONNXRUNTIME "ONNX Runtime backend (P2)"    OFF)
option(ENGINE_BUILD_CLI          "Build audiocore_cli"          ON)
option(ENGINE_BUILD_SERVER       "Build audiocore_server"       ON)
option(ENGINE_BUILD_TESTS        "Build unit tests"             OFF)

# ── ggml (subtree in third_party/ggml) ───────────────────────────────────
# Pinned via git submodule. Same source llama.cpp / stable-diffusion.cpp use.
add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)

# ── audiocore framework library ──────────────────────────────────────────
add_library(audiocore_framework STATIC
    src/framework/io/gguf_reader.cpp
    src/framework/io/gguf_reader_ext.cpp
    src/framework/io/weight_loader.cpp
    src/framework/core/backend.cpp
    src/framework/core/session.cpp
    src/framework/runtime/registry.cpp
    # Per-family code is added here as families land:
    # src/models/moss_tts/loader.cpp
    # src/models/moss_tts/session.cpp
    # src/models/ace_step/loader.cpp
    # src/models/ace_step/session.cpp
)
target_include_directories(audiocore_framework
    PUBLIC include third_party)
target_link_libraries(audiocore_framework PUBLIC ggml gguf)
if(ENGINE_ENABLE_CUDA)
    target_link_libraries(audiocore_framework PUBLIC ggml-cuda)
endif()

# ── Binaries ─────────────────────────────────────────────────────────────
if(ENGINE_BUILD_CLI)
    add_executable(audiocore_cli src/cli/main.cpp)
    target_link_libraries(audiocore_cli PRIVATE audiocore_framework)
endif()

if(ENGINE_BUILD_SERVER)
    add_executable(audiocore_server src/server/main.cpp)
    target_link_libraries(audiocore_server PRIVATE audiocore_framework)
endif()

if(ENGINE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
