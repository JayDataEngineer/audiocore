# tests/CMakeLists.txt — unit + e2e tests.
#
# Each test is its own executable so a crash in one doesn't hide the rest and
# ctest can run them in parallel. All link against audiocore_framework (which
# already brings in libllama + ggml).
#
# Synthesizing tiny GGUFs in-process keeps tests hermetic — no fixture files
# checked in, no pip deps.

# Shared helper lib: synthetic GGUF writer + the tiny test framework (header).
add_library(audiocore_test_helpers STATIC synthetic_gguf.cpp)
target_include_directories(audiocore_test_helpers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(audiocore_test_helpers PUBLIC audiocore_framework)

# Helper function: declare a ctest entry that runs the given source.
function(audiocore_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE audiocore_test_helpers)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# Variant that needs a path to a built binary (e.g. for subprocess e2e tests).
function(audiocore_test_with_bin name bin_target)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE audiocore_test_helpers)
    # $<TARGET_FILE:...> gives the absolute path to the built binary, so the
    # test can locate the converter regardless of CMAKE_RUNTIME_OUTPUT_DIRECTORY.
    target_compile_definitions(${name} PRIVATE
        AUDIOCORE_CONVERT_ACESTEP_PATH="$<TARGET_FILE:${bin_target}>")
    # Make sure the binary we shell out to is built before the test runs.
    add_dependencies(${name} ${bin_target})
    add_test(NAME ${name} COMMAND ${name})
endfunction()

audiocore_test(test_gguf_reader)
audiocore_test(test_registry)
audiocore_test(test_projection)
audiocore_test(test_acestep_units)
audiocore_test(test_server_e2e)
audiocore_test(test_sampler)
audiocore_test(test_moss_units)
audiocore_test(test_speaker_encoder_units)
audiocore_test_with_bin(test_convert_acestep convert_acestep)

# End-to-end MOSS-TTS test — requires ~7.5GB model files + ~7GB ONNX codec.
# Not registered as a ctest entry (too heavy / needs external weights).
# Run manually: ./test_moss_e2e
add_executable(test_moss_e2e test_moss_e2e.cpp)
target_link_libraries(test_moss_e2e PRIVATE audiocore_framework)

# End-to-end MOSS-TTS voice-clone test — exercises the real codec encoder
# path. Requires test_moss_e2e to have produced test_moss_output.wav in the
# working directory; otherwise soft-skips. Run manually: ./test_moss_voice_clone_e2e
add_executable(test_moss_voice_clone_e2e test_moss_voice_clone_e2e.cpp)
target_link_libraries(test_moss_voice_clone_e2e PRIVATE audiocore_framework)

# End-to-end Qwen3-TTS test — loads talker GGUF + predictor GGUF + ONNX codec.
# Requires ~1.4GB of model files from hf.co/wkwong/Lunavox-Qwen3-TTS-GGUF.
# Run manually: ./test_qwen3tts_e2e
# Set QWEN3TTS_DIR / QWEN3TTS_NGPU / QWEN3TTS_DEVICE env vars to configure.
add_executable(test_qwen3tts_e2e test_qwen3tts_e2e.cpp)
target_link_libraries(test_qwen3tts_e2e PRIVATE audiocore_framework)

# End-to-end ACE-Step test — loads 4 GGUFs (DiT + LM + TE + VAE), ~5.3GB total.
# Run manually: ./test_acestep_e2e
# Set ACESTEP_DIR / ACESTEP_NGPU / ACESTEP_DEVICE env vars to configure.
add_executable(test_acestep_e2e test_acestep_e2e.cpp)
target_link_libraries(test_acestep_e2e PRIVATE audiocore_framework)

add_executable(test_gguf_dump test_gguf_dump.cpp)
target_link_libraries(test_gguf_dump PRIVATE audiocore_framework)

# Verifies that unsupported modes fail fast with a clear error rather than
# silently degrading through system-prompt impersonation (GAP_TRACKER.md §1.2).
add_executable(test_moss_modes test_moss_modes.cpp)
target_link_libraries(test_moss_modes PRIVATE audiocore_framework)
