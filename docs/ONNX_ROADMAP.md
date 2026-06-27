# ONNX Runtime roadmap (Phase 2)

## Goal

Add ONNX Runtime as an alternative backend, alongside ggml CUDA/CPU. The
same model families (MOSS, ACE-Step, future additions) run on either backend
selected via `server.json`'s `backend` field.

## Why

- **CPU deployment.** ONNX Runtime has better CPU coverage than ggml on
  non-x86_64 platforms (ARM NEON, Apple Silicon, RISC-V).
- **Mobile/edge.** ORT Mobile is a real path; ggml-cpu on Android/iOS is
  fighting upstream.
- **Operator coverage.** Some audio ops (e.g. CustomFlashAttention variants
  used by newer Qwen models) land in ORT before ggml.

## What has to be true at the framework seam

The two abstractions in `docs/ARCHITECTURE.md` are designed to make this
a no-op for model code:

1. **`TensorStorage`** is already format-neutral. An `OnnxReader` that
   enumerates ORT initializer tensors produces the same shape as
   `GgufReader`. Model code that asks for tensors by name works identically.

2. **`Backend`** is already execution-runtime-neutral. The interface is
   `Backend::execute(graph_handle, io_bindings)`. Today the only
   implementation is `ggml::Backend` whose `graph_handle` is a
   `ggml_cgraph*`. Phase 2 adds `onnx::Backend` whose `graph_handle` is an
   `Ort::Session*`. Model code builds the appropriate handle type based on
   which backend it was loaded against, but the run-time call is identical.

## Phase 2 deliverables

1. `src/framework/io/onnx_reader.cpp` — reads `.onnx` files via
   `onnx::ModelProto`, produces `vector<TensorStorage>` for every initializer.
2. `src/framework/core/backend_onnx.cpp` — `Backend` impl wrapping
   `Ort::Session::Run()`. Maps `io_bindings` (an `Ort::IoBinding` under the
   hood) to ORT input/output names.
3. `ENGINE_ENABLE_ONNXRUNTIME` CMake flag + vendored `onnxruntime` shared
   lib in `third_party/`.
4. `tools/convert_gguf_to_onnx.py` — for users who want to migrate from
   our GGUF format to ONNX without re-exporting from PyTorch.
5. Parity tests for at least one family on both backends — accept up to
   1e-3 numerical drift, identical output shape and range.

## Non-goals

- We are NOT building a general ONNX model zoo. ONNX is a backend, not a
  model format we curate. The library of supported families is the same
  on both backends.
- We are NOT trying to be faster than ggml on CUDA. ggml CUDA is the
  performance path; ORT is the portability path.

## Open questions

- **Codec weights on ORT.** MOSS's audio codec is a 1.6B-param transformer.
  On ggml we run it on CPU per OpenMOSS-Team convention. On ORT we could
  run it on CPU or GPU; need a perf measurement before committing.
- **ACE-Step DiT on ORT.** Diffusion transformers lean heavily on ggml's
  custom attention kernels; ORT may be significantly slower here. Need to
  benchmark before claiming ORT is viable for ACE-Step.
