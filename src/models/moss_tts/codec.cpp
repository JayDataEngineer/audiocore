// codec.cpp — MOSS audio codec decode via ONNX Runtime.
//
// Wraps the ONNX Runtime C++ API for the MOSS-Audio-Tokenizer decoder graph.
// The decoder graph takes (codes, n_quantizers) and returns waveform.
// Encoder is not needed for TTS generation — only decoder.

#include "audiocore/models/moss_tts/codec.h"
#include "audiocore/models/moss_tts/delay_state.h"  // N_VQ

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#if __has_include(<onnxruntime_cxx_api.h>)
#include <onnxruntime_cxx_api.h>
#define HAS_ONNX_CXX_API 1
#else
#define HAS_ONNX_CXX_API 0
#endif

namespace audiocore::moss {

#if HAS_ONNX_CXX_API

// ── Pimpl: hide Ort types from the header ─────────────────────────────────
struct OnnxDecoderImpl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "moss_codec"};
    Ort::SessionOptions session_opts;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    int64_t n_quantizers = 32;
};

OnnxDecoder::OnnxDecoder() = default;
OnnxDecoder::~OnnxDecoder() = default;  // unique_ptr handles destruction

bool OnnxDecoder::load(const std::string& onnx_path, bool use_gpu,
                       std::string* error) {
    if (impl_) {
        if (error) *error = "OnnxDecoder already loaded";
        return false;
    }

    auto p = std::make_unique<OnnxDecoderImpl>();
    p->session_opts.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    p->session_opts.SetIntraOpNumThreads(4);

#if defined(USE_CUDA) && __has_include(<onnxruntime_cuda_provider_factory.h>)
    if (use_gpu) {
        try {
            // NOTE: OrtCUDAProviderOptionsV2 requires
            // onnxruntime_cuda_provider_factory.h, which ships with the CUDA
            // build of ONNX Runtime. If that header isn't available this code
            // is skipped and we fall back to CPU.
            OrtCUDAProviderOptionsV2 cuda_opts;
            p->session_opts.AppendExecutionProvider_CUDA_V2(cuda_opts);
        } catch (const Ort::Exception&) {
            // CUDA provider not available — fall through to CPU
        }
    }
#else
    (void)use_gpu;
#endif

    try {
        p->session = std::make_unique<Ort::Session>(
            p->env, onnx_path.c_str(), p->session_opts);
    } catch (const Ort::Exception& e) {
        if (error) *error = std::string("Failed to load ONNX model: ") + e.what();
        return false;
    }

    impl_ = std::move(p);
    return true;
}

bool OnnxDecoder::decode(const std::vector<std::vector<int32_t>>& codes,
                          std::vector<float>* pcm_out,
                          std::string* error) {
    if (!impl_ || !impl_->session) {
        if (error) *error = "OnnxDecoder not loaded";
        return false;
    }

    const int64_t n_frames = static_cast<int64_t>(codes.size());
    if (n_frames == 0) {
        pcm_out->clear();
        return true;
    }

    const int64_t n_vq = N_VQ;
    Ort::Session& session = *impl_->session;
    Ort::AllocatorWithDefaultOptions& alloc = impl_->allocator;

    // Transpose codes from (n_frames, n_vq) to (1, n_vq, n_frames).
    std::vector<int64_t> codes_raw(static_cast<size_t>(n_vq * n_frames));
    for (int64_t q = 0; q < n_vq; ++q) {
        for (int64_t t = 0; t < n_frames; ++t) {
            codes_raw[static_cast<size_t>(q * n_frames + t)] =
                static_cast<int64_t>(
                    codes[static_cast<size_t>(t)][static_cast<size_t>(q)]);
        }
    }

    // Build input tensor using pre-allocated data + CPU memory info.
    // The CreateTensor(ptr, count, shape, ndim) overload with MemoryInfo
    // is the portable way to wrap an existing buffer.
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> codes_shape = {n_vq, 1, n_frames};
    Ort::Value codes_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, codes_raw.data(), codes_raw.size(),
        codes_shape.data(), codes_shape.size());

    // n_quantizers scalar — onnx schema expects int64.
    int64_t nq_val = n_vq;
    std::vector<int64_t> nq_shape = {};  // scalar
    Ort::Value nq_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, &nq_val, 1, nq_shape.data(), nq_shape.size());

    // Input / output names — handle any number of model inputs dynamically.
    std::vector<Ort::AllocatedStringPtr> input_name_ptrs;
    std::vector<const char*> input_names;
    {
        const size_t n_inputs = session.GetInputCount();
        input_name_ptrs.reserve(n_inputs);
        input_names.reserve(n_inputs);
        for (size_t i = 0; i < n_inputs; ++i) {
            input_name_ptrs.push_back(session.GetInputNameAllocated(i, alloc));
            input_names.push_back(input_name_ptrs.back().get());
        }
    }

    // Build input values in the same order as input_names.
    std::vector<Ort::Value> input_values;
    input_values.reserve(input_names.size());
    input_values.push_back(std::move(codes_tensor));
    if (input_names.size() >= 2) {
        input_values.push_back(std::move(nq_tensor));
    }

    // Output names: keep allocations alive until after Run.
    std::vector<Ort::AllocatedStringPtr> output_name_ptrs;
    std::vector<const char*> output_names;
    {
        const size_t n_outputs = session.GetOutputCount();
        output_name_ptrs.reserve(n_outputs);
        output_names.reserve(n_outputs);
        for (size_t i = 0; i < n_outputs; ++i) {
            output_name_ptrs.push_back(session.GetOutputNameAllocated(i, alloc));
            output_names.push_back(output_name_ptrs.back().get());
        }
    }

    // Run inference
    Ort::RunOptions run_opts;
    std::vector<Ort::Value> outputs;
    try {
        outputs = session.Run(run_opts, input_names.data(), input_values.data(),
                              input_values.size(),
                              output_names.data(), output_names.size());
    } catch (const Ort::Exception& e) {
        if (error) *error = std::string("ONNX Run failed: ") + e.what();
        return false;
    }

    if (outputs.empty()) {
        if (error) *error = "ONNX Run returned no outputs";
        return false;
    }

    // Parse output waveform
    Ort::Value& wav_output = outputs[0];
    auto type_info = wav_output.GetTensorTypeAndShapeInfo();
    auto shape = type_info.GetShape();
    int64_t n_samples = shape.empty() ? 0 : shape.back();

    // If there's a second output with actual length, use it
    if (outputs.size() >= 2) {
        auto& len_output = outputs[1];
        auto len_type = len_output.GetTensorTypeAndShapeInfo();
        auto len_shape = len_type.GetShape();
        if (!len_shape.empty() && len_shape[0] > 0) {
            int64_t* len_data = len_output.GetTensorMutableData<int64_t>();
            if (len_data && *len_data > 0 && *len_data <= n_samples)
                n_samples = *len_data;
        }
    }

    // Get PCM data
    float* pcm_data = wav_output.GetTensorMutableData<float>();
    pcm_out->resize(static_cast<size_t>(n_samples));
    std::copy(pcm_data, pcm_data + n_samples, pcm_out->data());

    return true;
}

#else  // !HAS_ONNX_CXX_API

OnnxDecoder::OnnxDecoder() = default;
OnnxDecoder::~OnnxDecoder() {}

bool OnnxDecoder::load(const std::string&, bool, std::string* error) {
    if (error)
        *error = "built without ONNX Runtime C++ API. "
                 "Install libonnxruntime-dev and recompile.";
    return false;
}

bool OnnxDecoder::decode(const std::vector<std::vector<int32_t>>&,
                          std::vector<float>*,
                          std::string* error) {
    if (error) *error = "ONNX Runtime not available in this build";
    return false;
}

#endif  // HAS_ONNX_CXX_API

}  // namespace audiocore::moss
