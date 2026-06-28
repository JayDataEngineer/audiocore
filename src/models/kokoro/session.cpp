// session.cpp — Kokoro TTS inference (phonemize → tokenize → ONNX → PCM).

#include "audiocore/models/kokoro/family.h"

#include "tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

namespace audiocore::kokoro {

// ===========================================================================
// Construction / destruction
// ===========================================================================

KokoroSession::KokoroSession()
    : tokenizer_(std::make_unique<PhonemeTokenizer>()) {}

KokoroSession::~KokoroSession() {
    const OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (ort) {
        if (ort_session_) ort->ReleaseSession(reinterpret_cast<OrtSession*>(ort_session_));
        if (ort_env_) ort->ReleaseEnv(reinterpret_cast<OrtEnv*>(ort_env_));
        if (ort_mem_info_) ort->ReleaseMemoryInfo(reinterpret_cast<OrtMemoryInfo*>(ort_mem_info_));
    }
}

// ===========================================================================
// Voice listing
// ===========================================================================

std::vector<std::string> KokoroSession::list_voices() const {
    std::vector<std::string> names;
    names.reserve(voices_.size());
    for (const auto& [name, _] : voices_) {
        names.push_back(name);
    }
    std::sort(names.begin(), names.end());
    return names;
}

// ===========================================================================
// ONNX inference — 3 inputs: tokens, style, speed → 1 output: audio
// ===========================================================================

bool KokoroSession::infer(const std::vector<int64_t>& tokens,
                           const std::vector<float>& style,
                           float speed,
                           std::vector<float>& pcm_out,
                           std::string* error) {
    if (!onnx_loaded_) {
        if (error) *error = "ONNX model not loaded";
        return false;
    }

    // Use the C++ wrapper API for tensor creation (matches codec.cpp approach).
    // We keep the session as a raw C pointer (stored as void* in pimpl).
    auto* session = static_cast<OrtSession*>(ort_session_);
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

    // Build input tensors using the C++ API which is well-tested in codec.cpp.
    // 1. tokens: shape [1, seq_len], int64
    const int64_t seq_len = static_cast<int64_t>(tokens.size());
    const std::array<int64_t, 2> tokens_shape = {1, seq_len};
    Ort::Value tokens_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, const_cast<int64_t*>(tokens.data()), tokens.size(),
        tokens_shape.data(), tokens_shape.size());

    // 2. style: shape [1, 256], float32
    const std::array<int64_t, 2> style_shape = {1, static_cast<int64_t>(style.size())};
    Ort::Value style_tensor = Ort::Value::CreateTensor<float>(
        mem_info, const_cast<float*>(style.data()), style.size(),
        style_shape.data(), style_shape.size());

    // 3. speed: shape [1], float32
    float speed_val = speed;
    const std::array<int64_t, 1> speed_shape = {1};
    Ort::Value speed_tensor = Ort::Value::CreateTensor<float>(
        mem_info, &speed_val, 1,
        speed_shape.data(), speed_shape.size());

    // Get input names via C API (C++ wrapper has different name handling).
    const OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtAllocator* alloc = nullptr;
    ort->GetAllocatorWithDefaultOptions(&alloc);

    size_t n_inputs = 0;
    OrtStatus* st = ort->SessionGetInputCount(session, &n_inputs);
    if (st) {
        if (error) *error = ort->GetErrorMessage(st);
        ort->ReleaseStatus(st);
        return false;
    }

    std::vector<char*> input_names(n_inputs, nullptr);
    std::vector<OrtValue*> input_values(n_inputs, nullptr);
    bool tokens_placed = false, style_placed = false, speed_placed = false;

    for (size_t i = 0; i < n_inputs; ++i) {
        ort->SessionGetInputName(session, i, alloc, &input_names[i]);
        std::string iname(input_names[i] ? input_names[i] : "");
        if (iname == "tokens" || iname == "input_ids") {
            input_values[i] = tokens_tensor;
            tokens_placed = true;
        } else if (iname == "style") {
            input_values[i] = style_tensor;
            style_placed = true;
        } else if (iname == "speed") {
            input_values[i] = speed_tensor;
            speed_placed = true;
        }
    }

    if (!tokens_placed || !style_placed || !speed_placed) {
        std::string missing;
        if (!tokens_placed) missing += " tokens";
        if (!style_placed)  missing += " style";
        if (!speed_placed)  missing += " speed";
        if (error) *error = "ONNX model missing expected inputs:" + missing;
        for (size_t i = 0; i < n_inputs; ++i)
            if (input_names[i]) ort->AllocatorFree(alloc, const_cast<char*>(input_names[i]));
        return false;
    }

    // Get output names.
    size_t n_outputs = 0;
    st = ort->SessionGetOutputCount(session, &n_outputs);
    if (st) {
        if (error) *error = ort->GetErrorMessage(st);
        ort->ReleaseStatus(st);
        for (size_t i = 0; i < n_inputs; ++i)
            if (input_names[i]) ort->AllocatorFree(alloc, const_cast<char*>(input_names[i]));
        return false;
    }

    std::vector<char*> output_names(n_outputs, nullptr);
    for (size_t i = 0; i < n_outputs; ++i) {
        ort->SessionGetOutputName(session, i, alloc, &output_names[i]);
    }

    // Run inference.
    OrtRunOptions* run_opts = nullptr;
    ort->CreateRunOptions(&run_opts);

    OrtValue* output_tensor_raw = nullptr;
    st = ort->Run(session, run_opts,
                  input_names.data(), input_values.data(), n_inputs,
                  output_names.data(), 1, &output_tensor_raw);

    ort->ReleaseRunOptions(run_opts);

    // Free name strings.
    for (size_t i = 0; i < n_inputs; ++i)
        if (input_names[i]) ort->AllocatorFree(alloc, const_cast<char*>(input_names[i]));
    for (size_t i = 0; i < n_outputs; ++i)
        if (output_names[i]) ort->AllocatorFree(alloc, const_cast<char*>(output_names[i]));

    if (st) {
        if (error) {
            const char* msg = ort->GetErrorMessage(st);
            *error = std::string("ONNX Run failed: ") + (msg ? msg : "unknown");
        }
        ort->ReleaseStatus(st);
        return false;
    }

    // Wrap output in C++ Value for automatic cleanup.
    Ort::Value output_tensor(output_tensor_raw);

    // Get output data.
    auto shape_info = output_tensor.GetTensorTypeAndShapeInfo();
    const size_t total_samples = shape_info.GetElementCount();
    float* output_data = output_tensor.GetTensorMutableData<float>();
    pcm_out.assign(output_data, output_data + total_samples);

    return true;
}

// ===========================================================================
// run_tts — main inference pipeline
// ===========================================================================

bool KokoroSession::run_tts(const void* request, void* response,
                             std::string* error) {
    const auto& req = *static_cast<const TtsRequest*>(request);
    auto& resp = *static_cast<TtsResponse*>(response);

    // Validate request.
    if (req.text.empty()) {
        resp.error = "empty input text";
        if (error) *error = resp.error;
        return false;
    }
    if (req.speed < 0.5f || req.speed > 2.0f) {
        resp.error = "speed must be between 0.5 and 2.0";
        if (error) *error = resp.error;
        return false;
    }

    // Find voice.
    auto voice_it = voices_.find(req.voice);
    if (voice_it == voices_.end()) {
        resp.error = "unknown voice '" + req.voice + "'";
        if (error) *error = resp.error;
        return false;
    }
    const VoiceData& voice_data = voice_it->second;

    // Initialize tokenizer on first use.
    if (!tokenizer_initialized_) {
        if (!tokenizer_->initialize(error)) {
            resp.error = *error;
            return false;
        }
        tokenizer_initialized_ = true;
    }

    // Step 1: phonemize + tokenize.
    std::vector<int64_t> tokens;
    if (req.is_phonemes) {
        // Input is already a phoneme string.
        tokens = tokenizer_->tokenize(req.text);
        // Pad.
        std::vector<int64_t> padded;
        padded.reserve(tokens.size() + 2);
        padded.push_back(0);
        padded.insert(padded.end(), tokens.begin(), tokens.end());
        padded.push_back(0);

        // Truncate if needed.
        if (static_cast<int32_t>(tokens.size()) > cfg_.max_phoneme_len - 2) {
            tokens.resize(cfg_.max_phoneme_len - 2);
        }
        tokens = padded;
    } else {
        tokens = tokenizer_->phonemize_and_tokenize(req.text, req.language,
                                                     cfg_.max_phoneme_len);
    }

    if (tokens.size() < 3) {  // need at least [0, token, 0]
        resp.error = "text too short after tokenization";
        if (error) *error = resp.error;
        return false;
    }

    // Step 2: select style vector for this token length.
    // voice_data is (510, 256). We index by token count (before padding?).
    // Python: voice[len(tokens)] where tokens is the UNPADDED token list.
    // tokens[1:-1] strips the [0, ..., 0] padding.
    const int32_t content_len = static_cast<int32_t>(tokens.size()) - 2;
    const int32_t style_idx = std::min(content_len, voice_data.n_styles - 1);

    if (style_idx < 0 || style_idx >= voice_data.n_styles) {
        resp.error = "invalid style index";
        if (error) *error = resp.error;
        return false;
    }

    // Extract style row: one (256,) vector.
    std::vector<float> style(voice_data.style_dim);
    const float* base = voice_data.data.data() +
                        static_cast<size_t>(style_idx) * voice_data.style_dim;
    std::memcpy(style.data(), base,
                static_cast<size_t>(voice_data.style_dim) * sizeof(float));

    // Step 3: run ONNX inference.
    std::vector<float> audio;
    if (!infer(tokens, style, req.speed, audio, error)) {
        resp.error = *error;
        return false;
    }

    // Step 4: prepare response.
    resp.pcm_mono = std::move(audio);
    resp.sampling_rate = 24000;

    return true;
}

}  // namespace audiocore::kokoro
