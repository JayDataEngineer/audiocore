// codec.h — MOSS audio codec decode via ONNX Runtime.
//
// Wraps ONNX Runtime for the MOSS-Audio-Tokenizer decoder graph.
// Lightweight pimpl pattern — header exposes no Ort types.

#ifndef AUDIOCORE_MODELS_MOSS_TTS_CODEC_H
#define AUDIOCORE_MODELS_MOSS_TTS_CODEC_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace audiocore::moss {

struct OnnxDecoderImpl;

class OnnxDecoder {
public:
    OnnxDecoder();
    ~OnnxDecoder();

    OnnxDecoder(const OnnxDecoder&) = delete;
    OnnxDecoder& operator=(const OnnxDecoder&) = delete;

    // Load decoder.onnx. Returns false on failure.
    bool load(const std::string& onnx_path, bool use_gpu = true,
              std::string* error = nullptr);

    // Decode audio codes to waveform.
    //   codes: (n_frames, N_VQ) int32 in row-major
    //   pcm_out: float32 mono PCM at 24 kHz
    bool decode(const std::vector<std::vector<int32_t>>& codes,
                std::vector<float>* pcm_out,
                std::string* error = nullptr);

    bool loaded() const { return impl_ != nullptr; }

private:
    std::unique_ptr<OnnxDecoderImpl> impl_;
};

}  // namespace audiocore::moss

#endif  // AUDIOCORE_MODELS_MOSS_TTS_CODEC_H
