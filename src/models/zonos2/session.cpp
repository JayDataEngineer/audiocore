// session.cpp — ZONOS2 inference via subprocess HTTP.
//
// The heavy lifting is in loader.cpp (subprocess start / stop). This file
// implements the TTS inference call: serialize TtsRequest to JSON, POST it
// to the subprocess's /v1/audio/speech endpoint, and parse the returned WAV
// bytes into PCM float samples.

#include "audiocore/models/zonos2/family.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

namespace audiocore::zonos2 {

using nlohmann::json;

// ===========================================================================
// Lifetime
// ===========================================================================

Zonos2Session::Zonos2Session() = default;

Zonos2Session::~Zonos2Session() {
    stop_subprocess();
}

// ===========================================================================
// run_tts — forward TTS request to the subprocess HTTP server
// ===========================================================================

bool Zonos2Session::run_tts(const void* request, void* response,
                             std::string* error) {
    if (!loaded_ || !subprocess_running_) {
        if (error) *error = "zonos2: session not loaded";
        return false;
    }

    const auto* req = static_cast<const TtsRequest*>(request);
    auto*       res = static_cast<TtsResponse*>(response);

    if (req->text.empty()) {
        if (error) *error = "zonos2: empty text";
        return false;
    }

    // Build the JSON body matching the Mini-SGLang server's
    // /v1/audio/speech schema.
    json body = {
        {"input",       req->text},
        {"language",    req->language},
        {"temperature", req->temperature},
        {"top_p",       req->top_p},
        {"speed",       req->speed},
    };

    if (!req->speaker_embedding.empty()) {
        body["speaker_embedding"] = req->speaker_embedding;
    }

    httplib::Client cli("127.0.0.1", subprocess_port_);
    cli.set_read_timeout(180);  // TTS for long text can take minutes

    auto http_res = cli.Post("/v1/audio/speech",
                              body.dump(), "application/json");
    if (!http_res) {
        if (error) *error = "zonos2: subprocess HTTP request failed";
        return false;
    }
    if (http_res->status != 200) {
        if (error) {
            *error = "zonos2: subprocess returned HTTP " +
                     std::to_string(http_res->status);
        }
        return false;
    }

    // Parse the WAV response.
    if (!parse_wav_response(http_res->body, res->pcm_mono,
                            res->sampling_rate, error)) {
        return false;
    }
    return true;
}

// ===========================================================================
// parse_wav_response — extract mono PCM from a WAV byte blob
// ===========================================================================

bool Zonos2Session::parse_wav_response(const std::string& wav_bytes,
                                        std::vector<float>& pcm_out,
                                        int32_t& sampling_rate_out,
                                        std::string* error) {
    // Minimum valid WAV: 44-byte header.
    if (wav_bytes.size() < 44) {
        if (error) *error = "zonos2: WAV response too short (" +
                            std::to_string(wav_bytes.size()) + " bytes)";
        return false;
    }

    // Verify RIFF / WAVE magic.
    if (std::memcmp(wav_bytes.data(), "RIFF", 4) != 0 ||
        std::memcmp(wav_bytes.data() + 8, "WAVE", 4) != 0) {
        if (error) *error = "zonos2: response is not a valid WAV file";
        return false;
    }

    // Walk RIFF chunks to find fmt + data.
    // RIFF layout: 12-byte header, then chunks of (id[4], size[4], data[size]).
    size_t pos = 12;
    bool   found_fmt  = false;
    bool   found_data = false;

    uint16_t audio_format   = 0;
    uint16_t channels       = 0;
    uint16_t bits_per_sample = 0;
    int32_t  sampling_rate  = 0;
    size_t   data_offset    = 0;
    size_t   data_size      = 0;

    while (pos + 8 <= wav_bytes.size()) {
        const std::string chunk_id(wav_bytes.data() + pos, 4);
        uint32_t chunk_size = 0;
        std::memcpy(&chunk_size, wav_bytes.data() + pos + 4, 4);

        if (chunk_id == "fmt ") {
            if (pos + 8 + 16 > wav_bytes.size()) break;
            std::memcpy(&audio_format,  wav_bytes.data() + pos + 8,  2);
            std::memcpy(&channels,       wav_bytes.data() + pos + 10, 2);
            std::memcpy(&sampling_rate,  wav_bytes.data() + pos + 12, 4);
            std::memcpy(&bits_per_sample, wav_bytes.data() + pos + 22, 2);
            found_fmt = true;
        } else if (chunk_id == "data") {
            data_offset = pos + 8;
            data_size   = chunk_size;
            found_data  = true;
            break;  // data is typically the last chunk
        }

        pos += 8 + chunk_size;
        // Chunks are aligned to even offset.
        if (chunk_size % 2) ++pos;
    }

    if (!found_fmt) {
        if (error) *error = "zonos2: WAV missing fmt chunk";
        return false;
    }
    if (!found_data) {
        if (error) *error = "zonos2: WAV missing data chunk";
        return false;
    }
    if (data_offset + data_size > wav_bytes.size()) {
        if (error) *error = "zonos2: WAV data chunk truncated";
        return false;
    }
    if (audio_format != 1) {
        if (error) *error = "zonos2: WAV is not PCM format";
        return false;
    }
    if (channels != 1) {
        if (error) *error = "zonos2: expected mono WAV, got " +
                            std::to_string(channels) + " channels";
        return false;
    }
    if (bits_per_sample != 16 && bits_per_sample != 32) {
        if (error) {
            *error = "zonos2: unsupported WAV bit depth " +
                     std::to_string(bits_per_sample);
        }
        return false;
    }

    // Convert PCM data to float32 samples.
    const size_t n_samples = data_size / (bits_per_sample / 8);
    pcm_out.resize(n_samples);

    if (bits_per_sample == 16) {
        const auto* src = reinterpret_cast<const int16_t*>(
            wav_bytes.data() + data_offset);
        for (size_t i = 0; i < n_samples; ++i) {
            pcm_out[i] = src[i] / 32768.0f;
        }
    } else {
        // 32-bit PCM (common for DAC output before WAV encoding)
        const auto* src = reinterpret_cast<const int32_t*>(
            wav_bytes.data() + data_offset);
        for (size_t i = 0; i < n_samples; ++i) {
            pcm_out[i] = src[i] / 2147483648.0f;
        }
    }

    sampling_rate_out = sampling_rate;
    return true;
}

}  // namespace audiocore::zonos2
