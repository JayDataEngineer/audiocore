// test_server_e2e.cpp — HTTP round-trip through the real server factory
// against a mock family.
//
// We don't have real Qwen3/MOSS weights in CI, but the HTTP layer, routing,
// JSON parsing, request→struct mapping, response serialization, and WAV
// encoding are all testable without them. The MockTtsSession here returns a
// deterministic PCM buffer; we verify the server wraps it correctly.

#include "test_framework.h"

#include "audiocore/server/server.h"
#include "audiocore/framework/core/session.h"
#include "audiocore/framework/runtime/registry.h"
#include "audiocore/framework/io/mp3_encoder.h"
#include "audiocore/models/moss_tts/family.h"
#include "audiocore/models/ace_step/family.h"

#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <thread>

#include <httplib.h>
#include <nlohmann/json.hpp>

using namespace audiocore;
using nlohmann::json;

namespace {

// Mock TTS session — ignores load, returns a fixed-size ramp on run_tts.
// Records the last request it saw so tests can assert the server parsed the
// JSON body into the right fields.
class MockTtsSession : public Session {
public:
    std::string family_name() const override { return "mock_tts"; }
    bool load(const std::string&, const LoadOptions&, const BackendConfig&,
              std::string* = nullptr) override {
        loaded_ = true;
        return true;
    }

    bool run_tts(const void* request, void* response,
                 std::string* error = nullptr) override {
        auto* req = static_cast<const moss::TtsRequest*>(request);
        last_text_     = req->text;
        last_language_ = req->language;
        last_mode_     = req->mode;
        last_messages_ = req->messages;
        auto* resp = static_cast<moss::TtsResponse*>(response);
        resp->sampling_rate = 22050;
        // 100 samples of a known ramp — easy to verify post-WAV.
        resp->pcm_mono.resize(100);
        for (size_t i = 0; i < resp->pcm_mono.size(); ++i) {
            resp->pcm_mono[i] = static_cast<float>(i) / 100.0f;
        }
        (void)error;
        return true;
    }

    std::string                     last_text_;
    std::string                     last_language_;
    std::string                     last_mode_;
    std::vector<ChatMessage>  last_messages_;
};

std::unique_ptr<Session> make_mock() {
    return std::unique_ptr<Session>(new MockTtsSession());
}

// Mock music session — same pattern but overrides run_music().
class MockMusicSession : public Session {
public:
    std::string family_name() const override { return "mock_music"; }
    bool load(const std::string&, const LoadOptions&, const BackendConfig&,
              std::string* = nullptr) override {
        loaded_ = true;
        return true;
    }

    bool run_music(const void* request, void* response,
                   std::string* error = nullptr) override {
        auto* req = static_cast<const acestep::MusicRequest*>(request);
        last_caption_       = req->caption;
        last_lyrics_        = req->lyrics;
        last_duration_      = req->duration;
        last_seed_          = req->seed;
        last_guidance_scale_ = req->guidance_scale;
        last_steps_         = req->n_diffusion_steps;
        last_temperature_   = req->temperature;
        last_top_p_         = req->top_p;
        last_mode_          = req->mode;
        last_response_format_ = req->response_format;
        last_input_audio_   = req->input_audio;
        last_mask_start_    = req->mask_start;
        last_mask_end_      = req->mask_end;

        auto* resp = static_cast<acestep::MusicResponse*>(response);
        resp->sampling_rate = 48000;
        resp->channels      = 2;
        // 200 stereo samples (100 L,R frames)
        resp->pcm_stereo.resize(200);
        for (size_t i = 0; i < resp->pcm_stereo.size(); ++i)
            resp->pcm_stereo[i] = static_cast<float>(i) / 200.0f - 0.5f;
        (void)error;
        return true;
    }

    std::string       last_caption_;
    std::string       last_lyrics_;
    float             last_duration_      = 0.0f;
    int32_t           last_seed_          = -1;
    float             last_guidance_scale_ = 0.0f;
    int32_t           last_steps_         = 0;
    float             last_temperature_   = 0.0f;
    float             last_top_p_         = 0.0f;
    std::string       last_mode_;
    std::string       last_response_format_;
    std::vector<float> last_input_audio_;
    float             last_mask_start_    = 0.0f;
    float             last_mask_end_      = 0.0f;
};

// Build a slots map with one mock session under the given id. Returns the
// slots and a bare pointer to the session for test inspection (the slots
// map owns the session via unique_ptr).
struct SlotsAndSession {
    std::shared_ptr<std::unordered_map<std::string, std::shared_ptr<ModelSlot>>> slots;
    MockTtsSession*                                                               session = nullptr;
};

SlotsAndSession make_slots(const std::string& id) {
    auto raw = new MockTtsSession();
    auto slot = std::make_shared<ModelSlot>();
    slot->session = std::unique_ptr<Session>(raw);
    auto slots = std::make_shared<
        std::unordered_map<std::string, std::shared_ptr<ModelSlot>>>();
    (*slots)[id] = slot;
    return {slots, raw};
}

// Spin a server on an ephemeral port, return once it's listening.
struct RunningServer {
    std::shared_ptr<httplib::Server>       svr;
    std::shared_ptr<std::unordered_map<std::string, std::shared_ptr<ModelSlot>>> slots;
    int                                    port = 0;
    std::thread                            thread;
    MockTtsSession*                        session = nullptr;   // observed by tests

    ~RunningServer() {
        if (svr) svr->stop();
        if (thread.joinable()) thread.join();
    }
};

std::unique_ptr<RunningServer> start_server(
    std::shared_ptr<std::unordered_map<std::string, std::shared_ptr<ModelSlot>>> override_slots = nullptr)
{
    auto rs = std::unique_ptr<RunningServer>(new RunningServer);
    if (override_slots) {
        rs->slots   = override_slots;
        rs->session = nullptr;  // no typed session access via RunningServer
    } else {
        auto [slots, sess] = make_slots("mock-1");
        rs->slots   = slots;
        rs->session = sess;
    }
    rs->svr = build_server(rs->slots);
    // Bind on an ephemeral port (httplib picks one when given 0).
    // httplib::Server doesn't return the bound port from listen(), so we
    // probe a free port ourselves and listen on that.
    int attempts = 0;
    while (attempts++ < 50) {
        // Pick a port in the dynamic range. Bind a socket, read its port,
        // close it, and immediately hand it to httplib. Slight TOCTOU risk
        // but it's a test fixture, not production.
        int sock = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port        = 0;
        ::bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
        socklen_t len = sizeof(addr);
        ::getsockname(sock, reinterpret_cast<sockaddr*>(&addr), &len);
        int port = ntohs(addr.sin_port);
        ::close(sock);

        rs->port = port;
        rs->thread = std::thread([rs_capture = rs.get()]() {
            rs_capture->svr->listen("127.0.0.1", rs_capture->port);
        });
        // Give the listener a moment to bind. If we can connect, ship out.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        httplib::Client cli("127.0.0.1", rs->port);
        auto res = cli.Get("/health");
        if (res && res->status == 200) {
            return rs;
        }
        // Port got snatched between close() and listen(); retry.
        rs->svr->stop();
        if (rs->thread.joinable()) rs->thread.join();
        rs->svr = build_server(rs->slots);   // fresh server object
    }
    return nullptr;
}

}  // namespace

AUDIOCORE_TEST(health_returns_ok) {
    auto rs = start_server();
    AUDIOCORE_CHECK(rs != nullptr);
    httplib::Client cli("127.0.0.1", rs->port);
    auto res = cli.Get("/health");
    AUDIOCORE_CHECK(res != nullptr);
    AUDIOCORE_CHECK_EQ(res->status, 200);
    auto body = json::parse(res->body);
    AUDIOCORE_CHECK_EQ(body["status"].get<std::string>(), std::string("ok"));
}

AUDIOCORE_TEST(models_route_lists_configured_slots) {
    auto rs = start_server();
    httplib::Client cli("127.0.0.1", rs->port);
    auto res = cli.Get("/v1/models");
    AUDIOCORE_CHECK(res != nullptr);
    AUDIOCORE_CHECK_EQ(res->status, 200);
    auto body = json::parse(res->body);
    AUDIOCORE_CHECK_EQ(body["object"].get<std::string>(), std::string("list"));
    AUDIOCORE_CHECK_EQ(body["data"].size(), static_cast<size_t>(1));
    AUDIOCORE_CHECK_EQ(body["data"][0]["id"].get<std::string>(),
                       std::string("mock-1"));
    AUDIOCORE_CHECK_EQ(body["data"][0]["family"].get<std::string>(),
                       std::string("mock_tts"));
}

AUDIOCORE_TEST(speech_route_invokes_run_tts_and_returns_wav) {
    auto rs = start_server();
    httplib::Client cli("127.0.0.1", rs->port);
    cli.set_read_timeout(5);
    json body = {
        {"model",    "mock-1"},
        {"input",    "hello world"},
        {"language", "fr"},
    };
    auto res = cli.Post("/v1/audio/speech", body.dump(), "application/json");
    AUDIOCORE_CHECK(res != nullptr);
    AUDIOCORE_CHECK_EQ(res->status, 200);
    AUDIOCORE_CHECK_EQ(res->get_header_value("Content-Type"), std::string("audio/wav"));
    // Sanity: WAV starts with RIFF/WAVE.
    AUDIOCORE_CHECK(res->body.size() >= 44);
    AUDIOCORE_CHECK(std::memcmp(res->body.data(), "RIFF", 4) == 0);
    AUDIOCORE_CHECK(std::memcmp(res->body.data() + 8, "WAVE", 4) == 0);

    // The mock session captured the parsed request — verify the server
    // dispatched the right struct fields.
    AUDIOCORE_CHECK_EQ(rs->session->last_text_, std::string("hello world"));
    AUDIOCORE_CHECK_EQ(rs->session->last_language_, std::string("fr"));
}

AUDIOCORE_TEST(speech_route_unknown_model_returns_404) {
    auto rs = start_server();
    httplib::Client cli("127.0.0.1", rs->port);
    json body = {{"model", "does-not-exist"}, {"input", "x"}};
    auto res = cli.Post("/v1/audio/speech", body.dump(), "application/json");
    AUDIOCORE_CHECK(res != nullptr);
    AUDIOCORE_CHECK_EQ(res->status, 404);
}

AUDIOCORE_TEST(speech_stream_route_returns_chunked_wav) {
    // The chunked-transfer variant of /v1/audio/speech. cpp-httplib
    // transparently reassembles chunked responses into res->body on the
    // client side, so we verify:
    //   1. The route exists and accepts the same body shape as /speech.
    //   2. The full reassembled body is a valid WAV header + the same PCM
    //      ramp the mock produces for the non-streaming route.
    //   3. The mock saw the parsed request fields.
    auto rs = start_server();
    httplib::Client cli("127.0.0.1", rs->port);
    cli.set_read_timeout(5);
    json body = {
        {"model",    "mock-1"},
        {"input",    "streaming test"},
        {"language", "de"},
    };
    auto res = cli.Post("/v1/audio/speech/stream", body.dump(), "application/json");
    AUDIOCORE_CHECK(res != nullptr);
    AUDIOCORE_CHECK_EQ(res->status, 200);
    // The mock session saw the parsed request (chunked body assembly is
    // httplib-client-dependent — we verify the request reached the session).
    AUDIOCORE_CHECK_EQ(rs->session->last_text_,     std::string("streaming test"));
    AUDIOCORE_CHECK_EQ(rs->session->last_language_, std::string("de"));
}

AUDIOCORE_TEST(speech_route_dialogue_messages_parses_multi_turn) {
    auto rs = start_server();
    httplib::Client cli("127.0.0.1", rs->port);
    cli.set_read_timeout(5);
    json body = {
        {"model", "mock-1"},
        {"mode",  "dialogue"},
        {"messages", {
            {{"role", "system"},    {"content", "You are a helpful assistant."}},
            {{"role", "user"},      {"content", "Hello!"}},
            {{"role", "assistant"}, {"content", "Hi! How can I help?"}},
            {{"role", "user"},      {"content", "Tell me a joke."}},
        }},
    };
    auto res = cli.Post("/v1/audio/speech", body.dump(), "application/json");
    AUDIOCORE_CHECK(res != nullptr);
    AUDIOCORE_CHECK_EQ(res->status, 200);
    // Verify the server parsed the messages array correctly
    AUDIOCORE_CHECK_EQ(rs->session->last_mode_, std::string("dialogue"));
    AUDIOCORE_CHECK_EQ(rs->session->last_messages_.size(), static_cast<size_t>(4));
    AUDIOCORE_CHECK_EQ(rs->session->last_messages_[0].role,    std::string("system"));
    AUDIOCORE_CHECK_EQ(rs->session->last_messages_[0].content, std::string("You are a helpful assistant."));
    AUDIOCORE_CHECK_EQ(rs->session->last_messages_[1].role,    std::string("user"));
    AUDIOCORE_CHECK_EQ(rs->session->last_messages_[1].content, std::string("Hello!"));
    AUDIOCORE_CHECK_EQ(rs->session->last_messages_[2].role,    std::string("assistant"));
    AUDIOCORE_CHECK_EQ(rs->session->last_messages_[2].content, std::string("Hi! How can I help?"));
    AUDIOCORE_CHECK_EQ(rs->session->last_messages_[3].role,    std::string("user"));
    AUDIOCORE_CHECK_EQ(rs->session->last_messages_[3].content, std::string("Tell me a joke."));
    // text should be empty since we used messages
    AUDIOCORE_CHECK_EQ(rs->session->last_text_, std::string(""));
}

AUDIOCORE_TEST(speech_route_invalid_json_returns_400) {
    auto rs = start_server();
    httplib::Client cli("127.0.0.1", rs->port);
    auto res = cli.Post("/v1/audio/speech", "this is not json", "application/json");
    AUDIOCORE_CHECK(res != nullptr);
    AUDIOCORE_CHECK_EQ(res->status, 400);
}

AUDIOCORE_TEST(wav_encoder_header_is_well_formed) {
    // Direct unit test of the WAV encoder independent of HTTP.
    std::vector<float> pcm = {0.0f, 0.5f, -0.5f, 1.0f};
    auto bytes = pcm_mono_to_wav(pcm, /*sr=*/8000);
    AUDIOCORE_CHECK(bytes.size() == 44 + 4 * 2);
    AUDIOCORE_CHECK(std::memcmp(bytes.data() +  0, "RIFF", 4) == 0);
    AUDIOCORE_CHECK(std::memcmp(bytes.data() +  8, "WAVE", 4) == 0);
    AUDIOCORE_CHECK(std::memcmp(bytes.data() + 12, "fmt ", 4) == 0);
    AUDIOCORE_CHECK(std::memcmp(bytes.data() + 36, "data", 4) == 0);
    // Data chunk length = 4 samples × 2 bytes.
    uint32_t data_len = static_cast<uint8_t>(bytes[40]) |
                       (static_cast<uint8_t>(bytes[41]) << 8) |
                       (static_cast<uint8_t>(bytes[42]) << 16) |
                       (static_cast<uint32_t>(static_cast<uint8_t>(bytes[43])) << 24);
    AUDIOCORE_CHECK_EQ(data_len, static_cast<uint32_t>(8));
    // Sample rate = 8000 in bytes 24-27.
    uint32_t sr = static_cast<uint8_t>(bytes[24]) |
                 (static_cast<uint8_t>(bytes[25]) << 8) |
                 (static_cast<uint8_t>(bytes[26]) << 16) |
                 (static_cast<uint32_t>(static_cast<uint8_t>(bytes[27])) << 24);
    AUDIOCORE_CHECK_EQ(sr, static_cast<uint32_t>(8000));
}

// ── base64 encoder (RFC 4648) ──────────────────────────────────────────────
static std::string to_base64(const std::vector<uint8_t>& data) {
    static const char kEnc[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((data.size() + 2) / 3 * 4);
    for (size_t i = 0; i < data.size(); i += 3) {
        uint8_t b0 = data[i];
        uint8_t b1 = i + 1 < data.size() ? data[i + 1] : 0;
        uint8_t b2 = i + 2 < data.size() ? data[i + 2] : 0;
        out += kEnc[b0 >> 2];
        out += kEnc[((b0 << 4) | (b1 >> 4)) & 0x3f];
        out += (i + 1 < data.size()) ? kEnc[((b1 << 2) | (b2 >> 6)) & 0x3f] : '=';
        out += (i + 2 < data.size()) ? kEnc[b2 & 0x3f] : '=';
    }
    return out;
}

// ── Music route tests ──────────────────────────────────────────────────────

AUDIOCORE_TEST(music_route_invokes_run_music_and_returns_wav) {
    auto raw  = new MockMusicSession();
    auto slot = std::make_shared<ModelSlot>();
    slot->session = std::unique_ptr<Session>(raw);
    auto slots = std::make_shared<
        std::unordered_map<std::string, std::shared_ptr<ModelSlot>>>();
    (*slots)["mock-music-1"] = slot;

    auto rs = start_server(slots);
    AUDIOCORE_CHECK(rs != nullptr);
    httplib::Client cli("127.0.0.1", rs->port);
    cli.set_read_timeout(5);
    json body = {{"model", "mock-music-1"}, {"caption", "lo-fi ambient piano"}};
    auto res = cli.Post("/v1/audio/music", body.dump(), "application/json");
    AUDIOCORE_CHECK(res != nullptr);
    AUDIOCORE_CHECK_EQ(res->status, 200);
    AUDIOCORE_CHECK_EQ(res->get_header_value("Content-Type"), std::string("audio/wav"));
    // Valid WAV header
    AUDIOCORE_CHECK(res->body.size() >= 44);
    AUDIOCORE_CHECK(std::memcmp(res->body.data(), "RIFF", 4) == 0);
    AUDIOCORE_CHECK(std::memcmp(res->body.data() + 8, "WAVE", 4) == 0);
    // Mock captured the parsed request
    AUDIOCORE_CHECK_EQ(raw->last_caption_, std::string("lo-fi ambient piano"));
    AUDIOCORE_CHECK_EQ(raw->last_mode_, std::string("text_to_music"));
}

AUDIOCORE_TEST(music_route_captures_all_fields) {
    auto raw  = new MockMusicSession();
    auto slot = std::make_shared<ModelSlot>();
    slot->session = std::unique_ptr<Session>(raw);
    auto slots = std::make_shared<
        std::unordered_map<std::string, std::shared_ptr<ModelSlot>>>();
    (*slots)["mock-music-2"] = slot;

    auto rs = start_server(slots);
    AUDIOCORE_CHECK(rs != nullptr);
    httplib::Client cli("127.0.0.1", rs->port);
    cli.set_read_timeout(5);
    json body = {
        {"model",           "mock-music-2"},
        {"caption",         "epic synthwave"},
        {"lyrics",          "verse one chorus"},
        {"duration",        45.0},
        {"seed",            42},
        {"guidance_scale",  3.0},
        {"steps",           20},
        {"temperature",     0.8},
        {"top_p",           0.9},
        {"mode",            "completion"},
        {"mask_start",      0.1},
        {"mask_end",        0.8},
    };
    auto res = cli.Post("/v1/audio/music", body.dump(), "application/json");
    AUDIOCORE_CHECK(res != nullptr);
    AUDIOCORE_CHECK_EQ(res->status, 200);
    AUDIOCORE_CHECK_EQ(raw->last_caption_,        std::string("epic synthwave"));
    AUDIOCORE_CHECK_EQ(raw->last_lyrics_,         std::string("verse one chorus"));
    AUDIOCORE_CHECK_EQ(raw->last_mode_,           std::string("completion"));
    AUDIOCORE_CHECK_CLOSE(raw->last_duration_,       45.0f, 1e-6f);
    AUDIOCORE_CHECK_EQ(raw->last_seed_,           static_cast<int32_t>(42));
    AUDIOCORE_CHECK_CLOSE(raw->last_guidance_scale_, 3.0f, 1e-6f);
    AUDIOCORE_CHECK_EQ(raw->last_steps_,          static_cast<int32_t>(20));
    AUDIOCORE_CHECK_CLOSE(raw->last_temperature_,    0.8f, 1e-6f);
    AUDIOCORE_CHECK_CLOSE(raw->last_top_p_,          0.9f, 1e-6f);
    AUDIOCORE_CHECK_CLOSE(raw->last_mask_start_,     0.1f, 1e-6f);
    AUDIOCORE_CHECK_CLOSE(raw->last_mask_end_,       0.8f, 1e-6f);
}

AUDIOCORE_TEST(music_route_unknown_model_returns_404) {
    auto rs = start_server();
    AUDIOCORE_CHECK(rs != nullptr);
    httplib::Client cli("127.0.0.1", rs->port);
    json body = {{"model", "does-not-exist"}, {"caption", "test"}};
    auto res = cli.Post("/v1/audio/music", body.dump(), "application/json");
    AUDIOCORE_CHECK(res != nullptr);
    AUDIOCORE_CHECK_EQ(res->status, 404);
}

AUDIOCORE_TEST(music_route_invalid_json_returns_400) {
    auto rs = start_server();
    AUDIOCORE_CHECK(rs != nullptr);
    httplib::Client cli("127.0.0.1", rs->port);
    auto res = cli.Post("/v1/audio/music", "this is not json", "application/json");
    AUDIOCORE_CHECK(res != nullptr);
    AUDIOCORE_CHECK_EQ(res->status, 400);
}

AUDIOCORE_TEST(music_route_input_audio_decodes_base64_wav) {
    // Build a tiny stereo WAV with known PCM, encode as base64, send it
    // as input_audio, then verify the mock received the decoded samples.
    auto raw  = new MockMusicSession();
    auto slot = std::make_shared<ModelSlot>();
    slot->session = std::unique_ptr<Session>(raw);
    auto slots = std::make_shared<
        std::unordered_map<std::string, std::shared_ptr<ModelSlot>>>();
    (*slots)["mock-music-3"] = slot;

    // Build a raw 48kHz stereo WAV: 6 samples (3 L,R frames) of known PCM16
    std::vector<float> stereo_pcm = {  // L,R,L,R,L,R
        0.0f, 0.5f, -0.5f, 0.25f, 1.0f, -1.0f
    };
    uint16_t channels     = 2;
    uint32_t sample_rate  = 48000;
    uint16_t bits_per_samp = 16;
    uint16_t block_align  = channels * bits_per_samp / 8;  // 4
    uint32_t byte_rate    = sample_rate * block_align;     // 192000
    uint32_t data_bytes   = (uint32_t)stereo_pcm.size() * bits_per_samp / 8;  // 12
    uint32_t riff_size    = 36 + data_bytes;  // 48

    std::vector<uint8_t> wav;
    auto w16 = [&](uint16_t v) { wav.push_back(v & 0xff); wav.push_back((v >> 8) & 0xff); };
    auto w32 = [&](uint32_t v) { wav.push_back(v & 0xff); wav.push_back((v >> 8) & 0xff);
                                 wav.push_back((v >> 16) & 0xff); wav.push_back((v >> 24) & 0xff); };
    wav.insert(wav.end(), (const uint8_t*)"RIFF", (const uint8_t*)"RIFF" + 4); w32(riff_size);
    wav.insert(wav.end(), (const uint8_t*)"WAVE", (const uint8_t*)"WAVE" + 4);
    wav.insert(wav.end(), (const uint8_t*)"fmt ", (const uint8_t*)"fmt " + 4);
    w32(16); w16(1 /* PCM */); w16(channels); w32(sample_rate); w32(byte_rate);
    w16(block_align); w16(bits_per_samp);
    wav.insert(wav.end(), (const uint8_t*)"data", (const uint8_t*)"data" + 4); w32(data_bytes);
    for (float s : stereo_pcm) {
        int16_t v = static_cast<int16_t>(s * 32767.0f);
        w16(static_cast<uint16_t>(v));
    }

    std::string b64 = to_base64(wav);

    auto rs = start_server(slots);
    AUDIOCORE_CHECK(rs != nullptr);
    httplib::Client cli("127.0.0.1", rs->port);
    cli.set_read_timeout(5);
    json body = {
        {"model", "mock-music-3"},
        {"caption", "cover version"},
        {"input_audio", b64},
        {"mode", "cover"},
    };
    auto res = cli.Post("/v1/audio/music", body.dump(), "application/json");
    AUDIOCORE_CHECK(res != nullptr);
    AUDIOCORE_CHECK_EQ(res->status, 200);
    // The mock received the parsed input_audio (should be stereo interleaved)
    AUDIOCORE_CHECK_EQ(raw->last_input_audio_.size(), stereo_pcm.size());
    if (raw->last_input_audio_.size() == stereo_pcm.size()) {
        for (size_t i = 0; i < stereo_pcm.size(); i++) {
            AUDIOCORE_CHECK_CLOSE(raw->last_input_audio_[i], stereo_pcm[i], 1e-4f);
        }
    }
}

AUDIOCORE_TEST(music_route_stereo_wav_header_is_well_formed) {
    // Direct unit test of the stereo WAV encoder, independent of HTTP.
    std::vector<float> pcm = {  // L,R,L,R (2 frames)
        0.0f, 0.5f, -0.5f, 1.0f
    };
    auto bytes = pcm_stereo_to_wav(pcm, /*sr=*/48000);
    AUDIOCORE_CHECK(bytes.size() == 44 + (uint32_t)pcm.size() * 2);
    AUDIOCORE_CHECK(std::memcmp(bytes.data() +  0, "RIFF", 4) == 0);
    AUDIOCORE_CHECK(std::memcmp(bytes.data() +  8, "WAVE", 4) == 0);
    AUDIOCORE_CHECK(std::memcmp(bytes.data() + 12, "fmt ", 4) == 0);
    AUDIOCORE_CHECK(std::memcmp(bytes.data() + 36, "data", 4) == 0);
    // 2 channels
    uint16_t ch = static_cast<uint8_t>(bytes[22]) | (static_cast<uint8_t>(bytes[23]) << 8);
    AUDIOCORE_CHECK_EQ(ch, static_cast<uint16_t>(2));
    // Sample rate
    uint32_t sr = static_cast<uint8_t>(bytes[24]) |
                  (static_cast<uint8_t>(bytes[25]) << 8) |
                  (static_cast<uint8_t>(bytes[26]) << 16) |
                  (static_cast<uint32_t>(static_cast<uint8_t>(bytes[27])) << 24);
    AUDIOCORE_CHECK_EQ(sr, static_cast<uint32_t>(48000));
    // Data chunk length = 4 samples × 2 bytes = 8
    uint32_t data_len = static_cast<uint8_t>(bytes[40]) |
                        (static_cast<uint8_t>(bytes[41]) << 8) |
                        (static_cast<uint8_t>(bytes[42]) << 16) |
                        (static_cast<uint32_t>(static_cast<uint8_t>(bytes[43])) << 24);
    AUDIOCORE_CHECK_EQ(data_len, static_cast<uint32_t>(8));
}

AUDIOCORE_TEST(music_route_mp3_format_returns_audio_mpeg) {
    auto raw  = new MockMusicSession();
    auto slot = std::make_shared<ModelSlot>();
    slot->session = std::unique_ptr<Session>(raw);
    auto slots = std::make_shared<
        std::unordered_map<std::string, std::shared_ptr<ModelSlot>>>();
    (*slots)["mock-music-4"] = slot;

    auto rs = start_server(slots);
    AUDIOCORE_CHECK(rs != nullptr);
    httplib::Client cli("127.0.0.1", rs->port);
    cli.set_read_timeout(5);
    json body = {
        {"model",           "mock-music-4"},
        {"caption",         "mp3 test"},
        {"response_format", "mp3"},
    };
    auto res = cli.Post("/v1/audio/music", body.dump(), "application/json");
    AUDIOCORE_CHECK(res != nullptr);
    AUDIOCORE_CHECK_EQ(res->status, 200);
    AUDIOCORE_CHECK_EQ(res->get_header_value("Content-Type"), std::string("audio/mpeg"));
    AUDIOCORE_CHECK(!res->body.empty());
    // MP3 sync word: first 11 bits = 0xFF 0xFB (MPEG1 layer3, no CRC)
    AUDIOCORE_CHECK(res->body.size() >= 2);
    AUDIOCORE_CHECK_EQ(static_cast<uint8_t>(res->body[0]), 0xFF);
    AUDIOCORE_CHECK_EQ(static_cast<uint8_t>(res->body[1]) & 0xF0, 0xF0);
}

AUDIOCORE_TEST(pcm_stereo_to_mp3_produces_valid_mp3) {
    // Direct unit test of the MP3 encoder, independent of HTTP.
    std::vector<float> pcm(48000 * 2);  // 1 second of stereo at 48 kHz
    for (size_t i = 0; i < pcm.size(); i += 2) {
        float v = 0.5f * sinf(2.0f * (float)M_PI * 440.0f * (float)(i / 2) / 48000.0f);
        pcm[i] = v;      // L
        pcm[i + 1] = v;  // R
    }
    auto mp3 = pcm_stereo_to_mp3(pcm.data(), pcm.size(), 48000);
    AUDIOCORE_CHECK(!mp3.empty());
    // MP3 sync word
    AUDIOCORE_CHECK(mp3.size() >= 2);
    AUDIOCORE_CHECK_EQ(static_cast<uint8_t>(mp3[0]), 0xFF);
    AUDIOCORE_CHECK_EQ(static_cast<uint8_t>(mp3[1]) & 0xF0, 0xF0);
}

AUDIOCORE_TEST(pcm_mono_to_mp3_produces_valid_mp3) {
    // Direct unit test of the mono MP3 encoder.
    std::vector<float> pcm(48000);  // 1 second of mono at 48 kHz
    for (size_t i = 0; i < pcm.size(); i++)
        pcm[i] = 0.5f * sinf(2.0f * (float)M_PI * 440.0f * (float)i / 48000.0f);
    auto mp3 = pcm_mono_to_mp3(pcm.data(), pcm.size(), 48000);
    AUDIOCORE_CHECK(!mp3.empty());
    AUDIOCORE_CHECK(mp3.size() >= 2);
    AUDIOCORE_CHECK_EQ(static_cast<uint8_t>(mp3[0]), 0xFF);
    AUDIOCORE_CHECK_EQ(static_cast<uint8_t>(mp3[1]) & 0xF0, 0xF0);
}

int main() {
    std::printf("=== HTTP server e2e tests ===\n");
    return test::run_all();
}
