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
#include "audiocore/models/moss_tts/family.h"

#include <atomic>
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
        last_text_     = static_cast<const moss::TtsRequest*>(request)->text;
        last_language_ = static_cast<const moss::TtsRequest*>(request)->language;
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

    std::string last_text_;
    std::string last_language_;
};

std::unique_ptr<Session> make_mock() {
    return std::unique_ptr<Session>(new MockTtsSession());
}

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

std::unique_ptr<RunningServer> start_server() {
    auto rs = std::unique_ptr<RunningServer>(new RunningServer);
    auto [slots, sess] = make_slots("mock-1");
    rs->slots   = slots;
    rs->session = sess;
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

int main() {
    std::printf("=== HTTP server e2e tests ===\n");
    return test::run_all();
}
