// loader.cpp — ZONOS2 subprocess lifecycle + family registration.
//
// This file implements Zonos2Session::load(), which starts the official
// ZONOS2 Mini-SGLang server as a child process and waits for it to
// report healthy. The actual model inference runs entirely in the
// subprocess; the C++ side just forwards TTS requests over HTTP.
//
// Command launched (configurable via LoadOptions::extras):
//   python3 -m minisgl --model-path <model_path> --port <port> ...
//
// Runtime deps for the subprocess (not built here):
//   - Python 3.10+, torch, transformers, zonos, dac

#include "audiocore/models/zonos2/family.h"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <httplib.h>

#include "audiocore/framework/runtime/registry.h"

namespace audiocore::zonos2 {

// ===========================================================================
// Free-port helper
// ===========================================================================

int Zonos2Session::find_free_port() {
    int sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return -1;

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (::bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(sock);
        return -1;
    }

    socklen_t len = sizeof(addr);
    if (::getsockname(sock, reinterpret_cast<sockaddr*>(&addr), &len) < 0) {
        ::close(sock);
        return -1;
    }

    int port = ntohs(addr.sin_port);
    ::close(sock);
    return port;
}

// ===========================================================================
// Subprocess lifecycle
// ===========================================================================

bool Zonos2Session::start_subprocess(std::string* error) {
    // Pick a port.
    subprocess_port_ = cfg_.port > 0 ? cfg_.port : find_free_port();
    if (subprocess_port_ <= 0) {
        if (error) *error = "zonos2: failed to find a free TCP port";
        return false;
    }

    pid_t pid = ::fork();
    if (pid < 0) {
        if (error) {
            *error = std::string("zonos2: fork failed: ") + ::strerror(errno);
        }
        return false;
    }

    if (pid == 0) {
        // ── Child process ──────────────────────────────────────────
        // Run the Mini-SGLang server.
        const std::string port_str  = std::to_string(subprocess_port_);
        const std::string model     = cfg_.model_path;
        const std::string device    = cfg_.device;

        // Redirect stdout/stderr to stderr (same terminal as parent).
        // We don't close stdin so the subprocess can read if needed.
        ::dup2(STDERR_FILENO, STDOUT_FILENO);

        ::execlp(cfg_.python_bin.c_str(),
                 cfg_.python_bin.c_str(),
                 "-m", "minisgl",
                 "--model-path", model.c_str(),
                 "--port", port_str.c_str(),
                 "--device", device.c_str(),
                 nullptr);

        // exec failed — print to stderr and exit.
        std::fprintf(stderr, "zonos2: exec(%s) failed: %s\n",
                     cfg_.python_bin.c_str(), ::strerror(errno));
        ::_exit(127);
    }

    // ── Parent process ────────────────────────────────────────────
    subprocess_pid_ = pid;
    subprocess_url_ = "http://127.0.0.1:" + std::to_string(subprocess_port_);
    std::fprintf(stderr, "zonos2: started subprocess (PID %d) on %s\n",
                 pid, subprocess_url_.c_str());
    return true;
}

bool Zonos2Session::wait_for_ready(std::string* error) {
    httplib::Client cli("127.0.0.1", subprocess_port_);
    cli.set_read_timeout(2);
    cli.set_connection_timeout(1);

    const int max_attempts = cfg_.ready_timeout_sec * 2;  // every 500 ms
    for (int i = 0; i < max_attempts; ++i) {
        // Check if the subprocess is still alive.
        int status = 0;
        pid_t r = ::waitpid(subprocess_pid_, &status, WNOHANG);
        if (r == subprocess_pid_) {
            // Subprocess exited before becoming ready.
            if (error) {
                *error = "zonos2: subprocess died before becoming ready "
                         "(exit status " + std::to_string(WEXITSTATUS(status)) + ")";
            }
            subprocess_pid_ = -1;
            return false;
        }

        auto res = cli.Get("/health");
        if (res && res->status == 200) {
            std::fprintf(stderr, "zonos2: subprocess ready after ~%d ms\n",
                         i * 500);
            return true;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // Timed out.
    stop_subprocess();
    if (error) {
        *error = "zonos2: subprocess did not become ready within " +
                 std::to_string(cfg_.ready_timeout_sec) + "s";
    }
    return false;
}

void Zonos2Session::stop_subprocess() {
    if (!subprocess_running_ || subprocess_pid_ <= 0) return;
    subprocess_running_ = false;

    std::fprintf(stderr, "zonos2: stopping subprocess (PID %d)...\n",
                 subprocess_pid_);

    // 1. Send a graceful /shutdown POST so the Python process can release
    //    GPU memory and flush logs.
    if (!subprocess_url_.empty()) {
        httplib::Client cli("127.0.0.1", subprocess_port_);
        cli.set_read_timeout(2);
        cli.Post("/shutdown", "", "application/json");
    }

    // 2. SIGTERM — allow 1 s to exit cleanly.
    ::kill(subprocess_pid_, SIGTERM);
    for (int i = 0; i < 5; ++i) {
        int status = 0;
        pid_t r = ::waitpid(subprocess_pid_, &status, WNOHANG);
        if (r == subprocess_pid_) {
            std::fprintf(stderr, "zonos2: subprocess exited cleanly\n");
            subprocess_pid_ = -1;
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    // 3. SIGKILL — force terminate.
    std::fprintf(stderr, "zonos2: force-killing subprocess (PID %d)...\n",
                 subprocess_pid_);
    ::kill(subprocess_pid_, SIGKILL);
    ::waitpid(subprocess_pid_, nullptr, 0);
    subprocess_pid_ = -1;
}

// ===========================================================================
// load()
// ===========================================================================

bool Zonos2Session::load(const std::string& model_path,
                          const LoadOptions& opts,
                          const BackendConfig& backend_cfg,
                          std::string* error) {
    cfg_.model_path = model_path;

    // Parse extras.
    auto it = opts.extras.find("python_bin");
    if (it != opts.extras.end()) cfg_.python_bin = it->second;

    it = opts.extras.find("device");
    if (it != opts.extras.end()) {
        cfg_.device = it->second;
    } else {
        // Translate BackendConfig to device string.
        if (backend_cfg.kind == BackendKind::ggml_cpu) {
            cfg_.device = "cpu";
        } else {
            cfg_.device = "cuda:" + std::to_string(backend_cfg.device_id);
        }
    }

    it = opts.extras.find("port");
    if (it != opts.extras.end()) {
        cfg_.port = std::stoi(it->second);
    }

    it = opts.extras.find("ready_timeout");
    if (it != opts.extras.end()) {
        cfg_.ready_timeout_sec = std::stoi(it->second);
    }

    std::fprintf(stderr, "zonos2: loading model '%s' on %s (port %d)\n",
                 cfg_.model_path.c_str(), cfg_.device.c_str(),
                 cfg_.port > 0 ? cfg_.port : 0);

    if (!start_subprocess(error)) return false;
    if (!wait_for_ready(error)) return false;

    subprocess_running_ = true;
    loaded_ = true;
    std::fprintf(stderr, "zonos2: model ready on %s\n",
                 subprocess_url_.c_str());
    return true;
}

// ===========================================================================
// Factory + family registration
// ===========================================================================

namespace {
std::unique_ptr<Session> make_zonos2_session() {
    return std::unique_ptr<Session>(new Zonos2Session());
}
}  // namespace

AUDIOCORE_REGISTER_FAMILY(zonos2, make_zonos2_session)
AUDIOCORE_EXTERN_C_GUARD(zonos2, make_zonos2_session)

}  // namespace audiocore::zonos2
