// qwen3tts_proxy.cpp — Translation layer implementation.
//
// Translates our TtsRequest into the reference qwen_tts server's
// OpenAI-compatible JSON format, POSTs to /v1/audio/speech, and
// returns the WAV bytes.
//
// The reference API flow (synchronous — no job polling):
//   POST /v1/audio/speech { "input":"text","voice":"speaker",... }
//     → 200 OK audio/wav (binary)
//
// Supported JSON fields by the reference server (see qwen_tts_server.c):
//   text | input      — synthesis text (required)
//   speaker | voice   — speaker name or numeric ID
//   language          — language code ("en","zh","ja",...)
//   instruct          — natural-language style instruction (1.7B)
//   temperature       — sampling temperature [0..2]
//   top_p, top_k      — nucleus / top-k sampling
//   max_tokens        — max generation length
//   seed              — random seed (-1 = time-based)
//   volume            — output gain (1.0 = unchanged)
//   rate              — speaking rate (1.0 = unchanged, pitch-preserving)
//   response_format   — "wav" (only WAV supported on this path)

#include "audiocore/server/qwen3tts_proxy.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace audiocore {

using json = nlohmann::json;

// ── Helpers ────────────────────────────────────────────────────────────

// Parse host:port from a URL like "http://localhost:8086".
static void parse_url(const std::string& url, std::string& host, uint16_t& port) {
    host = url;
    port = 8086;
    size_t scheme = host.find("://");
    if (scheme != std::string::npos) host = host.substr(scheme + 3);
    size_t colon = host.rfind(':');
    if (colon != std::string::npos) {
        try {
            port = (uint16_t)std::stoi(host.substr(colon + 1));
        } catch (...) {}
        host = host.substr(0, colon);
    }
    // Strip any trailing slash / path.
    size_t slash = host.find('/');
    if (slash != std::string::npos) host = host.substr(0, slash);
}

// Resolve a voice field. The webapp sends "Vivian", a numeric ID, a
// .voice file path, or a .qvoice graft. The reference server wants a
// speaker name or numeric ID. If the field looks like a file path
// (contains '/' or ends in .voice/.qvoice), pass the basename without
// extension — the reference server treats unknown speakers as IDs.
static std::string resolve_speaker(const std::string& voice_path,
                                   const std::string& speaker_name) {
    if (!speaker_name.empty()) return speaker_name;
    if (!voice_path.empty()) {
        // Voice file: take basename, drop extension.
        std::string v = voice_path;
        size_t slash = v.find_last_of('/');
        if (slash != std::string::npos) v = v.substr(slash + 1);
        size_t dot = v.find_last_of('.');
        if (dot != std::string::npos) v = v.substr(0, dot);
        return v;
    }
    return "";
}

// ── Public API ─────────────────────────────────────────────────────────

std::vector<uint8_t> qwen3tts_proxy_generate(
        const Qwen3TtsProxyConfig& config,
        const TtsRequest& req,
        std::string* error) {

    std::string host;
    uint16_t port;
    parse_url(config.server_url, host, port);

    httplib::Client cli(host, port);
    cli.set_connection_timeout(30);
    cli.set_read_timeout(config.timeout_seconds);
    cli.set_write_timeout(30);

    fprintf(stderr, "[qwen3tts_proxy] → qwen_tts-server at %s:%d\n",
            host.c_str(), port);

    // Build the request JSON — translate our TtsRequest to the reference schema.
    json j;
    // Prefer the explicit text; if messages are populated, join them with
    // newlines so the reference synthesizes the full dialogue in one pass.
    if (!req.text.empty()) {
        j["input"] = req.text;
    } else if (!req.messages.empty()) {
        std::string joined;
        for (size_t i = 0; i < req.messages.size(); ++i) {
            if (i) joined += "\n";
            joined += req.messages[i].content;
        }
        j["input"] = joined;
    } else {
        if (error) *error = "no text to synthesize";
        return {};
    }

    std::string speaker = resolve_speaker(req.voice_path, req.speaker_name);
    if (!speaker.empty())  j["voice"] = speaker;
    if (!req.language.empty()) j["language"] = req.language;
    if (!req.instruct.empty()) j["instruct"] = req.instruct;

    // Sampling parameters.
    if (req.temperature > 0)   j["temperature"] = req.temperature;
    if (req.top_p > 0)         j["top_p"]       = req.top_p;
    if (req.text_top_k > 0)    j["top_k"]       = req.text_top_k;
    if (req.max_new_tokens > 0) j["max_tokens"] = req.max_new_tokens;
    if (req.seed != 0)         j["seed"]        = req.seed;
    if (std::fabs(req.repetition_penalty - 1.0f) > 0.001f)
        j["rep-penalty"] = req.repetition_penalty;

    // Rate / volume — map our speed → their rate.
    if (std::fabs(req.speed - 1.0f) > 0.001f) j["rate"] = req.speed;

    // Emotion + voice strength (forwarded to reference server if supported).
    if (!req.emotion.empty()) j["emotion"] = req.emotion;
    if (std::fabs(req.voice_strength - 1.0f) > 0.001f)
        j["voice_strength"] = req.voice_strength;

    j["response_format"] = "wav";

    std::string body = j.dump();
    fprintf(stderr, "[qwen3tts_proxy] POST /v1/audio/speech (input='%s...', voice='%s')\n",
            req.text.substr(0, 50).c_str(), speaker.c_str());

    auto res = cli.Post("/v1/audio/speech", body, "application/json");
    if (!res) {
        if (error) *error = "qwen_tts-server connection failed";
        return {};
    }
    if (res->status != 200) {
        if (error) *error = "qwen_tts-server returned status " +
                            std::to_string(res->status) + ": " + res->body;
        return {};
    }

    // The reference server returns audio/wav as the response body.
    std::vector<uint8_t> wav(res->body.begin(), res->body.end());
    fprintf(stderr, "[qwen3tts_proxy] ← %zu bytes WAV\n", wav.size());
    return wav;
}

bool qwen3tts_proxy_health(const Qwen3TtsProxyConfig& config) {
    std::string host;
    uint16_t port;
    parse_url(config.server_url, host, port);
    httplib::Client cli(host, port);
    cli.set_connection_timeout(5);
    cli.set_read_timeout(5);
    auto res = cli.Get("/v1/health");
    return res && res->status == 200;
}

// ── CLI runner ────────────────────────────────────────────────────────

// Run the qwen_tts CLI binary with the given argv and environment.
// Returns the exit code, or -1 on fork/exec failure.
// stdout/stderr are inherited (visible in server logs).
int qwen3tts_run_cli(const std::string& binary,
                   const std::vector<std::string>& argv,
                   const std::vector<std::string>& envp) {
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        // Child: set environment, exec.
        for (const auto& e : envp) {
            auto eq = e.find('=');
            if (eq != std::string::npos)
                setenv(e.substr(0, eq).c_str(), e.substr(eq + 1).c_str(), 1);
        }
        std::vector<const char*> cargv;
        for (const auto& a : argv) cargv.push_back(a.c_str());
        cargv.push_back(nullptr);
        execv(binary.c_str(), const_cast<char* const*>(cargv.data()));
        _exit(127);
    }
    // Parent: wait.
    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return -1;
}

// Read an entire file into a byte vector.
std::vector<uint8_t> qwen3tts_read_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz);
    size_t rd = fread(buf.data(), 1, sz, f);
    fclose(f);
    buf.resize(rd);
    return buf;
}

// Escape a string for safe shell embedding (single-quote style).
static std::string shell_escape(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

// Build the CLI environment (LD_LIBRARY_PATH, GPU vars).
std::vector<std::string> qwen3tts_build_cli_env(
        const Qwen3TtsProxyConfig& config) {
    std::vector<std::string> env;
    // LD_LIBRARY_PATH: prepend library_dir if set.
    if (!config.model_dir.empty()) {
        // The binary's directory is the default library dir.
        std::string bin_dir = config.binary_path;
        auto sl = bin_dir.find_last_of('/');
        if (sl != std::string::npos) bin_dir = bin_dir.substr(0, sl);
        std::string lp = bin_dir;
        const char* existing = getenv("LD_LIBRARY_PATH");
        if (existing && strlen(existing) > 0)
            lp += std::string(":") + existing;
        env.push_back("LD_LIBRARY_PATH=" + lp);
    }
    if (config.use_gpu) {
        env.push_back("GGML_CUDA_NO_PINNED=0");
        env.push_back("CUDA_VISIBLE_DEVICES=" + std::to_string(config.gpu_device));
    }
    return env;
}

// ── CLI: generate speech from a .qvoice file ──────────────────────────

std::vector<uint8_t> qwen3tts_cli_generate(
        const Qwen3TtsProxyConfig& config,
        const TtsRequest& req,
        const std::string& voice_qvoice_path,
        std::string* error) {

    if (config.binary_path.empty() || config.customvoice_dir.empty()) {
        if (error) *error = "CLI runner not configured (binary_path/customvoice_dir empty)";
        return {};
    }

    std::string tmp_out = "/tmp/audiocore_cli_out_" + std::to_string(getpid()) + ".wav";

    // Detect if this is a heavy WDELTA voice (file > 100MB).
    // Heavy voices need --load-voice WITHOUT --icl-only so weights get applied.
    // Lite grafts (< 100MB) need --icl-only to skip weight-swap.
    struct stat vstat;
    bool is_heavy = (stat(voice_qvoice_path.c_str(), &vstat) == 0 && vstat.st_size > 100 * 1024 * 1024);

    // Build argv — official Qwen3-TTS generation_config.json defaults.
    std::vector<std::string> argv = {
        config.binary_path,
        "-d", config.customvoice_dir,
        "--load-voice", voice_qvoice_path,
    };
    if (!is_heavy) {
        argv.push_back("--icl-only");
    }
    argv.push_back("--text");
    argv.push_back(req.text);
    argv.push_back("-o");
    argv.push_back(tmp_out);
    argv.push_back("-T");
    argv.push_back(std::to_string(req.temperature > 0 ? req.temperature : 0.9f));
    argv.push_back("-p");
    argv.push_back(std::to_string(req.top_p > 0 ? req.top_p : 1.0f));
    argv.push_back("-k");
    argv.push_back(std::to_string(req.text_top_k > 0 ? req.text_top_k : 50));
    argv.push_back("--rep-penalty");
    argv.push_back(std::to_string(req.repetition_penalty > 0 ? req.repetition_penalty : 1.05f));
    if (config.use_gpu) {
        argv.push_back("--backend");
        argv.push_back("cuda");
    }
    if (!req.language.empty()) {
        argv.push_back("--language");
        argv.push_back(req.language);
    }
    if (!req.emotion.empty()) {
        argv.push_back("--emotion");
        argv.push_back(req.emotion);
    }
    if (!req.expr_file.empty()) {
        argv.push_back("--expr");
        argv.push_back(req.expr_file);
        if (std::fabs(req.expr_weight - 1.0f) > 0.001f) {
            argv.push_back("--expr-weight");
            argv.push_back(std::to_string(req.expr_weight));
        }
    }
    if (std::fabs(req.voice_strength - 1.0f) > 0.001f) {
        argv.push_back("--voice-strength");
        argv.push_back(std::to_string(req.voice_strength));
    }
    if (!req.instruct.empty()) {
        argv.push_back("--instruct");
        argv.push_back(req.instruct);
    }
    if (req.seed != 0) {
        argv.push_back("--seed");
        argv.push_back(std::to_string(req.seed));
    }

    fprintf(stderr, "[qwen3tts_cli] generate: voice=%s emotion=%s strength=%.2f text='%s...'\n",
            voice_qvoice_path.c_str(), req.emotion.c_str(), req.voice_strength,
            req.text.substr(0, 40).c_str());

    auto env = qwen3tts_build_cli_env(config);
    int rc = qwen3tts_run_cli(config.binary_path, argv, env);

    if (rc != 0) {
        unlink(tmp_out.c_str());
        if (error) *error = "qwen_tts CLI exited with code " + std::to_string(rc);
        return {};
    }

    auto wav = qwen3tts_read_file(tmp_out);
    unlink(tmp_out.c_str());
    if (wav.empty()) {
        if (error) *error = "CLI produced no output WAV";
        return {};
    }
    fprintf(stderr, "[qwen3tts_cli] ← %zu bytes WAV\n", wav.size());
    return wav;
}

// ── CLI: export a voice (VoiceDesign gen + Base bake) ─────────────────

bool qwen3tts_cli_export_voice(
        const Qwen3TtsProxyConfig& config,
        const std::string& name,
        const std::string& instruct,
        const std::string& text,
        std::string* error,
        bool wdelta) {

    if (config.binary_path.empty() || config.voicedesign_dir.empty() || config.base_dir.empty()) {
        if (error) *error = "CLI export not configured (binary_path/voicedesign_dir/base_dir empty)";
        return false;
    }
    if (config.voices_dir.empty()) {
        if (error) *error = "voices_dir not configured";
        return false;
    }

    // Ensure voices_dir exists.
    mkdir(config.voices_dir.c_str(), 0755);

    std::string tmp_ref = "/tmp/audiocore_export_ref_" + std::to_string(getpid()) + ".wav";

    // Auto-number on name collision: foo → foo_2 → foo_3 → ...
    std::string out_name = name;
    std::string out_qvoice = config.voices_dir + "/" + out_name + ".qvoice";
    {
        struct stat st;
        int seq = 2;
        while (stat(out_qvoice.c_str(), &st) == 0) {
            out_name = name + "_" + std::to_string(seq++);
            out_qvoice = config.voices_dir + "/" + out_name + ".qvoice";
        }
    }

    // Step 1: VoiceDesign generate reference WAV.
    {
        std::string synth_text = text.empty()
            ? "Hello, I am a friendly voice assistant. How can I help you today?"
            : text;

        std::vector<std::string> argv = {
            config.binary_path,
            "-d", config.voicedesign_dir,
            "--voice-design",
            "--instruct", instruct,
            "--text", synth_text,
            "-o", tmp_ref,
            "-T", "0.9",
            "-p", "1.0",
            "-k", "50",
            "--rep-penalty", "1.05",
        };
        if (config.use_gpu) {
            argv.push_back("--backend");
            argv.push_back("cuda");
        }

        fprintf(stderr, "[qwen3tts_cli] export step1: VoiceDesign gen instruct='%s...'\n",
                instruct.substr(0, 40).c_str());

        auto env = qwen3tts_build_cli_env(config);
        int rc = qwen3tts_run_cli(config.binary_path, argv, env);

        if (rc != 0) {
            unlink(tmp_ref.c_str());
            if (error) *error = "VoiceDesign CLI exited with code " + std::to_string(rc);
            return false;
        }
    }

    // Step 2: Bake with Base model.
    {
        std::vector<std::string> argv = {
            config.binary_path,
            "-d", config.base_dir,
            "--ref-audio", tmp_ref,
            "--voice-name", out_name,
            "--save-voice", out_qvoice,
        };
        if (wdelta) {
            argv.push_back("--target-cv");
        }
        if (config.use_gpu) {
            argv.push_back("--backend");
            argv.push_back("cuda");
        }

        fprintf(stderr, "[qwen3tts_cli] export step2: bake%s → %s\n",
                wdelta ? " (WDELTA)" : "", out_qvoice.c_str());

        auto env = qwen3tts_build_cli_env(config);
        int rc = qwen3tts_run_cli(config.binary_path, argv, env);
        unlink(tmp_ref.c_str());

        if (rc != 0) {
            if (error) *error = "bake CLI exited with code " + std::to_string(rc);
            return false;
        }
    }

    // Verify the .qvoice was created.
    struct stat st;
    if (stat(out_qvoice.c_str(), &st) != 0 || st.st_size == 0) {
        if (error) *error = "export produced no .qvoice file";
        return false;
    }

    fprintf(stderr, "[qwen3tts_cli] export done: %s (%ld bytes)\n",
            out_qvoice.c_str(), (long)st.st_size);
    return true;
}

// ── List saved voices ──────────────────────────────────────────────────

std::vector<VoiceInfo> qwen3tts_list_voices(
        const Qwen3TtsProxyConfig& config) {
    std::vector<VoiceInfo> result;
    if (config.voices_dir.empty()) return result;

    DIR* dir = opendir(config.voices_dir.c_str());
    if (!dir) return result;

    struct dirent* ent;
    while ((ent = readdir(dir)) != nullptr) {
        std::string name = ent->d_name;
        // Only .qvoice files.
        if (name.size() < 8 || name.substr(name.size() - 7) != ".qvoice")
            continue;
        std::string full = config.voices_dir + "/" + name;
        struct stat st;
        if (stat(full.c_str(), &st) != 0) continue;
        VoiceInfo vi;
        vi.name = name.substr(0, name.size() - 7);  // strip .qvoice
        vi.path = full;
        vi.size_bytes = st.st_size;
        result.push_back(std::move(vi));
    }
    closedir(dir);
    return result;
}

}  // namespace audiocore
