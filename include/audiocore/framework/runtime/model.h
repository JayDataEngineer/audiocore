#ifndef AUDIOCORE_FRAMEWORK_RUNTIME_MODEL_H
#define AUDIOCORE_FRAMEWORK_RUNTIME_MODEL_H

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace audiocore {

// ── Capabilities ──────────────────────────────────────────────────────────

enum class VoiceTaskKind {
    Tts,
    VoiceCloning,
    MusicGen,
    Sfx,
    Alignment,
};

struct CapabilitySet {
    std::vector<VoiceTaskKind> tasks;
    std::vector<std::string> languages;
    bool supports_voice_cloning = false;
    bool supports_style = false;
};

struct ModelMetadata {
    std::string family;
    std::string variant;
    std::string description;
};

struct ModelInspection {
    ModelMetadata metadata;
    CapabilitySet capabilities;
    std::filesystem::path model_path;
};

struct ModelLoadRequest {
    std::filesystem::path model_path;
    std::string family_hint;
    std::unordered_map<std::string, std::string> options;
};

// ── Session interface ─────────────────────────────────────────────────────

struct TaskSpec {
    VoiceTaskKind task = VoiceTaskKind::Tts;
};

struct SessionOptions {
    std::unordered_map<std::string, std::string> options;
};

class ITaskSession {
public:
    virtual ~ITaskSession() = default;
    virtual std::string family() const = 0;
    virtual VoiceTaskKind task_kind() const = 0;
};

class IOfflineTaskSession : public virtual ITaskSession {
public:
    ~IOfflineTaskSession() override = default;
    // Run a synchronous TTS/music task. I/O structs defined in tasks.h.
    // Returns true on success, false + error message on failure.
    virtual bool run_tts(const void* request, void* response,
                         std::string* error = nullptr) = 0;
    virtual bool run_music(const void* request, void* response,
                           std::string* error = nullptr) = 0;
};

// ── Loaded model interface ────────────────────────────────────────────────

class ILoadedModel {
public:
    virtual ~ILoadedModel() = default;
    virtual const ModelMetadata& metadata() const noexcept = 0;
    virtual const CapabilitySet& capabilities() const noexcept = 0;
    virtual std::unique_ptr<IOfflineTaskSession> create_session(
        const TaskSpec& task,
        const SessionOptions& options) const = 0;
};

// ── Model loader interface ────────────────────────────────────────────────

class IModelLoader {
public:
    virtual ~IModelLoader() = default;
    virtual std::string family() const = 0;
    virtual bool can_load(const ModelLoadRequest& request) const = 0;
    virtual ModelInspection inspect(const ModelLoadRequest& request) const = 0;
    virtual std::unique_ptr<ILoadedModel> load(
        const ModelLoadRequest& request) const = 0;
};

}  // namespace audiocore

#endif  // AUDIOCORE_FRAMEWORK_RUNTIME_MODEL_H
