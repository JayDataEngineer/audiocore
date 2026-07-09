#ifndef AUDIOCORE_FRAMEWORK_RUNTIME_REGISTRY_H
#define AUDIOCORE_FRAMEWORK_RUNTIME_REGISTRY_H

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace audiocore {

class Session;

// ── Capabilities (new interface, adapted from audio.cpp) ──────────────────

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

// ── Session interfaces (new, adapted from audio.cpp) ──────────────────────

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
    virtual bool run_tts(const void* request, void* response,
                         std::string* error = nullptr) = 0;
    virtual bool run_music(const void* request, void* response,
                           std::string* error = nullptr) = 0;
    // Compute a speaker embedding from a reference WAV for voice cloning.
    // Only qwen3_tts implements this (ECAPA-TDNN encoder); others return {}.
    virtual std::vector<float> compute_embedding(const std::string& wav_path,
                                                  std::string* error = nullptr) {
        if (error) *error = "compute_embedding not supported by this family";
        return {};
    }
};

// ── Loaded model interface ────────────────────────────────────────────────

class ILoadedModel {
public:
    virtual ~ILoadedModel() = default;
    virtual const ModelMetadata& metadata() const noexcept = 0;
    virtual const CapabilitySet& capabilities() const noexcept = 0;
    // Returns a non-owning pointer to the session. The session's lifetime
    // is tied to this ILoadedModel — destroy the model to free the session.
    virtual IOfflineTaskSession* create_session(
        const TaskSpec& task,
        const SessionOptions& options) const = 0;
};

// ── Model loader interface (new, adapted from audio.cpp) ──────────────────

class IModelLoader {
public:
    virtual ~IModelLoader() = default;
    virtual std::string family() const = 0;
    virtual bool can_load(const ModelLoadRequest& request) const = 0;
    virtual ModelInspection inspect(const ModelLoadRequest& request) const = 0;
    virtual std::unique_ptr<ILoadedModel> load(
        const ModelLoadRequest& request) const = 0;
};

// ── ModelRegistry (new, adapted from audio.cpp) ───────────────────────────

class ModelRegistry {
public:
    void register_loader(std::shared_ptr<IModelLoader> loader);

    bool empty() const noexcept;
    size_t size() const noexcept;
    std::vector<std::string> families() const;
    bool supports_family(const std::string& family) const noexcept;

    ModelInspection inspect(const ModelLoadRequest& request) const;
    std::unique_ptr<ILoadedModel> load(const ModelLoadRequest& request) const;
    std::unique_ptr<ILoadedModel> load(const std::filesystem::path& model_path) const;

private:
    const IModelLoader* find_loader(const ModelLoadRequest& request) const;
    std::vector<std::shared_ptr<IModelLoader>> loaders_;
};

// Build a registry wrapping all registered families.
ModelRegistry make_default_registry();

// ── OLD FamilyRegistry (backward compat — kept for existing family code) ──

using FamilyFactory = std::function<std::unique_ptr<Session>()>;

class FamilyRegistry {
public:
    static FamilyRegistry& instance();
    void register_family(const std::string& name, FamilyFactory factory);
    std::unique_ptr<Session> create(const std::string& family) const;
    std::vector<std::string> list() const;
private:
    std::unordered_map<std::string, FamilyFactory> families_;
};

struct FamilyRegistrar {
    FamilyRegistrar(const std::string& name, FamilyFactory factory);
};

}  // namespace audiocore

// Static-registration macro (backward compat).
#define AUDIOCORE_REGISTER_FAMILY(name, factory)               \
    namespace {                                                 \
    ::audiocore::FamilyRegistrar                                \
        g_register_##name(#name, factory);                     \
    }

#define AUDIOCORE_EXTERN_C_GUARD(name, factory)                               \
    extern "C" void audiocore_register_##name() {                             \
        static bool done_##name = false;                                      \
        if (!done_##name) {                                                   \
            ::audiocore::FamilyRegistry::instance().register_family(           \
                #name, factory);                                              \
            done_##name = true;                                               \
        }                                                                     \
    }

#endif  // AUDIOCORE_FRAMEWORK_RUNTIME_REGISTRY_H
