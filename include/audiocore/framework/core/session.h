// session.h — loaded model + task session.
//
// Session IS-A IOfflineTaskSession. There is one session hierarchy, not two.
// Family implementations (Qwen3TtsSession, AceStepSession, MossSession, etc.)
// inherit from Session, which provides load() + family() + run_tts() +
// run_music() + compute_embedding(). The server holds IOfflineTaskSession*
// and calls through the virtual interface — no adapter layer.
//
// One Session per (model_path, load_options) pair. Owns the WeightLoader,
// the Backend, and whatever graph / KV cache / codec state the family
// needs. Multiple concurrent requests on the same Session serialize
// through the backend; the server creates multiple Sessions (one per
// configured model id) to get concurrency.

#ifndef AUDIOCORE_FRAMEWORK_CORE_SESSION_H
#define AUDIOCORE_FRAMEWORK_CORE_SESSION_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "audiocore/framework/core/backend.h"
#include "audiocore/framework/io/weight_loader.h"
#include "audiocore/framework/runtime/registry.h"  // IOfflineTaskSession

namespace audiocore {

struct LoadOptions {
    std::string language;          // MOSS / qwen3_tts language hint
    std::string voice_path;        // reference audio for cloning
    bool        keep_in_vram = true;
    // family-specific keys live in extras (parsed by the family loader).
    std::unordered_map<std::string, std::string> extras;
};

// Session is the concrete base for every family implementation. It inherits
// the task interface from IOfflineTaskSession so the server can hold
// IOfflineTaskSession* directly — no adapter, no forwarding, one hierarchy.
class Session : public IOfflineTaskSession {
public:
    // --- ITaskSession overrides ---
    // family() is the identity method; subclasses must implement it.
    // task_kind() defaults to Tts; families that do something else override.
    VoiceTaskKind task_kind() const override { return VoiceTaskKind::Tts; }

    // --- IOfflineTaskSession overrides (default implementations) ---
    // These mirror the old virtuals with default "not supported" behavior.
    // Families override the ones they implement.
    bool run_tts(const void* request, void* response,
                 std::string* error = nullptr) override { return false; }
    bool run_music(const void* request, void* response,
                   std::string* error = nullptr) override { return false; }
    using IOfflineTaskSession::compute_embedding;  // inherit default impl

    // --- Session-only methods (not part of IOfflineTaskSession) ---

    // Load weights via a WeightLoader chosen by file magic. The session
    // then asks the loader for the tensors it needs (by name) and builds
    // whatever family-specific graph state it requires.
    virtual bool load(const std::string& model_path,
                      const LoadOptions& opts,
                      const BackendConfig& backend_cfg,
                      std::string* error = nullptr) = 0;
    bool loaded() const { return loaded_; }

protected:
    // Primary weight loader (single-file families). The mmap backing any
    // tensor pointers the family code captured during load() stays alive
    // for the session's lifetime because loader_ holds the WeightLoader.
    std::unique_ptr<WeightLoader> loader_;
    // Multi-file families (e.g. ACE-Step has 4 GGUFs) keep their additional
    // readers here so the same lifetime guarantee applies to every file.
    std::vector<std::unique_ptr<WeightLoader>> extra_loaders_;
    std::unique_ptr<Backend>      backend_;
    bool loaded_ = false;
};

}  // namespace audiocore

#endif  // AUDIOCORE_FRAMEWORK_CORE_SESSION_H
