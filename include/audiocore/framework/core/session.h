// session.h — loaded model + task session.
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

#include "audiocore/framework/core/backend.h"
#include "audiocore/framework/io/weight_loader.h"

namespace audiocore {

struct LoadOptions {
    std::string language;          // MOSS / qwen3_tts language hint
    std::string voice_path;        // reference audio for cloning
    bool        keep_in_vram = true;
    // family-specific keys live in extras (parsed by the family loader).
    std::unordered_map<std::string, std::string> extras;
};

class Session {
public:
    virtual ~Session() = default;

    // family_name is what FamilyRegistry used to construct this instance.
    virtual std::string family_name() const = 0;

    // Load weights via a WeightLoader chosen by file magic. The session
    // then asks the loader for the tensors it needs (by name) and builds
    // whatever family-specific graph state it requires.
    virtual bool load(const std::string& model_path,
                      const LoadOptions& opts,
                      const BackendConfig& backend_cfg,
                      std::string* error = nullptr) = 0;

    // Task entry points. Not every family implements every task — MOSS
    // implements tts/clon/sfx, ACE-Step implements music generation. The
    // server dispatches by task name → method.
    //
    // Concrete I/O structs (TtsRequest, MusicRequest, …) are defined in
    // headers under include/audiocore/framework/runtime/tasks.h (to be
    // added when the first family lands).
    virtual bool run_tts(const void* request, void* response,
                         std::string* error = nullptr)   { return false; }
    virtual bool run_music(const void* request, void* response,
                           std::string* error = nullptr) { return false; }
    virtual bool run_asr(const void* request, void* response,
                         std::string* error = nullptr)   { return false; }

    bool loaded() const { return loaded_; }

protected:
    std::unique_ptr<WeightLoader> loader_;
    std::unique_ptr<Backend>      backend_;
    bool loaded_ = false;
};

}  // namespace audiocore

#endif  // AUDIOCORE_FRAMEWORK_CORE_SESSION_H
