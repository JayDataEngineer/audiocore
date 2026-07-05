#include "audiocore/framework/runtime/registry.h"

#include <algorithm>
#include <stdexcept>

#include "audiocore/framework/core/session.h"
#include "audiocore/framework/runtime/discovery.h"

#include "ggml.h"
#include "ggml-backend.h"

namespace audiocore {

// ── Old FamilyRegistry implementation (backward compat) ───────────────────

FamilyRegistry& FamilyRegistry::instance() {
    static FamilyRegistry r;
    return r;
}

void FamilyRegistry::register_family(const std::string& name, FamilyFactory factory) {
    families_[name] = std::move(factory);
}

std::unique_ptr<Session> FamilyRegistry::create(const std::string& family) const {
    auto it = families_.find(family);
    if (it != families_.end())
        return it->second();
    return nullptr;
}

std::vector<std::string> FamilyRegistry::list() const {
    std::vector<std::string> names;
    names.reserve(families_.size());
    for (const auto& [name, _] : families_)
        names.push_back(name);
    std::sort(names.begin(), names.end());
    return names;
}

FamilyRegistrar::FamilyRegistrar(const std::string& name, FamilyFactory factory) {
    FamilyRegistry::instance().register_family(name, std::move(factory));
}

// ── Adapter: wrap an existing family's Session into the new interfaces ─────

namespace {

class OfflineSessionAdapter final : public IOfflineTaskSession {
public:
    OfflineSessionAdapter(Session* session, std::string family)
        : session_(session), family_(std::move(family)) {}

    std::string family() const override { return family_; }
    VoiceTaskKind task_kind() const override { return VoiceTaskKind::Tts; }

    bool run_tts(const void* request, void* response,
                 std::string* error) override {
        return session_->run_tts(request, response, error);
    }

    bool run_music(const void* request, void* response,
                   std::string* error) override {
        return session_->run_music(request, response, error);
    }

private:
    Session* session_;
    std::string family_;
};

class LoadedModelAdapter final : public ILoadedModel {
public:
    LoadedModelAdapter(
        std::unique_ptr<Session> session,
        ModelMetadata metadata,
        CapabilitySet capabilities)
        : session_(std::move(session))
        , metadata_(std::move(metadata))
        , capabilities_(std::move(capabilities)) {}

    const ModelMetadata& metadata() const noexcept override { return metadata_; }
    const CapabilitySet& capabilities() const noexcept override { return capabilities_; }

    std::unique_ptr<IOfflineTaskSession> create_session(
        const TaskSpec& task,
        const SessionOptions& options) const override;

private:
    std::unique_ptr<Session> session_;
    ModelMetadata metadata_;
    CapabilitySet capabilities_;
};

// Adapter loader that wraps the old FamilyRegistry-based creation path.
class FamilyRegistryLoader final : public IModelLoader {
public:
    explicit FamilyRegistryLoader(std::string family)
        : family_(std::move(family)) {}

    std::string family() const override { return family_; }

    bool can_load(const ModelLoadRequest& request) const override {
        // Accept if the path exists and the family hint matches,
        // or if no hint is given and the path is a directory with our name.
        if (!request.family_hint.empty() && request.family_hint != family_)
            return false;
        if (!std::filesystem::exists(request.model_path))
            return false;
        return true;
    }

    ModelInspection inspect(const ModelLoadRequest& request) const override {
        ModelInspection ins;
        ins.metadata.family = family_;
        ins.metadata.variant = "gguf";
        ins.model_path = request.model_path;

        // Populate capabilities from the family metadata.
        if (family_ == "moss_tts") {
            ins.capabilities.tasks = {VoiceTaskKind::Tts, VoiceTaskKind::VoiceCloning};
            ins.capabilities.languages = {"en", "zh", "ja", "ko", "fr", "de",
                                          "es", "pt", "it", "pl", "ru", "ar"};
            ins.capabilities.supports_voice_cloning = true;
        } else if (family_ == "ace_step") {
            ins.capabilities.tasks = {VoiceTaskKind::MusicGen};
            ins.capabilities.languages = {"en", "zh", "ja", "ko", "es"};
        } else if (family_ == "qwen3_tts") {
            ins.capabilities.tasks = {VoiceTaskKind::Tts, VoiceTaskKind::VoiceCloning};
            ins.capabilities.languages = {"en", "zh", "ja", "ko", "fr", "de",
                                          "it", "pt", "ru", "es"};
            ins.capabilities.supports_voice_cloning = true;
        }
        return ins;
    }

    std::unique_ptr<ILoadedModel> load(const ModelLoadRequest& request) const override {
        // Find the model path by scanning the directory for .gguf files.
        std::filesystem::path resolved = request.model_path;
        if (std::filesystem::is_directory(resolved)) {
            // Auto-discover the first .gguf in the directory.
            for (auto& entry : std::filesystem::directory_iterator(resolved)) {
                if (entry.path().extension() == ".gguf") {
                    resolved = entry.path();
                    break;
                }
            }
        }

        auto session = FamilyRegistry::instance().create(family_);
        if (!session)
            throw std::runtime_error("unknown family: " + family_);

        // Build family-specific load options.
        LoadOptions opts;
        opts.extras = request.options;
        if (auto it = request.options.find("voice_path"); it != request.options.end())
            opts.voice_path = it->second;
        if (auto it = request.options.find("language"); it != request.options.end())
            opts.language = it->second;

        BackendConfig bc;
        bc.kind = BackendKind::ggml_cuda;
        if (auto it = request.options.find("backend"); it != request.options.end()) {
            if (it->second == "ggml_cpu")    bc.kind = BackendKind::ggml_cpu;
            else if (it->second == "ggml_vulkan") bc.kind = BackendKind::ggml_vulkan;
            else if (it->second == "ggml_metal")  bc.kind = BackendKind::ggml_metal;
            else bc.kind = BackendKind::ggml_cuda;
        }
        bc.device_id = 0;
        if (auto it = request.options.find("device"); it != request.options.end())
            bc.device_id = std::stoi(it->second);

        // GPU availability probe — mirror qwen_voice's load_session() logic.
        // The make_backend() call later does ggml_backend_dev_by_type(GPU)
        // → dev_init(); on some builds that silently returns null if the
        // ggml backend registry hasn't been touched yet (the talker load
        // via llama.cpp uses a separate path). Iterating dev_count/dev_get
        // here forces registration, and we fall back to CPU if no GPU is
        // actually present so the codec + speaker encoder still bind.
        if (bc.kind == BackendKind::ggml_cuda ||
            bc.kind == BackendKind::ggml_vulkan) {
            bool gpu_ok = false;
            int n_devs = ggml_backend_dev_count();
            for (int i = 0; i < n_devs; i++) {
                ggml_backend_dev_t d = ggml_backend_dev_get(i);
                if (d && ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                    gpu_ok = true;
                    break;
                }
            }
            if (!gpu_ok) {
                bc.kind = BackendKind::ggml_cpu;
                opts.extras["n_gpu_layers"] = "0";
            }
        }

        std::string err;
        if (!session->load(resolved.string(), opts, bc, &err))
            throw std::runtime_error("load failed: " + err);

        // Determine variant from the filename.
        std::string variant = "gguf";
        auto stem = resolved.stem().string();
        if (stem.find("Q8_0") != std::string::npos) variant = "Q8_0";
        else if (stem.find("Q4_K") != std::string::npos) variant = "Q4_K";
        else if (stem.find("F16") != std::string::npos) variant = "F16";

        ModelMetadata meta;
        meta.family = family_;
        meta.variant = variant;
        meta.description = family_ + " (GGUF)";

        // Infer capabilities from inspect().
        auto ins = inspect(request);

        return std::make_unique<LoadedModelAdapter>(
            std::move(session), std::move(meta), std::move(ins.capabilities));
    }

private:
    std::string family_;
};

}  // namespace

// Out-of-line definition now that both LoadedModelAdapter and
// OfflineSessionAdapter are complete types.
std::unique_ptr<IOfflineTaskSession> LoadedModelAdapter::create_session(
    const TaskSpec& task,
    const SessionOptions& options) const {
    (void)task;
    (void)options;
    return std::make_unique<OfflineSessionAdapter>(session_.get(), metadata_.family);
}

// ── ModelRegistry implementation ──────────────────────────────────────────

void ModelRegistry::register_loader(std::shared_ptr<IModelLoader> loader) {
    if (!loader)
        throw std::invalid_argument("loader must not be null");
    loaders_.push_back(std::move(loader));
}

bool ModelRegistry::empty() const noexcept { return loaders_.empty(); }
size_t ModelRegistry::size() const noexcept { return loaders_.size(); }

std::vector<std::string> ModelRegistry::families() const {
    std::vector<std::string> names;
    names.reserve(loaders_.size());
    for (auto& l : loaders_)
        names.push_back(l->family());
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());
    return names;
}

bool ModelRegistry::supports_family(const std::string& family) const noexcept {
    for (auto& l : loaders_)
        if (l->family() == family) return true;
    return false;
}

ModelInspection ModelRegistry::inspect(const ModelLoadRequest& request) const {
    auto* loader = find_loader(request);
    if (!loader)
        throw std::runtime_error("no loader for: " + request.model_path.string());
    return loader->inspect(request);
}

std::unique_ptr<ILoadedModel> ModelRegistry::load(const ModelLoadRequest& request) const {
    auto* loader = find_loader(request);
    if (!loader)
        throw std::runtime_error("no loader for: " + request.model_path.string());
    return loader->load(request);
}

std::unique_ptr<ILoadedModel> ModelRegistry::load(const std::filesystem::path& path) const {
    ModelLoadRequest req;
    req.model_path = path;
    return load(req);
}

const IModelLoader* ModelRegistry::find_loader(const ModelLoadRequest& request) const {
    for (auto& loader : loaders_) {
        if (!request.family_hint.empty() && loader->family() != request.family_hint)
            continue;
        if (loader->can_load(request))
            return loader.get();
    }
    return nullptr;
}

// ── Default registry builder ──────────────────────────────────────────────

ModelRegistry make_default_registry() {
    ModelRegistry reg;

    // Each compiled-in family gets a FamilyRegistryLoader adapter.
    // This uses the existing FamilyRegistry underneath, so existing
    // AUDIOCORE_REGISTER_FAMILY calls in loader.cpp still work.
    for (auto& f : FamilyRegistry::instance().list()) {
        reg.register_loader(std::make_shared<FamilyRegistryLoader>(f));
    }

    return reg;
}

}  // namespace audiocore
