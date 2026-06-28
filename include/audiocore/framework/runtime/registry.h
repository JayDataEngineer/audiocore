// registry.h — model family registry.
//
// One place where a new model family is declared. Adding MOSS or ACE-Step
// = one REGISTER_FAMILY call in their loader.cpp. The server / CLI / future
// pool dispatcher all look up families through here.

#ifndef AUDIOCORE_FRAMEWORK_RUNTIME_REGISTRY_H
#define AUDIOCORE_FRAMEWORK_RUNTIME_REGISTRY_H

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace audiocore {

class Session;

// A FamilyFactory builds a fresh, unloaded Session for a given family.
// The Session will then be handed a WeightLoader + Backend and asked to
// load + run.
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

// Static-registration helper. Place one of these in each family's
// loader.cpp so the family is registered at static-init time. Pattern is
// identical to llama.cpp's llama_model_loader register calls.
struct FamilyRegistrar {
    FamilyRegistrar(const std::string& name, FamilyFactory factory) {
        FamilyRegistry::instance().register_family(name, std::move(factory));
    }
};

#define AUDIOCORE_REGISTER_FAMILY(name, factory)               \
    namespace {                                                 \
    ::audiocore::FamilyRegistrar                                \
        g_register_##name(#name, factory);                     \
    }

// Registration anchor that doubles as a linkage guard. Each family's loader.cpp
// calls this after AUDIOCORE_REGISTER_FAMILY so main.cpp can force the
// object file to be linked (static archives drop unreferenced TUs).
#define AUDIOCORE_EXTERN_C_GUARD(name, factory)                               \
    extern "C" void audiocore_register_##name() {                             \
        static bool done_##name = false;                                      \
        if (!done_##name) {                                                   \
            ::audiocore::FamilyRegistry::instance().register_family(           \
                #name, factory);                                              \
            done_##name = true;                                               \
        }                                                                     \
    }

}  // namespace audiocore

#endif  // AUDIOCORE_FRAMEWORK_RUNTIME_REGISTRY_H
