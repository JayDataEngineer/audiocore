// registry.cpp — family registry singleton.
//
// Families register themselves at static-init time via
// AUDIOCORE_REGISTER_FAMILY (see registry.h). The server / CLI looks them
// up by name through FamilyRegistry::create().

#include "audiocore/framework/runtime/registry.h"

#include "audiocore/framework/core/session.h"   // complete type for unique_ptr<Session> destruction

#include <algorithm>

namespace audiocore {

FamilyRegistry& FamilyRegistry::instance() {
    static FamilyRegistry r;
    return r;
}

void FamilyRegistry::register_family(const std::string& name,
                                     FamilyFactory factory) {
    families_[name] = std::move(factory);
}

std::unique_ptr<Session> FamilyRegistry::create(const std::string& family) const {
    auto it = families_.find(family);
    if (it == families_.end()) return nullptr;
    return it->second();
}

std::vector<std::string> FamilyRegistry::list() const {
    std::vector<std::string> out;
    out.reserve(families_.size());
    for (const auto& [name, _] : families_) out.push_back(name);
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace audiocore
