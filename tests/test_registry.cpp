// test_registry.cpp — FamilyRegistry behavior.

#include "test_framework.h"

#include "audiocore/framework/core/session.h"
#include "audiocore/framework/runtime/registry.h"

#include <string>

using namespace audiocore;

namespace {

// A throwaway family so we can register/create without loading real weights.
class DummySession : public Session {
public:
    std::string family() const override { return "dummy"; }
    bool load(const std::string&, const LoadOptions&, const BackendConfig&,
              std::string* = nullptr) override { return true; }
};

std::unique_ptr<Session> make_dummy() {
    return std::unique_ptr<Session>(new DummySession());
}

}  // namespace

AUDIOCORE_TEST(register_then_create_returns_session) {
    auto& reg = FamilyRegistry::instance();
    reg.register_family("dummy_under_test", make_dummy);
    auto sess = reg.create("dummy_under_test");
    AUDIOCORE_CHECK(sess != nullptr);
    AUDIOCORE_CHECK_EQ(sess->family(), std::string("dummy"));
}

AUDIOCORE_TEST(create_unknown_family_returns_null) {
    auto& reg = FamilyRegistry::instance();
    auto sess = reg.create("no_such_family_xyz_12345");
    AUDIOCORE_CHECK(sess == nullptr);
}

AUDIOCORE_TEST(reregister_overwrites_factory) {
    // Same name registered twice — last factory wins (matches std::unordered_map
    // behavior of operator[]).
    auto& reg = FamilyRegistry::instance();
    reg.register_family("dummy_overwrite", []() -> std::unique_ptr<Session> {
        return std::unique_ptr<Session>(new DummySession());
    });
    class OtherSession : public Session {
    public:
        std::string family() const override { return "other"; }
        bool load(const std::string&, const LoadOptions&, const BackendConfig&,
                  std::string* = nullptr) override { return true; }
    };
    reg.register_family("dummy_overwrite", []() -> std::unique_ptr<Session> {
        return std::unique_ptr<Session>(new OtherSession());
    });
    auto sess = reg.create("dummy_overwrite");
    AUDIOCORE_CHECK(sess != nullptr);
    AUDIOCORE_CHECK_EQ(sess->family(), std::string("other"));
}

AUDIOCORE_TEST(list_contains_registered_family) {
    auto& reg = FamilyRegistry::instance();
    reg.register_family("dummy_listed", make_dummy);
    auto names = reg.list();
    bool found = false;
    for (const auto& n : names) if (n == "dummy_listed") found = true;
    AUDIOCORE_CHECK(found);
}

int main() {
    std::printf("=== FamilyRegistry tests ===\n");
    return test::run_all();
}
