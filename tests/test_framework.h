// test_framework.h — minimal header-only test framework for audiocore.
//
// No external deps. Each test is a free function registered at static-init
// time; `run_all` invokes them in registration order and reports. Failures
// throw; the runner catches per-test so one failure doesn't hide the rest.
//
// Why custom instead of GoogleTest: audiocore builds against a vendored
// ggml/libllama subtree and we don't want to add a FetchContent dependency
// just for tests. The surface here is intentionally tiny: TEST, CHECK,
// CHECK_EQ, FAIL.

#ifndef AUDIOCORE_TESTS_TEST_FRAMEWORK_H
#define AUDIOCORE_TESTS_TEST_FRAMEWORK_H

#include <cstdio>
#include <exception>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace audiocore::test {

// Stringify a CHECK_EQ operand for the diagnostic message. Works for anything
// that streams to std::ostream (builtins via overloads, std::string, etc.).
template <typename T>
std::string to_dbg(const T& v) {
    std::ostringstream os;
    os << v;
    return os.str();
}

struct Failure : std::exception {
    std::string file;
    int         line;
    std::string expr;
    std::string msg;
    std::string full;

    Failure(std::string f, int l, std::string e, std::string m)
        : file(std::move(f)), line(l), expr(std::move(e)), msg(std::move(m)) {
        full = file + ":" + std::to_string(line) + ": CHECK failed";
        if (!expr.empty()) full += ": " + expr;
        if (!msg.empty())  full += " — " + msg;
    }
    const char* what() const noexcept override { return full.c_str(); }
};

using TestFn = void(*)();

struct Test {
    const char* name;
    TestFn      fn;
};

inline std::vector<Test>& registry() {
    static std::vector<Test> v;
    return v;
}

inline int register_test(const char* name, TestFn fn) {
    registry().push_back({name, fn});
    return 0;   // used only for static init
}

inline int run_all() {
    int failed = 0;
    int passed = 0;
    for (const Test& t : registry()) {
        try {
            t.fn();
            std::printf("  [PASS]  %s\n", t.name);
            ++passed;
        } catch (const Failure& f) {
            std::printf("  [FAIL]  %s\n", t.name);
            std::printf("          %s\n", f.what());
            ++failed;
        } catch (const std::exception& e) {
            std::printf("  [FAIL]  %s\n", t.name);
            std::printf("          unhandled exception: %s\n", e.what());
            ++failed;
        }
    }
    std::printf("\n  %d passed, %d failed\n", passed, failed);
    return failed ? 1 : 0;
}

}  // namespace audiocore::test

#define AUDIOCORE_TEST_(name, uniq)                                        \
    static void name();                                                    \
    namespace {                                                            \
    struct Registrar_##uniq {                                              \
        Registrar_##uniq() {                                               \
            ::audiocore::test::register_test(#name, &name);                \
        }                                                                  \
    };                                                                     \
    static Registrar_##uniq g_reg_##uniq;                                  \
    }                                                                      \
    static void name()

#define AUDIOCORE_TEST(name) AUDIOCORE_TEST_(name, name)

#define AUDIOCORE_CHECK(cond)                                              \
    do {                                                                   \
        if (!(cond)) {                                                     \
            throw ::audiocore::test::Failure(__FILE__, __LINE__, #cond, "");\
        }                                                                  \
    } while (0)

#define AUDIOCORE_CHECK_EQ(a, b)                                           \
    do {                                                                   \
        auto _a = (a);                                                     \
        auto _b = (b);                                                     \
        if (!(_a == _b)) {                                                 \
            throw ::audiocore::test::Failure(                              \
                __FILE__, __LINE__, #a " == " #b,                          \
                "got: " + ::audiocore::test::to_dbg(_a) +                  \
                ", want: " + ::audiocore::test::to_dbg(_b));               \
        }                                                                  \
    } while (0)

#define AUDIOCORE_FAIL(msg)                                                \
    throw ::audiocore::test::Failure(__FILE__, __LINE__, "", (msg))

#endif  // AUDIOCORE_TESTS_TEST_FRAMEWORK_H
