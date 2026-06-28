// test_gguf_reader.cpp — round-trip tests for GgufReader against synthetic GGUFs.

#include "test_framework.h"
#include "synthetic_gguf.h"

#include "audiocore/framework/io/gguf_reader.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace audiocore;

namespace {

using test::tspec;
using test::kv_i32;
using test::kv_str;
using test::kv_bool;

bool f32_close(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) <= tol + tol * std::fabs(b);
}

}  // namespace

// Basic: KV round-trips with the type that was written.
AUDIOCORE_TEST(reads_back_i32_kv) {
    test::SyntheticGguf spec;
    spec.kvs = {kv_i32("moss.n_vq", 32), kv_i32("moss.sr", 24000)};
    auto path = test::write_synthetic_gguf("i32_kv", spec);

    GgufReader r;
    std::string err;
    AUDIOCORE_CHECK(r.load(path, &err));
    if (!err.empty()) std::fprintf(stderr, "load error: %s\n", err.c_str());
    int32_t n_vq = 0;
    AUDIOCORE_CHECK(r.get_kv_i32("moss.n_vq", &n_vq));
    AUDIOCORE_CHECK_EQ(n_vq, 32);
    int32_t sr = 0;
    AUDIOCORE_CHECK(r.get_kv_i32("moss.sr", &sr));
    AUDIOCORE_CHECK_EQ(sr, 24000);
    std::remove(path.c_str());
}

AUDIOCORE_TEST(reads_back_string_kv) {
    test::SyntheticGguf spec;
    spec.kvs = {kv_str("general.architecture", "qwen3")};
    auto path = test::write_synthetic_gguf("str_kv", spec);

    GgufReader r;
    AUDIOCORE_CHECK(r.load(path));
    std::string arch;
    AUDIOCORE_CHECK(r.get_kv_str("general.architecture", &arch));
    AUDIOCORE_CHECK_EQ(arch, std::string("qwen3"));
    std::remove(path.c_str());
}

AUDIOCORE_TEST(reads_back_bool_kv) {
    test::SyntheticGguf spec;
    spec.kvs = {kv_bool("moss.codec.present", true)};
    auto path = test::write_synthetic_gguf("bool_kv", spec);

    GgufReader r;
    AUDIOCORE_CHECK(r.load(path));
    bool present = false;
    AUDIOCORE_CHECK(r.get_kv_bool("moss.codec.present", &present));
    AUDIOCORE_CHECK(present);
    std::remove(path.c_str());
}

AUDIOCORE_TEST(missing_kv_returns_false) {
    auto path = test::write_synthetic_gguf("empty", test::SyntheticGguf{});
    GgufReader r;
    AUDIOCORE_CHECK(r.load(path));
    int32_t x = 999;
    AUDIOCORE_CHECK(!r.get_kv_i32("does.not.exist", &x));
    AUDIOCORE_CHECK_EQ(x, 999);  // untouched on miss
    std::remove(path.c_str());
}

AUDIOCORE_TEST(wrong_type_kv_returns_false) {
    test::SyntheticGguf spec;
    spec.kvs = {
        kv_str("as_int", "hello"),
        kv_i32("as_str", 42),
    };
    auto path = test::write_synthetic_gguf("wrong_type", spec);
    GgufReader r;
    AUDIOCORE_CHECK(r.load(path));
    int32_t i = 0;
    AUDIOCORE_CHECK(!r.get_kv_i32("as_int", &i));   // written as string
    std::string s;
    AUDIOCORE_CHECK(!r.get_kv_str("as_str", &s));   // written as int
    std::remove(path.c_str());
}

AUDIOCORE_TEST(reads_back_tensor_via_materialize) {
    // 2×3 tensor filled with 0,10,20,30,40,50 (row-major ne[0]=3, ne[1]=2).
    test::SyntheticGguf spec;
    spec.tensors = {tspec("weights", {3, 2}, 0.0f, 10.0f)};
    auto path = test::write_synthetic_gguf("tensor_f32", spec);

    GgufReader r;
    AUDIOCORE_CHECK(r.load(path));
    AUDIOCORE_CHECK_EQ(r.tensors().size(), static_cast<size_t>(1));
    const auto& t = r.tensors().front();
    AUDIOCORE_CHECK_EQ(t.name, std::string("weights"));
    AUDIOCORE_CHECK_EQ(t.ne[0], static_cast<int64_t>(3));
    AUDIOCORE_CHECK_EQ(t.ne[1], static_cast<int64_t>(2));
    std::vector<float> buf(6);
    std::string err;
    AUDIOCORE_CHECK(r.materialize(t, buf.data(), &err));
    for (int i = 0; i < 6; ++i) {
        AUDIOCORE_CHECK(f32_close(buf[i], static_cast<float>(i * 10)));
    }
    std::remove(path.c_str());
}

AUDIOCORE_TEST(tensor_data_ptr_aliases_mmap) {
    // tensor_data_ptr should return non-null (the mmap path is hit for
    // well-formed files via Path 1 / gguf_init_from_file) and alias the
    // same bytes materialize() copies.
    test::SyntheticGguf spec;
    spec.tensors = {tspec("t", {4}, 1.0f, 1.0f)};
    auto path = test::write_synthetic_gguf("mmap", spec);

    GgufReader r;
    AUDIOCORE_CHECK(r.load(path));
    const auto& t = r.tensors().front();
    const void* ptr = r.tensor_data_ptr(t);
    AUDIOCORE_CHECK(ptr != nullptr);
    const float* f = static_cast<const float*>(ptr);
    AUDIOCORE_CHECK(f32_close(f[0], 1.0f));
    AUDIOCORE_CHECK(f32_close(f[3], 4.0f));
    std::remove(path.c_str());
}

AUDIOCORE_TEST(detects_gguf_magic) {
    auto path = test::write_synthetic_gguf("magic", test::SyntheticGguf{});
    AUDIOCORE_CHECK(GgufReader::is_gguf_file(path));
    std::remove(path.c_str());
}

int main() {
    std::printf("=== GgufReader tests ===\n");
    return test::run_all();
}
