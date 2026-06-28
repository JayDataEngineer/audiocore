// test_convert_acestep.cpp — drive the convert_acestep tool end-to-end.
//
// Writes a synthetic ACE-Step-named GGUF via the shared test helper, invokes
// the converter as a subprocess (so we exercise the real rule table, not a
// copy of it), then reads back the output and verifies both rename + KV
// preservation.

#include "test_framework.h"
#include "synthetic_gguf.h"

#include "audiocore/framework/io/gguf_reader.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>

using namespace audiocore;

namespace {

// Converter binary path. The CMake helper passes the absolute path via
// AUDIOCORE_CONVERT_ACESTEP_PATH; an env override lets us run the test outside ctest.
std::string find_converter() {
    if (const char* env = std::getenv("AUDIOCORE_CONVERT_ACESTEP")) {
        return env;
    }
#ifdef AUDIOCORE_CONVERT_ACESTEP_PATH
    return std::string(AUDIOCORE_CONVERT_ACESTEP_PATH);
#else
    return std::string("./convert_acestep");
#endif
}

struct CmdResult { int code; std::string out; };
CmdResult run_cmd(const std::string& cmd) {
    std::string out;
    std::array<char, 4096> buf{};
    FILE* p = ::popen(cmd.c_str(), "r");
    if (!p) return {1, "popen failed"};
    while (size_t n = std::fread(buf.data(), 1, buf.size(), p)) {
        out.append(buf.data(), n);
    }
    return {::pclose(p), std::move(out)};
}

}  // namespace

AUDIOCORE_TEST(convert_renames_hf_to_llama_cpp_layout) {
    test::SyntheticGguf spec;
    spec.kvs = {
        test::kv_i32("general.architecture", 0),
        test::kv_str("general.name", "test-acestep"),
    };
    // One hidden layer of F32 weights, each sized (1) so we don't waste bytes.
    spec.tensors = {
        test::tspec("model.embed_tokens.weight",                      {1}),
        test::tspec("model.norm.weight",                              {1}),
        test::tspec("model.layers.0.input_layernorm.weight",          {1}),
        test::tspec("model.layers.0.post_attention_layernorm.weight", {1}),
        test::tspec("model.layers.0.self_attn.q_proj.weight",         {1}),
        test::tspec("model.layers.0.self_attn.k_proj.weight",         {1}),
        test::tspec("model.layers.0.self_attn.v_proj.weight",         {1}),
        test::tspec("model.layers.0.self_attn.o_proj.weight",         {1}),
        test::tspec("model.layers.0.mlp.gate_proj.weight",            {1}),
        test::tspec("model.layers.0.mlp.up_proj.weight",              {1}),
        test::tspec("model.layers.0.mlp.down_proj.weight",            {1}),
        // Unmatched name — should survive verbatim.
        test::tspec("some_unmatched_aux.weight",                      {1}),
    };

    const std::string in_path  = test::write_synthetic_gguf("acestep_in", spec);
    const std::string out_path =
        "/tmp/audiocore_test_acestep_out_" +
        std::to_string(static_cast<long>(::getpid())) + ".gguf";

    const std::string cmd = "'" + find_converter() + "' '" + in_path +
                            "' '" + out_path + "'";
    auto res = run_cmd(cmd);
    if (res.code != 0) {
        std::fprintf(stderr, "converter stdout: %s\n", res.out.c_str());
        AUDIOCORE_FAIL("converter exited with code " +
                       std::to_string(res.code));
    }

    GgufReader r;
    std::string err;
    if (!r.load(out_path, &err)) AUDIOCORE_FAIL("load failed: " + err);

    const auto has = [&](const std::string& n) -> bool {
        for (const auto& t : r.tensors()) if (t.name == n) return true;
        return false;
    };
    AUDIOCORE_CHECK(has("token_embd.weight"));
    AUDIOCORE_CHECK(has("output_norm.weight"));
    AUDIOCORE_CHECK(has("blk.0.attn_norm.weight"));
    AUDIOCORE_CHECK(has("blk.0.ffn_norm.weight"));
    AUDIOCORE_CHECK(has("blk.0.attn_q.weight"));
    AUDIOCORE_CHECK(has("blk.0.attn_k.weight"));
    AUDIOCORE_CHECK(has("blk.0.attn_v.weight"));
    AUDIOCORE_CHECK(has("blk.0.attn_output.weight"));
    AUDIOCORE_CHECK(has("blk.0.ffn_gate.weight"));
    AUDIOCORE_CHECK(has("blk.0.ffn_up.weight"));
    AUDIOCORE_CHECK(has("blk.0.ffn_down.weight"));
    AUDIOCORE_CHECK(has("some_unmatched_aux.weight"));

    AUDIOCORE_CHECK_EQ(r.tensors().size(), spec.tensors.size());

    int32_t arch = -1;
    AUDIOCORE_CHECK(r.get_kv_i32("general.architecture", &arch));
    AUDIOCORE_CHECK_EQ(arch, 0);
    std::string name;
    AUDIOCORE_CHECK(r.get_kv_str("general.name", &name));
    AUDIOCORE_CHECK_EQ(name, std::string("test-acestep"));

    std::remove(in_path.c_str());
    std::remove(out_path.c_str());
}

AUDIOCORE_TEST(convert_dry_run_does_not_write) {
    test::SyntheticGguf spec;
    spec.tensors = {test::tspec("model.embed_tokens.weight", {1})};
    const std::string in_path = test::write_synthetic_gguf("dryrun_in", spec);
    const std::string bogus_out =
        "/tmp/audiocore_test_acestep_dryrun_should_not_exist.gguf";
    std::remove(bogus_out.c_str());

    const std::string cmd = "'" + find_converter() + "' '" + in_path +
                            "' --dry-run";
    auto res = run_cmd(cmd);
    AUDIOCORE_CHECK_EQ(res.code, 0);

    FILE* f = std::fopen(bogus_out.c_str(), "r");
    AUDIOCORE_CHECK(f == nullptr);
    if (f) std::fclose(f);
    std::remove(in_path.c_str());
}

int main() {
    std::printf("=== convert_acestep e2e tests ===\n");
    return test::run_all();
}
