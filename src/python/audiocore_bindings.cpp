#include <memory>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "audiocore/framework/core/backend.h"
#include "audiocore/framework/core/session.h"
#include "audiocore/framework/runtime/registry.h"
#include "audiocore/framework/runtime/tasks.h"

namespace py = pybind11;
using namespace audiocore;

// ── Expilicit registration calls (defined in each loader.cpp via            ──
//    AUDIOCORE_EXTERN_C_GUARD). These guard against the linker dropping       ──
//    static initializers from the static archive.                            ──

extern "C" void audiocore_register_moss_tts();
extern "C" void audiocore_register_ace_step();
extern "C" void audiocore_register_qwen3_tts();

static void register_all_families() {
    audiocore_register_moss_tts();
    audiocore_register_ace_step();
    audiocore_register_qwen3_tts();
}

// ── PythonSession wrapper ──────────────────────────────────────────────────

class PythonSession {
public:
    PythonSession(std::unique_ptr<Session> session)
        : session_(std::move(session)) {}

    static PythonSession create(const std::string& family) {
        auto s = FamilyRegistry::instance().create(family);
        if (!s) {
            throw std::runtime_error("unknown family: " + family +
                                     " (did you call audiocore.init()?)");
        }
        return PythonSession(std::move(s));
    }

    void load(const std::string& model_path,
              const std::string& backend = "ggml_cuda",
              int device = 0,
              int threads = 4,
              const std::unordered_map<std::string, std::string>& extras = {}) {
        BackendKind bk;
        if (backend == "ggml_cuda")       bk = BackendKind::ggml_cuda;
        else if (backend == "ggml_cpu")   bk = BackendKind::ggml_cpu;
        else if (backend == "ggml_vulkan")bk = BackendKind::ggml_vulkan;
        else if (backend == "ggml_metal") bk = BackendKind::ggml_metal;
        else throw std::runtime_error("unknown backend: " + backend);

        BackendConfig bc;
        bc.kind       = bk;
        bc.device_id  = device;
        bc.n_threads  = threads;
        bc.cuda_graphs = true;

        LoadOptions opts;
        opts.extras = extras;
        std::string err;
        if (!session_->load(model_path, opts, bc, &err)) {
            throw std::runtime_error("load failed: " + err);
        }
    }

    std::pair<std::vector<float>, int32_t>
    run_tts(const std::string& text,
            const std::string& voice = "",
            float temperature = 0.8f,
            float top_p = 0.9f,
            float speed = 1.0f,
            int32_t seed = 0,
            const std::string& language = "",
            const std::string& mode = "tts",
            const std::string& reference_audio = "",
            const std::string& reference_text = "",
            const std::string& speaker_name = "",
            const std::string& instruct = "",
            float repetition_penalty = 1.05f,
            const std::vector<float>& speaker_embedding = {}) {
        TtsRequest req;
        req.text               = text;
        req.voice_path         = voice;
        req.temperature        = temperature;
        req.top_p              = top_p;
        req.speed              = speed;
        req.seed               = seed;
        req.language           = language;
        req.mode               = mode;
        req.reference_audio    = reference_audio;
        req.reference_text     = reference_text;
        req.speaker_name       = speaker_name;
        req.instruct           = instruct;
        req.repetition_penalty = repetition_penalty;
        // Pre-computed speaker embedding (Stage 17b). When non-empty,
        // bypasses reference_audio load + ECAPA — this is the "voice
        // caching" path (compute once, reuse across many calls).
        req.speaker_embedding  = speaker_embedding;

        TtsResponse resp;
        std::string err;
        if (!session_->run_tts(&req, &resp, &err)) {
            throw std::runtime_error("inference failed: " + err);
        }
        return {std::move(resp.pcm_mono), resp.sampling_rate};
    }

    std::string family_name() const {
        return session_->family_name();
    }

    // Compute a speaker embedding from a WAV file, for the voice-caching
    // pattern: compute once via this call, save the returned vector, then
    // pass it to run_tts(speaker_embedding=...) on every subsequent call.
    // Currently only qwen3_tts implements this (ECAPA-TDNN encoder).
    std::vector<float> compute_embedding(const std::string& wav_path) {
        std::string err;
        auto emb = session_->compute_embedding(wav_path, &err);
        if (emb.empty()) {
            throw std::runtime_error(
                "compute_embedding failed: " + err +
                " (only qwen3_tts with a loaded speaker_encoder GGUF "
                "supports this call)");
        }
        return emb;
    }

    bool loaded() const { return session_->loaded(); }

private:
    std::unique_ptr<Session> session_;
};

// ── Pybind11 module ────────────────────────────────────────────────────────

PYBIND11_MODULE(audiocore, m) {
    m.doc() = "audiocore — unified audio GGUF inference engine";

    // Module-level init (must call before creating sessions)
    m.def("init", &register_all_families,
          "Register all model families (moss_tts, ace_step, qwen3_tts). "
          "Required once before creating any session.");

    m.def("list_families", []() {
        return FamilyRegistry::instance().list();
    }, "List registered model family names.");

    // Session wrapper
    py::class_<PythonSession>(m, "Session")
        .def_static("create", &PythonSession::create,
                    py::arg("family"),
                    "Create a model session for the given family. "
                    "Call audiocore.init() first.")

        .def("load", &PythonSession::load,
             py::arg("model_path"),
             py::arg("backend") = "ggml_cuda",
             py::arg("device") = 0,
             py::arg("threads") = 4,
             py::arg("extras") = std::unordered_map<std::string, std::string>{},
             "Load model weights from the given path. "
             "`extras` is a family-specific string→string map (e.g. "
             "qwen3_tts accepts talker_path, predictor_path, codec_path, "
             "speaker_encoder_path).")

        .def("run_tts", &PythonSession::run_tts,
             py::arg("text"),
             py::arg("voice") = "",
             py::arg("temperature") = 0.8f,
             py::arg("top_p") = 0.9f,
             py::arg("speed") = 1.0f,
             py::arg("seed") = 0,
             py::arg("language") = "",
             py::arg("mode") = "tts",
             py::arg("reference_audio") = "",
             py::arg("reference_text") = "",
             py::arg("speaker_name") = "",
             py::arg("instruct") = "",
             py::arg("repetition_penalty") = 1.05f,
             py::arg("speaker_embedding") = std::vector<float>{},
             "Run TTS inference. Returns (pcm_f32, sample_rate). "
             "`speaker_embedding` (optional, Stage 17b): pre-computed "
             "ECAPA-TDNN vector — bypasses reference_audio load + ECAPA. "
             "Voice-caching pattern: compute once, reuse across many calls.")

        .def("family_name", &PythonSession::family_name,
             "Get the model family name.")

        .def("compute_embedding", &PythonSession::compute_embedding,
             py::arg("wav_path"),
             "Compute a speaker embedding from a WAV file (qwen3_tts only, "
             "requires speaker_encoder GGUF loaded). Returns a list[float] "
             "suitable for passing to run_tts(speaker_embedding=...). "
             "Voice-caching pattern: compute once, save the vector, reuse "
             "across thousands of run_tts calls without re-running ECAPA.")

        .def("loaded", &PythonSession::loaded,
             "Check if model weights are loaded.");
}
