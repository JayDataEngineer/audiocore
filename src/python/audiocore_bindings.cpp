#include <memory>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "audiocore/framework/core/backend.h"
#include "audiocore/framework/core/session.h"
#include "audiocore/framework/runtime/registry.h"
#include "audiocore/framework/runtime/tasks.h"
#include "audiocore/models/ace_step/family.h"

namespace py = pybind11;
using namespace audiocore;

// ── Expilicit registration calls (defined in each loader.cpp via            ──
//    AUDIOCORE_EXTERN_C_GUARD). These guard against the linker dropping       ──
//    static initializers from the static archive.                            ──

extern "C" void audiocore_register_moss_tts();
extern "C" void audiocore_register_ace_step();
extern "C" void audiocore_register_qwen3_tts();
extern "C" void audiocore_register_moss_sfx_v2();

static void register_all_families() {
    audiocore_register_moss_tts();
    audiocore_register_ace_step();
    audiocore_register_qwen3_tts();
    audiocore_register_moss_sfx_v2();
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

    std::string family() const {
        return session_->family();
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

    // Run music generation (ACE-Step family). Returns stereo PCM at 48 kHz.
    std::tuple<std::vector<float>, int32_t, int32_t>
    run_music(const std::string& caption,
              const std::string& lyrics = "",
              float duration = 30.0f,
              int32_t seed = 0,
              float guidance_scale = 1.0f,
              int32_t n_diffusion_steps = 0,
              float temperature = 0.85f,
              float top_p = 0.9f,
              float lm_cfg_scale = 2.0f,
              const std::string& mode = "text_to_music",
              const std::string& vocal_language = "en",
              int32_t bpm = 0,
              const std::string& keyscale = "",
              const std::string& timesignature = "") {
        acestep::MusicRequest req;
        req.caption           = caption;
        req.lyrics            = lyrics;
        req.duration          = duration;
        req.seed              = seed;
        req.guidance_scale    = guidance_scale;
        req.n_diffusion_steps = n_diffusion_steps;
        req.temperature       = temperature;
        req.top_p             = top_p;
        req.lm_cfg_scale      = lm_cfg_scale;
        req.mode              = mode;
        req.vocal_language    = vocal_language;
        req.bpm               = bpm;
        req.keyscale          = keyscale;
        req.timesignature     = timesignature;

        acestep::MusicResponse resp;
        std::string err;
        if (!session_->run_music(&req, &resp, &err)) {
            throw std::runtime_error("music generation failed: " + err);
        }
        return {std::move(resp.pcm_stereo), resp.sampling_rate, resp.channels};
    }

    bool loaded() const { return session_ && session_->loaded(); }

    /// Explicitly destroy the session and free all VRAM.
    ///
    /// The GGML CUDA backend allocates model weights via cudaMalloc, which
    /// is NOT tracked by PyTorch's caching allocator.  Python's GC may not
    /// collect the pybind11 wrapper immediately after the last reference is
    /// dropped, leaving VRAM occupied.  This method forces immediate
    /// destruction of the C++ Session (which destroys the Backend, which
    /// calls ggml_backend_free → cudaFree).
    void close() {
        session_.reset();  // destroys Session → destroys Backend → frees VRAM
    }

private:
    std::unique_ptr<Session> session_;
};

// ── Pybind11 module ────────────────────────────────────────────────────────

PYBIND11_MODULE(audiocore, m) {
    m.doc() = "audiocore — unified audio GGUF inference engine";

    // Module-level init (must call before creating sessions)
    m.def("init", &register_all_families,
          "Register all model families (moss_tts, ace_step, qwen3_tts, "
          "moss_sfx_v2). Required once before creating any session.");

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

        .def("family", &PythonSession::family,
             "Get the model family name.")

        .def("compute_embedding", &PythonSession::compute_embedding,
             py::arg("wav_path"),
             "Compute a speaker embedding from a WAV file (qwen3_tts only, "
             "requires speaker_encoder GGUF loaded). Returns a list[float] "
             "suitable for passing to run_tts(speaker_embedding=...). "
             "Voice-caching pattern: compute once, save the vector, reuse "
             "across thousands of run_tts calls without re-running ECAPA.")

        .def("run_music", &PythonSession::run_music,
             py::arg("caption"),
             py::arg("lyrics") = "",
             py::arg("duration") = 30.0f,
             py::arg("seed") = 0,
             py::arg("guidance_scale") = 1.0f,
             py::arg("n_diffusion_steps") = 0,
             py::arg("temperature") = 0.85f,
             py::arg("top_p") = 0.9f,
             py::arg("lm_cfg_scale") = 2.0f,
             py::arg("mode") = "text_to_music",
             py::arg("vocal_language") = "en",
             py::arg("bpm") = 0,
             py::arg("keyscale") = "",
             py::arg("timesignature") = "",
             "Run music generation (ACE-Step). "
             "Returns (pcm_stereo_f32, sample_rate, channels). "
             "caption is the text prompt; lyrics optional. "
             "n_diffusion_steps=0 uses variant default (turbo=8, sft=50).")

        .def("loaded", &PythonSession::loaded,
             "Check if model weights are loaded.")

        .def("close", &PythonSession::close,
             "Destroy the session and free all VRAM. "
             "The session cannot be used after calling this.");
}
