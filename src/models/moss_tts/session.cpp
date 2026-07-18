// session.cpp — MOSS-TTS end-to-end generation pipeline (upstream-honest port).
//
// Prompts and the codec-encode path are verbatim ports of
// pwilkin/openmoss (Apache-2.0) src/pipeline.cpp + src/codec.cpp::encode().
// Earlier revisions of this file impersonated SFX / VoiceDesign / Dialogue /
// Realtime modes by routing requests through the flagship backbone with a
// system-prompt disguise (silently producing off-distribution output), and
// implemented voice cloning via a `.codes` binary loader. That surface has
// been removed: unsupported modes now fail fast, voice cloning now reads a
// WAV and runs the real codec encoder, and plain TTS now builds the upstream
// `<user_inst>` template the model was trained on. See GAPS.md §1 for the
// per-mode support matrix.
//
// Pipeline:
//   1. Build the upstream chat prompt:
//        <|im_start|>user\n<user_inst>\n- Reference(s):\n[S1]:\n
//          <audio_start><|audio_user_slot|>×T_ref<|audio_user_slot|>×(n_vq-1)
//          <audio_end>\n- Instruction:\n…\n- Tokens:\n…\n- Quality:\n…
//          \n- Sound Event:\nNone\n- Ambient Sound:\nNone\n
//          \n- Language:\n…\n- Text:\n<text>\n</user_inst><|im_end|>
//          \n<|im_start|>assistant\n<audio_start>
//      — or the same body without the Reference(s)/[S1]: block when no
//      reference audio is supplied.
//   2. Tokenize the prompt and build the (S, 1+n_vq) grid of int32 tokens:
//      text ids in column 0, AUDIO_PAD_CODE elsewhere; the delay-pattern-
//      shifted reference codes are spliced into the rows that fall between
//      the <audio_start>/<audio_end> markers inside the user block.
//   3. Compute summed text+audio input embeddings and prefill the backbone.
//   4. Autoregressive loop:
//      a. Pull text logits + hidden state from backbone
//      b. Project through audio_head.{i}.weight → per-stream codec logits
//      c. delay_step → sample next (1+n_vq) token vector
//      d. Embed the token vector (text_embed + Σ audio_embed[i])
//      e. Forward single embedding through backbone (incremental KV)
//   5. Apply de-delay pattern to audio channels.
//   6. Decode codec tokens → PCM via MossCodecGraphs (the same ggml port
//      openmoss uses). Missing codec tensors is a hard error.
//   7. Write PCM into response.
//
// Per-frame streaming (mode="streaming" with a non-null req->stream callback):
// the AR loop decodes newly available codec frames incrementally and emits
// PCM through the callback. The first frame becomes available after n_vq
// delay steps (~1.3 s at 80 ms/frame) — this is the cold-start of the Delay
// architecture. It is NOT the upstream MossTTSRealtime architecture, which
// we do not ship (see GAPS.md §1.1).

#include "audiocore/models/moss_tts/family.h"
#include "audiocore/models/moss_tts/projection.h"
#include "audiocore/models/moss_tts/delay_state.h"
#include "audiocore/framework/io/wav.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml.h"
#include "llama.h"

namespace audiocore::moss {

namespace {

// ── Upstream prompt builder (verbatim port of openmoss/src/pipeline.cpp) ─────

// Convert a token id to its string form. Mirrors the Python helper.
std::string id_to_token(qwen3::Runner* backbone, int32_t id) {
    std::string piece;
    std::string err;
    if (backbone->token_to_piece(id, &piece, &err) && !piece.empty()) {
        return piece;
    }
    // token_to_piece returns empty for SPECIAL/control tokens (llama.cpp
    // convention: special tokens have no textual piece). The MOSS prompt
    // is built as a STRING that we re-tokenize, so we MUST substitute the
    // actual special-token literal here — otherwise the prompt structure
    // markers (`<|im_start|>`, `<|audio_start|>`, etc.) get dropped and
    // the model sees a structureless blob of user text. With no valid
    // dialogue framing the AR loop degenerates within ~5 steps and emits
    // 30 seconds of mechanical buzz instead of speech.
    //
    // The literals below are the actual strings stored at these IDs in
    // the MOSS-TTS GGUF (tokenizer.ggml.tokens array, verified by reading
    // the array directly — llama_cpp.tokenize(special=True) confirms each
    // literal round-trips to exactly one token of the expected id).
    // MOSS defines its own special-token names; do NOT assume the upstream
    // Qwen2.5 tokenizer layout (e.g. id 151654 is `<|audio_user_slot|>`
    // here, not `<|audio_pad|>`).
    switch (id) {
        case 151643: return "<|endoftext|>";
        case 151644: return "<|im_start|>";
        case 151645: return "<|im_end|>";
        case 151652: return "<|audio_start|>";
        case 151653: return "<|audio_end|>";
        case 151654: return "<|audio_user_slot|>";
        case 151656: return "<|audio_assistant_gen_slot|>";
        case 151662: return "<|audio_assistant_delay_slot|>";
    }
    return std::string("<UNKNOWN_TOKEN_") + std::to_string(id) + ">";
}

std::string default_or_none(const std::optional<std::string>& s) {
    return s ? *s : "None";
}

std::string default_or_none(const std::optional<int>& v) {
    return v ? std::to_string(*v) : "None";
}

// Build the literal token-string form of the reference-audio block:
//   <audio_start><user_slot>…<user_slot>…<audio_end>
// — exactly what `_replace_audio_placeholders` produces upstream when the
// reference is for a `user` role. The block has length T_ref + n_vq + 1.
std::string build_reference_audio_block(qwen3::Runner* backbone,
                                         const MossConfig& d,
                                         int32_t T_ref) {
    const std::string audio_start = id_to_token(backbone, d.tok_audio_start);
    const std::string audio_end   = id_to_token(backbone, d.tok_audio_end);
    const std::string user_slot   = id_to_token(backbone, d.tok_user_slot);
    std::string s;
    s += audio_start;
    for (int t = 0; t < T_ref; ++t) s += user_slot;
    for (int i = 0; i < d.n_vq - 1; ++i) s += user_slot;
    s += audio_end;
    return s;
}

// Build the user-instruction body that wraps the synthesis target.
//   reference_block: literal token string for the encoded reference audio,
//                    or empty when no reference is supplied.
std::string build_user_inst(const std::optional<std::string>& instruction,
                             const std::optional<int>&        tokens,
                             const std::optional<std::string>& quality,
                             const std::optional<std::string>& language,
                             const std::string& text,
                             const std::string& reference_block) {
    std::string s;
    s += "<user_inst>\n";
    s += "- Reference(s):\n";
    if (reference_block.empty()) s += "None\n";
    else                          s += "[S1]:\n" + reference_block + "\n";
    s += "- Instruction:\n" + default_or_none(instruction) + "\n";
    s += "- Tokens:\n"      + default_or_none(tokens)      + "\n";
    s += "- Quality:\n"     + default_or_none(quality)     + "\n";
    s += "- Sound Event:\nNone\n";
    s += "- Ambient Sound:\nNone\n";
    s += "- Language:\n"    + default_or_none(language)    + "\n";
    s += "- Text:\n"        + text                         + "\n";
    s += "</user_inst>";
    return s;
}

// Build the full assistant-prompt string:
//   <im_start>user\n…<im_end>\n<im_start>assistant\n<audio_start>
std::string build_prompt_text(qwen3::Runner* backbone,
                               const MossConfig& d,
                               const std::optional<std::string>& instruction,
                               const std::optional<int>&        tokens,
                               const std::optional<std::string>& quality,
                               const std::optional<std::string>& language,
                               const std::string& text,
                               const std::string& reference_block) {
    const std::string im_start    = id_to_token(backbone, d.tok_im_start);
    const std::string im_end      = id_to_token(backbone, d.tok_im_end);
    const std::string audio_start = id_to_token(backbone, d.tok_audio_start);

    std::string body = build_user_inst(instruction, tokens, quality, language,
                                        text, reference_block);
    std::string out;
    out += im_start + "user\n" + body + im_end + "\n"
         + im_start + "assistant\n" + audio_start;
    return out;
}

// Apply the delay-pattern shift to a (n_vq, T_ref) row-major code matrix.
//   Output: (T_ref + n_vq - 1, n_vq) row-major, where row r col i =
//     codes[i, r - i]   if 0 <= r - i < T_ref
//     pad_code          otherwise
// — equivalent to MossTTSDelayProcessor.apply_delay_pattern.
std::vector<int32_t> apply_delay_pattern(const int32_t* codes,
                                          int32_t n_vq, int32_t T_ref,
                                          int32_t pad_code) {
    const int64_t T = int64_t(T_ref) + int64_t(n_vq) - 1;
    std::vector<int32_t> out(size_t(T) * size_t(n_vq), pad_code);
    for (int32_t i = 0; i < n_vq; ++i) {
        for (int32_t t = 0; t < T_ref; ++t) {
            out[size_t(i + t) * size_t(n_vq) + size_t(i)] = codes[i * T_ref + t];
        }
    }
    return out;
}

// Build the (S, 1+n_vq) prompt grid: text ids in column 0, audio_pad_code
// elsewhere — except for the reference-audio rows where we splice in the
// delay-pattern-shifted codes from the encoded reference.
std::vector<int32_t> build_prompt_grid(qwen3::Runner* backbone,
                                        const MossConfig& d,
                                        const std::optional<std::string>& instruction,
                                        const std::optional<int>&        tokens,
                                        const std::optional<std::string>& quality,
                                        const std::optional<std::string>& language,
                                        const std::string& text,
                                        const std::string& reference_block,
                                        const std::vector<int32_t>* ref_codes,
                                        int32_t T_ref,
                                        int32_t& n_pos_out) {
    const std::string prompt = build_prompt_text(backbone, d,
                                                  instruction, tokens, quality,
                                                  language, text, reference_block);
    std::vector<int32_t> ids;
    std::string tok_err;
    if (!backbone->tokenize(prompt, /*add_special=*/false,
                             /*parse_special=*/true, &ids, nullptr, &tok_err)
        || ids.empty()) {
        throw std::runtime_error("build_prompt_grid: tokenize failed: " + tok_err);
    }
    n_pos_out = int32_t(ids.size());
    const int32_t cols = 1 + d.n_vq;

    std::vector<int32_t> grid(size_t(n_pos_out) * size_t(cols), d.audio_pad_code);
    for (int32_t r = 0; r < n_pos_out; ++r) {
        grid[size_t(r) * size_t(cols) + 0] = ids[size_t(r)];
    }

    if (!ref_codes || T_ref <= 0) return grid;

    // Locate the (single) audio_start / audio_end pair that bounds the user
    // reference. The trailing audio_start the assistant turn ends with does
    // NOT have a matching audio_end and is therefore skipped naturally.
    int32_t a_start = -1, a_end = -1;
    for (int32_t r = 0; r < n_pos_out; ++r) {
        if (a_start < 0 && ids[size_t(r)] == d.tok_audio_start) {
            a_start = r;
        } else if (a_start >= 0 && ids[size_t(r)] == d.tok_audio_end) {
            a_end = r;
            break;
        }
    }
    if (a_start < 0 || a_end < 0) {
        throw std::runtime_error("build_prompt_grid: reference audio markers not found in tokenized prompt");
    }

    const int32_t span = a_end - a_start - 1;        // tokens strictly between markers
    const int32_t expected = T_ref + d.n_vq - 1;
    if (span != expected) {
        throw std::runtime_error("build_prompt_grid: reference audio span mismatch (got " +
                                  std::to_string(span) + ", expected " + std::to_string(expected) + ")");
    }

    const auto delayed = apply_delay_pattern(ref_codes->data(), d.n_vq, T_ref, d.audio_pad_code);
    for (int32_t k = 0; k < span; ++k) {
        const int32_t r = a_start + 1 + k;
        for (int32_t i = 0; i < d.n_vq; ++i) {
            grid[size_t(r) * size_t(cols) + 1 + i] =
                delayed[size_t(k) * size_t(d.n_vq) + size_t(i)];
        }
    }
    return grid;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// embed_one_step -- sum text_embed + audio_embed[stream] for one token vector
// ---------------------------------------------------------------------------
bool MossSession::embed_one_step(const int32_t* tokens,
                                  std::vector<float>* embd_out,
                                  std::string* error) {
    // tokens[0] = text token, tokens[1..n_vq] = audio codec tokens.
    const int32_t hs = backbone_->hidden_size();
    embd_out->assign(static_cast<size_t>(hs), 0.0f);

    // Text token embedding — pull from the backbone's token table.
    // (Community extras GGUFs omit token_embd.weight; the backbone is the
    // canonical source.)
    int32_t text_tok = tokens[0];
    if (text_tok >= 0) {
        std::string lk_err;
        if (!backbone_->embed_lookup(&text_tok, 1, embd_out->data(), &lk_err)) {
            if (error) *error = "embed_one_step: embed_lookup failed: " + lk_err;
            return false;
        }
    }

    // Sum audio embeddings for each stream
    for (int s = 0; s < cfg_.n_vq; ++s) {
        int32_t code = tokens[1 + s];
        if (code < 0) continue;
        ggml_tensor* W = audio_embed_[s];
        if (!W) continue;
        if (W->ne[0] != hs) continue;
        float* row = embd_out->data();
        const size_t row_off = static_cast<size_t>(code) * hs;
        if (row_off + static_cast<size_t>(hs) > ggml_nbytes(W) /
            (W->type == GGML_TYPE_F32 ? sizeof(float) : sizeof(ggml_fp16_t)))
            continue;
        if (W->type == GGML_TYPE_F32) {
            const float* wrow = static_cast<const float*>(W->data) + row_off;
            for (int j = 0; j < hs; ++j)
                row[j] += wrow[j];
        } else if (W->type == GGML_TYPE_F16) {
            const ggml_fp16_t* wrow = static_cast<const ggml_fp16_t*>(W->data) + row_off;
            for (int j = 0; j < hs; ++j)
                row[j] += ggml_fp16_to_fp32(wrow[j]);
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// run_tts — upstream-honest MOSS-TTS pipeline.
//
// Accepts exactly two modes:
//   • mode="tts"         — plain text-to-speech using the flagship backbone.
//   • mode="voice_clone" — read req->reference_audio WAV, run the real codec
//                          encoder, splice the delay-pattern-shifted codes
//                          into the user-side audio block.
//
// Any other mode (sfx / dialogue / voice_design / realtime) is a request for
// a model we do not ship. We fail fast with an error that names the missing
// checkpoint — see GAPS.md §1.2 for the full list.
// ---------------------------------------------------------------------------
bool MossSession::run_tts(const void* request, void* response,
                          std::string* error) {
    if (!loaded_) {
        if (error) *error = "MossSession not loaded";
        return false;
    }

    // Clear the KV cache before each request so sequence positions start at 0.
    // Without this, the cache fills up across requests and eventually runs out
    // of slots (n_ctx = 8192 by default), causing "failed to find a memory
    // slot" errors on subsequent calls.
    {
        llama_context* ctx = backbone_->raw_context();
        llama_memory_t mem = llama_get_memory(ctx);
        std::fprintf(stderr, "[moss_tts] run_tts: ctx=%p mem=%p\n",
                     (void*)ctx, (void*)mem);
        if (mem) {
            llama_pos before = llama_memory_seq_pos_max(mem, 0);
            bool ok = llama_memory_seq_rm(mem, -1, -1, -1);
            llama_pos after = llama_memory_seq_pos_max(mem, 0);
            std::fprintf(stderr, "[moss_tts] run_tts: seq_rm ok=%d pos_max(0) before=%d after=%d\n",
                         ok, before, after);
        } else {
            std::fprintf(stderr, "[moss_tts] run_tts: mem is NULL — clear is no-op!\n");
        }
    }

    const auto* req = static_cast<const TtsRequest*>(request);
    auto*       res = static_cast<TtsResponse*>(response);
    if (!req || !res) {
        if (error) *error = "null request/response";
        return false;
    }
    res->sampling_rate = cfg_.sampling_rate;

    if (req->text.empty() && req->messages.empty()) {
        if (error) *error = "empty text (or messages)";
        return false;
    }

    // ── Mode dispatch ──────────────────────────────────────────────────────
    const bool is_clone  = (req->mode == "voice_clone");
    const bool is_stream = (req->mode == "streaming");
    // moss_tts is the single family for all Delay-architecture checkpoints
    // (flagship TTS, VoiceGenerator, SoundEffect-v1, TTSD — they all share
    // the same code path). Each checkpoint supports its own mode:
    //
    //   n_vq=32 (MOSS-TTS-Delay v1.5):  tts / voice_clone / streaming
    //   n_vq=16 (MOSS-SoundEffect v1):  sfx (text→audio, same AR path as tts)
    //   n_vq=16 (MOSS-VoiceGenerator):  voice_design (via npy-loaded audio heads)
    //
    // SFX mode uses the identical generation path as TTS mode (text-to-audio,
    // no reference, non-streaming). The only difference is which checkpoint
    // is loaded — the model weights, not the mode string, determine the
    // output quality.
    //
    // Requesting a mode that the loaded checkpoint wasn't trained for is
    // a hard error.
    bool mode_ok = (req->mode == "tts" ||
                    req->mode == "voice_clone" ||
                    req->mode == "streaming" ||
                    req->mode == "sfx");
    if (cfg_.n_vq != 32) {
        // Non-flagship checkpoint — restrict to its intended modes.
        // SFX and VoiceGenerator both use n_vq=16; the user selects the
        // appropriate mode for the loaded checkpoint.
        mode_ok = (req->mode == "voice_design" || req->mode == "sfx");
    }

    if (!mode_ok) {
        if (error) *error =
            std::string("moss_tts: mode '") + req->mode
            + "' is not supported by this checkpoint (n_vq="
            + std::to_string(cfg_.n_vq) + "). "
            "Supported modes: "
            + (cfg_.n_vq == 32
                ? "tts, voice_clone, streaming"
                : "sfx (MOSS-SoundEffect), voice_design (MOSS-VoiceGenerator)")
            + ".";
        return false;
    }

    // Streaming mode requires the user to wire req->stream->on_audio.
    if (is_stream && (!req->stream || !req->stream->on_audio)) {
        if (error) *error =
            "moss_tts streaming requires a non-null stream callback "
            "(set stream.on_audio) for incremental codec decode.";
        return false;
    }

    // Voice-clone mode requires a reference WAV and the codec encoder.
    if (is_clone) {
        if (req->reference_audio.empty() && !req->voice_path.empty()) {
            // Back-compat: older API used voice_path for the same purpose.
        } else if (req->reference_audio.empty()) {
            if (error) *error =
                "voice_clone mode requires reference_audio (path to a WAV file, "
                "any sample rate; the loader resamples to the codec rate).";
            return false;
        }
        if (!codec_graphs_.is_present() || !codec_graphs_.encoder_present()) {
            if (error) *error =
                "voice_clone mode requires the codec encoder "
                "(moss.codec.enc.* tensors). This GGUF does not carry them.";
            return false;
        }
    }

    const int32_t hs = backbone_->hidden_size();
    const int32_t text_vocab = backbone_->vocab_size();
    const int32_t audio_vocab = cfg_.audio_vocab_size + 1;  // +1 for pad code

    // ── 0. Encode reference audio via the real codec encoder ───────────────
    std::vector<int32_t> ref_codes;        // (n_vq, T_ref) row-major
    int32_t T_ref = 0;
    std::string reference_block;

    if (is_clone) {
        const std::string& ref_path = !req->reference_audio.empty()
            ? req->reference_audio : req->voice_path;
        std::vector<float> ref_wav;
        try {
            ref_wav = io::read_wav_mono(ref_path, cfg_.sampling_rate);
        } catch (const std::exception& e) {
            if (error) *error = std::string("voice_clone: ") + e.what();
            return false;
        }
        if (ref_wav.empty()) {
            if (error) *error = "voice_clone: reference WAV is empty";
            return false;
        }
        try {
            ref_codes = codec_graphs_.encode(ref_wav.data(),
                                              static_cast<int64_t>(ref_wav.size()),
                                              T_ref);
        } catch (const std::exception& e) {
            if (error) *error = std::string("voice_clone: codec encode failed: ") + e.what();
            return false;
        }
        if (T_ref <= 0 || ref_codes.size() != size_t(cfg_.n_vq) * size_t(T_ref)) {
            if (error) *error =
                "voice_clone: codec encode returned unexpected shape (T_ref=" +
                std::to_string(T_ref) + ", n_vq=" + std::to_string(cfg_.n_vq) + ")";
            return false;
        }
        std::fprintf(stderr, "[moss_tts] encoded reference: %d frames (%.2fs)\n",
                     T_ref, T_ref * 1920.0 / double(cfg_.sampling_rate));
        reference_block = build_reference_audio_block(backbone_.get(), cfg_, T_ref);
    }

    // ── 1. Build the upstream prompt grid ───────────────────────────────────
    // The upstream MOSS template uses single-turn user/assistant tags; it is
    // NOT the chat template. We ignore req->messages for prompt construction
    // (the flagship Delay backbone is single-turn), but allow callers to pass
    // multi-turn text via req->text. Multi-turn dialogue requires MOSS-TTSD.
    const std::string text_to_speak = req->text.empty() && !req->messages.empty()
        ? req->messages.back().content : req->text;
    if (text_to_speak.empty()) {
        if (error) *error = "empty text";
        return false;
    }

    // Map the unified TtsRequest fields onto the upstream optional slots.
    std::optional<std::string> opt_instruction;
    if (!req->instruct.empty()) opt_instruction = req->instruct;
    std::optional<std::string> opt_language;
    if (!req->language.empty()) opt_language = req->language;
    std::optional<std::string> opt_quality;
    if (!req->quality.empty()) opt_quality = req->quality;
    std::optional<int> opt_duration_tokens;
    if (req->duration_tokens > 0) opt_duration_tokens = req->duration_tokens;

    int32_t n_text = 0;
    std::vector<int32_t> grid;
    try {
        grid = build_prompt_grid(backbone_.get(), cfg_,
                                  opt_instruction,
                                  opt_duration_tokens,
                                  opt_quality,
                                  opt_language,
                                  text_to_speak,
                                  reference_block,
                                  T_ref > 0 ? &ref_codes : nullptr,
                                  T_ref,
                                  n_text);
    } catch (const std::exception& e) {
        if (error) *error = std::string("moss_tts: ") + e.what();
        return false;
    }
    if (n_text <= 0) {
        if (error) *error = "moss_tts: prompt grid is empty";
        return false;
    }

    // Re-pack grid into the (S, 1+n_vq) vector-of-vectors that init_delay_state
    // expects.
    const int32_t total_S = n_text;
    std::vector<std::vector<int32_t>> all_ids(static_cast<size_t>(total_S),
        std::vector<int32_t>(1 + cfg_.n_vq, AUDIO_PAD_CODE));
    const int32_t cols = 1 + cfg_.n_vq;
    for (int32_t r = 0; r < total_S; ++r) {
        for (int32_t c = 0; c < cols; ++c) {
            all_ids[size_t(r)][size_t(c)] =
                grid[size_t(r) * size_t(cols) + size_t(c)];
        }
    }

    // ── 2. Build prompt embeddings ──────────────────────────────────────────
    // We need the per-position input embedding = text_tok_embed + Σ_s audio_embed[s][code_s].
    // The text-token table is the Qwen3 backbone's token_embd.weight (kept
    // inside the runner); the audio codebook tables live in the MOSS extras
    // GGUF (audio_embed_[s]). The legacy code assumed token_embd_ (an
    // extras-side tensor also named "token_embd.weight") was always present
    // — but community extras GGUFs that ship only the codec + audio
    // embed/head tensors leave it null, silently producing an all-zero
    // prompt embedding and a degenerate AR loop (30 s of mechanical buzz).
    // Use the backbone's embed_lookup() to populate the text half of every
    // row first, then sum in audio codebook rows for reference-audio positions.
    std::vector<float> prompt_embeds(static_cast<size_t>(total_S) * hs, 0.0f);
    {
        std::vector<int32_t> text_only(static_cast<size_t>(total_S));
        for (int32_t i = 0; i < total_S; ++i)
            text_only[size_t(i)] = all_ids[size_t(i)][0];
        std::string emb_err;
        if (!backbone_->embed_lookup(text_only.data(), total_S,
                                      prompt_embeds.data(), &emb_err)) {
            if (error) *error = "moss_tts: embed_lookup failed: " + emb_err;
            return false;
        }
    }
    // Sum audio codebook embeddings for reference-audio rows (those whose
    // text channel is the user-slot marker). Text rows have AUDIO_PAD_CODE
    // in channels 1..n_vq and contribute nothing here.
    for (int32_t i = 0; i < total_S; i++) {
        const int32_t text_tok = all_ids[size_t(i)][0];
        const bool is_ref_row = (text_tok == cfg_.tok_user_slot);
        if (!is_ref_row) continue;
        float* row = prompt_embeds.data() + static_cast<size_t>(i) * hs;
        for (int s = 0; s < cfg_.n_vq; ++s) {
            int32_t code = all_ids[size_t(i)][1 + s];
            if (code < 0 || code == cfg_.audio_pad_code) continue;
            ggml_tensor* W = audio_embed_[s];
            if (!W) continue;
            if (W->ne[0] != hs) continue;
            const size_t row_off = static_cast<size_t>(code) * hs;
            if (row_off + static_cast<size_t>(hs) > ggml_nbytes(W) /
                (W->type == GGML_TYPE_F32 ? sizeof(float) : sizeof(ggml_fp16_t)))
                continue;
            if (W->type == GGML_TYPE_F32) {
                const float* wrow = static_cast<const float*>(W->data) + row_off;
                for (int j = 0; j < hs; ++j)
                    row[j] += wrow[j];
            } else if (W->type == GGML_TYPE_F16) {
                const ggml_fp16_t* wrow = static_cast<const ggml_fp16_t*>(W->data) + row_off;
                for (int j = 0; j < hs; ++j)
                    row[j] += ggml_fp16_to_fp32(wrow[j]);
            }
        }
    }

    // ── 3. Prefill through backbone ─────────────────────────────────────────
    {
        llama_memory_t debug_mem = llama_get_memory(backbone_->raw_context());
        if (debug_mem) {
            std::fprintf(stderr, "[moss_tts] run_tts: prefill at pos=0, seq_pos_max(0)=%d\n",
                         llama_memory_seq_pos_max(debug_mem, 0));
        }
    }
    std::vector<float> hidden_buf(static_cast<size_t>(total_S) * hs);
    if (!backbone_->forward_embeddings(prompt_embeds.data(), total_S, 0,
                                        hidden_buf.data(), error)) {
        return false;
    }

    // ── 4. Delay state from full multi-channel prompt ──────────────────────
    DelayState state = init_delay_state(all_ids, cfg_.n_vq);

    SamplingConfig samp_cfg;
    samp_cfg.text_temperature  = req->text_temperature > 0.0f ? req->text_temperature : 1.5f;
    samp_cfg.text_top_p        = req->text_top_p  > 0.0f ? req->text_top_p  : 1.0f;
    samp_cfg.text_top_k        = req->text_top_k  > 0    ? req->text_top_k  : 50;
    samp_cfg.audio_temperature = req->temperature > 0.0f ? req->temperature : 1.7f;
    samp_cfg.audio_top_p       = req->top_p       > 0.0f ? req->top_p       : 0.8f;
    samp_cfg.audio_top_k       = 25;
    samp_cfg.audio_repetition_penalty = 1.0f;
    samp_cfg.rng.seed = req->seed != 0 ? req->seed : 0;
    if (req->seed == 0) {
        std::random_device rd;
        samp_cfg.rng.seed = (static_cast<int64_t>(rd()) << 32) ^ rd();
    }

    // ── 5. Autoregressive generation loop ──────────────────────────────────
    // Compute frame rate from config: prefer explicit frame_rate from GGUF,
    // then fall back to sampling_rate / downsample_rate.
    const float effective_fps = cfg_.frame_rate > 0.0f
        ? cfg_.frame_rate
        : static_cast<float>(cfg_.sampling_rate) / static_cast<float>(cfg_.downsample_rate);
    const int32_t max_steps = req->max_new_tokens > 0
        ? req->max_new_tokens
        : static_cast<int32_t>(effective_fps * 30.0f) + cfg_.n_vq;  // 30 s + delay flush

    std::vector<float> step_embd(static_cast<size_t>(hs));
    std::vector<float> step_hidden(static_cast<size_t>(hs));
    std::vector<float> audio_logits_buf(
        static_cast<size_t>(cfg_.n_vq) * audio_vocab);

    int32_t pos = total_S;
    bool generated = false;
    const bool streaming = is_stream && req->stream && req->stream->on_audio;
    int32_t decoded_frames = 0;

    for (int32_t step = 0; step < max_steps; ++step) {
        const float* last_hidden = (step == 0)
            ? (hidden_buf.data() + static_cast<size_t>(total_S - 1) * hs)
            : step_hidden.data();

        // Logits index into the most recent decode batch:
        //   step == 0 → the prefill batch (total_S tokens, last_only=true).
        //               output_ids[total_S - 1] is the only logits-enabled
        //               row, so get_logits_ith(total_S - 1) resolves to it.
        //   step > 0  → a 1-token AR forward; output_ids[0] is the only row.
        const float* text_logits = backbone_->get_logits_ith(
            (step == 0) ? (total_S - 1) : 0);
        if (!text_logits) {
            if (error) *error = "get_logits_ith returned null";
            return false;
        }

        ProjectionRefs refs;
        refs.n_tokens    = 1;
        refs.hidden_size = hs;
        refs.n_vq        = cfg_.n_vq;
        refs.vocab       = audio_vocab;
        refs.hidden      = last_hidden;
        refs.heads       = audio_head_;

        bool proj_ok = project_logits_cgraph(refs, audio_logits_buf.data(), error);
        if (!proj_ok) {
            std::string ref_err;
            proj_ok = project_logits_reference(refs, audio_logits_buf.data(), &ref_err);
            if (!proj_ok) {
                if (error) *error = "audio head projection failed: " + ref_err;
                return false;
            }
        }

        std::vector<int32_t> next_ids = delay_step(
            state, text_logits, text_vocab,
            audio_logits_buf.data(), audio_vocab,
            samp_cfg);

        embed_one_step(next_ids.data(), &step_embd, error);

        if (!backbone_->forward_embeddings(step_embd.data(), 1, pos,
                                            step_hidden.data(), error)) {
            return false;
        }
        pos++;
        generated = true;

        if (state.is_stopping) break;

        // ── Per-frame streaming: decode newly available frames ────────────
        if (streaming && codec_graphs_.is_present()) {
            const int32_t n_delayed = static_cast<int32_t>(state.audio_buf.size());
            if (n_delayed >= cfg_.n_vq) {
                auto dedelayed = apply_de_delay_pattern(state.audio_buf, cfg_.n_vq);
                const int32_t n_available = static_cast<int32_t>(dedelayed.size());
                if (n_available > decoded_frames) {
                    for (int32_t i = decoded_frames; i < n_available; i++) {
                        for (int32_t s = 0; s < cfg_.n_vq; s++) {
                            if (dedelayed[size_t(i)][size_t(s)] >= cfg_.audio_vocab_size)
                                dedelayed[size_t(i)][size_t(s)] = 0;
                        }
                    }
                    const int32_t n_new = n_available - decoded_frames;
                    std::vector<int32_t> flat(
                        static_cast<size_t>(cfg_.n_vq) * static_cast<size_t>(n_new));
                    for (int32_t t = 0; t < n_new; ++t) {
                        const auto& frame = dedelayed[decoded_frames + t];
                        for (int32_t v = 0; v < cfg_.n_vq; ++v) {
                            flat[size_t(v) * size_t(n_new) + size_t(t)] = frame[size_t(v)];
                        }
                    }
                    try {
                        auto pcm = codec_graphs_.decode(flat.data(), cfg_.n_vq, n_new);
                        if (!pcm.empty()) {
                            if (!req->stream->on_audio(pcm.data(), pcm.size())) {
                                if (error) *error = "moss_tts: streaming aborted by client";
                                return false;
                            }
                        }
                    } catch (const std::exception& e) {
                        std::fprintf(stderr, "moss_tts: streaming codec decode failed "
                                             "at step %d (%s); continuing\n",
                                     step, e.what());
                    }
                    decoded_frames = n_available;
                }
            }
        }
    }

    if (!generated) {
        if (error) *error = "zero generation steps";
        return false;
    }

    // ── 6. Extract audio from generation output ────────────────────────────
    auto audio_channels = state.audio_buf;
    auto segments = extract_audio_segments(audio_channels, cfg_.n_vq);

    if (streaming && codec_graphs_.is_present() && !segments.empty()) {
        const int32_t n_remaining = static_cast<int32_t>(segments.size()) - decoded_frames;
        if (n_remaining > 0) {
            for (int32_t i = decoded_frames; i < static_cast<int32_t>(segments.size()); i++) {
                for (int32_t s = 0; s < cfg_.n_vq; s++) {
                    if (segments[size_t(i)][size_t(s)] >= cfg_.audio_vocab_size)
                        segments[size_t(i)][size_t(s)] = 0;
                }
            }
            const int32_t n_new = n_remaining;
            std::vector<int32_t> flat(
                static_cast<size_t>(cfg_.n_vq) * static_cast<size_t>(n_new));
            for (int32_t t = 0; t < n_new; ++t) {
                const auto& frame = segments[decoded_frames + t];
                for (int32_t v = 0; v < cfg_.n_vq; ++v) {
                    flat[size_t(v) * size_t(n_new) + size_t(t)] = frame[size_t(v)];
                }
            }
            try {
                auto pcm = codec_graphs_.decode(flat.data(), cfg_.n_vq, n_new);
                if (!pcm.empty()) {
                    if (!req->stream->on_audio(pcm.data(), pcm.size())) {
                        if (error) *error = "moss_tts: streaming aborted by client";
                        return false;
                    }
                }
            } catch (const std::exception& e) {
                std::fprintf(stderr, "moss_tts: streaming final codec decode failed (%s)\n", e.what());
            }
        }
        res->pcm_mono.clear();
        res->sampling_rate = cfg_.sampling_rate;
        return true;
    }

    if (segments.empty()) {
        if (error) *error = "no audio tokens generated";
        return false;
    }

    for (auto& frame : segments) {
        for (auto& code : frame) {
            if (code >= cfg_.audio_vocab_size) code = 0;
        }
    }

    // ── 7. Decode codec → PCM ──────────────────────────────────────────────
    // Missing codec tensors is now a hard error. The flagship TTS backbone
    // requires the codec decoder to produce audio; a community GGUF without
    // moss.codec.* tensors cannot serve audio and should fail fast so callers
    // know to convert the codec alongside the backbone.
    if (!codec_graphs_.is_present()) {
        if (error) *error =
            "moss_tts: codec tensors are not bound (no moss.codec.* tensors "
            "in the GGUF). The MOSS-TTS flagship backbone requires the codec "
            "decoder to produce PCM. Re-convert the GGUF with the codec "
            "tensors included.";
        return false;
    }

    const int32_t T_audio = static_cast<int32_t>(segments.size());
    if (T_audio <= 0) {
        if (error) *error = "no audio frames to decode";
        return false;
    }

    std::vector<int32_t> flat(
        static_cast<size_t>(cfg_.n_vq) * static_cast<size_t>(T_audio));
    for (int32_t t = 0; t < T_audio; ++t) {
        const auto& frame = segments[size_t(t)];
        for (int32_t v = 0; v < cfg_.n_vq; ++v) {
            flat[size_t(v) * size_t(T_audio) + size_t(t)] = frame[size_t(v)];
        }
    }
    try {
        res->pcm_mono = codec_graphs_.decode(flat.data(), cfg_.n_vq, T_audio);
    } catch (const std::exception& e) {
        if (error) *error = std::string("moss_tts codec decode failed: ") + e.what();
        return false;
    }
    res->sampling_rate = cfg_.sampling_rate;

    // Non-streaming post-hoc: emit audio in chunks via the callback if one
    // was supplied alongside mode="tts" / "voice_clone".
    if (req->stream && req->stream->on_audio && !res->pcm_mono.empty()) {
        constexpr size_t kChunk = 32768;
        const float* pcm = res->pcm_mono.data();
        size_t total = res->pcm_mono.size();
        for (size_t off = 0; off < total; off += kChunk) {
            size_t n = std::min(kChunk, total - off);
            if (!req->stream->on_audio(pcm + off, n)) {
                if (error) *error = "moss_tts: streaming aborted by client";
                return false;
            }
        }
    }
    return true;
}

bool MossSession::decode_codec(const int32_t* codec_tokens, int32_t n_tokens,
                                std::vector<float>* pcm_out,
                                std::string* error) {
    // Stage 16: route through the ggml port when codec tensors are bound.
    // `codec_tokens` here is (n_tokens, n_vq) row-major — flatten into the
    // (n_vq, T_audio) layout MossCodecGraphs::decode expects.
    if (codec_graphs_.is_present() && n_tokens > 0) {
        std::vector<int32_t> flat(static_cast<size_t>(cfg_.n_vq) *
                                   static_cast<size_t>(n_tokens));
        for (int32_t v = 0; v < cfg_.n_vq; ++v) {
            for (int32_t t = 0; t < n_tokens; ++t) {
                flat[size_t(v) * size_t(n_tokens) + size_t(t)] =
                    codec_tokens[size_t(t) * size_t(cfg_.n_vq) + size_t(v)];
            }
        }
        try {
            *pcm_out = codec_graphs_.decode(flat.data(), cfg_.n_vq, n_tokens);
            return true;
        } catch (const std::exception& e) {
            if (error) *error = std::string("codec decode failed: ") + e.what();
            return false;
        }
    }

    // No silence fallback — surface the misconfiguration as an error so the
    // caller knows to reconvert the GGUF with codec tensors included.
    if (error) *error =
        "moss_tts: codec tensors are not bound; cannot decode audio";
    return false;
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
MossSession::~MossSession() {
    if (owns_ext_ctx_ && ext_ctx_) {
        ggml_free(ext_ctx_);
        ext_ctx_ = nullptr;
    }
}

}  // namespace audiocore::moss
