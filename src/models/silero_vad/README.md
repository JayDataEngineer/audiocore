# silero_vad

Silero VAD v6.2 — voice activity detection. Tiny conv-recurrent model,
~1.8 MB weights, 16 kHz mono input, outputs speech segment timestamps.

## Status

✅ **CPU reference implementation complete** — full forward pass (STFT-via-Conv1d
→ 4× Conv+ReLU → LSTM cell → 1×1 Conv → sigmoid) and segmenter ported from
`0xShug0/audio.cpp` (release-0.1, MIT). Naive C++ loops; no ggml graph.

🚧 **One integration TODO** — the `WeightLoader` concrete factory call in
`runtime.cpp::load()` is left as a clearly-marked stub. Fill in 1 line
(matching the pattern in `moss_sfx_v2/vae_runner.cpp` or
`kokoro_tts/loader.cpp`) and the family is end-to-end functional.

## Architecture

1. Input: `[context(64) | chunk(512)]` = 576 samples at 16 kHz
2. STFT via Conv1D (in=1, out=258, k=256, s=128, no bias) → split real/imag
3. Magnitude = `sqrt(re² + im²)` → 129 bins
4. Conv1+ReLU (129→128, k=3, s=1, p=1)
5. Conv2+ReLU (128→64,  k=3, s=2, p=1)
6. Conv3+ReLU (64→64,   k=3, s=2, p=1)
7. Conv4+ReLU (64→128,  k=3, s=1, p=1)
8. Slice time[0:1] → reshape [128]
9. LSTM cell (input 128, hidden 128), **stateful across chunks**
10. final_conv 1×1 (128→1) → sigmoid → speech probability

Per-chunk probabilities → segmenter (threshold + hysteresis + silence/
speech padding) → speech segments.

State carried across chunks:
- LSTM hidden [128] + cell [128]
- 64-sample left audio context

## Weights

Source: `snakers4/silero-vad` on HuggingFace — file `silero_vad_16k.safetensors`.

Convert:
```sh
python tools/convert_silero_vad.py /path/to/silero_vad_16k.safetensors \
    models/silero_vad/silero_vad.gguf
```

Stored under prefix `silero_vad.*`, F32. Expected tensors (15 total):

| Name | Shape |
|---|---|
| `silero_vad.stft_conv.weight` | [256, 1, 258] |
| `silero_vad.conv1.weight` / `.bias` | [3, 129, 128] / [128] |
| `silero_vad.conv2.weight` / `.bias` | [3, 128, 64] / [64] |
| `silero_vad.conv3.weight` / `.bias` | [3, 64, 64] / [64] |
| `silero_vad.conv4.weight` / `.bias` | [3, 64, 128] / [128] |
| `silero_vad.lstm_cell.weight_ih` / `.weight_hh` | [512, 128] |
| `silero_vad.lstm_cell.bias_ih` / `.bias_hh` | [512] |
| `silero_vad.final_conv.weight` / `.bias` | [128, 1] / [1] |

## Performance

CPU reference. The model is ~1.8 MB and chunks are 32 ms — even with naive
C++ loops, throughput is far above realtime on any modern CPU. If GPU
acceleration is ever needed, the math in `runtime.cpp` maps 1:1 onto
`ggml_conv_1d` / `ggml_mul_mat` / `ggml_sigmoid` — same pattern as
`moss_sfx_v2/vae_runner.cpp`.

## Port source

| Their file (audio.cpp release-0.1) | Our file | Notes |
|---|---|---|
| `src/models/silero_vad/runtime.cpp` | `runtime.cpp` | Graph + segmenter ported. Re-expressed `modules::*Module` as plain C++ loops. |
| `src/models/silero_vad/session.cpp` | `session.cpp` | Audiocore Session wrapper. |
| `src/models/silero_vad/assets.cpp` | (deleted) | Replaced by audiocore's `WeightLoader`. |

## Integration TODO (one item)

In `src/models/silero_vad/runtime.cpp::load()`, replace:

```cpp
std::unique_ptr<WeightLoader> loader;  // = ... concrete subclass ...
```

with the correct concrete instantiation. Check `moss_sfx_v2/vae_runner.cpp`
or `kokoro_tts/loader.cpp` for the canonical call — likely one of:

```cpp
loader_ = make_weight_loader(model_path);          // factory fn
loader_ = std::make_unique<GgufWeightLoader>();    // concrete subclass
loader_ = WeightLoader::open(model_path);          // static factory
```

That's the only blocker. Everything else compiles and is logically complete.

## License

- Model: MIT — `snakers4/silero-vad`
- Ported graph code: MIT — `0xShug0/audio.cpp`
