"""Minimal ONNX audio tokenizer wrapper for MOSS-TTS inference pipeline."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

N_QUANTIZERS = 32
DOWNSAMPLE_RATE = 320


def _load_ort_session(path: str, use_gpu: bool) -> ort.InferenceSession:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    return ort.InferenceSession(path, providers=providers)


class OnnxAudioTokenizer:
    """MOSS audio tokenizer using ONNX Runtime."""

    def __init__(self, encoder_path: str, decoder_path: str, use_gpu: bool = True):
        self.encoder = _load_ort_session(encoder_path, use_gpu)
        self.decoder = _load_ort_session(decoder_path, use_gpu)
        self._enc_inputs = [i.name for i in self.encoder.get_inputs()]
        self._enc_outputs = [o.name for o in self.encoder.get_outputs()]
        self._dec_inputs = [i.name for i in self.decoder.get_inputs()]
        self._dec_outputs = [o.name for o in self.decoder.get_outputs()]

    def encode(self, waveform: np.ndarray, n_quantizers: int = N_QUANTIZERS) -> np.ndarray:
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, np.newaxis, :]
        elif waveform.ndim == 2:
            waveform = waveform[np.newaxis, :]
        T = waveform.shape[-1]
        padded = ((T + DOWNSAMPLE_RATE - 1) // DOWNSAMPLE_RATE) * DOWNSAMPLE_RATE
        if padded != T:
            waveform = np.concatenate(
                [waveform, np.zeros((1, 1, padded - T), dtype=np.float32)], axis=-1,
            )
        waveform = waveform.astype(np.float32)
        nq = np.array(n_quantizers, dtype=np.int64)
        outputs = self.encoder.run(self._enc_outputs, {self._enc_inputs[0]: waveform, self._enc_inputs[1]: nq})
        return outputs[0][:, 0, :int(outputs[1][0])].T.astype(np.int64)

    def decode(self, audio_codes: np.ndarray, n_quantizers: int = N_QUANTIZERS) -> np.ndarray:
        if audio_codes.ndim == 2:
            if audio_codes.shape[1] == N_QUANTIZERS and audio_codes.shape[0] != N_QUANTIZERS:
                audio_codes = audio_codes.T
            audio_codes = audio_codes[:, np.newaxis, :]
        codes = audio_codes.astype(np.int64)
        nq = np.array(n_quantizers, dtype=np.int64)
        outputs = self.decoder.run(self._dec_outputs, {self._dec_inputs[0]: codes, self._dec_inputs[1]: nq})
        return outputs[0][0, 0, :int(outputs[1][0])].astype(np.float32)

    def close(self):
        pass
