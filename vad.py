"""VAD frontend — attempts webrtcvad (lightweight), falls back to silero-vad."""

from __future__ import annotations

import numpy as np

_use_webrtc = False
_get_speech_ts = None
_load_vad = None

try:
    import webrtcvad
    _use_webrtc = True
except Exception:
    try:
        import silero_vad as _sv
        _get_speech_ts = _sv.get_speech_timestamps
        _load_vad = _sv.load_silero_vad
    except Exception:
        pass


def load_vad():
    if _use_webrtc:
        return webrtcvad.Vad(2)
    if _load_vad is not None:
        return _load_vad()
    raise ImportError("No VAD backend (install webrtcvad or silero-vad)")


def get_speech_timestamps(audio: np.ndarray, vad_model, sampling_rate: int):
    if _use_webrtc:
        return _webrtc_get_ts(audio, vad_model, sampling_rate)
    if _get_speech_ts is not None:
        return _get_speech_ts(audio, vad_model, sampling_rate=sampling_rate)
    raise ImportError("No VAD backend")


def _webrtc_get_ts(audio: np.ndarray, vad, sr: int):
    if sr not in (8000, 16000, 32000, 48000):
        from scipy.signal import resample
        target_len = int(len(audio) * 16000 / sr)
        audio = resample(audio, target_len).astype(np.int16)
        sr = 16000
    audio = (audio / np.max(np.abs(audio) + 1e-12) * 32767).astype(np.int16)
    frame_samples = int(sr * 30 / 1000)
    is_speech = []
    for i in range(0, len(audio), frame_samples):
        chunk = audio[i : i + frame_samples]
        if len(chunk) < frame_samples:
            chunk = np.pad(chunk, (0, frame_samples - len(chunk)))
        is_speech.append(vad.is_speech(chunk.tobytes(), sr))
    timestamps, start = [], None
    for idx, speaking in enumerate(is_speech):
        if speaking and start is None:
            start = idx * frame_samples
        elif not speaking and start is not None:
            timestamps.append({"start": start, "end": idx * frame_samples})
            start = None
    if start is not None:
        timestamps.append({"start": start, "end": len(audio)})
    return timestamps
