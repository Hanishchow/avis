"""Mel-spectrogram feature extraction for bird audio."""

import numpy as np
import librosa
from config import SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, DURATION, FMIN, FMAX


def extract_mel_spectrogram(audio, sr=None, n_mels=None, n_fft=None,
                            hop_length=None, target_duration=None,
                            fmin=None, fmax=None):
    if sr is None:
        sr = SAMPLE_RATE
    if n_mels is None:
        n_mels = N_MELS
    if n_fft is None:
        n_fft = N_FFT
    if hop_length is None:
        hop_length = HOP_LENGTH
    if target_duration is None:
        target_duration = DURATION
    if fmin is None:
        fmin = FMIN
    if fmax is None:
        fmax = FMAX
    """Extract mel-spectrogram from audio array.
    
    Returns spectrogram of shape (n_mels, n_frames) or None if audio too short.
    """
    expected_samples = int(target_duration * sr)
    if len(audio) < expected_samples:
        return None
    audio = audio[:expected_samples]
    S = librosa.feature.melspectrogram(
        y=audio.astype(np.float32), sr=sr, n_mels=n_mels,
        n_fft=n_fft, hop_length=hop_length,
        fmin=fmin, fmax=fmax
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def extract_batch(files, sr=None):
    if sr is None:
        sr = SAMPLE_RATE
    """Extract mel-spectrograms for a list of audio files."""
    specs = []
    for path in files:
        audio, _ = librosa.load(path, sr=sr, mono=True)
        spec = extract_mel_spectrogram(audio, sr=sr)
        if spec is not None:
            specs.append(spec)
    return np.array(specs)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test_bird.wav"
    try:
        audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        spec = extract_mel_spectrogram(audio, sr)
        if spec is not None:
            print(f"Mel-spectrogram shape: {spec.shape}")
        else:
            print("Audio too short")
    except Exception as e:
        print(f"Error: {e}")
