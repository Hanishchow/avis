"""Mel-spectrogram feature extraction for bird audio."""

import numpy as np
import librosa


def extract_mel_spectrogram(audio, sr=44100, n_mels=128, n_fft=2048,
                            hop_length=512, target_duration=3.0,
                            fmin=500, fmax=16000):
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


def extract_batch(files, sr=44100):
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
        audio, sr = librosa.load(path, sr=44100, mono=True)
        spec = extract_mel_spectrogram(audio, sr)
        if spec is not None:
            print(f"Mel-spectrogram shape: {spec.shape}")
        else:
            print("Audio too short")
    except Exception as e:
        print(f"Error: {e}")
