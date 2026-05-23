"""Model loading and inference functions."""

import os
import numpy as np
from feature_extraction import extract_mel_spectrogram
from fallback_model import load_fallback


class BirdDetector:
    """Loads model and provides inference for bird detection."""

    def __init__(self, model_path="bird_sound_classifier.h5"):
        self.model = load_fallback()
        self.model_path = model_path

    def predict(self, audio, sr=44100):
        """Run inference on audio array.
        
        Returns:
            bird_probability: float 0-1
            logits: array of per-class scores [bird, none] or [N species + 1 unknown]
        """
        spec = extract_mel_spectrogram(audio, sr=sr)
        if spec is None:
            return 0.0, np.array([0.0, 1.0])

        spec = spec[np.newaxis, ..., np.newaxis]
        pred = self.model.predict(spec, verbose=0)
        bird_prob = float(pred[0][0]) if pred.shape[-1] >= 2 else 0.5
        logits = pred[0]
        return bird_prob, logits

    def persist(self, path=None):
        """Save model to disk."""
        path = path or self.model_path
        self.model.save(path)
        print(f"Model saved to {path}")


if __name__ == "__main__":
    import librosa
    detector = BirdDetector()
    audio, sr = librosa.load("test_bird.wav", sr=44100, mono=True)
    bird_prob, logits = detector.predict(audio, sr)
    print(f"Bird probability: {bird_prob:.3f}")
    print(f"Logits: {logits}")
