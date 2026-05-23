"""Ensemble inference combining BirdNET and custom NN via MLP stacking."""

import os
import numpy as np
from birdnet_integration import BirdNETProcessor
from model_inference import BirdDetector


class EnsembleDetector:
    """Combines BirdNET and custom NN outputs using confidence stacking MLP."""

    def __init__(self, mlp_path="mlp_stacker.h5"):
        self.birdnet = BirdNETProcessor()
        self.nn_detector = BirdDetector()
        self.mlp = None
        self.mlp_path = mlp_path
        self._load_mlp()

    def _load_mlp(self):
        import tensorflow as tf
        if os.path.exists(self.mlp_path):
            try:
                self.mlp = tf.keras.models.load_model(self.mlp_path)
                print(f"MLP stacker loaded from {self.mlp_path}")
            except Exception as e:
                print(f"Could not load MLP: {e}")

    def _predict_mlp(self, birdnet_logits, nn_bird_prob):
        """Run MLP inference on combined features."""
        if self.mlp is None or len(birdnet_logits) < 3:
            return self._fallback_combine(birdnet_logits, nn_bird_prob)
        features = np.concatenate([birdnet_logits[:3], [nn_bird_prob]])
        features = features[np.newaxis, :]
        pred = self.mlp.predict(features, verbose=0)
        return float(pred[0][0]) if pred.shape[-1] >= 1 else nn_bird_prob

    def _fallback_combine(self, birdnet_logits, nn_bird_prob):
        """Weighted average fallback when MLP not available."""
        birdnet_conf = birdnet_logits[0] if len(birdnet_logits) > 0 else 0.0
        return 0.7 * birdnet_conf + 0.3 * nn_bird_prob

    def detect(self, audio, sr=44100):
        """Run ensemble detection on audio array.
        
        Returns:
            combined_confidence: float 0-1
            species: str or None
            birdnet_detections: list
        """
        nn_bird_prob, nn_logits = self.nn_detector.predict(audio, sr)
        temp_path = "_ensemble_temp.wav"
        import scipy.io.wavfile as wav
        wav.write(temp_path, sr, (audio * 32767).astype(np.int16))
        detections = self.birdnet.analyze_file(temp_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        birdnet_logits = self.birdnet.get_top3_logits(detections)
        combined = self._predict_mlp(birdnet_logits, nn_bird_prob)
        species = detections[0]["common_name"] if detections else None
        return combined, species, detections


def apply_detection_threshold(confidence, threshold=0.35):
    """Return True if confidence exceeds threshold."""
    return confidence >= threshold


def save_detection(audio, sr, species, timestamp, output_dir="detections",
                   generate_spectrogram=False):
    """Save detected audio as WAV with species name and timestamp."""
    os.makedirs(output_dir, exist_ok=True)
    species_str = species.replace(" ", "_") if species else "unknown"
    filename = f"{species_str}_{timestamp}.wav"
    filepath = os.path.join(output_dir, filename)
    import scipy.io.wavfile as wav
    wav.write(filepath, sr, (audio * 32767).astype(np.int16))
    print(f"Saved: {filepath}")
    if generate_spectrogram:
        _save_spectrogram(audio, sr, filepath.replace(".wav", ".png"))
    return filepath


def _save_spectrogram(audio, sr, output_path):
    """Generate and save a spectrogram image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa.display
    S = librosa.feature.melspectrogram(y=audio.astype(np.float32), sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel-spectrogram")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Spectrogram saved: {output_path}")
