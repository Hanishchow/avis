"""Tests for the neural network pipeline."""

import numpy as np
from bird_classifier_cnn import build_cnn
from model_inference import BirdDetector


def test_model_compiles():
    model = build_cnn()
    test_input = np.random.randn(1, 128, 259, 1)
    output = model.predict(test_input, verbose=0)
    assert output.shape == (1, 2), f"Unexpected output shape: {output.shape}"
    assert 0.0 <= output[0][0] <= 1.0
    assert abs(output[0].sum() - 1.0) < 1e-5
    print("PASS: test_model_compiles")


def test_detector_runs():
    detector = BirdDetector()
    sr = 44100
    audio = np.random.randn(sr * 3).astype(np.float32)
    bird_prob, logits = detector.predict(audio, sr)
    assert 0.0 <= bird_prob <= 1.0, f"Bird prob out of range: {bird_prob}"
    assert len(logits) >= 2, f"Not enough logits: {len(logits)}"
    print(f"PASS: test_detector_runs (bird_prob={bird_prob:.3f})")


if __name__ == "__main__":
    test_model_compiles()
    test_detector_runs()
    print("\nAll NN tests passed!")
