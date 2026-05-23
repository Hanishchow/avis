"""Test the retrained CNN on real bird audio."""
import json
import numpy as np
import librosa
from model_inference import BirdDetector
from birdnet_integration import BirdNETProcessor
from filtering_pipeline import full_filter_pipeline
from ensemble_inference import EnsembleDetector, apply_detection_threshold

# Load test data
print("=== Test Set Evaluation ===")
with open("real_dataset/test.json") as f:
    test_records = json.load(f)

detector = BirdDetector()
correct = 0
bird_probs = []

for rec in test_records:
    spec = np.load(rec["spec"])
    label = rec["label"]
    spec_input = spec[np.newaxis, :, :, np.newaxis]
    pred = detector.model.predict(spec_input, verbose=0)
    bird_prob = float(pred[0][0]) if pred[0].shape == (2,) else float(pred[0][1])
    bird_probs.append(bird_prob)
    pred_label = 1 if bird_prob > 0.5 else 0
    if pred_label == label:
        correct += 1

print(f"Test accuracy: {correct}/{len(test_records)} = {correct/len(test_records)*100:.1f}%")
print(f"Avg bird prob for bird: {np.mean([b for r,b in zip(test_records, bird_probs) if r['label']==1]):.3f}")
print(f"Avg bird prob for noise: {np.mean([b for r,b in zip(test_records, bird_probs) if r['label']==0]):.3f}")

# Test on the Delicate Prinia recording
print("\n=== Delicate Prinia Inference ===")
audio, sr = librosa.load("test_bird_real.wav", sr=44100, mono=True)
filtered = full_filter_pipeline(audio, fs=sr, stationary_wiener=True)

bird_prob, logits = detector.predict(filtered, sr)
print(f"CNN bird probability: {bird_prob*100:.1f}%")

# Full ensemble
ensemble = EnsembleDetector()
confidence, species, detections = ensemble.detect(filtered, sr)
print(f"Ensemble confidence: {confidence*100:.1f}%")
print(f"Above 35% threshold: {apply_detection_threshold(confidence, 0.35)}")
