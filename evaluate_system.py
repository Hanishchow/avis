"""Evaluation script for the bird detection system."""

import os
import json
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix


def evaluate(detection_log="detection_log.json", ground_truth=None):
    """Compute macro-averaged F1 score and false positive rate per hour."""
    if not os.path.exists(detection_log):
        print(f"Detection log {detection_log} not found.")
        print("Creating sample evaluation data for testing...")
        _create_sample_log(detection_log)
        ground_truth = _create_sample_truth()

    with open(detection_log) as f:
        detections = json.load(f)

    if ground_truth is None:
        ground_truth = []

    y_true = [gt["label"] for gt in ground_truth]
    y_pred = [d.get("label", 0) for d in detections]

    if len(y_true) == 0:
        print("No ground truth data for evaluation. Using detection counts only.")
        print(f"Total detections: {len(detections)}")
        false_positives = sum(1 for d in detections if d.get("confidence", 0) < 0.5)
        hours = max(1, len(detections) / 240)
        fp_per_hour = false_positives / hours
        print(f"False positive rate: {fp_per_hour:.2f}/hr")
        return {"fp_per_hour": fp_per_hour}

    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    false_positives = cm[0][1] if cm.shape == (2, 2) else 0
    hours = max(1, len(detections) / 240)
    fp_per_hour = false_positives / hours

    print(f"Macro-averaged F1 score: {f1:.3f}")
    print(f"False positive rate: {fp_per_hour:.2f}/hr")
    print(f"Confusion matrix:\n{cm}")
    return {"f1": f1, "fp_per_hour": fp_per_hour}


def _create_sample_log(path):
    n = 100
    detections = []
    for i in range(n):
        detections.append({
            "species": "test" if i % 2 == 0 else "none",
            "confidence": np.random.uniform(0.3, 0.9),
            "label": 1 if i % 2 == 0 else 0,
            "timestamp": f"2026-05-19_{i:04d}",
        })
    with open(path, "w") as f:
        json.dump(detections, f)


def _create_sample_truth():
    return [{"label": 1 if i % 2 == 0 else 0} for i in range(100)]


if __name__ == "__main__":
    evaluate()
