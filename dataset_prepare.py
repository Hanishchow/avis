"""Download and prepare bird vocalization dataset."""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split


def prepare_synthetic_dataset(output_dir="bird_dataset"):
    """Create a small synthetic dataset for testing the pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    sr = 44100
    n_samples = 100
    duration = 3.0
    np.random.seed(42)

    labels = ["bird", "none"]
    data = []
    for i in range(n_samples):
        t = np.linspace(0, duration, int(sr * duration))
        is_bird = np.random.rand() > 0.5
        if is_bird:
            freq = 2000 + np.random.rand() * 3000
            signal = np.sin(2 * np.pi * freq * t) * 0.5
            signal += np.sin(2 * np.pi * freq * 2 * t) * 0.25
            signal += np.random.randn(len(t)) * 0.05
        else:
            signal = np.random.randn(len(t)) * 0.3
        signal = (signal / np.max(np.abs(signal)) * 32767).astype(np.int16)
        path = os.path.join(output_dir, f"sample_{i:04d}.npy")
        np.save(path, signal)
        data.append({"path": path, "label": 0 if not is_bird else 1})

    train, test = train_test_split(data, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.25, random_state=42)

    with open(os.path.join(output_dir, "train.json"), "w") as f:
        json.dump(train, f)
    with open(os.path.join(output_dir, "val.json"), "w") as f:
        json.dump(val, f)
    with open(os.path.join(output_dir, "test.json"), "w") as f:
        json.dump(test, f)

    print(f"Created synthetic dataset in '{output_dir}/'")
    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return output_dir


if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "bird_dataset"
    prepare_synthetic_dataset(output_dir)
