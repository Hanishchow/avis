"""Prepare training data for the confidence stacking MLP."""

import os
import json
import numpy as np
import librosa
from birdnet_integration import BirdNETProcessor
from model_inference import BirdDetector


def generate_mlp_data(data_dir="bird_dataset", output_csv="mlp_training_data.csv",
                      num_samples=200):
    """Generate MLP training data pairing BirdNET logits + NN probability.
    
    Each sample: [birdnet_logit_1, birdnet_logit_2, birdnet_logit_3, nn_bird_prob]
    Label: 1 if bird present, 0 otherwise.
    """
    birdnet = BirdNETProcessor()
    nn = BirdDetector()

    records = []
    temp_path = "_mlp_temp.wav"
    sr = 44100

    for i in range(num_samples):
        t = np.linspace(0, 3, int(sr * 3))
        is_bird = np.random.rand() > 0.5
        if is_bird:
            freq = 2000 + np.random.rand() * 3000
            audio = np.sin(2 * np.pi * freq * t) * 0.5
            audio += np.sin(2 * np.pi * freq * 2 * t) * 0.25
        else:
            audio = np.random.randn(len(t)) * 0.3

        import scipy.io.wavfile as wav
        wav.write(temp_path, sr, (audio / np.max(np.abs(audio)) * 32767).astype(np.int16))

        detections = birdnet.analyze_file(temp_path)
        birdnet_logits = birdnet.get_top3_logits(detections)
        nn_prob, _ = nn.predict(audio, sr)

        features = np.concatenate([birdnet_logits, [nn_prob]])
        records.append(list(features) + [1 if is_bird else 0])

    os.remove(temp_path)

    import csv
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["birdnet_1", "birdnet_2", "birdnet_3", "nn_prob", "label"])
        writer.writerows(records)

    print(f"MLP training data saved to {output_csv} ({len(records)} samples)")
    return output_csv


if __name__ == "__main__":
    generate_mlp_data()
