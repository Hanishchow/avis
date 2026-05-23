"""BirdNET integration wrapper - resamples audio and runs BirdNET inference."""

import numpy as np
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer


class BirdNETProcessor:
    """Wraps BirdNET library for species identification."""

    def __init__(self, lat=12.97, lon=77.59, min_conf=0.35):
        self.analyzer = Analyzer()
        self.lat = lat
        self.lon = lon
        self.min_conf = min_conf

    def analyze_file(self, filepath):
        """Run BirdNET on a WAV file. Returns top-3 detection logits."""
        recording = Recording(
            self.analyzer, filepath,
            lat=self.lat, lon=self.lon,
            min_conf=self.min_conf,
        )
        recording.analyze()
        return recording.detections

    def get_top3_logits(self, detections):
        """Extract top-3 logits from birdnet detections.
        
        Returns array of shape (3,) with confidence values.
        """
        logits = np.zeros(3)
        for i, d in enumerate(detections[:3]):
            logits[i] = d.get("confidence", 0.0)
        return logits


if __name__ == "__main__":
    processor = BirdNETProcessor()
    detections = processor.analyze_file("test_bird.wav")
    print(f"Detections: {len(detections)}")
    for d in detections:
        print(f"  {d['common_name']}: {d['confidence']*100:.1f}%")
