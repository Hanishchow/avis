"""BirdNET integration wrapper - resamples audio and runs BirdNET inference."""

import numpy as np
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from config import LAT, LON, BIRDNET_MIN_CONF, TOP_K_LOGITS


class BirdNETProcessor:
    """Wraps BirdNET library for species identification."""

    def __init__(self, lat=None, lon=None, min_conf=None):
        self.lat = lat if lat is not None else LAT
        self.lon = lon if lon is not None else LON
        self.min_conf = min_conf if min_conf is not None else BIRDNET_MIN_CONF
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
        """Extract top-k logits from birdnet detections."""
        logits = np.zeros(TOP_K_LOGITS)
        for i, d in enumerate(detections[:3]):
            logits[i] = d.get("confidence", 0.0)
        return logits


if __name__ == "__main__":
    processor = BirdNETProcessor()
    detections = processor.analyze_file("test_bird.wav")
    print(f"Detections: {len(detections)}")
    for d in detections:
        print(f"  {d['common_name']}: {d['confidence']*100:.1f}%")
