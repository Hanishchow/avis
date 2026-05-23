"""Test baseline BirdNET recorder functionality."""
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.io.wavfile import write as write_wav
from birdnetlib.analyzer import Analyzer
from birdnetlib import Recording


def test_analyzer_runs():
    """Verify the BirdNET analyzer loads and processes a file without errors."""
    analyzer = Analyzer()
    print("Analyzer loaded successfully.")

    # Try test_bird.wav first
    input_file = None
    for candidate in [
        "test_bird.wav",
        "silent_test.wav",
        os.path.join(os.path.dirname(__file__), "..", "test_bird.wav"),
        os.path.join(os.path.dirname(__file__), "..", "silent_test.wav"),
    ]:
        if os.path.exists(candidate):
            input_file = candidate
            break

    if input_file is None:
        sr = 44100
        data = np.zeros(sr * 3, dtype=np.int16)
        input_file = os.path.join(tempfile.gettempdir(), "silent_test_birdnet.wav")
        write_wav(input_file, sr, data)

    print(f"Processing file: {input_file}")
    recording = Recording(analyzer, input_file, lat=12.97, lon=77.59, min_conf=0.35)
    recording.analyze()

    print(f"Detections: {len(recording.detections)}")
    for d in recording.detections:
        print(f"  {d['common_name']}: {d['confidence']*100:.1f}%")

    if input_file and "silent_test" in input_file and os.path.exists(input_file):
        if input_file.startswith(tempfile.gettempdir()):
            os.remove(input_file)

    return True


if __name__ == "__main__":
    print("Testing BirdNET baseline recorder...")
    try:
        test_analyzer_runs()
        print("\nTEST PASSED: BirdNET analyzer runs without errors.")
        sys.exit(0)
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
