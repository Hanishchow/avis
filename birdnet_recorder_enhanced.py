"""Advanced bioacoustic bird detection system - continuous monitoring loop."""

import sys
import os
import time
import logging
from datetime import datetime
from collections import deque

import numpy as np

from filtering_pipeline import full_filter_pipeline, wiener_filter_multi_band
from ensemble_inference import (
    EnsembleDetector,
    apply_detection_threshold,
    save_detection,
)
from silero_vad import load_silero_vad, get_speech_timestamps


# --- CONFIGURABLE CONSTANTS ---
DEVICE_INDEX = 1
SAMPLE_RATE = 44100
CHANNELS = 2
WINDOW_DURATION = 15  # seconds
HIGH_PASS_CUTOFF = 300
LOW_PASS_CUTOFF = 22000
DETECTION_THRESHOLD = 0.35
SPEECH_PROB_THRESHOLD = 0.4
OUTPUT_DIR = "detections"
LOG_FILE = "birdnet_enhanced.log"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(),
    ],
)


def check_speech_obscuration(audio, sr):
    """Check if audio contains speech > threshold using silero-vad (requires 16kHz)."""
    try:
        vad_model = load_silero_vad()
        # Resample to 16kHz if needed (silero-vad supports 8kHz and 16kHz only)
        if sr != 16000:
            from scipy.signal import resample
            target_len = int(len(audio) * 16000 / sr)
            audio_16k = resample(audio, target_len).astype(audio.dtype)
            vad_sr = 16000
        else:
            audio_16k = audio
            vad_sr = sr
        speech_ts = get_speech_timestamps(audio_16k, vad_model, sampling_rate=vad_sr)
        total_speech = sum(ts["end"] - ts["start"] for ts in speech_ts)
        speech_ratio = total_speech / len(audio_16k) if len(audio_16k) > 0 else 0
        return speech_ratio > SPEECH_PROB_THRESHOLD, speech_ts
    except Exception as e:
        logging.warning(f"VAD error: {e}")
        return False, []


def main():
    logging.info("Starting Advanced Bioacoustic Bird Detection System")
    logging.info(f"Config: sr={SAMPLE_RATE}, window={WINDOW_DURATION}s, threshold={DETECTION_THRESHOLD}")

    detector = EnsembleDetector()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize audio stream
    import pyaudio
    p = pyaudio.PyAudio()

    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=DEVICE_INDEX,
            frames_per_buffer=8192,
        )
    except Exception as e:
        logging.error(f"Audio stream init failed: {e}")
        logging.info("Falling back to simulated audio for testing")
        stream = None

    frame_buffer = deque(maxlen=int(SAMPLE_RATE / 8192 * WINDOW_DURATION))
    frames_per_window = int(SAMPLE_RATE / 8192 * WINDOW_DURATION)

    # Latency benchmarking
    latency_history = deque(maxlen=100)
    window_start_time = time.time()

    logging.info("Monitoring started. Press Ctrl+C to stop.")
    try:
        while True:
            if stream:
                data = stream.read(8192, exception_on_overflow=False)
                frame_buffer.append(np.frombuffer(data, dtype=np.int16))
            else:
                # Simulated audio for testing without microphone
                frame_buffer.append(np.random.randn(8192).astype(np.int16))
                time.sleep(0.1)

            if len(frame_buffer) >= frames_per_window:
                audio = np.concatenate(list(frame_buffer))
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                loop_start = time.perf_counter()

                # 1. Apply noise filtering
                filtered = full_filter_pipeline(
                    audio, fs=SAMPLE_RATE,
                    hp_cutoff=HIGH_PASS_CUTOFF,
                    lp_cutoff=LOW_PASS_CUTOFF,
                    stationary_wiener=True,
                )

                # 2. Run ensemble detection
                confidence, species, detections = detector.detect(filtered, SAMPLE_RATE)

                # 3. Check speech obscuration
                is_speech, _ = check_speech_obscuration(filtered, SAMPLE_RATE)
                if is_speech:
                    logging.info(f"Speech detected ({timestamp}) - obscuring audio")
                    continue

                # 4. Apply detection threshold
                if apply_detection_threshold(confidence, DETECTION_THRESHOLD):
                    species_name = species or "unknown"
                    save_detection(
                        filtered, SAMPLE_RATE,
                        species_name, timestamp,
                        output_dir=OUTPUT_DIR,
                    )
                    logging.info(f"Detection: {species_name} ({confidence*100:.1f}%)")
                else:
                    logging.debug(f"No detection (confidence={confidence:.3f})")

                # Clear buffer for next window
                frame_buffer.clear()

                elapsed = time.perf_counter() - loop_start
                latency_history.append(elapsed)
                logging.debug(f"Window processed in {elapsed*1000:.1f}ms")

                # Log latency stats every 10 windows
                if len(latency_history) >= 10 and len(latency_history) % 10 == 0:
                    avg_latency = sum(latency_history) / len(latency_history)
                    max_latency = max(latency_history)
                    logging.info(
                        f"Latency stats (last {len(latency_history)} windows): "
                        f"avg={avg_latency*1000:.1f}ms, max={max_latency*1000:.1f}ms, "
                        f"window_size={WINDOW_DURATION}s"
                    )
                    if avg_latency > WINDOW_DURATION:
                        logging.warning(
                            f"Average latency ({avg_latency*1000:.1f}ms) exceeds "
                            f"window duration ({WINDOW_DURATION}s)! Increase WINDOW_DURATION "
                            f"or optimize filters."
                        )

                # Track uptime
                uptime = time.time() - window_start_time
                if uptime >= 3600:
                    logging.info("1-hour stability milestone reached.")
                    window_start_time = time.time()

    except KeyboardInterrupt:
        logging.info("Monitoring stopped by user.")
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()


if __name__ == "__main__":
    main()
