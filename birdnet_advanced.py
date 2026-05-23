import pyaudio
import wave
import numpy as np
import os
import time
import json
import threading
import queue
from datetime import datetime
from scipy.signal import butter, lfilter, sosfilt, sosfilt_zi, spectrogram
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
import librosa
import noisereduce as nr
import tensorflow as tf
from collections import deque

# --- CONFIGURATION ---
INDEX = 1
RATE = 44100
CHANNELS = 2
RECORD_SECONDS = 15
TEMP_FILENAME = "temp_capture.wav"
NOISE_REDUCE_STRENGTH = 0.8
CONFIDENCE_THRESHOLD = 0.35
LATITUDE = 12.97
LONGITUDE = 77.59

# Advanced features
ENABLE_SPECTROGRAM = True
ENABLE_NEURAL_NETWORK = True
ENABLE_CONTINUOUS_MONITORING = True
OUTPUT_DIR = "recordings"
LOG_FILE = "detection_log.json"
MAX_LOG_ENTRIES = 1000

# Neural network model path
MODEL_PATH = "bird_classifier.h5"

# Statistics tracking
stats = {
    "total_recordings": 0,
    "total_detections": 0,
    "species_counts": {},
    "start_time": None,
    "detections": []
}

# Detection history for pattern analysis
detection_history = deque(maxlen=100)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos', analog=False)
    return sos


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def apply_advanced_filter(data, fs=44100):
    """Multi-stage noise reduction pipeline"""
    # Stage 1: High-pass filter
    sos_high = butter_highpass(50, fs, order=4)
    zi_high = sosfilt_zi(sos_high)
    filtered, _ = sosfilt(sos_high, data, zi=zi_high * data[0])

    # Stage 2: Bandpass for bird range
    sos_band = butter_bandpass(1000, 8000, fs, order=4)
    zi_band = sosfilt_zi(sos_band)
    filtered, _ = sosfilt(sos_band, filtered, zi=zi_band * filtered[0])

    # Stage 3: Spectral noise reduction
    try:
        y_float = filtered.astype(np.float32) / 32768.0
        reduced = nr.reduce_noise(y=y_float, sr=fs, prop_decrease=NOISE_REDUCE_STRENGTH)
        filtered = (reduced * 32767).astype(np.int16)
    except Exception as e:
        print(f"Noise reduction failed: {e}")

    return filtered


def compute_rms_level(signal):
    """Compute RMS amplitude for audio level monitoring"""
    return np.sqrt(np.mean(signal.astype(np.float32) ** 2))


def compute_spectral_centroid(signal, sr=44100):
    """Compute spectral centroid for frequency analysis"""
    S = np.abs(librosa.fft_librosa.fft(signal.astype(np.float32), n_fft=2048))
    freqs = librosa.fft_librosa.fft_frequencies(sr=sr, n_fft=2048)
    centroid = np.sum(freqs * S) / np.sum(S)
    return centroid


def save_spectrogram(signal, sr, filename):
    """Generate and save spectrogram visualization"""
    try:
        f, t, Sxx = spectrogram(signal, sr, nperseg=1024, noverlap=512)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title(f'Spectrogram - {filename}')
        plt.colorbar(label='Power [dB]')
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"Spectrogram save failed: {e}")
        return False


def update_stats(species_name, confidence):
    """Update running statistics"""
    stats["total_detections"] += 1
    stats["total_recordings"] += 1
    stats["species_counts"][species_name] = stats["species_counts"].get(species_name, 0) + 1
    stats["detections"].append({
        "species": species_name,
        "confidence": confidence,
        "timestamp": datetime.now().isoformat()
    })
    detection_history.append({
        "species": species_name,
        "confidence": confidence,
        "time": time.time()
    })


def save_stats():
    """Persist statistics to log file"""
    with open(LOG_FILE, 'w') as f:
        json.dump(stats, f, indent=2)


def load_stats():
    """Load previous statistics"""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            data = json.load(f)
            stats["total_recordings"] = data.get("total_recordings", 0)
            stats["total_detections"] = data.get("total_detections", 0)
            stats["species_counts"] = data.get("species_counts", {})
            stats["detections"] = data.get("detections", [])


def analyze_audio_quality(signal, sr=44100):
    """Analyze audio quality metrics"""
    rms = compute_rms_level(signal)
    centroid = compute_spectral_centroid(signal, sr)
    peak = np.max(np.abs(signal))
    snr_estimate = 20 * np.log10(peak / (np.std(signal) + 1e-10))
    return {
        "rms": rms,
        "spectral_centroid": centroid,
        "peak": peak,
        "snr_estimate": snr_estimate
    }


def create_detection_filename(species_name):
    """Generate unique filename with timestamp"""
    safe_name = species_name.replace(" ", "_").replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_name}_{timestamp}.wav"


def create_neural_network():
    """Create TensorFlow neural network model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(52,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def load_or_create_model():
    """Load existing model or create new one"""
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        print("Creating new model")
        model = create_neural_network()
        model.save(MODEL_PATH)
        return model


def extract_mfcc_features(signal, sr=44100, n_mfcc=13):
    """Extract MFCC features for neural network"""
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    features = np.hstack([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.max(mfcc, axis=1),
        np.min(mfcc, axis=1)
    ])
    return features.reshape(1, -1)


def record_and_analyze(analyzer, nn_model=None):
    """Main recording and analysis pipeline"""
    p = pyaudio.PyAudio()
    quality_metrics = None

    try:
        stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE,
                        input=True, input_device_index=INDEX, frames_per_buffer=8192)

        print(f"[LISTENING] {RECORD_SECONDS}s window active...")
        frames = []
        for _ in range(0, int(RATE / 8192 * RECORD_SECONDS)):
            data = stream.read(8192, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.int16))

        stream.stop_stream()
        stream.close()

        # Analyze audio quality
        full_signal = np.concatenate(frames)
        quality_metrics = analyze_audio_quality(full_signal, RATE)
        print(f"  Audio RMS: {quality_metrics['rms']:.2f}, "
              f"Centroid: {quality_metrics['spectral_centroid']:.0f}Hz, "
              f"SNR: {quality_metrics['snr_estimate']:.1f}dB")

        # Apply filtering
        filtered_signal = apply_advanced_filter(full_signal, fs=RATE)

        # Normalize
        max_val = np.max(np.abs(filtered_signal))
        if max_val > 0:
            normalized_signal = (filtered_signal / max_val * 32767).astype(np.int16)
        else:
            normalized_signal = filtered_signal

        # Save temp file
        with wave.open(TEMP_FILENAME, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(normalized_signal.tobytes())

        # BirdNET analysis
        recording = Recording(analyzer, TEMP_FILENAME, lat=LATITUDE, lon=LONGITUDE,
                             min_conf=CONFIDENCE_THRESHOLD)
        recording.analyze()

        # Neural network prediction
        nn_confidence = 0.5
        if nn_model is not None:
            try:
                features = extract_mfcc_features(normalized_signal.astype(np.float32))
                nn_confidence = nn_model.predict(features, verbose=0)[0][0]
            except Exception as e:
                print(f"NN prediction error: {e}")

        if recording.detections:
            best = recording.detections[0]
            bird_name = best['common_name']
            confidence = best['confidence']

            # Combined confidence
            combined_conf = 0.7 * confidence + 0.3 * nn_confidence

            if combined_conf >= CONFIDENCE_THRESHOLD:
                filename = create_detection_filename(bird_name)
                filepath = os.path.join(OUTPUT_DIR, filename)

                with wave.open(filepath, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(2)
                    wf.setframerate(RATE)
                    wf.writeframes(normalized_signal.tobytes())

                # Save spectrogram
                if ENABLE_SPECTROGRAM:
                    spec_file = os.path.join(OUTPUT_DIR, f"{bird_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_spec.png")
                    save_spectrogram(normalized_signal, RATE, spec_file)

                update_stats(bird_name, combined_conf)
                print(f"✅ DETECTED: {bird_name} (BirdNET: {confidence*100:.1f}%, NN: {nn_confidence*100:.1f}%, Combined: {combined_conf*100:.1f}%)")
                print(f"💾 SAVED: {filename}")

                if os.path.exists(TEMP_FILENAME):
                    os.remove(TEMP_FILENAME)
            else:
                print(f"⚠️ Low confidence: {bird_name} ({combined_conf*100:.1f}%)")
                if os.path.exists(TEMP_FILENAME):
                    os.remove(TEMP_FILENAME)
        else:
            print("❌ No detection")
            if os.path.exists(TEMP_FILENAME):
                os.remove(TEMP_FILENAME)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        p.terminate()


def continuous_monitoring(analyzer, nn_model):
    """Run continuous monitoring loop"""
    print("Starting continuous monitoring...")
    print("Press Ctrl+C to stop\n")

    if stats["start_time"] is None:
        stats["start_time"] = datetime.now().isoformat()

    try:
        while True:
            record_and_analyze(analyzer, nn_model)
            # Brief pause between recordings
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        save_stats()
        print(f"\n=== Session Summary ===")
        print(f"Total recordings: {stats['total_recordings']}")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Species detected: {len(stats['species_counts'])}")
        if stats['species_counts']:
            print("Species counts:")
            for species, count in sorted(stats['species_counts'].items(), key=lambda x: -x[1]):
                print(f"  {species}: {count}")


def print_system_info():
    """Print system and configuration info"""
    print("=" * 60)
    print("BIRDNET ADVANCED BIOACOUSTIC MONITOR")
    print("=" * 60)
    print(f"Sample Rate: {RATE} Hz")
    print(f"Channels: {CHANNELS}")
    print(f"Record Duration: {RECORD_SECONDS}s")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Noise Reduction: {NOISE_REDUCE_STRENGTH}")
    print(f"Neural Network: {'Enabled' if ENABLE_NEURAL_NETWORK else 'Disabled'}")
    print(f"Spectrogram: {'Enabled' if ENABLE_SPECTROGRAM else 'Disabled'}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load previous stats
    load_stats()

    # Print system info
    print_system_info()

    # Initialize
    print("Loading BirdNET Analyzer...")
    analyzer = Analyzer()

    nn_model = None
    if ENABLE_NEURAL_NETWORK:
        try:
            nn_model = load_or_create_model()
            print("Neural network model loaded/created.")
        except Exception as e:
            print(f"Neural network init failed: {e}")

    # Run monitoring
    continuous_monitoring(analyzer, nn_model)
