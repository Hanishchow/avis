import pyaudio
import wave
import numpy as np
import os
import time
import json
import tensorflow as tf
from datetime import datetime
from scipy.signal import butter, lfilter, sosfilt, sosfilt_zi, spectrogram
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
import librosa
import noisereduce as nr
from collections import deque

# === CONFIGURATION ===
INDEX = 1
RATE = 44100
CHANNELS = 2
RECORD_SECONDS = 15
TEMP_FILENAME = "temp_capture.wav"
NOISE_REDUCE_STRENGTH = 0.8
CONFIDENCE_THRESHOLD = 0.35
LATITUDE = 12.97
LONGITUDE = 77.59
OUTPUT_DIR = "recordings"
LOG_FILE = "detection_log.json"
MODEL_PATH = "bird_classifier.h5"
ENABLE_NEURAL_NETWORK = True
ENABLE_SPECTROGRAM = True

# Statistics
stats = {"total_recordings": 0, "total_detections": 0, "species_counts": {}, "start_time": None, "detections": []}
detection_history = deque(maxlen=100)

# === FILTER FUNCTIONS ===
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    sos = butter(order, [lowcut/nyq, highcut/nyq], btype='band', output='sos', analog=False)
    return sos

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff/nyq, btype='high', analog=False)
    return b, a

def apply_advanced_filter(data, fs=44100):
    sos_high = butter_highpass(50, fs, order=4)
    zi_high = sosfilt_zi(sos_high)
    filtered, _ = sosfilt(sos_high, data, zi=zi_high * data[0])
    sos_band = butter_bandpass(1000, 8000, fs, order=4)
    zi_band = sosfilt_zi(sos_band)
    filtered, _ = sosfilt(sos_band, filtered, zi=zi_band * filtered[0])
    try:
        y_float = filtered.astype(np.float32) / 32768.0
        reduced = nr.reduce_noise(y=y_float, sr=fs, prop_decrease=NOISE_REDUCE_STRENGTH)
        filtered = (reduced * 32767).astype(np.int16)
    except Exception as e:
        print(f"Noise reduction failed: {e}")
    return filtered

def compute_rms(signal):
    return np.sqrt(np.mean(signal.astype(np.float32) ** 2))

def compute_spectral_centroid(signal, sr=44100):
    S = np.abs(np.fft.fft(signal.astype(np.float32), n_fft=2048))
    freqs = np.fft.fftfreq(2048, 1/sr)[:1024]
    freqs = freqs[:len(S)//2]
    S = S[:len(S)//2]
    centroid = np.sum(freqs * S) / (np.sum(S) + 1e-10)
    return centroid

def analyze_audio_quality(signal, sr=44100):
    rms = compute_rms(signal)
    centroid = compute_spectral_centroid(signal, sr)
    peak = np.max(np.abs(signal))
    snr = 20 * np.log10(peak / (np.std(signal) + 1e-10))
    return {"rms": rms, "centroid": centroid, "peak": peak, "snr": snr}

def save_spectrogram(signal, sr, filename):
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
        print(f"Spectrogram error: {e}")
        return False

def extract_mfcc_features(signal, sr=44100, n_mfcc=13):
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    features = np.hstack([np.mean(mfcc, axis=1), np.std(mfcc, axis=1), np.max(mfcc, axis=1), np.min(mfcc, axis=1)])
    return features.reshape(1, -1)

def create_neural_network():
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
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    model = create_neural_network()
    model.save(MODEL_PATH)
    return model

def update_stats(species_name, confidence):
    stats["total_detections"] += 1
    stats["total_recordings"] += 1
    stats["species_counts"][species_name] = stats["species_counts"].get(species_name, 0) + 1
    stats["detections"].append({"species": species_name, "confidence": confidence, "timestamp": datetime.now().isoformat()})
    detection_history.append({"species": species_name, "confidence": confidence, "time": time.time()})

def save_stats():
    with open(LOG_FILE, 'w') as f:
        json.dump(stats, f, indent=2)

def load_stats():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            data = json.load(f)
            stats["total_recordings"] = data.get("total_recordings", 0)
            stats["total_detections"] = data.get("total_detections", 0)
            stats["species_counts"] = data.get("species_counts", {})
            stats["detections"] = data.get("detections", [])

# === MAIN RECORDING FUNCTION ===
def record_and_analyze(analyzer, nn_model=None):
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE,
                        input=True, input_device_index=INDEX, frames_per_buffer=8192)
        print(f"[LISTENING] {RECORD_SECONDS}s window...")
        frames = []
        for _ in range(0, int(RATE / 8192 * RECORD_SECONDS)):
            data = stream.read(8192, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.int16))
        stream.stop_stream()
        stream.close()

        full_signal = np.concatenate(frames)
        quality = analyze_audio_quality(full_signal, RATE)
        print(f"  RMS: {quality['rms']:.2f}, Centroid: {quality['centroid']:.0f}Hz, SNR: {quality['snr']:.1f}dB")

        filtered = apply_advanced_filter(full_signal, fs=RATE)
        max_val = np.max(np.abs(filtered))
        normalized = (filtered / max_val * 32767).astype(np.int16) if max_val > 0 else filtered

        with wave.open(TEMP_FILENAME, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(normalized.tobytes())

        recording = Recording(analyzer, TEMP_FILENAME, lat=LATITUDE, lon=LONGITUDE, min_conf=CONFIDENCE_THRESHOLD)
        recording.analyze()

        nn_conf = 0.5
        if nn_model is not None:
            try:
                features = extract_mfcc_features(normalized.astype(np.float32))
                nn_conf = nn_model.predict(features, verbose=0)[0][0]
            except Exception as e:
                print(f"NN error: {e}")

        if recording.detections:
            best = recording.detections[0]
            bird_name = best['common_name']
            conf = best['confidence']
            combined = 0.7 * conf + 0.3 * nn_conf

            if combined >= CONFIDENCE_THRESHOLD:
                safe_name = bird_name.replace(" ", "_").replace("/", "_")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{safe_name}_{timestamp}.wav"
                filepath = os.path.join(OUTPUT_DIR, filename)

                with wave.open(filepath, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(2)
                    wf.setframerate(RATE)
                    wf.writeframes(normalized.tobytes())

                if ENABLE_SPECTROGRAM:
                    spec_file = os.path.join(OUTPUT_DIR, f"{safe_name}_{timestamp}_spec.png")
                    save_spectrogram(normalized, RATE, spec_file)

                update_stats(bird_name, combined)
                print(f"✅ {bird_name} (BirdNET: {conf*100:.1f}%, NN: {nn_conf*100:.1f}%, Combined: {combined*100:.1f}%)")
                print(f"💾 {filename}")
            else:
                print(f"⚠️ Low confidence: {bird_name} ({combined*100:.1f}%)")
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

# === MONITORING LOOP ===
def continuous_monitoring(analyzer, nn_model):
    print("Starting continuous monitoring... Press Ctrl+C to stop\n")
    if stats["start_time"] is None:
        stats["start_time"] = datetime.now().isoformat()

    try:
        while True:
            record_and_analyze(analyzer, nn_model)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        save_stats()
        print(f"\n=== Summary ===")
        print(f"Total recordings: {stats['total_recordings']}")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Species: {len(stats['species_counts'])}")
        for species, count in sorted(stats['species_counts'].items(), key=lambda x: -x[1]):
            print(f"  {species}: {count}")

# === MAIN ===
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    load_stats()

    print("=" * 60)
    print("BIRDNET ADVANCED BIOACOUSTIC MONITOR")
    print("=" * 60)
    print(f"Rate: {RATE}Hz, Channels: {CHANNELS}, Duration: {RECORD_SECONDS}s")
    print(f"Threshold: {CONFIDENCE_THRESHOLD}, Output: {OUTPUT_DIR}")
    print("=" * 60)

    analyzer = Analyzer()
    nn_model = None
    if ENABLE_NEURAL_NETWORK:
        try:
            nn_model = load_or_create_model()
            print("Neural network loaded/created.")
        except Exception as e:
            print(f"NN init failed: {e}")

    continuous_monitoring(analyzer, nn_model)
