import pyaudio
import wave
import numpy as np
import os
from datetime import datetime
from scipy.signal import butter, lfilter, sosfilt, sosfilt_zi
import tensorflow as tf
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

# --- CONFIGURATION ---
INDEX = 1
RATE = 44100  # Recommended to keep at 44.1kHz for standard hardware compatibility
CHANNELS = 2
RECORD_SECONDS = 15 
TEMP_FILENAME = "temp_capture.wav" # Temporary file for AI to read
MODEL_PATH = "bird_sound_classifier.h5"  # Path to our simple neural network model
CONFIDENCE_THRESHOLD = 0.35
LATITUDE = 12.97  # Bangalore latitude
LONGITUDE = 77.59  # Bangalore longitude

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Create a bandpass filter for bird vocalizations (typically 1-8 kHz)"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos', analog=False)
    return sos

def butter_highpass(cutoff, fs, order=5):
    """High-pass filter to remove low-frequency noise (like fan noise)"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='high', output='sos', analog=False)
    return sos

def butter_lowpass(cutoff, fs, order=5):
    """Low-pass filter to remove high-frequency noise"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='low', output='sos', analog=False)
    return sos

def apply_advanced_filter(data, fs=RATE):
    """Apply multiple filters for noise reduction"""
    # Remove low-frequency fan noise (< 100 Hz)
    sos_high = butter_highpass(100, fs, order=4)
    zi_high = sosfilt_zi(sos_high)
    filtered_data, _ = sosfilt(sos_high, data, zi=zi_high * data[0])
    
    # Apply bandpass for bird vocalizations (1-8 kHz) 
    sos_band = butter_bandpass(1000, 8000, fs, order=4)
    zi_band = sosfilt_zi(sos_band)
    filtered_data, _ = sosfilt(sos_band, filtered_data, zi=zi_band * filtered_data[0])
    
    # Remove very high frequency noise (>12 kHz)
    sos_low = butter_lowpass(12000, fs, order=4)
    zi_low = sosfilt_zi(sos_low)
    filtered_data, _ = sosfilt(sos_low, filtered_data, zi=zi_low * filtered_data[0])
    
    return filtered_data.astype(np.int16)

def extract_features(signal, sample_rate=RATE, n_mfCC=13):
    """Extract MFCC features for neural network input"""
    from python_speech_features import mfcc
    # Convert to mono if stereo
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)
    
    # Normalize signal
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))
    
    # Extract MFCC features
    mfcc_features = mfcc(signal, samplerate=sample_rate, numcep=n_mfCC)
    
    # Take statistical features (mean, std, max, min) of each MFCC coefficient
    features = np.hstack([
        np.mean(mfcc_features, axis=0),
        np.std(mfcc_features, axis=0),
        np.max(mfcc_features, axis=0),
        np.min(mfcc_features, axis=0)
    ])
    
    return features.reshape(1, -1)  # Reshape for model prediction

def create_simple_nn_model(input_shape=(52,)):
    """Create a simple neural network for bird sound classification"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification: bird vs noise
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_or_create_model(model_path=MODEL_PATH):
    """Load existing model or create a new one"""
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        return tf.keras.models.load_model(model_path)
    else:
        print("Creating new simple neural network model")
        model = create_simple_nn_model()
        # Save untrained model for now - in practice, you'd train it with bird sound data
        model.save(model_path)
        return model

def record_and_analyze(analyzer, nn_model):
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE,
                        input=True, input_device_index=INDEX, frames_per_buffer=8192)
        
        print(f"\n[LISTENING] {RECORD_SECONDS}s window active...")
        frames = []
        for _ in range(0, int(RATE / 8192 * RECORD_SECONDS)):
            data = stream.read(8192, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.int16))
        
        stream.stop_stream()
        stream.close()
    except Exception as e:
        print(f"Error accessing microphone: {e}")
        return
    finally:
        p.terminate()

    # --- SIGNAL PROCESSING ---
    full_signal = np.concatenate(frames)
    
    # Apply advanced filtering
    filtered_signal = apply_advanced_filter(full_signal, fs=RATE)
    
    # Normalization for cleaner AI read
    max_val = np.max(np.abs(filtered_signal))
    if max_val > 0:
        normalized_signal = (filtered_signal / max_val * 32767).astype(np.int16)
    else:
        normalized_signal = filtered_signal

    # Save to temporary file for BirdNET to analyze
    with wave.open(TEMP_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(normalized_signal.tobytes())

    # --- FEATURE EXTRACTION FOR NEURAL NETWORK ---
    # Extract features for NN prediction
    try:
        features = extract_features(normalized_signal.astype(np.float32))
        nn_prediction = nn_model.predict(features, verbose=0)[0][0]
        print(f"NN Confidence (bird vs noise): {nn_prediction:.3f}")
    except Exception as e:
        print(f"Feature extraction error: {e}")
        nn_prediction = 0.5  # Default to uncertain if feature extraction fails

    # --- AI ANALYSIS WITH BIRDNET ---
    try:
        recording = Recording(
            analyzer, 
            TEMP_FILENAME, 
            lat=LATITUDE, 
            lon=LONGITUDE, 
            min_conf=CONFIDENCE_THRESHOLD
        )
        recording.analyze()
        
        if recording.detections and nn_prediction > 0.4:  # Both BirdNET and NN agree
            # Get the highest confidence detection for the filename
            best_detection = recording.detections[0]
            bird_name = best_detection['common_name'].replace(" ", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create a unique filename: e.g., "Common_Myna_20260513_134500.wav"
            new_filename = f"{bird_name}_{timestamp}.wav"
            
            # Save the finalized recording with the new name
            with wave.open(new_filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(RATE)
                wf.writeframes(normalized_signal.tobytes())
                
            print(f"✅ DETECTED: {best_detection['common_name']} "
                  f"(BirdNET: {best_detection['confidence']*100:.1f}%, "
                  f"NN: {nn_prediction*100:.1f}%)")
            print(f"💾 SAVED AS: {new_filename}")
        else:
            print("❌ No confident detection. Deleting temporary audio...")
            if os.path.exists(TEMP_FILENAME):
                os.remove(TEMP_FILENAME)
                
    except Exception as e:
        print(f"BirdNET analysis error: {e}")
        # Clean up temp file on error
        if os.path.exists(TEMP_FILENAME):
            os.remove(TEMP_FILENAME)

def main():
    print("Loading BirdNET Analyzer...")
    analyzer = Analyzer()
    
    print("Loading/Creating Neural Network Model...")
    nn_model = load_or_create_model()
    
    print("Starting bioacoustic monitoring...")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            record_and_analyze(analyzer, nn_model)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

if __name__ == "__main__":
    main()