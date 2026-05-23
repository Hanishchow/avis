# Advanced Bioacoustic Bird Detection System

An intelligent bird detection pipeline combining **BirdNET** identification with a custom **CNN neural network** via confidence stacking MLP, featuring multi-stage noise filtering and privacy-compliant speech obscuration.

## Architecture

```
Microphone → Noise Filtering Pipeline → BirdNET + Custom CNN
                                              ↓
                                    MLP Confidence Stacking
                                              ↓
                                    Detection Threshold (35%)
                                              ↓
                              WAV Saving / Spectrogram / Speech VAD
```

### Noise Filtering Pipeline
1. **High-pass filter** (300 Hz cutoff) – removes wind, traffic rumble
2. **Bandpass filter** (500–22000 Hz) – focuses on bird frequency range
3. **Multi-band adaptive Wiener filter** – attenuates wind/rain noise
4. **Spectral reduction** (via `noisereduce`) – broadband noise gate
5. **Notch filters** (4.5–9.5 kHz) – removes insect chorus (crickets, cicadas)
6. **Amplitude normalization** – ensures consistent signal levels

### Neural Network
- **Input**: Mel-spectrograms (128 bands, 3-second windows)
- **Architecture**: CNN (Conv2D layers + pooling + dense head)
- **Output**: Bird probability + per-species logits
- **Fallback**: Random-weight model if no trained weights found

### Ensemble Detection
- BirdNET provides top-3 species confidence scores
- Custom CNN provides bird/not-bird probability + logits
- Lightweight MLP combines these into final confidence
- Detection triggered when combined confidence ≥ 35%

### Privacy Compliance (GDPR/CCPA)
- Silero-VAD detects human speech in audio segments
- Segments with >40% speech probability are automatically discarded

## Files

| File | Purpose |
|------|---------|
| `birdnet_recorder_enhanced.py` | Main monitoring loop |
| `filtering_pipeline.py` | Multi-stage noise reduction |
| `feature_extraction.py` | Mel-spectrogram extraction |
| `bird_classifier_cnn.py` | CNN model definition |
| `train_cnn.py` | CNN training script |
| `fallback_model.py` | Fallback model loader |
| `model_inference.py` | Model inference wrapper |
| `birdnet_integration.py` | BirdNET library wrapper |
| `prepare_mlp_data.py` | MLP training data generator |
| `train_mlp.py` | MLP stacker training script |
| `ensemble_inference.py` | Combined detection + output |
| `evaluate_system.py` | Performance evaluation |
| `verify_env.py` | Environment dependency check |

## Configuration

Edit constants at the top of `birdnet_recorder_enhanced.py`:

```python
DEVICE_INDEX = 1          # Audio input device
SAMPLE_RATE = 44100       # Sampling rate
WINDOW_DURATION = 15      # Analysis window (seconds)
DETECTION_THRESHOLD = 0.35 # Confidence threshold
HIGH_PASS_CUTOFF = 300    # HP filter frequency
LOW_PASS_CUTOFF = 22000   # LP filter frequency
```

## Requirements

```
pip install -r requirements.txt
```

Requires Python 3.10+. See `verify_env.py` for dependency validation.

## Training

1. Prepare dataset: `python dataset_prepare.py`
2. Train CNN: `python train_cnn.py`
3. Generate MLP data: `python prepare_mlp_data.py`
4. Train MLP: `python train_mlp.py`

## Usage

```bash
python birdnet_recorder_enhanced.py
```

Press Ctrl+C to stop.

## Ethical Considerations

- **Do not deploy near active bird nests** during breeding season without appropriate permits.
- **Do not play audio to birds** – this system is listen-only.
- **Privacy compliance**: Audio segments containing human speech are automatically obscured per GDPR/CCPA guidelines.
- Ensure compliance with local wildlife recording regulations before field deployment.

## Expected Performance

- **Detection accuracy**: 80%+ for clear recordings (SNR > 20 dB)
- **False positive rate**: <0.5 false detections per hour in quiet conditions
- **Latency**: <5 seconds per 15-second audio window on modern CPU
- **F1 score**: ≥0.75 macro-averaged on temporal test set

## Data Sources

- **BirdNET**: Pretrained model covering ~10,000 bird species
- **Training datasets**: Xeno-Canto API (requires API key), BirdCLEF challenges
- **Synthetic fallback**: Built-in synthetic data generator for pipeline testing
