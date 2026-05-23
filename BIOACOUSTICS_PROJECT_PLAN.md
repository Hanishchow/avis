# Advanced Bioacoustic Bird Detection System - Project Plan

## Overview
This project implements an advanced real-time bird detection system using audio processing, machine learning, and the BirdNET library. The system continuously monitors audio input, applies sophisticated noise reduction techniques, and uses both traditional AI (BirdNET) and neural network approaches for accurate bird species identification.

## Project Components

### 1. Core Functionality (`birdnet_recorder.py`)
- Real-time audio capture from microphone
- Basic high-pass filtering to remove low-frequency noise
- BirdNET analysis for bird detection
- Automatic saving of detected bird calls with timestamped filenames

### 2. Enhanced System (`birdnet_recorder_enhanced.py`)
- Advanced multi-stage filtering pipeline:
  - High-pass filter (50Hz) to remove rumble/wind noise
  - Bandpass filter (1-8 kHz) focused on bird vocalization range
  - Spectral noise reduction using noisereduce library
- Neural network integration for improved detection confidence
- Combined decision making (70% BirdNET, 30% neural network)
- Spectrogram visualization for debugging and analysis
- Model persistence for neural network weights

### 3. Alternative Implementation (`bird_detector_enhanced.py`)
- Additional filtering approaches
- Alternative neural network architectures
- Different feature extraction methods

## Technical Specifications

### Audio Processing Pipeline
1. **Input**: 16-bit PCM audio at 44.1kHz, stereo/mono configurable
2. **Pre-processing**:
   - High-pass filtering (>50Hz) to eliminate low-frequency noise
   - Bandpass filtering (1000-8000Hz) to focus on bird vocalization range
   - Optional spectral noise reduction
3. **Normalization**: Amplitude normalization to maximize dynamic range
4. **Analysis Pipeline**:
   - Traditional: BirdNET library for species identification
   - Neural Network: Custom CNN for bird/not-bird classification
   - Fusion: Weighted combination of both approaches

### Neural Network Architecture
- Input: Mel-spectrogram features (128 mel bands, temporal frames)
- Layers:
  - Conv2D (16 filters, 3x3) + MaxPooling
  - Conv2D (32 filters, 3x3) + MaxPooling
  - Flatten
  - Dense (64 units, ReLU) + Dropout (0.5)
  - Dense (1 unit, Sigmoid) for binary classification
- Training: Binary cross-entropy loss, Adam optimizer

### Detection Logic
- BirdNET provides species identification and confidence
- Neural network provides bird/not-bird probability
- Combined confidence = 0.7 × BirdNET_confidence + 0.3 × NN_confidence
- Detection threshold: 35% combined confidence
- Output: Saved WAV file with species name and timestamp

## Installation Requirements

### Core Dependencies
```bash
pip install pyaudio wave numpy scipy birdnetlib
```

### Enhanced Dependencies
```bash
pip install pyaudio numpy scipy librosa noisereduce tensorflow birdnetlib matplotlib
```

### System Requirements
- Python 3.7+
- Working microphone (audio input device)
- Sufficient RAM (>4GB recommended) for neural network processing
- Optional: GPU for accelerated neural network inference

## Configuration Parameters

### Audio Settings
- `INDEX`: Audio input device index (default: 1)
- `RATE`: Sample rate in Hz (default: 44100)
- `CHANNELS`: Audio channels (1 for mono, 2 for stereo)
- `RECORD_SECONDS`: Analysis window duration (default: 15s)

### Filtering Parameters
- `LOW_CUT`: High-pass filter cutoff (default: 1000 Hz)
- `HIGH_CUT`: Low-pass filter cutoff (default: 8000 Hz)
- `NOISE_REDUCE_STRENGTH`: Spectral subtraction strength (0-1, default: 0.8)

### Detection Thresholds
- `min_conf`: BirdNET minimum confidence (default: 0.35)
- Neural network threshold: Integrated into combined confidence
- Final detection threshold: 0.35 (35%) combined confidence

## File Output
- Detected calls saved as: `{SPECIES_NAME}_{TIMESTAMP}.wav`
- Timestamp format: YYYYMMDD_HHMMSS
- Example: `American_Robin_20260516_143022.wav`
- Optional spectrograms: `{SPECIES_NAME}_{TIMESTAMP}_spec.png`

## Usage Instructions

### Basic Usage
```bash
python birdnet_recorder.py
```

### Enhanced Usage
```bash
python birdnet_recorder_enhanced.py
```

### Controls
- Press `Ctrl+C` to stop monitoring gracefully
- System will save neural network model on exit if newly created

## Project Phases

### Phase 1: Foundation (Completed)
- Basic audio capture and BirdNET integration
- Simple noise filtering
- Automatic file saving

### Phase 2: Enhancement (Current)
- Advanced multi-stage filtering
- Neural network integration
- Combined decision making
- Spectrogram visualization

### Phase 3: Expansion (Future)
- Training pipeline for neural network with labeled data
- Species-specific neural networks
- Real-time web dashboard for monitoring
- Integration with birding APIs (eBird, Xeno-Canto)
- Power optimization for field deployment
- Weatherproof enclosure designs

## Research Applications
- Citizen science bird monitoring
- Ecological research and biodiversity assessment
- Urban wildlife studies
- Migration pattern analysis
- Habitat quality indicator species tracking
- Nocturnal flight call monitoring

## Limitations and Considerations
- Environmental noise (wind, rain, traffic) affects accuracy
- Simultaneous bird calls may confuse identification
- Microphone quality significantly impacts results
- Neural network requires training data for species-specific accuracy
- Battery life considerations for field deployment
- Legal considerations for wildlife recording in some jurisdictions

## Future Enhancements
1. **Transfer Learning**: Use pretrained audio models (YAMNet, VGGish)
2. **Ensemble Methods**: Multiple model voting for increased robustness
3. **Temporal Analysis**: Track bird presence over extended periods
4. **Geolocation Integration**: GPS tagging of recordings
5. **Cloud Integration**: Automatic upload to research databases
6. **Power Optimization**: Duty cycling, low-power hardware options
7. **Species Expansion**: Region-specific model training
8. **Real-time Feedback**: LED/audio indicators for detection events

## Maintenance
- Regularly update BirdNET library for improved models
- Retrain neural network with new field data
- Monitor false positive/negative rates for threshold tuning
- Keep dependencies updated for security and performance
- Backup recorded data regularly for research purposes

## Safety and Ethics
- Minimize disturbance to wildlife during recording
- Follow local regulations regarding wildlife monitoring
- Share data responsibly with scientific communities
- Consider data privacy if recording in public spaces