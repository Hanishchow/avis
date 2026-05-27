"""Generate MLP data: BirdNET on base segments, NN on all variants."""
import os, csv, glob, numpy as np
import librosa
import soundfile as sf
from birdnet_integration import BirdNETProcessor
from model_inference import BirdDetector
from config import (
    SAMPLE_RATE, N_SAMPLES, SEG_DIR, MLP_TRAINING_DATA,
    NOISE_PREFIXES_TUPLE, NORMALIZE_TARGET_MAX,
)

SR = SAMPLE_RATE

print("Loading models...", flush=True)
birdnet = BirdNETProcessor(min_conf=0.0)
nn = BirdDetector()

seg_dir = SEG_DIR
wav_files = sorted(glob.glob(os.path.join(seg_dir, "*.wav")))

def is_noise_file(fname):
    name = os.path.basename(fname)
    return any(name.startswith(p) for p in NOISE_PREFIXES_TUPLE)

def get_species(fname):
    name = os.path.basename(fname)
    parts = name.split("_")
    if is_noise_file(fname):
        return "noise"
    return "_".join(parts[:-1])

temp_path = "_mlp_temp.wav"
all_records = []
noise_birdnet_cache = [0.0, 0.0, 0.0]  # Most noise segments have BN=0

print(f"Processing {len(wav_files)} segment files...", flush=True)

# Phase 1: BirdNET for ALL bird segments (base + pitch variants)
# For noise segments, skip BirdNET (use cached zeros)
bird_birdnet_cache = {}  # species -> [base_logits, pitch2_logits, pitch-2_logits]

for wav_path in wav_files:
    species = get_species(wav_path)
    label = 0 if species == "noise" else 1

    audio, sr = librosa.load(wav_path, sr=SR, mono=True)
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    audio = audio[:N_SAMPLES]

    base_birdnet = list(noise_birdnet_cache) if label == 0 else None
    pitch2_birdnet = list(noise_birdnet_cache) if label == 0 else None
    pitch_neg2_birdnet = list(noise_birdnet_cache) if label == 0 else None

    if label == 1:
        sf.write(temp_path, audio, SR)
        detections = birdnet.analyze_file(temp_path)
        base_birdnet = list(birdnet.get_top3_logits(detections))
        print(f"  BN base {species:20s}: {base_birdnet}", flush=True)

        # Pitch +2
        aug = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
        sf.write(temp_path, aug, SR)
        detections = birdnet.analyze_file(temp_path)
        pitch2_birdnet = list(birdnet.get_top3_logits(detections))

        # Pitch -2
        aug = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2)
        sf.write(temp_path, aug, SR)
        detections = birdnet.analyze_file(temp_path)
        pitch_neg2_birdnet = list(birdnet.get_top3_logits(detections))

    nn_prob, _ = nn.predict(audio, SR)
    all_records.append(base_birdnet + [float(nn_prob), int(label), species])

    # Pitch variants
    for bn, n_steps in [(pitch2_birdnet, 2), (pitch_neg2_birdnet, -2)]:
        aug = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
        nn_prob, _ = nn.predict(aug, SR)
        all_records.append(bn + [float(nn_prob), int(label), species])

    # Time-stretch (NN only, reuse base BirdNET)
    aug = librosa.effects.time_stretch(audio, rate=0.8)
    if len(aug) >= N_SAMPLES:
        aug = aug[:N_SAMPLES]
        nn_prob, _ = nn.predict(aug, SR)
        all_records.append(base_birdnet + [float(nn_prob), int(label), species])

    # Noise-added (NN only)
    noise = np.random.randn(N_SAMPLES) * 0.1
    aug = audio + noise
    nn_prob, _ = nn.predict(aug, SR)
    all_records.append(base_birdnet + [float(nn_prob), int(label), species])

    # Pitch +4 (NN only)
    aug = librosa.effects.pitch_shift(audio, sr=sr, n_steps=4)
    nn_prob, _ = nn.predict(aug, SR)
    all_records.append(base_birdnet + [float(nn_prob), int(label), species])

if os.path.exists(temp_path):
    os.remove(temp_path)

with open(MLP_TRAINING_DATA, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["birdnet_1", "birdnet_2", "birdnet_3", "nn_prob", "label", "species"])
    writer.writerows(all_records)

bird_count = sum(1 for r in all_records if r[4] == 1)
noise_count = sum(1 for r in all_records if r[4] == 0)
print(f"\nMLP data saved: {MLP_TRAINING_DATA} ({len(all_records)} samples, {bird_count} bird, {noise_count} noise)", flush=True)

# Show unique vectors
unique_vecs = set(tuple(r[:4]) for r in all_records)
print(f"Unique feature vectors: {len(unique_vecs)}", flush=True)
