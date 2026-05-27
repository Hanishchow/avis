"""Prepare dataset with heavy data augmentation AND real noise."""
import os, json, librosa, numpy as np
from sklearn.model_selection import train_test_split
from feature_extraction import extract_mel_spectrogram
import soundfile as sf
from config import (
    SAMPLE_RATE, DURATION, N_SAMPLES, AUDIO_DIR, SEG_DIR, SPEC_DIR,
    NOISE_PREFIXES, SPEC_H, SPEC_W, NUM_SYNTHETIC_NOISE, SYNTHETIC_NOISE_TYPES,
    NOISE_AMPLITUDE, NOISE_NORMALIZE_CEILING, DATASET_TEST_SPLIT,
    DATASET_VAL_SPLIT, DATASET_RANDOM_STATE, LOAD_MAX_DURATION, N_MELS,
)

SR = SAMPLE_RATE
DUR = int(DURATION)
N_SAMPLES = SR * DUR

audio_dir = AUDIO_DIR
seg_dir = SEG_DIR
spec_dir = SPEC_DIR
os.makedirs(seg_dir, exist_ok=True)
os.makedirs(spec_dir, exist_ok=True)

# Step 1: Segment all MP3s into 3s chunks
mp3_files = [f for f in os.listdir(audio_dir) if f.endswith(".mp3") and not f.endswith(".tmp")]

all_bird_segments = []
all_noise_segments = []
for mp3 in mp3_files:
    species_name = mp3.rsplit("_", 1)[0]
    is_noise = any(species_name.startswith(p) for p in NOISE_PREFIXES)
    filepath = os.path.join(audio_dir, mp3)
    try:
        audio, sr = librosa.load(filepath, sr=SR, mono=True, duration=LOAD_MAX_DURATION)
    except:
        continue
    for i in range(len(audio) // N_SAMPLES):
        seg = audio[i * N_SAMPLES : (i + 1) * N_SAMPLES]
        seg_name = f"{species_name}_{i:03d}.wav"
        seg_path = os.path.join(seg_dir, seg_name)
        sf.write(seg_path, seg, SR)
        if is_noise:
            all_noise_segments.append({"species": "noise", "file": seg_path})
        else:
            all_bird_segments.append({"species": species_name, "file": seg_path})

print(f"Bird segments: {len(all_bird_segments)}, Noise segments: {len(all_noise_segments)}")

# Step 2: Heavily augmented bird segments (15x each)
augmented = []
for entry in all_bird_segments:
    audio, sr = librosa.load(entry["file"], sr=SR, mono=True)
    if len(audio) < N_SAMPLES:
        continue
    audio = audio[:N_SAMPLES]
    augmented.append((audio, entry["species"]))
    augmented.append((librosa.effects.pitch_shift(audio, sr=sr, n_steps=2), entry["species"]))
    augmented.append((librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2), entry["species"]))
    augmented.append((librosa.effects.time_stretch(audio, rate=0.8), entry["species"]))
    augmented.append((librosa.effects.time_stretch(audio, rate=1.2), entry["species"]))
    for level in [0.05, 0.1, 0.2]:
        noise = np.random.randn(len(audio)) * level
        augmented.append((audio + noise, entry["species"]))
    filtered = audio.copy()
    filtered[int(len(filtered)*0.3):int(len(filtered)*0.7)] *= 0.3
    augmented.append((filtered, entry["species"]))
    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=1)
    augmented.append((shifted + np.random.randn(len(shifted))*0.1, entry["species"]))

print(f"Augmented bird segments: {len(augmented)}")

# Step 3: Real noise segments (no augmentation — already diverse)
for entry in all_noise_segments:
    audio, sr = librosa.load(entry["file"], sr=SR, mono=True)
    if len(audio) >= N_SAMPLES:
        augmented.append((audio[:N_SAMPLES], "noise"))

# Step 4: Synthetic noise for variety
for i in range(NUM_SYNTHETIC_NOISE):
    noise_type = np.random.choice(SYNTHETIC_NOISE_TYPES)
    if noise_type == "white":
        noise = np.random.randn(N_SAMPLES) * NOISE_AMPLITUDE
    elif noise_type == "pink":
        noise = np.cumsum(np.random.randn(N_SAMPLES)) / np.sqrt(N_SAMPLES) * NOISE_AMPLITUDE
    elif noise_type == "brown":
        noise = np.cumsum(np.cumsum(np.random.randn(N_SAMPLES))) / N_SAMPLES * NOISE_AMPLITUDE
    elif noise_type == "wind":
        wind = np.sin(np.linspace(0, 10*np.pi, N_SAMPLES)) * NOISE_AMPLITUDE
        noise = np.random.randn(N_SAMPLES) * 0.1 + wind
    else:
        noise = (np.random.rand(N_SAMPLES) > 0.995).astype(np.float32) * np.random.randn(N_SAMPLES) * 0.5
    noise = noise / max(np.abs(noise).max(), 1e-8) * NOISE_NORMALIZE_CEILING
    augmented.append((noise.astype(np.float32), "noise"))

print(f"Total augmented: {len(augmented)}")

# Step 4: Extract spectrograms
dataset = []
for audio, species in augmented:
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    else:
        audio = audio[:N_SAMPLES]
    
    spec = extract_mel_spectrogram(audio, SR)
    if spec is None or spec.shape != (SPEC_H, SPEC_W):
        continue
    
    label = 0 if species == "noise" else 1
    fname = f"{species}_{np.random.randint(99999):05d}.npy"
    spec_path = os.path.join(spec_dir, fname)
    np.save(spec_path, spec.astype(np.float32))
    dataset.append({"spec": spec_path, "label": label, "species": species})

# Step 5: Split
labels = np.array([d["label"] for d in dataset])
tv, test = train_test_split(dataset, test_size=DATASET_TEST_SPLIT, stratify=labels, random_state=DATASET_RANDOM_STATE)
tv_labels = np.array([d["label"] for d in tv])
train, val = train_test_split(tv, test_size=DATASET_VAL_SPLIT, stratify=tv_labels, random_state=DATASET_RANDOM_STATE)

splits = {"train": train, "val": val, "test": test}
for name, data in splits.items():
    with open(f"real_dataset/{name}.json", "w") as f:
        json.dump(data, f, indent=2)
    b = sum(1 for d in data if d["label"] == 1)
    n = sum(1 for d in data if d["label"] == 0)
    print(f"  {name}: {len(data)} ({b} bird, {n} noise)")
