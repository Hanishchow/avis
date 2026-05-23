"""Prepare dataset with heavy data augmentation."""
import os, json, librosa, numpy as np
from sklearn.model_selection import train_test_split
from feature_extraction import extract_mel_spectrogram
import soundfile as sf

SR = 44100
DUR = 3
N_SAMPLES = SR * DUR

audio_dir = "real_dataset/audio"
seg_dir = "real_dataset/segments"
spec_dir = "real_dataset/spectrograms"
os.makedirs(seg_dir, exist_ok=True)
os.makedirs(spec_dir, exist_ok=True)

# Step 1: Segment all MP3s into 3s chunks
mp3_files = [f for f in os.listdir(audio_dir) if f.endswith(".mp3")]

all_segments = []
for mp3 in mp3_files:
    species_name = mp3.rsplit("_", 1)[0]
    filepath = os.path.join(audio_dir, mp3)
    try:
        audio, sr = librosa.load(filepath, sr=SR, mono=True, duration=60)
    except:
        continue
    for i in range(len(audio) // N_SAMPLES):
        seg = audio[i * N_SAMPLES : (i + 1) * N_SAMPLES]
        seg_name = f"{species_name}_{i:03d}.wav"
        seg_path = os.path.join(seg_dir, seg_name)
        sf.write(seg_path, seg, SR)
        all_segments.append({"species": species_name, "file": seg_path})

# Step 2: Heavily augmented bird segments (15x each)
print(f"Base bird segments: {len(all_segments)}")
augmented = []
for entry in all_segments:
    audio, sr = librosa.load(entry["file"], sr=SR, mono=True)
    if len(audio) < N_SAMPLES:
        continue
    audio = audio[:N_SAMPLES]
    
    # 1. Original
    augmented.append((audio, entry["species"]))
    
    # 2. Pitch shift +-2 semitones
    augmented.append((librosa.effects.pitch_shift(audio, sr=sr, n_steps=2), entry["species"]))
    augmented.append((librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2), entry["species"]))
    
    # 3. Time stretch 0.8x and 1.2x
    augmented.append((librosa.effects.time_stretch(audio, rate=0.8), entry["species"]))
    augmented.append((librosa.effects.time_stretch(audio, rate=1.2), entry["species"]))
    
    # 4. Add noise
    for level in [0.05, 0.1, 0.2]:
        noise = np.random.randn(len(audio)) * level
        augmented.append((audio + noise, entry["species"]))
    
    # 5. Frequency masking (simulate distant bird)
    filtered = audio.copy()
    filtered[int(len(filtered)*0.3):int(len(filtered)*0.7)] *= 0.3
    augmented.append((filtered, entry["species"]))
    
    # 6. Combined: pitch + noise
    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=1)
    augmented.append((shifted + np.random.randn(len(shifted))*0.1, entry["species"]))

print(f"Augmented + base bird segments: {len(augmented)}")

# Step 3: Synthetic noise (more variety)
num_noise = 120
for i in range(num_noise):
    noise_type = np.random.choice(["white", "pink", "brown", "wind", "crackle"])
    if noise_type == "white":
        noise = np.random.randn(N_SAMPLES) * 0.3
    elif noise_type == "pink":
        noise = np.cumsum(np.random.randn(N_SAMPLES)) / np.sqrt(N_SAMPLES) * 0.3
    elif noise_type == "brown":
        noise = np.cumsum(np.cumsum(np.random.randn(N_SAMPLES))) / N_SAMPLES * 0.3
    elif noise_type == "wind":
        wind = np.sin(np.linspace(0, 10*np.pi, N_SAMPLES)) * 0.3
        noise = np.random.randn(N_SAMPLES) * 0.1 + wind
    else:  # crackle
        noise = (np.random.rand(N_SAMPLES) > 0.995).astype(np.float32) * np.random.randn(N_SAMPLES) * 0.5
    noise = noise / max(np.abs(noise).max(), 1e-8) * 0.3
    augmented.append((noise.astype(np.float32), "noise"))

print(f"Total with noise: {len(augmented)}")

# Step 4: Extract spectrograms
dataset = []
for audio, species in augmented:
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    else:
        audio = audio[:N_SAMPLES]
    
    spec = extract_mel_spectrogram(audio, SR)
    if spec is None or spec.shape != (128, 259):
        print(f"  Bad spec shape: {spec.shape if spec is not None else None}")
        continue
    
    label = 0 if species == "noise" else 1
    fname = f"{species}_{np.random.randint(99999):05d}.npy"
    spec_path = os.path.join(spec_dir, fname)
    np.save(spec_path, spec.astype(np.float32))
    dataset.append({"spec": spec_path, "label": label, "species": species})

# Step 5: Split
labels = np.array([d["label"] for d in dataset])
tv, test = train_test_split(dataset, test_size=0.15, stratify=labels, random_state=42)
tv_labels = np.array([d["label"] for d in tv])
train, val = train_test_split(tv, test_size=0.1765, stratify=tv_labels, random_state=42)

splits = {"train": train, "val": val, "test": test}
for name, data in splits.items():
    with open(f"real_dataset/{name}.json", "w") as f:
        json.dump(data, f, indent=2)
    b = sum(1 for d in data if d["label"] == 1)
    n = sum(1 for d in data if d["label"] == 0)
    print(f"  {name}: {len(data)} ({b} bird, {n} noise)")
