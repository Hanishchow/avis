"""Generate MLP training data from real audio segments."""
import os, json, csv, numpy as np
import librosa
import soundfile as sf
from birdnet_integration import BirdNETProcessor
from model_inference import BirdDetector

SR = 44100
DUR = 3
N_SAMPLES = SR * DUR

print("Loading models...", flush=True)
birdnet = BirdNETProcessor(min_conf=0.0)
nn = BirdDetector()

# Use train + val splits (not test) to generate MLP data
all_records = []
for split in ["train", "val"]:
    path = f"real_dataset/{split}.json"
    with open(path) as f:
        data = json.load(f)
    for entry in data:
        # Load original audio from the .npy spectrogram source
        # We need the WAV, not the spectrogram. The spectrograms are extracted from
        # real_dataset/segments/*.wav but we don't have the mapping.
        # Instead, regenerate 3s audio for each spectrogram file
        spec = np.load(entry["spec"])
        label = entry["label"]
        species = entry["species"]
        
        # Reconstruct audio from the species info
        # Try to find the original segment file
        # The spectrogram name encodes the species
        fname = os.path.basename(entry["spec"]).replace(".npy", "")
        parts = fname.split("_")
        species = "_".join(parts[:-1])
        seg_id = parts[-1]
        seg_path = f"real_dataset/segments/{species}_{seg_id}.wav"
        
        if os.path.exists(seg_path):
            audio, sr = librosa.load(seg_path, sr=SR, mono=True)
        else:
            # Try to find any segment for this species
            seg_dir = "real_dataset/segments"
            candidates = [f for f in os.listdir(seg_dir) if f.startswith(species + "_") and f.endswith(".wav")]
            if candidates:
                audio, sr = librosa.load(os.path.join(seg_dir, candidates[0]), sr=SR, mono=True)
            else:
                # Skip - no audio available
                continue
        
        if len(audio) < N_SAMPLES:
            audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
        audio = audio[:N_SAMPLES]
        
        # Write temp WAV for BirdNET
        temp_path = "_mlp_temp.wav"
        sf.write(temp_path, audio, SR)
        detections = birdnet.analyze_file(temp_path)
        birdnet_logits = birdnet.get_top3_logits(detections)
        nn_prob, _ = nn.predict(audio, SR)
        
        features = [float(birdnet_logits[0]), float(birdnet_logits[1]), float(birdnet_logits[2]), float(nn_prob)]
        all_records.append(features + [int(label), species])
        print(f"  {split}: {species} birdnet={birdnet_logits} nn={nn_prob:.3f} label={label}", flush=True)

if os.path.exists(temp_path):
    os.remove(temp_path)

csv_path = "real_dataset/mlp_training_data.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["birdnet_1", "birdnet_2", "birdnet_3", "nn_prob", "label", "species"])
    writer.writerows(all_records)

print(f"\nMLP data saved: {csv_path} ({len(all_records)} samples)", flush=True)
bird_count = sum(1 for r in all_records if r[4] == 1)
noise_count = sum(1 for r in all_records if r[4] == 0)
print(f"  Bird: {bird_count}, Noise: {noise_count}", flush=True)
