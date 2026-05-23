"""Download multiple species from Xeno-Canto (with robust timeouts)."""
import requests
import os
import json
import time

API_KEY = "dc05237b5fb65a8d3f6c19665aa4c0a2fa39a6f7"

species = [
    ("Acridotheres", "tristis", "Common_Myna"),
    ("Passer", "domesticus", "House_Sparrow"),
    ("Corvus", "splendens", "House_Crow"),
    ("Pycnonotus", "cafer", "Red-vented_Bulbul"),
    ("Columba", "livia", "Rock_Dove"),
    ("Copsychus", "saularis", "Oriental_Magpie-Robin"),
    ("Prinia", "socialis", "Ashy_Prinia"),
    ("Dicrurus", "macrocercus", "Black_Drongo"),
]

OUT_DIR = "real_dataset"
os.makedirs(f"{OUT_DIR}/audio", exist_ok=True)

def download_file(url, path, timeout=30):
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return os.path.getsize(path)
    except Exception as e:
        if os.path.exists(path):
            os.remove(path)
        return None

all_meta = []
existing = set()
if os.path.exists(f"{OUT_DIR}/metadata/manifest.json"):
    with open(f"{OUT_DIR}/metadata/manifest.json") as f:
        all_meta = json.load(f)
    existing = {e["file"] for e in all_meta}

for genus, sp, en_name in species:
    count_existing = sum(1 for e in all_meta if e["species"] == en_name)
    if count_existing >= 10:
        print(f"{en_name}: already have {count_existing}, skipping")
        continue

    print(f"\n{en_name} ({genus} {sp}): need {10-count_existing} more...")
    params = {"query": f"gen:{genus} sp:{sp}", "key": API_KEY, "page": "1", "len": "40"}
    try:
        r = requests.get("https://xeno-canto.org/api/3/recordings", params=params, timeout=15)
        data = r.json()
    except Exception as e:
        print(f"  Search failed: {e}")
        continue

    count = count_existing
    for rec in data.get("recordings", []):
        if count >= 10:
            break
        file_url = rec["file"]
        rid = rec["id"]
        quality = rec.get("q", "?")
        filename = f"{OUT_DIR}/audio/{en_name}_{rid}.mp3"
        filepath = filename

        if filepath in existing:
            count += 1
            continue

        print(f"  Downloading [{count+1}/10] {rid} (q={quality})...", end=" ", flush=True)
        size = download_file(file_url, filepath, timeout=45)
        if size:
            print(f"{size//1024}KB")
            entry = {"species": en_name, "id": rid, "file": filepath, "quality": quality}
            all_meta.append(entry)
            existing.add(filepath)
            count += 1
        else:
            print("FAILED (timeout/error)")
        time.sleep(1)  # be polite

with open(f"{OUT_DIR}/metadata/manifest.json", "w") as f:
    json.dump(all_meta, f, indent=2)

bird_count = sum(1 for e in all_meta if e["species"] != "noise")
noise_count = sum(1 for e in all_meta if e["species"] == "noise")
print(f"\nDone: {bird_count} bird recordings, {noise_count} noise recordings")
for e in all_meta:
    print(f"  {e['species']:30s} {e['id']}")
