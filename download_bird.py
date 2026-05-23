"""Download multiple bird species from Xeno-Canto for evaluation."""
import requests
import os

API_KEY = "dc05237b5fb65a8d3f6c19665aa4c0a2fa39a6f7"
OUT_DIR = "bird_dataset"
os.makedirs(OUT_DIR, exist_ok=True)

species = [
    "en:Acridotheres tristis",
    "en:Corvus splendens",
    "en:Pycnonotus cafer",
    "en:Passer domesticus",
    "en:Columba livia",
    "en:Copsychus saularis",
    "en:Prinia socialis",
    "en:Dicrurus macrocercus",
]

for sp in species:
    params = {"query": sp, "key": API_KEY, "page": "1", "len": "30"}
    r = requests.get("https://xeno-canto.org/api/3/recordings", params=params, timeout=15)
    data = r.json()
    num = int(data.get("numRecordings", 0))
    en_name = sp.split(":")[1].replace(" ", "_")
    print(f"\n{sp}: {num} recordings")
    count = 0
    for rec in data.get("recordings", []):
        if count >= 2:
            break
        file_url = rec["file"]
        quality = rec.get("q", "?")
        filename = f"{OUT_DIR}/{en_name}_{rec['id']}.mp3"
        if not os.path.exists(filename):
            try:
                audio = requests.get(file_url, timeout=30)
                with open(filename, "wb") as f:
                    f.write(audio.content)
                print(f"  Downloaded {rec['id']} (qual={quality}, {len(audio.content)} bytes)")
                count += 1
            except Exception as e:
                print(f"  Failed {rec['id']}: {e}")
        else:
            print(f"  Skipped {rec['id']} (exists)")
            count += 1

print(f"\nDone. Files in {OUT_DIR}/")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f} ({os.path.getsize(os.path.join(OUT_DIR,f))//1024}KB)")
