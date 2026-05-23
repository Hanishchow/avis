"""Download more bird species - increased size limit."""
import requests
import os

API_KEY = "dc05237b5fb65a8d3f6c19665aa4c0a2fa39a6f7"
OUT = "real_dataset/audio"
os.makedirs(OUT, exist_ok=True)

species_queries = [
    ("Corvus", "splendens"),
    ("Pycnonotus", "cafer"),
    ("Columba", "livia"),
    ("Copsychus", "saularis"),
    ("Prinia", "socialis"),
    ("Dicrurus", "macrocercus"),
]

for genus, sp in species_queries:
    en_name = f"{genus}_{sp}"
    existing = len([f for f in os.listdir(OUT) if f.startswith(en_name) and f.endswith(".mp3")])
    if existing >= 5:
        print(f"{en_name}: {existing} files, skip", flush=True)
        continue

    r = requests.get("https://xeno-canto.org/api/3/recordings",
        params={"query": f"gen:{genus} sp:{sp}", "key": API_KEY, "page": "1", "len": "25"},
        timeout=15)
    data = r.json()
    print(f"\n{en_name}: {data.get('numRecordings', 0)} available", flush=True)

    count = existing
    for rec in data.get("recordings", []):
        if count >= 5:
            break
        rid = rec["id"]
        quality = rec.get("q", "?")
        dur = rec.get("length", "0:00")
        filename = f"{OUT}/{en_name}_{rid}.mp3"
        if os.path.exists(filename):
            count += 1
            continue
        try:
            r2 = requests.get(rec["file"], timeout=15)
            if len(r2.content) > 8_000_000:
                print(f"  SKIP {rid}: {dur}, {len(r2.content)//1024}KB", flush=True)
                continue
            with open(filename, "wb") as f:
                f.write(r2.content)
            print(f"  [{count+1}/5] {rid} ({dur}, q={quality}, {len(r2.content)//1024}KB)", flush=True)
            count += 1
        except Exception as e:
            print(f"  FAIL {rid}: {e}", flush=True)

print(f"\nFinal files:", flush=True)
for f in sorted(os.listdir(OUT)):
    print(f"  {f} ({os.path.getsize(os.path.join(OUT,f))//1024}KB)", flush=True)
