import requests, os
key = "dc05237b5fb65a8d3f6c19665aa4c0a2fa39a6f7"
OUT = "real_dataset/audio"
os.makedirs(OUT, exist_ok=True)

# Pick species where we have <5 files
needed = []
existing = {}
for f in os.listdir(OUT):
    prefix = f.rsplit("_", 1)[0]
    existing[prefix] = existing.get(prefix, 0) + 1

genus_sp = [
    ("Pycnonotus", "cafer"),
    ("Columba", "livia"),
    ("Copsychus", "saularis"),
    ("Prinia", "socialis"),
    ("Dicrurus", "macrocercus"),
]

for genus, sp in genus_sp:
    en = f"{genus}_{sp}"
    if existing.get(en, 0) >= 3:
        print(f"{en}: have {existing[en]}, ok", flush=True)
    else:
        needed.append((genus, sp, en))

if not needed:
    print("All species have enough files!", flush=True)
else:
    genus, sp, en = needed[0]
    print(f"Downloading {en}... (need {3 - existing.get(en, 0)} more)", flush=True)
    r = requests.get("https://xeno-canto.org/api/3/recordings",
        params={"query": f"gen:{genus} sp:{sp}", "key": key, "page": "1", "len": "15"},
        timeout=15)
    data = r.json()
    count = existing.get(en, 0)
    for rec in data.get("recordings", []):
        if count >= 3:
            break
        rid = rec["id"]
        dur = rec.get("length", "0:00")
        quality = rec.get("q", "?")
        path = os.path.join(OUT, f"{en}_{rid}.mp3")
        if os.path.exists(path):
            count += 1
            continue
        try:
            r2 = requests.get(rec["file"], timeout=15)
            if len(r2.content) > 8_000_000:
                print(f"  SKIP {rid}: {dur}, {len(r2.content)//1024}KB", flush=True)
                continue
            with open(path, "wb") as f:
                f.write(r2.content)
            print(f"  GOT {rid}: {dur} q={quality} {len(r2.content)//1024}KB", flush=True)
            count += 1
        except Exception as e:
            print(f"  FAIL {rid}: {e}", flush=True)

print("Done", flush=True)
for f in sorted(os.listdir(OUT)):
    print(f"  {f} ({os.path.getsize(os.path.join(OUT,f))//1024}KB)", flush=True)
