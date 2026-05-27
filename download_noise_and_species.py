"""Download real environmental noise + new bird species from Xeno-Canto."""
import requests, os, time, json
from config import XENO_CANTO_API_KEY, AUDIO_DIR, DOWNLOAD_TIMEOUT, DOWNLOAD_CHUNK_SIZE, DOWNLOAD_SLEEP, NOISE_QUERIES, BIRD_QUERIES, XENO_CANTO_PAGE_SIZE

os.makedirs(AUDIO_DIR, exist_ok=True)

QUERIES = {
    "noise": NOISE_QUERIES,
    "birds": BIRD_QUERIES,
}

def download_file(url, path, timeout=None):
    if timeout is None:
        timeout = DOWNLOAD_TIMEOUT
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
        return os.path.getsize(path)
    except Exception as e:
        if os.path.exists(path):
            os.remove(path)
        print(f"    FAILED: {e}")
        return None

def search_and_download(category, query, label, max_files=3):
    print(f"\n[{category}] {label} (query: {query})")
    params = {"query": query, "key": XENO_CANTO_API_KEY, "page": "1", "len": XENO_CANTO_PAGE_SIZE}
    try:
        r = requests.get("https://xeno-canto.org/api/3/recordings", params=params, timeout=15)
        data = r.json()
    except Exception as e:
        print(f"  Search FAILED: {e}")
        return []
    num = int(data.get("numRecordings", 0))
    print(f"  Found {num} recordings")
    count = 0
    downloaded = []
    for rec in data.get("recordings", []):
        if count >= max_files:
            break
        file_url = rec.get("file")
        if not file_url or file_url == "None" or not file_url.startswith("http"):
            continue
        rid = rec["id"]
        quality = rec.get("q", "?")
        filename = f"{AUDIO_DIR}/{label}_{rid}.mp3"
        filepath = filename
        if os.path.exists(filepath):
            print(f"  [{count+1}/{max_files}] {rid} (q={quality}) — already exists")
            count += 1
            downloaded.append(filepath)
            continue
        print(f"  [{count+1}/{max_files}] {rid} (q={quality})...", end=" ", flush=True)
        size = download_file(file_url, filepath)
        if size:
            print(f"{size//1024}KB")
            count += 1
            downloaded.append(filepath)
        time.sleep(DOWNLOAD_SLEEP)
    return downloaded

all_results = {}
for category, items in QUERIES.items():
    all_results[category] = []
    for query, label in items:
        files = search_and_download(category, query, label, max_files=3)
        all_results[category].extend(files)

print(f"\n=== Download Summary ===")
print(f"Noise files: {len(all_results.get('noise', []))}")
print(f"Bird files:  {len(all_results.get('birds', []))}")
for f in sorted(os.listdir(AUDIO_DIR)):
    sz = os.path.getsize(os.path.join(AUDIO_DIR, f)) // 1024
    print(f"  {f} ({sz}KB)")
