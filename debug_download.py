import requests
import sys

API_KEY = "dc05237b5fb65a8d3f6c19665aa4c0a2fa39a6f7"
print("Testing API query...", flush=True)

r = requests.get("https://xeno-canto.org/api/3/recordings",
    params={"query": "gen:Corvus sp:splendens", "key": API_KEY, "page": "1", "len": "5"},
    timeout=10)
print(f"Status: {r.status_code}", flush=True)
d = r.json()
print(f"Recordings: {d.get('numRecordings', 0)}", flush=True)
for rec in d.get("recordings", [])[:3]:
    print(f"  {rec['id']}: {rec.get('en','?')} len={rec.get('length','?')} q={rec.get('q','?')}", flush=True)
    file_url = rec["file"]
    print(f"  Download URL: {file_url}", flush=True)
    try:
        t0 = __import__("time").time()
        audio_r = requests.get(file_url, timeout=10, stream=True)
        size = 0
        for chunk in audio_r.iter_content(8192):
            if chunk:
                size += len(chunk)
            if size > 500000:
                break
        elapsed = __import__("time").time() - t0
        print(f"  Downloaded {size//1024}KB in {elapsed:.1f}s", flush=True)
    except Exception as e:
        print(f"  Download failed: {e}", flush=True)
        break
