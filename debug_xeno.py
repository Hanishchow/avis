import requests
api_key = "dc05237b5fb65a8d3f6c19665aa4c0a2fa39a6f7"

# Query must use tags format for API v3
# Try different tag formats
queries = [
    {"query": "grp:birds", "key": api_key},
    {"query": "cnt:india", "key": api_key},
    {"query": "en:Cardinalis cardinalis", "key": api_key},
]

for params in queries:
    r = requests.get(
        "https://xeno-canto.org/api/3/recordings",
        params=params,
        timeout=15
    )
    print(f"Query: {params['query']} -> Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        num = int(data.get("numRecordings", 0))
        print(f"  Recordings: {num}")
        if num > 0:
            rec = data["recordings"][0]
            print(f"  First: ID={rec['id']}, Name={rec.get('en','?')}, File={rec.get('file','?')}")
            break
    else:
        print(f"  Body: {r.text[:200]}")
