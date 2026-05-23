import requests
key = "dc05237b5fb65a8d3f6c19665aa4c0a2fa39a6f7"
tests = [
    "gen:Acridotheres sp:tristis",
    "gen:Acridotheres sp:grandis",
    "gen:Passer sp:domesticus",
    "gen:Corvus sp:splendens",
    "gen:Pycnonotus sp:cafer",
    "gen:Columba sp:livia",
    "gen:Copsychus sp:saularis",
    "gen:Prinia sp:socialis",
    "gen:Dicrurus sp:macrocercus",
    "gen:Cardinalis sp:cardinalis",
    "gen:Prinia",
    "q:Common Myna type:species",
]
for q in tests:
    r = requests.get("https://xeno-canto.org/api/3/recordings", params={"query": q, "key": key}, timeout=15)
    data = r.json()
    n = int(data.get("numRecordings", 0))
    status = "OK" if r.status_code == 200 else f"ERR {r.status_code}"
    print(f"[{status}] {q:45s} -> {n} recordings")
    if n > 0:
        for rec in data["recordings"][:1]:
            print(f"  {rec.get('en','?')} ({rec['id']})")
