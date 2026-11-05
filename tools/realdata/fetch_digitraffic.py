"""Download long-distance train runs from Fintraffic Digitraffic (CC BY 4.0).

Days: the 8th and 22nd of every month, January 2024 to December 2025 (48 days).
Endpoint: https://rata.digitraffic.fi/api/v1/trains/<date>
Keeps only trainCategory == "Long-distance" and stores each day gzipped.
"""
import gzip
import json
import sys
import time
import urllib.request
from pathlib import Path

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
HEADERS = {"Digitraffic-User": "flatland-railways-student-project", "Accept-Encoding": "gzip",
           "User-Agent": "flatland-railways-student-project/1.0"}
days = [f"{y}-{m:02d}-{d:02d}" for y in (2024, 2025) for m in range(1, 13) for d in (8, 22)]
for day in days:
    path = OUT / f"{day}.json.gz"
    if path.exists():
        continue
    req = urllib.request.Request(f"https://rata.digitraffic.fi/api/v1/trains/{day}", headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
    trains = [t for t in json.loads(raw) if t.get("trainCategory") == "Long-distance"]
    with gzip.open(path, "wt") as f:
        json.dump(trains, f)
    print(day, len(trains), "long-distance trains", flush=True)
    time.sleep(2)
