"""Lightweight web dashboard for the bird detection system.

Usage:
    python dashboard.py

Opens a browser at http://localhost:5000 with auto-refreshing status.
"""

import os
import json
import glob
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template_string

from config import OUTPUT_DIR, LOG_FILE, DETECTION_THRESHOLD

app = Flask(__name__)


# ── Helpers ─────────────────────────────────────────────

def read_detection_log():
    try:
        if os.path.exists("detection_log.json"):
            with open("detection_log.json") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def list_recent_detections(n=20):
    pattern = os.path.join(OUTPUT_DIR, "*.wav")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)[:n]
    entries = []
    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        parts = name.split("_", 1)
        species = parts[0] if len(parts) > 0 else "unknown"
        ts_str = parts[1] if len(parts) > 1 else ""
        mtime = os.path.getmtime(fp)
        entries.append({
            "species": species,
            "timestamp": ts_str,
            "datetime": datetime.fromtimestamp(mtime).isoformat(),
            "file": os.path.basename(fp),
            "size_kb": round(os.path.getsize(fp) / 1024, 1),
        })
    return entries


def tail_log(n=30):
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE) as f:
                lines = f.readlines()
            return [l.strip() for l in lines[-n:]]
    except Exception:
        pass
    return []


def get_species_stats():
    log = read_detection_log()
    counts = log.get("species_counts", {})
    total = sum(counts.values()) if counts else 0
    # Also count actual WAV files
    wav_count = len(glob.glob(os.path.join(OUTPUT_DIR, "*.wav")))
    return counts, max(total, wav_count)


# ── Routes ──────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    species_counts, total_detections = get_species_stats()
    recent = list_recent_detections()
    now = time.time()
    return jsonify({
        "total_detections": total_detections,
        "species_counts": species_counts,
        "num_species": len(species_counts),
        "recent_detections": recent,
        "threshold": DETECTION_THRESHOLD,
        "check_time": datetime.now().isoformat(),
    })


@app.route("/api/log")
def api_log():
    return jsonify({"lines": tail_log(50)})


@app.route("/")
def index():
    species_counts, total_detections = get_species_stats()
    recent = list_recent_detections()
    log_lines = tail_log()
    return render_template_string(HTML_TEMPLATE,
        total_detections=total_detections,
        species_counts=species_counts,
        recent=recent,
        log_lines=log_lines,
        threshold=DETECTION_THRESHOLD,
    )


# ── Template ────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BirdNET Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f172a; color: #e2e8f0; padding: 20px; }
  h1 { font-size: 1.5rem; margin-bottom: 8px; }
  .subtitle { color: #94a3b8; font-size: 0.85rem; margin-bottom: 20px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .card { background: #1e293b; border-radius: 10px; padding: 16px; border: 1px solid #334155; }
  .card h2 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; margin-bottom: 8px; }
  .stat-value { font-size: 2rem; font-weight: 700; }
  .stat-value.green { color: #22c55e; }
  .stat-value.yellow { color: #eab308; }
  .stat-value.blue { color: #3b82f6; }
  .species-list { display: flex; flex-wrap: wrap; gap: 6px; }
  .species-tag { background: #334155; padding: 4px 10px; border-radius: 12px; font-size: 0.8rem; }
  .species-tag .count { color: #22c55e; font-weight: 600; margin-left: 4px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th { text-align: left; color: #64748b; font-weight: 500; padding: 6px 8px; border-bottom: 1px solid #334155; }
  td { padding: 6px 8px; border-bottom: 1px solid #1e293b; }
  .log-box { background: #0f172a; border: 1px solid #334155; border-radius: 6px;
             padding: 10px; max-height: 300px; overflow-y: auto; font-family: 'Cascadia Code', 'Fira Code', monospace;
             font-size: 0.75rem; line-height: 1.5; }
  .log-box .info { color: #94a3b8; }
  .log-box .warn { color: #eab308; }
  .log-box .error { color: #ef4444; }
  .log-box .debug { color: #64748b; }
  .timestamp { color: #64748b; }
  .footer { text-align: center; color: #475569; font-size: 0.75rem; margin-top: 32px; }
  .no-data { color: #64748b; font-style: italic; }
</style>
</head>
<body>
<h1>BirdNET Bioacoustic Detection System</h1>
<p class="subtitle">Live dashboard — auto-refreshes every 5s &middot; threshold={{ threshold }}</p>

<div class="grid">
  <div class="card">
    <h2>Total Detections</h2>
    <div class="stat-value green">{{ total_detections }}</div>
  </div>
  <div class="card">
    <h2>Species Detected</h2>
    <div class="stat-value blue">{{ species_counts|length }}</div>
  </div>
  <div class="card">
    <h2>Threshold</h2>
    <div class="stat-value yellow">{{ threshold }}</div>
  </div>
</div>

<div class="card" style="margin-bottom: 16px;">
  <h2>Species Breakdown</h2>
  {% if species_counts %}
  <div class="species-list">
    {% for species, count in species_counts.items() %}
    <span class="species-tag">{{ species }}<span class="count">{{ count }}</span></span>
    {% endfor %}
  </div>
  {% else %}
  <p class="no-data">No detections yet</p>
  {% endif %}
</div>

<div class="grid" style="grid-template-columns: 1fr 1fr;">
  <div class="card">
    <h2>Recent Detections</h2>
    {% if recent %}
    <table>
      <thead><tr><th>Species</th><th>Time</th><th>Size</th></tr></thead>
      <tbody>
        {% for d in recent %}
        <tr>
          <td>{{ d.species.replace('_', ' ') }}</td>
          <td class="timestamp">{{ d.datetime[11:19] if d.datetime else '' }}</td>
          <td>{{ d.size_kb }}KB</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    {% else %}
    <p class="no-data">No detection files yet</p>
    {% endif %}
  </div>

  <div class="card">
    <h2>Recent Log</h2>
    <div class="log-box">
      {% for line in log_lines %}
        {% set cls = 'info' %}
        {% if '[WARNING]' in line or '[WARN]' in line %}{% set cls = 'warn' %}{% endif %}
        {% if '[ERROR]' in line %}{% set cls = 'error' %}{% endif %}
        {% if '[DEBUG]' in line %}{% set cls = 'debug' %}{% endif %}
        <div class="{{ cls }}">{{ line }}</div>
      {% endfor %}
    </div>
  </div>
</div>

<div class="footer">
  BirdNET Detection System &middot; Refresh every 5s
  <span id="check-time"></span>
</div>

<script>
  let checkTime = document.getElementById('check-time');
  function updateTime() { checkTime.textContent = ' · ' + new Date().toLocaleTimeString(); }
  setInterval(() => location.reload(), 5000);
  updateTime();
</script>
</body>
</html>"""


if __name__ == "__main__":
    print(f"BirdNET Dashboard starting at http://localhost:5000")
    print(f"Detection dir: {OUTPUT_DIR}, log: {LOG_FILE}")
    app.run(host="0.0.0.0", port=5000, debug=False)
