import os
import sys
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, url_for

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from storage.events_db import daily_counts, list_elders, list_recent_events, update_elder_name

DB_PATH = str(PROJECT_ROOT / "data" / "fall_events.db")

app = Flask(__name__, template_folder=str(PROJECT_ROOT / "web" / "templates"))


@app.route("/")
def index():
    limit = int(request.args.get("limit", 50))
    events = list_recent_events(DB_PATH, limit=limit)
    elders = list_elders(DB_PATH)
    counts = daily_counts(DB_PATH, days=7)
    total = sum(int(item.get("total", 0)) for item in counts)
    return render_template("index.html", events=events, elders=elders, counts=counts, total=total, limit=limit)


@app.route("/api/events")
def api_events():
    limit = int(request.args.get("limit", 50))
    events = list_recent_events(DB_PATH, limit=limit)
    return jsonify({"items": events, "count": len(events)})


@app.route("/api/elders")
def api_elders():
    elders = list_elders(DB_PATH)
    return jsonify({"items": elders, "count": len(elders)})


@app.route("/elders/rename", methods=["POST"])
def elders_rename():
    elder_code = str(request.form.get("elder_code") or "").strip()
    elder_name = str(request.form.get("elder_name") or "").strip()
    if elder_code:
        update_elder_name(DB_PATH, elder_code, elder_name)
    return redirect(url_for("index"))


if __name__ == "__main__":
    host = os.getenv("FALL_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("FALL_WEB_PORT", "5000"))
    app.run(host=host, port=port, debug=False)
