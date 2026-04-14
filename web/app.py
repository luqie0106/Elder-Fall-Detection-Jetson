import os
import sys
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, url_for

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from storage.events_db import clear_fall_events, list_recent_events

DB_PATH = str(PROJECT_ROOT / "data" / "fall_events.db")

app = Flask(__name__, template_folder=str(PROJECT_ROOT / "web" / "templates"))


def _clear_snapshot_files() -> None:
    snapshot_dir = PROJECT_ROOT / "web" / "static" / "faces"
    if not snapshot_dir.exists():
        return
    for file_path in snapshot_dir.iterdir():
        if file_path.name == ".gitkeep":
            continue
        if file_path.is_file():
            try:
                file_path.unlink()
            except OSError:
                pass


@app.route("/")
def index():
    limit = int(request.args.get("limit", 50))
    events = list_recent_events(DB_PATH, limit=limit)
    return render_template("index.html", events=events, limit=limit)


@app.route("/api/events")
def api_events():
    limit = int(request.args.get("limit", 50))
    events = list_recent_events(DB_PATH, limit=limit)
    return jsonify({"items": events, "count": len(events)})


@app.route("/events/clear", methods=["POST"])
def clear_events():
    clear_fall_events(DB_PATH)
    _clear_snapshot_files()
    return redirect(url_for("index"))


if __name__ == "__main__":
    host = os.getenv("FALL_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("FALL_WEB_PORT", "5001"))
    app.run(host=host, port=port, debug=False)
