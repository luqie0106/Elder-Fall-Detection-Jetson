from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from services.notifier import Notifier
from storage.events_db import insert_fall_event

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = str(PROJECT_ROOT / "data" / "fall_events.db")
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "alert_config.json"


def _load_config() -> dict[str, Any]:
    if DEFAULT_CONFIG_PATH.exists():
        try:
            return json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {
        "console": {"enabled": True},
        "webhook": {"enabled": False, "url": "", "timeout_sec": 4},
        "weixin": {"enabled": False},
        "phone_call": {"enabled": False},
    }


_notifier = Notifier(_load_config())


def report_fall_event(event: dict[str, Any]) -> dict[str, Any]:
    payload = dict(event)
    payload.setdefault("event_time", datetime.now().isoformat(timespec="seconds"))
    payload.setdefault("camera_id", "cam-0")

    status = _notifier.send_fall_alert(payload)
    payload["channel_status"] = json.dumps(status, ensure_ascii=False)

    event_id = insert_fall_event(DEFAULT_DB_PATH, payload)
    return {"event_id": event_id, "channels": status}
