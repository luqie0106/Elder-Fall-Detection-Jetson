from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any


class Notifier:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    def send_fall_alert(self, event: dict[str, Any]) -> dict[str, Any]:
        status: dict[str, Any] = {
            "console": False,
            "webhook": False,
            "weixin": False,
            "phone_call": False,
        }

        if self.config.get("console", {}).get("enabled", True):
            self._send_console(event)
            status["console"] = True

        if self.config.get("webhook", {}).get("enabled", False):
            status["webhook"] = self._send_webhook(event)

        if self.config.get("weixin", {}).get("enabled", False):
            status["weixin"] = self._send_weixin_placeholder(event)

        if self.config.get("phone_call", {}).get("enabled", False):
            status["phone_call"] = self._send_phone_call_placeholder(event)

        return status

    def _send_console(self, event: dict[str, Any]) -> None:
        ts = event.get("event_time") or datetime.now().isoformat(timespec="seconds")
        frame_idx = event.get("frame_idx", -1)
        camera_id = event.get("camera_id", "cam-0")
        print(f"[ALERT] Fall detected | time={ts} camera={camera_id} frame={frame_idx}")

    def _send_webhook(self, event: dict[str, Any]) -> bool:
        webhook_cfg = self.config.get("webhook", {})
        url = str(webhook_cfg.get("url") or "").strip()
        if not url:
            return False

        timeout_sec = int(webhook_cfg.get("timeout_sec", 4))
        body = {
            "msg_type": "fall_alert",
            "event": event,
        }
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url=url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as response:
                return 200 <= int(response.status) < 300
        except (urllib.error.URLError, TimeoutError, ValueError):
            return False

    def _send_weixin_placeholder(self, event: dict[str, Any]) -> bool:
        print("[ALERT] Weixin channel enabled but provider adapter not configured yet.")
        return False

    def _send_phone_call_placeholder(self, event: dict[str, Any]) -> bool:
        print("[ALERT] Phone-call channel enabled but provider adapter not configured yet.")
        return False
