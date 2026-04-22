from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import Any


def _ensure_db_dir(db_path: str) -> None:
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    existing = conn.execute(f"PRAGMA table_info({table})").fetchall()
    names = {row[1] for row in existing}
    if column not in names:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _next_elder_code(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT elder_code FROM elders ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if not row or not row[0]:
        return "E001"

    code = str(row[0])
    try:
        index = int(code[1:])
    except (ValueError, IndexError):
        index = 0
    return f"E{index + 1:03d}"


def init_db(db_path: str) -> None:
    _ensure_db_dir(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fall_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_time TEXT NOT NULL,
                camera_id TEXT,
                frame_idx INTEGER,
                cx REAL,
                cy REAL,
                bbox_json TEXT,
                elder_code TEXT,
                channel_status TEXT,
                extra_json TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS elders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                elder_code TEXT UNIQUE NOT NULL,
                elder_name TEXT,
                avatar_path TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS elder_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                elder_code TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS person_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_time TEXT NOT NULL,
                date_key TEXT NOT NULL,
                camera_id TEXT,
                track_token TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        _ensure_column(conn, "fall_events", "elder_code", "TEXT")
        _ensure_column(conn, "elders", "avatar_path", "TEXT")
        conn.commit()


def insert_person_entry(db_path: str, entry: dict[str, Any]) -> int:
    init_db(db_path)
    now = datetime.now().isoformat(timespec="seconds")
    entry_time = str(entry.get("entry_time") or now)
    date_key = str(entry_time[:10])

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO person_entries (
                entry_time, date_key, camera_id, track_token, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                entry_time,
                date_key,
                str(entry.get("camera_id") or "cam-0"),
                str(entry.get("track_token") or ""),
                now,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_today_overview(db_path: str) -> dict[str, int]:
    init_db(db_path)
    today = datetime.now().strftime("%Y-%m-%d")

    with sqlite3.connect(db_path) as conn:
        entered_row = conn.execute(
            "SELECT COUNT(*) FROM person_entries WHERE date_key = ?",
            (today,),
        ).fetchone()
        alarms_row = conn.execute(
            "SELECT COUNT(*) FROM fall_events WHERE substr(event_time, 1, 10) = ?",
            (today,),
        ).fetchone()

    entered_today = int(entered_row[0]) if entered_row and entered_row[0] is not None else 0
    alarms_today = int(alarms_row[0]) if alarms_row and alarms_row[0] is not None else 0
    return {"entered_today": entered_today, "alarms_today": alarms_today}


def ensure_elder(db_path: str, elder_code: str | None = None) -> dict[str, Any]:
    init_db(db_path)
    now = datetime.now().isoformat(timespec="seconds")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        code = str(elder_code).strip() if elder_code else ""
        if code:
            row = conn.execute(
                "SELECT id, elder_code, elder_name, avatar_path, created_at FROM elders WHERE elder_code = ?",
                (code,),
            ).fetchone()
            if row:
                return dict(row)
        else:
            code = _next_elder_code(conn)

        conn.execute(
            "INSERT INTO elders (elder_code, elder_name, avatar_path, created_at) VALUES (?, ?, ?, ?)",
            (code, "", "", now),
        )
        conn.commit()

        row = conn.execute(
            "SELECT id, elder_code, elder_name, avatar_path, created_at FROM elders WHERE elder_code = ?",
            (code,),
        ).fetchone()
        return dict(row) if row else {"elder_code": code, "elder_name": "", "avatar_path": "", "created_at": now}


def update_elder_avatar(db_path: str, elder_code: str, avatar_path: str) -> bool:
    init_db(db_path)
    code = str(elder_code).strip()
    if not code:
        return False

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "UPDATE elders SET avatar_path = ? WHERE elder_code = ?",
            (str(avatar_path).strip(), code),
        )
        conn.commit()
        return int(cursor.rowcount) > 0


def list_elders(db_path: str) -> list[dict[str, Any]]:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, elder_code, elder_name, avatar_path, created_at FROM elders ORDER BY id ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def update_elder_name(db_path: str, elder_code: str, elder_name: str) -> bool:
    init_db(db_path)
    code = str(elder_code).strip()
    if not code:
        return False

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "UPDATE elders SET elder_name = ? WHERE elder_code = ?",
            (str(elder_name).strip(), code),
        )
        conn.commit()
        return int(cursor.rowcount) > 0


def insert_fall_event(db_path: str, event: dict[str, Any]) -> int:
    init_db(db_path)
    now = datetime.now().isoformat(timespec="seconds")
    event_time = str(event.get("event_time") or now)
    elder_code = str(event.get("elder_code") or "").strip()
    if elder_code:
        ensure_elder(db_path, elder_code)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO fall_events (
                event_time, camera_id, frame_idx, cx, cy, bbox_json, elder_code, channel_status, extra_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(event.get("event_time") or now),
                str(event.get("camera_id") or "cam-0"),
                int(event.get("frame_idx") or -1),
                float(event.get("cx") or 0.0),
                float(event.get("cy") or 0.0),
                json.dumps(event.get("bbox") or []),
                elder_code,
                str(event.get("channel_status") or "pending"),
                json.dumps(event, ensure_ascii=False),
                now,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def clear_fall_events(db_path: str) -> int:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("DELETE FROM fall_events")
        conn.commit()
        return int(cursor.rowcount)


def list_recent_events(db_path: str, limit: int = 100) -> list[dict[str, Any]]:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        elder_rows = conn.execute(
            "SELECT elder_code, elder_name, avatar_path FROM elders"
        ).fetchall()
        elder_map = {
            str(r["elder_code"]): {
                "name": str(r["elder_name"] or ""),
                "avatar_path": str(r["avatar_path"] or ""),
            }
            for r in elder_rows
        }

        rows = conn.execute(
            """
            SELECT id, event_time, camera_id, frame_idx, cx, cy, bbox_json, elder_code, channel_status, extra_json, created_at
            FROM fall_events
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()

    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["bbox"] = json.loads(item.pop("bbox_json") or "[]")
        except json.JSONDecodeError:
            item["bbox"] = []
        try:
            extra = json.loads(item.pop("extra_json") or "{}")
        except json.JSONDecodeError:
            extra = {}
        item["alert_level"] = str(extra.get("alert_level") or "confirmed")
        item["snapshot_path"] = str(extra.get("snapshot_path") or "")
        item["video_path"] = str(extra.get("video_path") or "")
        code = str(item.get("elder_code") or "").strip()
        if code:
            elder_info = elder_map.get(code, {})
            name = str(elder_info.get("name") or "")
            item["elder_name"] = name if name else code
            item["avatar_path"] = str(elder_info.get("avatar_path") or "")
        else:
            item["elder_name"] = "未识别"
            item["avatar_path"] = ""
        result.append(item)
    return result


def daily_counts(db_path: str, days: int = 7) -> list[dict[str, Any]]:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT substr(event_time, 1, 10) AS day, COUNT(*) AS total
            FROM fall_events
            GROUP BY substr(event_time, 1, 10)
            ORDER BY day DESC
            LIMIT ?
            """,
            (int(days),),
        ).fetchall()
    return [dict(r) for r in rows]
