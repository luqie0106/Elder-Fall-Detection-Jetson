from __future__ import annotations

import importlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from storage.events_db import ensure_elder, init_db


def _load_face_recognition():
    try:
        return importlib.import_module("face_recognition")
    except Exception:
        return None


_face_recognition = _load_face_recognition()


class FaceRecognitionService:
    def __init__(self, db_path: str, tolerance: float = 0.42):
        self.db_path = db_path
        self.tolerance = tolerance
        init_db(self.db_path)

    @property
    def available(self) -> bool:
        return _face_recognition is not None

    def identify_or_register(self, frame_bgr: np.ndarray) -> str | None:
        if not self.available:
            return None

        face_recognition = _face_recognition
        if face_recognition is None:
            return None

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        location = self._extract_valid_face_location(frame_rgb)
        if location is None:
            return None

        encodings = face_recognition.face_encodings(frame_rgb, known_face_locations=[location])
        if not encodings:
            return None

        target_encoding = encodings[0]
        matched_code = self._match_encoding(target_encoding)
        if matched_code:
            return matched_code

        elder = ensure_elder(self.db_path, None)
        elder_code = str(elder["elder_code"])
        self._save_face_embedding(elder_code, target_encoding)
        return elder_code

    def identify_only(self, frame_bgr: np.ndarray) -> str | None:
        if not self.available:
            return None

        face_recognition = _face_recognition
        if face_recognition is None:
            return None

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        location = self._extract_valid_face_location(frame_rgb)
        if location is None:
            return None

        encodings = face_recognition.face_encodings(frame_rgb, known_face_locations=[location])
        if not encodings:
            return None

        return self._match_encoding(encodings[0])

    def attach_face_to_elder(self, frame_bgr: np.ndarray, elder_code: str) -> bool:
        if not self.available:
            return False
        code = str(elder_code).strip()
        if not code:
            return False

        face_recognition = _face_recognition
        if face_recognition is None:
            return False

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        location = self._extract_valid_face_location(frame_rgb)
        if location is None:
            return False

        encodings = face_recognition.face_encodings(frame_rgb, known_face_locations=[location])
        if not encodings:
            return False

        ensure_elder(self.db_path, code)
        self._save_face_embedding(code, encodings[0])
        return True

    def _match_encoding(self, target_encoding: np.ndarray) -> str | None:
        if _face_recognition is None:
            return None

        known = self._load_known_faces()
        if not known:
            return None

        known_codes = [item["elder_code"] for item in known]
        known_vectors = [item["encoding"] for item in known]
        distances = _face_recognition.face_distance(known_vectors, target_encoding)
        best_idx = int(np.argmin(distances))
        if float(distances[best_idx]) <= self.tolerance:
            return str(known_codes[best_idx])
        return None

    def _extract_valid_face_location(self, frame_rgb: np.ndarray):
        face_recognition = _face_recognition
        if face_recognition is None:
            return None

        locations = face_recognition.face_locations(frame_rgb, number_of_times_to_upsample=0, model="hog")
        if not locations:
            return None

        height, width = frame_rgb.shape[:2]
        min_edge = max(32, int(min(width, height) * 0.06))

        valid_locations = []
        for location in locations:
            top, right, bottom, left = location
            face_w = max(0, right - left)
            face_h = max(0, bottom - top)
            if face_w < min_edge or face_h < min_edge:
                continue
            ratio = face_w / max(1.0, float(face_h))
            if ratio < 0.55 or ratio > 1.85:
                continue
            valid_locations.append(location)

        if not valid_locations:
            return None

        best_location = max(valid_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))
        landmarks = face_recognition.face_landmarks(frame_rgb, [best_location])
        if not landmarks:
            return None

        landmark = landmarks[0]
        has_eyes = bool(landmark.get("left_eye")) and bool(landmark.get("right_eye"))
        has_nose = bool(landmark.get("nose_bridge")) or bool(landmark.get("nose_tip"))
        if not (has_eyes and has_nose):
            return None

        return best_location

    def _load_known_faces(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT elder_code, embedding_json FROM elder_faces ORDER BY id ASC"
            ).fetchall()

        result: list[dict[str, Any]] = []
        for row in rows:
            try:
                vec = np.array(json.loads(str(row["embedding_json"])), dtype=np.float32)
                if vec.size > 0:
                    result.append({"elder_code": str(row["elder_code"]), "encoding": vec})
            except json.JSONDecodeError:
                continue
        return result

    def _save_face_embedding(self, elder_code: str, encoding: np.ndarray) -> None:
        payload = json.dumps(encoding.astype(float).tolist())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO elder_faces (elder_code, embedding_json, created_at) VALUES (?, ?, datetime('now'))",
                (elder_code, payload),
            )
            conn.commit()


def get_default_face_service() -> FaceRecognitionService:
    project_root = Path(__file__).resolve().parents[1]
    db_path = str(project_root / "data" / "fall_events.db")
    return FaceRecognitionService(db_path=db_path)
