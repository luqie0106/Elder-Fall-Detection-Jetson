from __future__ import annotations

from ultralytics import YOLO
import cv2
import numpy as np
import math
import time
import os
import subprocess
import queue
import threading
import signal
import importlib
from pathlib import Path
from collections import deque
from typing import Any

try:
    _deepsort_mod = importlib.import_module("deep_sort_realtime.deepsort_tracker")
    DeepSort = getattr(_deepsort_mod, "DeepSort", None)
except Exception:
    DeepSort = None

try:
    from services.event_pipeline import report_fall_event
except Exception:
    report_fall_event = None

ENABLE_FACE_RECOG = str(os.getenv("FALL_ENABLE_FACE_RECOG", "0")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

get_default_face_service = None
if ENABLE_FACE_RECOG:
    try:
        from services.face_recognition_service import get_default_face_service
    except Exception:
        get_default_face_service = None

try:
    from storage.events_db import ensure_elder, update_elder_avatar, insert_person_entry
except Exception:
    ensure_elder = None
    update_elder_avatar = None
    insert_person_entry = None

try:
    from services.cpp_accel import load_cpp_accel
except Exception:
    load_cpp_accel = None

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = str(PROJECT_ROOT / "data" / "fall_events.db")
FACES_DIR = PROJECT_ROOT / "web" / "static" / "faces"

def _resolve_fall_model_path() -> tuple[str, str]:
    env_model = str(os.getenv("FALL_MODEL_PATH", "")).strip()
    default_model_name = str(os.getenv("FALL_MODEL_DEFAULT", "yolov8n-pose.pt")).strip() or "yolov8n-pose.pt"

    def _pick_local(candidate: str) -> str | None:
        model_path = Path(candidate).expanduser()
        if model_path.is_file():
            return str(model_path)
        if not model_path.is_absolute():
            project_model_path = (PROJECT_ROOT / model_path).resolve()
            if project_model_path.is_file():
                return str(project_model_path)
        return None

    if env_model:
        local_model = _pick_local(env_model)
        if local_model is not None:
            return local_model, "env-local"
        if "/" not in env_model and "\\" not in env_model:
            return env_model, "env-download"
        print(f"[Model] FALL_MODEL_PATH={env_model} 在本地未找到，继续按默认策略选择。")

    # 优先尊重 FALL_MODEL_DEFAULT：若本地存在同名文件，直接使用
    default_local = _pick_local(default_model_name)
    if default_local is not None:
        return default_local, "default-local"

    # 其次尝试同名 .engine（便于将 pt 导出后无缝切换）
    default_stem = Path(default_model_name).stem
    if default_stem:
        preferred_engine = _pick_local(f"{default_stem}.engine")
        if preferred_engine is not None:
            return preferred_engine, "default-engine-local"

    for pattern in ("yolo*-pose.engine", "yolo*-pose.pt"):
        for local_model in sorted(PROJECT_ROOT.glob(pattern)):
            if local_model.is_file():
                return str(local_model), "local"

    return default_model_name, "default-download"


MODEL_PATH, MODEL_SOURCE = _resolve_fall_model_path()
print(f"[Model] 使用模型: {MODEL_PATH} (source={MODEL_SOURCE})")

# 加载模型（一个模型同时支持检测+pose）
model = YOLO(MODEL_PATH)

face_service = None
if get_default_face_service is not None:
    try:
        face_service = get_default_face_service()
    except Exception:
        face_service = None
elif not ENABLE_FACE_RECOG:
    print("[face] 人脸识别已禁用（FALL_ENABLE_FACE_RECOG=0）。")


def _parse_camera_sources() -> list[str | int]:
    raw = str(os.getenv("FALL_CAMERA_SOURCES", "")).strip()
    if not raw:
        # Linux 上默认扫描 /dev/video*，避免仅尝试 index=0 导致可用设备被漏掉。
        if os.name == "posix":
            auto_sources: list[str | int] = []
            for device_path in sorted(Path("/dev").glob("video*")):
                name = device_path.name
                if len(name) > 5 and name[5:].isdigit():
                    auto_sources.append(str(device_path))
            if auto_sources:
                print(f"[camera] FALL_CAMERA_SOURCES 未设置，自动扫描到: {auto_sources}")
                return auto_sources
        return [0]
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    result: list[str | int] = []
    for part in parts:
        if part.lstrip("-").isdigit():
            result.append(int(part))
        else:
            result.append(part)
    return result or [0]


def _build_camera_attempts(source: str | int) -> list[tuple[str, str | int, int | None]]:
    attempts: list[tuple[str, str | int, int | None]] = []

    if isinstance(source, int):
        attempts.append(("index-default", source, None))
        attempts.append(("index-v4l2", source, cv2.CAP_V4L2))
        if os.name == "posix":
            dev_path = f"/dev/video{source}"
            attempts.append(("dev-default", dev_path, None))
            attempts.append(("dev-v4l2", dev_path, cv2.CAP_V4L2))
        return attempts

    source_str = str(source).strip()
    if source_str.lstrip("-").isdigit():
        return _build_camera_attempts(int(source_str))

    attempts.append(("source-default", source_str, None))

    if os.name == "posix" and source_str.startswith("/dev/video"):
        attempts.append(("source-v4l2", source_str, cv2.CAP_V4L2))

    return attempts


def _open_camera_source(source: str | int) -> tuple[Any | None, str]:
    failure_tags: list[str] = []
    for tag, actual_source, backend in _build_camera_attempts(source):
        try:
            if backend is None:
                cap = cv2.VideoCapture(actual_source)
            else:
                cap = cv2.VideoCapture(actual_source, backend)
        except Exception as exc:
            failure_tags.append(f"{tag}:{exc}")
            continue

        if cap is not None and cap.isOpened():
            backend_name = "default" if backend is None else str(backend)
            return cap, f"opened via {tag} source={actual_source} backend={backend_name}"

        if cap is not None:
            cap.release()
        failure_tags.append(tag)

    return None, ", ".join(failure_tags) if failure_tags else "no-attempt"


CAMERA_SOURCES = _parse_camera_sources()
camera_entries: list[dict] = []
for index, source in enumerate(CAMERA_SOURCES):
    cap, open_msg = _open_camera_source(source)
    if cap is not None:
        print(f"[camera] source={source} {open_msg}")
        camera_entries.append({
            "camera_key": f"cam-{index}",
            "camera_source": source,
            "capture": cap,
        })
    else:
        print(f"[camera] source={source} open failed ({open_msg})")

if not camera_entries:
    raise RuntimeError(
        "没有可用摄像头。请检查 FALL_CAMERA_SOURCES（例如 /dev/video1 或 0,1），"
        "并确认当前用户有 video 组权限。"
    )

camera_latest_frames: dict[str, np.ndarray | None] = {entry["camera_key"]: None for entry in camera_entries}
camera_locks: dict[str, threading.Lock] = {entry["camera_key"]: threading.Lock() for entry in camera_entries}
camera_stop_event = threading.Event()


def _camera_capture_worker(entry: dict) -> None:
    camera_key = str(entry["camera_key"])
    capture = entry["capture"]
    while not camera_stop_event.is_set():
        ret, frame = capture.read()
        if not ret:
            time.sleep(0.02)
            continue
        with camera_locks[camera_key]:
            camera_latest_frames[camera_key] = frame


camera_threads: list[threading.Thread] = []
for entry in camera_entries:
    t = threading.Thread(target=_camera_capture_worker, args=(entry,), daemon=True)
    t.start()
    camera_threads.append(t)

# 时间序列缓存（每个人独立）
MAX_HISTORY = 10

# 统计数据
fall_count = 0
person_count = 0
FALL_HOLD_TIME = 30  # 保持红框30帧（约1秒）
UNRECOVERED_ALERT_FRAMES = 120
ENTRY_STABLE_FRAMES = 8
DWELL_ALERT_SECONDS = 30.0

EVENT_CLIP_PRE_SECONDS = max(3, min(5, int(os.getenv("FALL_CLIP_PRE_SECONDS", "3"))))
EVENT_CLIP_POST_SECONDS = max(3, min(5, int(os.getenv("FALL_CLIP_POST_SECONDS", "3"))))
EVENT_CLIP_FPS = 15
EVENT_CLIP_EXT = str(os.getenv("FALL_CLIP_EXT", "mp4")).strip().lower().lstrip(".") or "mp4"
EVENT_CLIP_CODECS = [
    c.strip()
    for c in os.getenv("FALL_CLIP_CODECS", "H264,avc1,mp4v").split(",")
    if len(c.strip()) == 4
]
EVENT_CLIP_ASYNC_WRITE = str(os.getenv("FALL_CLIP_ASYNC_WRITE", "1")).strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
EVENT_CLIP_QUEUE_SIZE = max(1, int(os.getenv("FALL_CLIP_QUEUE_SIZE", "8")))
EVENT_CLIP_AUTO_FIX_BROWSER = str(os.getenv("FALL_CLIP_AUTO_FIX_BROWSER", "1")).strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
TRACK_PERSIST = len(camera_entries) <= 1
TRACKER_CFG = str(os.getenv("FALL_TRACKER", "bytetrack.yaml")).strip() or "bytetrack.yaml"
TRACKER_BACKEND = str(os.getenv("FALL_TRACKER_BACKEND", "ultralytics")).strip().lower()
TRACK_FRAME_STRIDE = max(1, int(os.getenv("FALL_TRACK_FRAME_STRIDE", "1")))
TRACK_REUSE_IOU_TH = float(os.getenv("FALL_TRACK_REUSE_IOU_TH", "0.35"))
DEEPSORT_MAX_AGE = int(os.getenv("FALL_DEEPSORT_MAX_AGE", "30"))
DEEPSORT_N_INIT = int(os.getenv("FALL_DEEPSORT_N_INIT", "3"))
DEEPSORT_MAX_IOU_DISTANCE = float(os.getenv("FALL_DEEPSORT_MAX_IOU_DISTANCE", "0.7"))

if TRACKER_BACKEND not in {"deepsort", "ultralytics"}:
    print(f"[WARN] 未知 TRACKER_BACKEND={TRACKER_BACKEND}，自动回退到 ultralytics。")
    TRACKER_BACKEND = "ultralytics"

FALL_PERF_INFER_ONLY = str(os.getenv("FALL_PERF_INFER_ONLY", "0")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DRAW_OVERLAYS = str(os.getenv("FALL_DRAW_OVERLAYS", "1")).strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
DRAW_SUPPORT_OBJECTS = str(os.getenv("FALL_DRAW_SUPPORT_OBJECTS", "1")).strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
DRAW_NORMAL_BOXES = str(os.getenv("FALL_DRAW_NORMAL_BOXES", "1")).strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
DRAW_DWELL_TEXT = str(os.getenv("FALL_DRAW_DWELL_TEXT", "0")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DRAW_KEYPOINT_MODE = str(os.getenv("FALL_DRAW_KEYPOINT_MODE", "fall-only")).strip().lower() or "fall-only"
if DRAW_KEYPOINT_MODE not in {"none", "fall-only", "all"}:
    DRAW_KEYPOINT_MODE = "fall-only"
DRAW_HUD_EVERY_N_FRAMES = max(1, int(os.getenv("FALL_DRAW_HUD_EVERY_N_FRAMES", "2")))
PROFILE_PRINT_INTERVAL = max(0, int(os.getenv("FALL_PROFILE_PRINT_INTERVAL", "0")))
USE_CPP_ACCEL = str(os.getenv("FALL_USE_CPP_ACCEL", "0")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

cpp_accel_module = None
CPP_ACCEL_ENABLED = False
if USE_CPP_ACCEL:
    if load_cpp_accel is None:
        print("[cpp] FALL_USE_CPP_ACCEL=1 但加载器不可用，继续使用 Python 路径。")
    else:
        cpp_accel_module, cpp_msg = load_cpp_accel()
        if cpp_accel_module is None:
            print(
                "[cpp] FALL_USE_CPP_ACCEL=1 但未加载到 cpp_accel_impl，继续使用 Python 路径。"
                "可执行: python3 cpp_accel/build_cpp_accel.py build_ext --inplace"
            )
            if cpp_msg:
                print(f"[cpp] load error: {cpp_msg}")
        else:
            CPP_ACCEL_ENABLED = True
            print("[cpp] C++ acceleration enabled (cpp_accel_impl)")

if TRACKER_BACKEND == "deepsort" and DeepSort is None:
    print("[WARN] DeepSORT 依赖未安装，自动回退到 ByteTrack(ultralytics tracker)。")
    TRACKER_BACKEND = "ultralytics"

camera_trackers: dict[str, Any] = {}
if TRACKER_BACKEND == "deepsort" and DeepSort is not None:
    try:
        for entry in camera_entries:
            camera_key = str(entry["camera_key"])
            camera_trackers[camera_key] = DeepSort(
                max_age=DEEPSORT_MAX_AGE,
                n_init=DEEPSORT_N_INIT,
                max_iou_distance=DEEPSORT_MAX_IOU_DISTANCE,
            )
    except Exception as error:
        print(f"[WARN] DeepSORT 初始化失败，自动回退到 ByteTrack: {error}")
        camera_trackers.clear()
        TRACKER_BACKEND = "ultralytics"

if TRACKER_BACKEND == "deepsort":
    print(
        f"[INFO] Tracker backend: deepsort (max_age={DEEPSORT_MAX_AGE}, "
        f"n_init={DEEPSORT_N_INIT}, max_iou_distance={DEEPSORT_MAX_IOU_DISTANCE})"
    )
else:
    print(f"[INFO] Tracker backend: ultralytics (tracker={TRACKER_CFG})")

# =====================
# 可调阈值（调参指南）
# =====================
# 说明：
# - “调大更紧”=更难触发（误报通常会降，但漏报可能增）
# - “调小更松”=更易触发（漏报通常会降，但误报可能增）

# 目标检测置信度阈值（调大更紧）
DETECTION_CONF_TH = 0.4

# 姿态判定：bbox高宽比阈值（ratio < 阈值 判 horizontal）
# 调大更松（更容易判 horizontal），调小更紧
BASE_HORIZONTAL_RATIO_TH = 0.8

# 摄像头俯仰角配置（单位：度）
# 系统会自动根据该角度放宽“弯腰”和“躺倒”的判定。
# 若设为 AUTO_DETECT_CAMERA_ANGLE = True，则实时动态估算角度。
AUTO_DETECT_CAMERA_ANGLE = False
CAMERA_PITCH_ANGLE = 0.0

# 自动估角与动态阈值补偿配置
PITCH_RATIO_BASELINE = 2.8
PITCH_BASELINE_DECAY = 0.002
PITCH_HISTORY_MIN = 24
PITCH_HISTORY_MAX = 150
PITCH_RATIO_PERCENTILE = 85
PITCH_MIN_DEG = 0.0
PITCH_MAX_DEG = 70.0
THRESH_COMP_MIN = 0.85
THRESH_COMP_MAX = 1.20

# 人体有效性过滤
# 最少关键点（调大更紧）
VALID_PERSON_MIN_KEYPOINTS = 4
# 最小人体高度（像素，调大更紧）
VALID_PERSON_MIN_HEIGHT = 50

# 跌倒序列判定
# 历史最少帧数（调大更紧，反应更慢）
FALL_SEQ_MIN_HISTORY = 3
# 前后窗口帧数（调大更紧）
FALL_SEQ_WINDOW = 2
# 头部累计下落阈值（像素，调大更紧）
FALL_SEQ_DROP_TH = 30
# 髋部累计下落阈值（像素，调大更紧）
FALL_SEQ_HIP_DROP_TH = 10
# 头部速度阈值（像素/帧，调大更紧）
FALL_SEQ_SPEED_TH = 6
# 冲击速度阈值（像素/帧，调大更紧）
FALL_SEQ_IMPACT_SPEED_TH = 9
# 冲击加速度阈值（像素/帧^2，调大更紧）
FALL_SEQ_IMPACT_ACCEL_TH = 1.5
# 速度计算使用的回看帧数（调大更稳更慢）
FALL_SEQ_SPEED_LOOKBACK = 3
# 侧向/后向摔倒：人体高度塌缩比例阈值（调小更松）
FALL_SEQ_BODY_HEIGHT_DROP_RATIO_TH = 0.16

# 支撑物重叠阈值（人体框与支撑物框）
# 调大更紧（更不容易判为“被支撑”）
SUPPORT_OVERLAP_TH = 0.2

# 下半身可见性门控
# 至少有效髋点数量（调大更紧）
LOWER_BODY_MIN_HIPS = 1
# 下半身总有效点数量（调大更紧）
LOWER_BODY_MIN_POINTS = 3
# 是否强制要求“下半身足够可见”才允许触发跌倒
REQUIRE_LOWER_BODY_FOR_FALL = True
# 是否强制要求“下半身几何关系可靠”才允许触发跌倒
REQUIRE_LOWER_BODY_GEOMETRY_FOR_FALL = True
# 下半身几何可靠性（调大更紧）
LOWER_BODY_MIN_HIP_ANKLE_DY_RATIO = 0.18
LOWER_BODY_KNEE_MID_TOL_RATIO = 0.12
# 核心躯干门控：至少1个肩点+1个髋点，才允许参与跌倒判定
CORE_BODY_MIN_SHOULDERS = 1
CORE_BODY_MIN_HIPS = 1

# 近景半身抑制：当“头部在人体框中占比过大”时，不触发跌倒
# 头部占比阈值（调小更紧，更容易抑制）
HEAD_DOMINANT_RATIO_TH = 0.46
# 头部宽高比下限（过窄通常不是近景头部）
HEAD_DOMINANT_MIN_ASPECT_RATIO = 0.55
# 近景上半身/截头画面抑制（与 head_dominant 互补）
CLOSEUP_BOTTOM_RATIO_TH = 0.88
CLOSEUP_MIN_HEIGHT_RATIO_TH = 0.40
CLOSEUP_HEAD_TOP_EDGE_RATIO_TH = 0.06
CLOSEUP_CLEAR_FALL_FRAMES = 2

# 人脸识别采样间隔（帧）
FACE_RECOG_INTERVAL = 15
# 人脸绑定最多尝试次数（每个track）
FACE_BIND_MAX_TRIES = 6
# 人脸关键点最小数量（鼻子/眼睛/耳朵）
FACE_MIN_KEYPOINTS_FOR_RECOG = 3
# 头像更新：最小质量分与提升阈值
AVATAR_MIN_SCORE = 0.18
AVATAR_UPDATE_SCORE_MARGIN = 0.06

# 坐姿门控（用于抑制坐姿误报）
# 躯干纵向占优比例阈值（调大更紧，不易判坐姿）
SITTING_TORSO_VERTICAL_RATIO_TH = 1.2
# 躯干横向最小基准（防抖常量）
SITTING_TORSO_DX_FLOOR = 5.0
# 髋-膝竖向距离占身高比例阈值（调大更松，更易判坐姿）
SITTING_SHORT_THIGH_RATIO_TH = 0.22

    # 同帧重复人合并阈值
# IoU阈值（调大更紧，重复合并更少）
DUPLICATE_IOU_TH = 0.35
# 中心点X/Y距离比例阈值（调大更松，更易判重复）
DUPLICATE_CENTER_DX_RATIO_TH = 0.10
DUPLICATE_CENTER_DY_RATIO_TH = 0.18

# 上下分裂场景阈值
# X重叠比例阈值（调大更紧）
DUPLICATE_X_OVERLAP_RATIO_TH = 0.65
# 纵向间隙比例阈值（调大更松）
DUPLICATE_VERTICAL_GAP_RATIO_TH = 0.10
# 上下分裂时中心X距离比例阈值（调大更松）
DUPLICATE_SPLIT_CENTER_DX_RATIO_TH = 0.12
# 包含关系比例阈值（调大更紧）
DUPLICATE_CONTAIN_RATIO_TH = 0.85

# 姿态与误判屏蔽的基础阈值（将被动态透视补偿）
BASE_BEND_HIP_ANKLE_RATIO = 0.28
BASE_BEND_SHOULDER_ANKLE_RATIO = 0.45

# 头脚高度接近判定（用于躺地确认）
# 头脚Y轴差值阈值 = max(绝对像素阈值, 身高比例阈值)
HEAD_FEET_CLOSE_RATIO_TH = 0.28
HEAD_FEET_CLOSE_ABS_PX_TH = 45
# 动态阈值上下限（相对基线倍数）
HEAD_FEET_DYNAMIC_MIN_SCALE = 0.60
HEAD_FEET_DYNAMIC_MAX_SCALE = 1.15

# 站立误判过滤（参考高度位移不足时跳过）
# 调大更紧（更容易跳过跌倒判定）
BASE_HEAD_DROP_IGNORE_TH = 5

# 状态机触发阈值
# 连续fall确认帧数（调大更紧）
FALL_CONFIRM_FRAMES = 2
# 仰角场景下更严格确认帧数
UPTILT_FALL_CONFIRM_FRAMES = 3
# 持续躺地触发帧数（调大更紧）
GROUND_FALL_FRAMES = 6
# 新轨迹最少稳定帧（调大更紧）
MIN_TRACK_STABLE_FRAMES = 3
# 仰角场景下最少稳定帧
UPTILT_MIN_TRACK_STABLE_FRAMES = 5
# 判定前至少连续横向帧数（调大更紧）
MIN_HORIZONTAL_FRAMES_FOR_FALL = 1
# 视频漏报补偿：低重心持续触发
LOW_CENTER_Y_RATIO_TH = 0.62
LOW_CENTER_BBOX_BOTTOM_RATIO_TH = 0.78
LOW_CENTER_HOLD_FRAMES = 4

# 事件去重与重复计数抑制
# 同一位置在短时间内重复触发，视为同一次摔倒
EVENT_DEDUP_FRAMES = 100
EVENT_DEDUP_CENTER_DIST_RATIO = 0.16
EVENT_DEDUP_IOU_TH = 0.25
# 同一 track 在冷却时间内不重复计数
FALL_RECOUNT_COOLDOWN_FRAMES = 210

# 水平躺倒宽松门控（用于前/后向倒地时头脚接近条件可能失效）
HORIZONTAL_RELAX_MIN_KEYPOINTS = 6
HORIZONTAL_RELAX_MIN_SIZE = 120
HORIZONTAL_RELAX_MAX_RATIO_TH = 0.72
HORIZONTAL_RELAX_SOFT_RATIO_MARGIN = 0.08
ENABLE_HORIZONTAL_RELAX_FALLBACK = True

# 侧身躺地门控（用于侧身对摄像头时提升召回）
# 条件：躯干轴近水平 + 人体主轴近水平 + 连续稳定若干帧
SIDE_LYING_TRUNK_DY_RATIO_TH = 0.34
SIDE_LYING_TRUNK_DX_RATIO_TH = 0.10
SIDE_LYING_AXIS_MAX_DEG = 55.0
SIDE_LYING_MIN_FRAMES = 2

# 已判跌倒后的稳定保持与恢复确认
FALLEN_MIN_HOLD_FRAMES = 20
RECOVERY_CONFIRM_FRAMES = 6

# 人体候选与轨迹保活
CORE_BODY_FALLBACK_LOWER_POINTS = 3
TRACK_STATE_TTL_FRAMES = 15
RECENT_IDENTITY_MAX_AGE_FRAMES = 90
RECENT_IDENTITY_MAX_DIST_RATIO = 0.12
RECENT_IDENTITY_POOL_MAX = 80

# 半蹲误报抑制（knee angle 越小表示屈膝越明显）
HALF_SQUAT_KNEE_ANGLE_TH = 136.0
HALF_SQUAT_HIP_ANKLE_DY_RATIO_TH = 0.18
HALF_SQUAT_HIP_ANKLE_DX_RATIO_TH = 0.32

# 跪姿误报抑制
KNEEL_KNEE_TO_ANKLE_DY_RATIO_TH = 0.16
KNEEL_HIP_TO_KNEE_DY_RATIO_TH = 0.10
NON_FALL_RESET_FRAMES = 2

# 仰角误报抑制
UPTILT_ORDER_RATIO_TH = 0.65
UPTILT_STRICT_HORIZONTAL_RATIO_TH = 0.60
UPTILT_STRICT_MIN_DEG = 50.0
HIGH_UPTILT_DEG_TH = 65.0
HIGH_UPTILT_STRICT_HORIZONTAL_RATIO_TH = 0.52
HIGH_UPTILT_GROUND_FRAMES = 18

# 真实躺地几何约束（大仰角场景）
TRUE_LYING_LEG_SPAN_RATIO_TH = 0.24
TRUE_LYING_TORSO_SPAN_RATIO_TH = 0.22

SUPPORT_OBJECT_LABELS = {"bed", "chair", "table", "dining table"}

# 每个 track_id 对应一套独立状态
person_states = {}
recent_fall_events = []
recent_identity_pool = []
frame_index = 0
fps_ema = 0.0
FPS_EMA_ALPHA = 0.2
inference_ms = 0.0
frame_ms = 0.0

camera_frame_buffers = {
    entry["camera_key"]: deque(maxlen=max(1, EVENT_CLIP_PRE_SECONDS * EVENT_CLIP_FPS))
    for entry in camera_entries
}
camera_pending_clips = {entry["camera_key"]: [] for entry in camera_entries}
camera_fall_counts = {entry["camera_key"]: 0 for entry in camera_entries}
camera_last_track_boxes: dict[str, dict[int, tuple[float, float, float, float]]] = {
    entry["camera_key"]: {} for entry in camera_entries
}
camera_virtual_track_next: dict[str, int] = {
    entry["camera_key"]: 100000 for entry in camera_entries
}
clip_write_queue: queue.Queue | None = None
clip_writer_stop_event = threading.Event()
clip_writer_thread: threading.Thread | None = None
warned_no_keypoints = False
interrupt_requested = False


def _handle_sigint(_signum, _frame) -> None:
    global interrupt_requested
    interrupt_requested = True


signal.signal(signal.SIGINT, _handle_sigint)

def get_posture(keypoints, horizontal_ratio_th):
    valid_xy = keypoints[(keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)]
    if len(valid_xy) >= 2:
        x_min, y_min = valid_xy.min(axis=0)
        x_max, y_max = valid_xy.max(axis=0)
        box_w = max(1.0, x_max - x_min)
        box_h = y_max - y_min
        ratio = box_h / box_w
        if ratio < horizontal_ratio_th:
            return "horizontal"

    nose = keypoints[0]
    left_ankle = keypoints[15]
    right_ankle = keypoints[16]

    # 身体高度
    body_height = max(left_ankle[1], right_ankle[1]) - nose[1]

    # 身体宽度
    x_coords = keypoints[:, 0]
    body_width = max(x_coords) - min(x_coords)

    if body_height < body_width:
        return "horizontal"
    else:
        return "vertical"

def valid_person(keypoints):
    """
    过滤无效人体（防止头/衣服误检）
    """

    if CPP_ACCEL_ENABLED and cpp_accel_module is not None:
        try:
            return bool(
                cpp_accel_module.valid_person(
                    np.asarray(keypoints, dtype=np.float32),
                    int(VALID_PERSON_MIN_KEYPOINTS),
                    float(VALID_PERSON_MIN_HEIGHT),
                )
            )
        except Exception:
            pass

    # 统计有效关键点（非0）
    valid_points = 0
    for x, y in keypoints:
        if x > 0 and y > 0:
            valid_points += 1

    # 少于4个关键点 → 忽略
    if valid_points < VALID_PERSON_MIN_KEYPOINTS:
        return False

    # 计算人体高度
    y_coords = keypoints[:, 1]
    height = max(y_coords) - min(y_coords)

    # 太小 → 忽略（远处/误检）
    if height < VALID_PERSON_MIN_HEIGHT:
        return False

    return True

def detect_fall_sequence(history):
    if len(history) < FALL_SEQ_MIN_HISTORY:
        return False

    postures = [h[0] for h in history]
    ref_positions = [h[1] for h in history]
    hip_positions = [h[2] for h in history]
    body_heights = [h[3] for h in history if len(h) > 3]

    # 1️⃣ 姿态变化：竖 → 横
    cond1 = ("vertical" in postures[:FALL_SEQ_WINDOW] and "horizontal" in postures[-FALL_SEQ_WINDOW:])

    # 2️⃣ 头部下降（修正方向）
    drop = ref_positions[-1] - ref_positions[0]
    cond2 = drop > FALL_SEQ_DROP_TH

    # 2.5️⃣ 髋部也应有明显下落（抑制俯拍下弯腰误判）
    hip_drop = hip_positions[-1] - hip_positions[0]
    cond2b = hip_drop > FALL_SEQ_HIP_DROP_TH

    # 3️⃣ 下降速度（关键🔥）
    speed = (ref_positions[-1] - ref_positions[-FALL_SEQ_SPEED_LOOKBACK]) / FALL_SEQ_SPEED_LOOKBACK
    cond3 = speed > FALL_SEQ_SPEED_TH

    # 4️⃣ 冲击门控：要求近几帧出现明显下落速度与加速度，抑制慢速弯腰/半蹲
    recent_ref = ref_positions[-max(4, FALL_SEQ_SPEED_LOOKBACK + 1):]
    frame_speeds = []
    for i in range(1, len(recent_ref)):
        frame_speeds.append(recent_ref[i] - recent_ref[i - 1])

    if len(frame_speeds) >= 2:
        max_speed = max(frame_speeds)
        max_accel = max(frame_speeds[i] - frame_speeds[i - 1] for i in range(1, len(frame_speeds)))
    elif len(frame_speeds) == 1:
        max_speed = frame_speeds[0]
        max_accel = 0.0
    else:
        max_speed = 0.0
        max_accel = 0.0

    cond4 = max_speed > FALL_SEQ_IMPACT_SPEED_TH or max_accel > FALL_SEQ_IMPACT_ACCEL_TH
    # 大幅下落兜底：即使冲击特征不明显，也允许触发（例如缓慢摔倒/遮挡）
    very_large_drop = drop > (1.4 * FALL_SEQ_DROP_TH)

    # 5️⃣ 侧向/后向摔倒常见特征：人体在图像中的纵向高度明显塌缩
    cond_shape = False
    if len(body_heights) >= 2:
        h0 = max(1.0, body_heights[0])
        h1 = max(1.0, body_heights[-1])
        shape_drop_ratio = (h0 - h1) / h0
        cond_shape = shape_drop_ratio > FALL_SEQ_BODY_HEIGHT_DROP_RATIO_TH

    # 主路径：完整证据
    strong_path = cond1 and cond2 and cond2b and cond3 and (cond4 or very_large_drop)
    # 恢复兜底（收紧）：仍要求髋部下落，避免站立前倾/抖动触发
    fallback_path = cond1 and cond2 and cond2b and cond3 and (cond4 or very_large_drop)

    # 侧后摔专用路径（收紧）：要求髋部同步下落，抑制站立误触发
    side_back_path = ("vertical" in postures[:FALL_SEQ_WINDOW]) and cond2 and cond2b and cond3 and cond_shape and (cond4 or very_large_drop)

    if strong_path or fallback_path or side_back_path:
        return True

    return False

def is_on_object(cx, cy, objects):
    for (x1, y1, x2, y2, cls) in objects:
        if x1 < cx < x2 and y1 < cy < y2:
            return True
    return False

def is_support_object_label(label):
    return label in SUPPORT_OBJECT_LABELS

def is_supported_by_object(keypoints, cx, cy, objects):
    support_objects = [obj for obj in objects if is_support_object_label(obj[4])]

    if is_on_object(cx, cy, support_objects):
        return True

    # 优先看躯干关键点是否落在支撑物上：肩(5,6) 髋(11,12)
    support_idxs = [5, 6, 11, 12]
    for idx in support_idxs:
        x, y = keypoints[idx]
        if x <= 0 or y <= 0:
            continue
        for (x1, y1, x2, y2, _) in support_objects:
            if x1 < x < x2 and y1 < y < y2:
                return True

    # 次级规则：人体框与支撑物框有明显重叠
    valid_xy = keypoints[(keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)]
    if len(valid_xy) >= 2:
        px1, py1 = valid_xy.min(axis=0)
        px2, py2 = valid_xy.max(axis=0)
        person_area = max(1.0, (px2 - px1) * (py2 - py1))
        for (x1, y1, x2, y2, _) in support_objects:
            ix1 = max(px1, x1)
            iy1 = max(py1, y1)
            ix2 = min(px2, x2)
            iy2 = min(py2, y2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter / person_area > SUPPORT_OVERLAP_TH:
                return True

    return False

def has_enough_lower_body(keypoints):
    # COCO关键点：11/12髋，13/14膝，15/16踝
    lower_idxs = [11, 12, 13, 14, 15, 16]
    valid_lower = 0
    for idx in lower_idxs:
        x, y = keypoints[idx]
        if x > 0 and y > 0:
            valid_lower += 1

    hips_valid = 0
    for idx in [11, 12]:
        x, y = keypoints[idx]
        if x > 0 and y > 0:
            hips_valid += 1

    # 至少1个髋点 + 总计至少3个下半身点，才做跌倒判断
    return hips_valid >= LOWER_BODY_MIN_HIPS and valid_lower >= LOWER_BODY_MIN_POINTS

def has_reliable_lower_body_geometry(keypoints):
    """
    下半身几何可信度校验：
    - 至少有髋和踝
    - 踝应明显低于髋
    - 若有膝点，膝应大致位于髋与踝之间
    """
    valid_xy = keypoints[(keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)]
    if len(valid_xy) < 4:
        return False

    _, y_min = valid_xy.min(axis=0)
    _, y_max = valid_xy.max(axis=0)
    body_h = max(1.0, y_max - y_min)

    hips = [keypoints[i][1] for i in [11, 12] if keypoints[i][0] > 0 and keypoints[i][1] > 0]
    knees = [keypoints[i][1] for i in [13, 14] if keypoints[i][0] > 0 and keypoints[i][1] > 0]
    ankles = [keypoints[i][1] for i in [15, 16] if keypoints[i][0] > 0 and keypoints[i][1] > 0]

    if not hips or not ankles:
        return False

    hip_y = float(np.mean(hips))
    ankle_y = float(np.mean(ankles))
    if (ankle_y - hip_y) <= LOWER_BODY_MIN_HIP_ANKLE_DY_RATIO * body_h:
        return False

    if knees:
        knee_y = float(np.mean(knees))
        tol = LOWER_BODY_KNEE_MID_TOL_RATIO * body_h
        if not (hip_y - tol <= knee_y <= ankle_y + tol):
            return False

    return True

def has_core_body(keypoints):
    shoulders_valid = 0
    for idx in [5, 6]:
        x, y = keypoints[idx]
        if x > 0 and y > 0:
            shoulders_valid += 1

    hips_valid = 0
    for idx in [11, 12]:
        x, y = keypoints[idx]
        if x > 0 and y > 0:
            hips_valid += 1

    return shoulders_valid >= CORE_BODY_MIN_SHOULDERS and hips_valid >= CORE_BODY_MIN_HIPS

def is_head_dominant_closeup(keypoints):
    """
    近景半身/大头特写抑制：
    - 估计头部区域（额顶到肩线）在人体框中的高度占比；
    - 占比过大时，认为是近景半身，不参与跌倒触发。
    """
    valid_xy = keypoints[(keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)]
    if len(valid_xy) < 4:
        return False

    x_min, y_min = valid_xy.min(axis=0)
    x_max, y_max = valid_xy.max(axis=0)
    body_h = max(1.0, y_max - y_min)

    face_pts = [keypoints[i] for i in [0, 1, 2, 3, 4] if keypoints[i][0] > 0 and keypoints[i][1] > 0]
    if len(face_pts) < 2:
        return False

    face_arr = np.array(face_pts, dtype=np.float32)
    face_x_min, face_y_min = face_arr.min(axis=0)
    face_x_max, _ = face_arr.max(axis=0)

    shoulder_ys = [keypoints[i][1] for i in [5, 6] if keypoints[i][0] > 0 and keypoints[i][1] > 0]
    if shoulder_ys:
        head_bottom_y = float(np.mean(shoulder_ys))
    else:
        # 无肩点时，使用脸部高度近似，避免误抑制
        face_h = max(1.0, float(face_arr[:, 1].max() - face_y_min))
        head_bottom_y = float(face_y_min + 2.2 * face_h)

    head_h = max(1.0, head_bottom_y - float(face_y_min))
    head_ratio = head_h / body_h

    face_w = max(1.0, float(face_x_max - face_x_min))
    head_aspect = face_w / head_h

    return head_ratio >= HEAD_DOMINANT_RATIO_TH and head_aspect >= HEAD_DOMINANT_MIN_ASPECT_RATIO

def is_probable_sitting(keypoints):
    # 用肩-髋-膝几何关系做“坐姿”门控，避免坐着被当成跌倒
    shoulders = []
    hips = []
    knees = []

    for idx in [5, 6]:
        x, y = keypoints[idx]
        if x > 0 and y > 0:
            shoulders.append((x, y))

    for idx in [11, 12]:
        x, y = keypoints[idx]
        if x > 0 and y > 0:
            hips.append((x, y))

    for idx in [13, 14]:
        x, y = keypoints[idx]
        if x > 0 and y > 0:
            knees.append((x, y))

    if len(shoulders) == 0 or len(hips) == 0 or len(knees) == 0:
        return False

    shoulder_center = np.mean(np.array(shoulders), axis=0)
    hip_center = np.mean(np.array(hips), axis=0)
    knee_center = np.mean(np.array(knees), axis=0)

    torso_dx = abs(hip_center[0] - shoulder_center[0])
    torso_dy = abs(hip_center[1] - shoulder_center[1])
    hip_knee_dy = abs(knee_center[1] - hip_center[1])

    valid_xy = keypoints[(keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)]
    if len(valid_xy) < 2:
        return False

    _, y_min = valid_xy.min(axis=0)
    _, y_max = valid_xy.max(axis=0)
    body_h = max(1.0, y_max - y_min)

    torso_vertical = torso_dy > SITTING_TORSO_VERTICAL_RATIO_TH * max(SITTING_TORSO_DX_FLOOR, torso_dx)
    short_thigh = hip_knee_dy < SITTING_SHORT_THIGH_RATIO_TH * body_h

    return torso_vertical and short_thigh

def joint_angle_deg(a, b, c):
    """返回夹角 ABC（单位：度）。"""
    ba = np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)
    bc = np.array(c, dtype=np.float32) - np.array(b, dtype=np.float32)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 180.0
    cos_v = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    cos_v = max(-1.0, min(1.0, cos_v))
    return float(np.degrees(np.arccos(cos_v)))

def is_probable_half_squat(keypoints):
    """
    半蹲门控：至少一条腿出现明显屈膝，且脚踝显著低于髋部，
    同时脚踝在水平方向接近髋部（人还在“竖向支撑”，不是躺倒）。
    用于抑制半蹲/前倾时被误判为摔倒。
    """
    valid_xy = keypoints[(keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)]
    if len(valid_xy) < 6:
        return False

    _, y_min = valid_xy.min(axis=0)
    _, y_max = valid_xy.max(axis=0)
    body_h = max(1.0, y_max - y_min)

    leg_triplets = [(11, 13, 15), (12, 14, 16)]  # (hip, knee, ankle)
    for hip_idx, knee_idx, ankle_idx in leg_triplets:
        hip = keypoints[hip_idx]
        knee = keypoints[knee_idx]
        ankle = keypoints[ankle_idx]

        if hip[0] <= 0 or hip[1] <= 0 or knee[0] <= 0 or knee[1] <= 0 or ankle[0] <= 0 or ankle[1] <= 0:
            continue

        knee_angle = joint_angle_deg(hip, knee, ankle)
        hip_ankle_dy = float(ankle[1] - hip[1])
        hip_ankle_dx = float(abs(ankle[0] - hip[0]))

        if (
            knee_angle < HALF_SQUAT_KNEE_ANGLE_TH
            and hip_ankle_dy > HALF_SQUAT_HIP_ANKLE_DY_RATIO_TH * body_h
            and hip_ankle_dx < HALF_SQUAT_HIP_ANKLE_DX_RATIO_TH * body_h
        ):
            return True

    return False

def is_probable_kneeling(keypoints):
    """
    跪姿门控：膝盖接近地面且与脚踝高度接近，髋部明显高于膝部。
    用于过滤单膝/双膝着地被当成倒地的场景。
    """
    valid_xy = keypoints[(keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)]
    if len(valid_xy) < 6:
        return False

    _, y_min = valid_xy.min(axis=0)
    _, y_max = valid_xy.max(axis=0)
    body_h = max(1.0, y_max - y_min)

    leg_triplets = [(11, 13, 15), (12, 14, 16)]  # (hip, knee, ankle)
    for hip_idx, knee_idx, ankle_idx in leg_triplets:
        hip = keypoints[hip_idx]
        knee = keypoints[knee_idx]
        ankle = keypoints[ankle_idx]

        if hip[0] <= 0 or hip[1] <= 0 or knee[0] <= 0 or knee[1] <= 0 or ankle[0] <= 0 or ankle[1] <= 0:
            continue

        knee_ankle_dy = abs(float(knee[1] - ankle[1]))
        hip_knee_dy = float(knee[1] - hip[1])

        knee_near_ankle = knee_ankle_dy < KNEEL_KNEE_TO_ANKLE_DY_RATIO_TH * body_h
        hip_above_knee = hip_knee_dy > KNEEL_HIP_TO_KNEE_DY_RATIO_TH * body_h

        if knee_near_ankle and hip_above_knee:
            return True

    return False

def is_uptilt_suspected(keypoints):
    """
    基于人体纵向关键点顺序判断是否可能是摄像头仰角场景。
    仰角时常出现“肩-髋-踝纵向顺序被压缩/错乱”，易触发误报。
    """
    shoulders = []
    hips = []
    ankles = []

    for idx in [5, 6]:
        x, y = keypoints[idx]
        if x > 0 and y > 0:
            shoulders.append(y)

    for idx in [11, 12]:
        x, y = keypoints[idx]
        if x > 0 and y > 0:
            hips.append(y)

    for idx in [15, 16]:
        x, y = keypoints[idx]
        if x > 0 and y > 0:
            ankles.append(y)

    if not shoulders or not hips or not ankles:
        return False

    shoulder_y = float(np.mean(shoulders))
    hip_y = float(np.mean(hips))
    ankle_y = float(np.mean(ankles))

    checks = [
        hip_y > shoulder_y,
        ankle_y > hip_y,
        ankle_y > shoulder_y,
    ]
    order_ratio = sum(1 for c in checks if c) / len(checks)
    return order_ratio < UPTILT_ORDER_RATIO_TH

def is_true_lying_geometry(keypoints):
    """
    高仰角下用于区分“前倾站立/半蹲”和“真正躺地”的几何约束：
    - 髋-踝纵向跨度应较小
    - 肩-髋纵向跨度应较小
    """
    valid_xy = keypoints[(keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)]
    if len(valid_xy) < 6:
        return False

    _, y_min = valid_xy.min(axis=0)
    _, y_max = valid_xy.max(axis=0)
    body_h = max(1.0, y_max - y_min)

    shoulders = [keypoints[i][1] for i in [5, 6] if keypoints[i][0] > 0 and keypoints[i][1] > 0]
    hips = [keypoints[i][1] for i in [11, 12] if keypoints[i][0] > 0 and keypoints[i][1] > 0]
    ankles = [keypoints[i][1] for i in [15, 16] if keypoints[i][0] > 0 and keypoints[i][1] > 0]

    if not shoulders or not hips or not ankles:
        return False

    shoulder_y = float(np.mean(shoulders))
    hip_y = float(np.mean(hips))
    ankle_y = float(np.mean(ankles))

    leg_span_ratio = abs(ankle_y - hip_y) / body_h
    torso_span_ratio = abs(hip_y - shoulder_y) / body_h

    return leg_span_ratio < TRUE_LYING_LEG_SPAN_RATIO_TH and torso_span_ratio < TRUE_LYING_TORSO_SPAN_RATIO_TH

def get_reference_y(keypoints):
    """
    选择更稳定的参考高度：优先用肩+髋中心，其次肩中心，再其次鼻子，最后退化为全体有效点均值。
    """
    shoulders = []
    hips = []

    for idx in [5, 6]:
        x, y = keypoints[idx]
        if x > 0 and y > 0:
            shoulders.append((x, y))

    for idx in [11, 12]:
        x, y = keypoints[idx]
        if x > 0 and y > 0:
            hips.append((x, y))

    if shoulders and hips:
        shoulder_center = np.mean(np.array(shoulders), axis=0)
        hip_center = np.mean(np.array(hips), axis=0)
        return float((shoulder_center[1] + hip_center[1]) / 2.0)

    if shoulders:
        shoulder_center = np.mean(np.array(shoulders), axis=0)
        return float(shoulder_center[1])

    nose = keypoints[0]
    if nose[0] > 0 and nose[1] > 0:
        return float(nose[1])

    valid_y = keypoints[(keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)][:, 1]
    if len(valid_y) > 0:
        return float(np.mean(valid_y))

    return 0.0

def is_standing_bending(keypoints, hip_th, shoulder_th):
    """
    判断是否是俯视角度下的站立低头 posture 或弯腰动作
    如果踝关节比髋关节或肩关节明显在下方，通常还在站立/弯腰，并未完全倒地
    """
    shoulders = []
    hips = []
    ankles = []
    for idx in [5, 6]:
        if keypoints[idx][0] > 0 and keypoints[idx][1] > 0:
            shoulders.append(keypoints[idx][1])
    for idx in [11, 12]:
        if keypoints[idx][0] > 0 and keypoints[idx][1] > 0:
            hips.append(keypoints[idx][1])
    for idx in [15, 16]:
        if keypoints[idx][0] > 0 and keypoints[idx][1] > 0:
            ankles.append(keypoints[idx][1])
            
    if not ankles:
        return False
        
    ankle_y = np.mean(ankles)
    
    # 身体整体高度
    y_coords = keypoints[(keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)][:, 1]
    if len(y_coords) < 2:
        return False
    body_h = max(1.0, np.max(y_coords) - np.min(y_coords))
    
    # 优先判断腿部是否直立：通过髋关节(11,12)到踝关节(15,16)的距离
    if hips:
        hip_y = np.mean(hips)
        if (ankle_y - hip_y) > hip_th * body_h:
            return True
    
    # 退场判断：如果没有检测到髋关节，使用肩膀
    if shoulders:
        shoulder_y = np.mean(shoulders)
        if (ankle_y - shoulder_y) > shoulder_th * body_h:
            return True
            
    return False

def get_head_feet_dynamic_scale(keypoints):
    """
    基于下肢在图像Y轴的展开程度估计俯拍强度：
    - 俯拍越强，下肢纵向展开通常越小 -> 收紧头脚接近阈值。
    返回值越小表示阈值越严格。
    """
    valid_xy = keypoints[(keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)]
    if len(valid_xy) < 2:
        return 1.0

    _, y_min = valid_xy.min(axis=0)
    _, y_max = valid_xy.max(axis=0)
    body_h = max(1.0, y_max - y_min)

    hips = []
    ankles = []
    for idx in [11, 12]:
        if keypoints[idx][0] > 0 and keypoints[idx][1] > 0:
            hips.append(keypoints[idx][1])
    for idx in [15, 16]:
        if keypoints[idx][0] > 0 and keypoints[idx][1] > 0:
            ankles.append(keypoints[idx][1])

    if not hips or not ankles:
        return 1.0

    hip_y = float(np.mean(hips))
    ankle_y = float(np.mean(ankles))
    leg_span_ratio = abs(ankle_y - hip_y) / body_h

    # 经验映射：比值越小，俯拍越强，阈值越收紧
    if leg_span_ratio < 0.20:
        scale = 0.60
    elif leg_span_ratio < 0.28:
        scale = 0.75
    elif leg_span_ratio < 0.36:
        scale = 0.90
    else:
        scale = 1.05

    return min(HEAD_FEET_DYNAMIC_MAX_SCALE, max(HEAD_FEET_DYNAMIC_MIN_SCALE, scale))

def is_head_feet_height_close(keypoints):
    """
    头部与脚踝在图像Y轴上的高度接近，常见于倒地/躺地姿态。
    仅作为辅助条件，避免单一姿态误判。
    """
    nose = keypoints[0]
    if nose[0] <= 0 or nose[1] <= 0:
        return False

    ankles = []
    for idx in [15, 16]:
        if keypoints[idx][0] > 0 and keypoints[idx][1] > 0:
            ankles.append(keypoints[idx][1])

    if not ankles:
        return False

    valid_xy = keypoints[(keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)]
    if len(valid_xy) < 2:
        return False

    _, y_min = valid_xy.min(axis=0)
    _, y_max = valid_xy.max(axis=0)
    body_h = max(1.0, y_max - y_min)

    ankle_y = float(np.mean(ankles))
    dy = abs(ankle_y - nose[1])
    dynamic_scale = get_head_feet_dynamic_scale(keypoints)
    close_th = max(HEAD_FEET_CLOSE_ABS_PX_TH * dynamic_scale,
                   HEAD_FEET_CLOSE_RATIO_TH * body_h * dynamic_scale)
    return dy <= close_th

def is_side_lying_pose(keypoints):
    """
    侧身躺地判据：
    1) 肩-髋躯干轴在图像中接近水平；
    2) 全身关键点主轴接近水平（PCA主方向）；
    用于补偿“侧身朝摄像头时 head-feet 高度关系不明显”的漏检。
    """
    valid_xy = keypoints[(keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)]
    if len(valid_xy) < 5:
        return False

    x_min, y_min = valid_xy.min(axis=0)
    x_max, y_max = valid_xy.max(axis=0)
    body_h = max(1.0, y_max - y_min)

    shoulders = []
    hips = []
    for idx in [5, 6]:
        x, y = keypoints[idx]
        if x > 0 and y > 0:
            shoulders.append((x, y))
    for idx in [11, 12]:
        x, y = keypoints[idx]
        if x > 0 and y > 0:
            hips.append((x, y))

    if not shoulders or not hips:
        return False

    shoulder_center = np.mean(np.array(shoulders), axis=0)
    hip_center = np.mean(np.array(hips), axis=0)
    trunk_dx = abs(float(hip_center[0] - shoulder_center[0]))
    trunk_dy = abs(float(hip_center[1] - shoulder_center[1]))

    trunk_horizontal = (
        trunk_dy <= SIDE_LYING_TRUNK_DY_RATIO_TH * body_h
        and trunk_dx >= SIDE_LYING_TRUNK_DX_RATIO_TH * body_h
    )

    centered = valid_xy - np.mean(valid_xy, axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        main_axis = vt[0]
        axis_angle = abs(float(np.degrees(np.arctan2(main_axis[1], main_axis[0]))))
        if axis_angle > 90.0:
            axis_angle = 180.0 - axis_angle
    except np.linalg.LinAlgError:
        return False

    axis_horizontal = axis_angle <= SIDE_LYING_AXIS_MAX_DEG
    return trunk_horizontal and axis_horizontal

pitch_history = []
pitch_baseline = None

def bbox_iou(box_a, box_b):
    if CPP_ACCEL_ENABLED and cpp_accel_module is not None:
        try:
            return float(cpp_accel_module.bbox_iou(box_a, box_b))
        except Exception:
            pass

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter_area / (area_a + area_b - inter_area + 1e-6)


def _assign_track_ids_from_previous(camera_key: str, boxes: list[tuple[float, float, float, float]]) -> list[int]:
    prev_boxes = camera_last_track_boxes.get(camera_key, {})
    used_prev_ids = set()
    assigned_ids: list[int] = []

    for box in boxes:
        best_id = -1
        best_iou = 0.0
        for prev_id, prev_box in prev_boxes.items():
            if prev_id in used_prev_ids:
                continue
            iou = bbox_iou(box, prev_box)
            if iou > best_iou:
                best_iou = iou
                best_id = int(prev_id)

        if best_id >= 0 and best_iou >= TRACK_REUSE_IOU_TH:
            assigned_ids.append(best_id)
            used_prev_ids.add(best_id)
        else:
            new_id = int(camera_virtual_track_next.get(camera_key, 100000))
            camera_virtual_track_next[camera_key] = new_id + 1
            assigned_ids.append(new_id)

    return assigned_ids


def _update_track_box_cache(camera_key: str, track_ids: list[int], boxes: list[tuple[float, float, float, float]]) -> None:
    updated: dict[int, tuple[float, float, float, float]] = {}
    for idx, box in enumerate(boxes):
        if idx >= len(track_ids):
            continue
        x1, y1, x2, y2 = box
        if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
            continue
        updated[int(track_ids[idx])] = (float(x1), float(y1), float(x2), float(y2))
    camera_last_track_boxes[camera_key] = updated

def is_duplicate_person_bbox(box_a, box_b, frame_w, frame_h):
    if CPP_ACCEL_ENABLED and cpp_accel_module is not None:
        try:
            return bool(
                cpp_accel_module.is_duplicate_person_bbox(
                    box_a,
                    box_b,
                    float(frame_w),
                    float(frame_h),
                    float(DUPLICATE_IOU_TH),
                    float(DUPLICATE_CENTER_DX_RATIO_TH),
                    float(DUPLICATE_CENTER_DY_RATIO_TH),
                    float(DUPLICATE_X_OVERLAP_RATIO_TH),
                    float(DUPLICATE_VERTICAL_GAP_RATIO_TH),
                    float(DUPLICATE_SPLIT_CENTER_DX_RATIO_TH),
                    float(DUPLICATE_CONTAIN_RATIO_TH),
                )
            )
        except Exception:
            pass

    iou = bbox_iou(box_a, box_b)
    if iou > DUPLICATE_IOU_TH:
        return True

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    aw = max(1.0, ax2 - ax1)
    ah = max(1.0, ay2 - ay1)
    bw = max(1.0, bx2 - bx1)
    bh = max(1.0, by2 - by1)

    acx = (ax1 + ax2) / 2
    acy = (ay1 + ay2) / 2
    bcx = (bx1 + bx2) / 2
    bcy = (by1 + by2) / 2

    # 情况1：中心非常接近（遮挡后重复框）
    if abs(acx - bcx) < DUPLICATE_CENTER_DX_RATIO_TH * frame_w and abs(acy - bcy) < DUPLICATE_CENTER_DY_RATIO_TH * frame_h:
        return True

    # 情况2：上下分裂，但X方向重合很高
    x_overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    x_overlap_ratio = x_overlap / max(1.0, min(aw, bw))
    vertical_gap = max(0.0, max(ay1, by1) - min(ay2, by2))
    if x_overlap_ratio > DUPLICATE_X_OVERLAP_RATIO_TH and vertical_gap < DUPLICATE_VERTICAL_GAP_RATIO_TH * frame_h and abs(acx - bcx) < DUPLICATE_SPLIT_CENTER_DX_RATIO_TH * frame_w:
        return True

    # 情况3：一个框几乎包含在另一个框内
    inner_left = max(ax1, bx1)
    inner_top = max(ay1, by1)
    inner_right = min(ax2, bx2)
    inner_bottom = min(ay2, by2)
    inner_area = max(0.0, inner_right - inner_left) * max(0.0, inner_bottom - inner_top)
    if inner_area / max(1.0, min(aw * ah, bw * bh)) > DUPLICATE_CONTAIN_RATIO_TH:
        return True

    return False

def save_fall_snapshot(frame, frame_idx: int, elder_code: str = "") -> str:
    if frame is None:
        return ""

    try:
        FACES_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        code_part = str(elder_code or "unknown").strip() or "unknown"
        file_name = f"fall_{ts}_{int(frame_idx):06d}_{code_part}.jpg"
        file_path = FACES_DIR / file_name
        ok = cv2.imwrite(str(file_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return ""
        return f"/static/faces/{file_name}"
    except Exception:
        return ""


def _probe_video_codec(file_path: Path) -> str:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
        )
        if result.returncode != 0:
            return ""
        return str(result.stdout or "").strip().lower()
    except Exception:
        return ""


def fix_video_for_browser(file_path: Path) -> bool:
    codec = _probe_video_codec(file_path)
    if "mpeg4" not in codec:
        return True

    tmp_path = file_path.with_name(f"{file_path.stem}_tmp{file_path.suffix}")
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(file_path),
                "-c:v",
                "libx264",
                "-preset",
                "superfast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=90,
        )
        if result.returncode != 0:
            print(f"[clip] ffmpeg transcode failed for {file_path.name}: {result.stderr.strip()}")
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            return False

        tmp_path.replace(file_path)
        print(f"[clip] browser-fix applied {file_path.name}: mpeg4 -> h264")
        return True
    except Exception as error:
        print(f"[clip] browser-fix exception for {file_path.name}: {error}")
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return False


def _write_clip_file(frames: list, file_path: Path, fps: int) -> bool:
    if not frames:
        return False
    h, w = frames[0].shape[:2]

    # 清理旧文件，避免残留空文件影响本次判定
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception:
            pass

    used_codec = ""
    write_ok = False
    for codec in EVENT_CLIP_CODECS or ["H264", "avc1", "mp4v"]:
        writer = cv2.VideoWriter(
            str(file_path),
            cv2.VideoWriter_fourcc(*codec),
            float(max(1, fps)),
            (int(w), int(h)),
        )
        if not writer.isOpened():
            writer.release()
            continue

        wrote_frames = 0
        for frame in frames:
            if frame is None or getattr(frame, "size", 0) == 0:
                continue
            fh, fw = frame.shape[:2]
            if fh != h or fw != w:
                frame = cv2.resize(frame, (int(w), int(h)))
            writer.write(frame)
            wrote_frames += 1
        writer.release()

        file_size = 0
        try:
            if file_path.exists():
                file_size = int(file_path.stat().st_size)
        except Exception:
            file_size = 0

        # 部分后端会“打开成功但写出空文件”，这里做最低可用校验
        if wrote_frames > 0 and file_size > 2048:
            used_codec = codec
            write_ok = True
            break

        try:
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass

    if not write_ok:
        print(
            f"[clip] failed to save {file_path.name} with codecs={EVENT_CLIP_CODECS or ['H264', 'avc1', 'mp4v']}"
        )
        return False

    if used_codec:
        print(f"[clip] saved {file_path.name} codec={used_codec}")
    if EVENT_CLIP_AUTO_FIX_BROWSER:
        fix_video_for_browser(file_path)
    return True


def _ensure_clip_writer_started() -> None:
    global clip_write_queue, clip_writer_thread
    if not EVENT_CLIP_ASYNC_WRITE:
        return
    if clip_writer_thread is not None and clip_writer_thread.is_alive():
        return

    clip_write_queue = queue.Queue(maxsize=EVENT_CLIP_QUEUE_SIZE)

    def _worker() -> None:
        while not clip_writer_stop_event.is_set():
            try:
                job = clip_write_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if job is None:
                clip_write_queue.task_done()
                break

            saved = _write_clip_file(job["frames"], job["file_path"], job["fps"])
            if not saved:
                print(f"[clip] async save failed: camera={job['camera_key']} file={job['file_path'].name}")
            clip_write_queue.task_done()

    clip_writer_thread = threading.Thread(target=_worker, daemon=True)
    clip_writer_thread.start()


def _dispatch_clip_write(camera_key: str, item: dict) -> None:
    if EVENT_CLIP_ASYNC_WRITE:
        _ensure_clip_writer_started()
        if clip_write_queue is not None:
            try:
                clip_write_queue.put_nowait({
                    "camera_key": camera_key,
                    "file_path": item["file_path"],
                    "frames": item["frames"],
                    "fps": EVENT_CLIP_FPS,
                })
                return
            except queue.Full:
                print(f"[clip] queue full, fallback sync write: {item['file_path'].name}")

    saved = _write_clip_file(item["frames"], item["file_path"], EVENT_CLIP_FPS)
    if not saved:
        print(f"[clip] save failed: camera={camera_key} file={item['file_path'].name}")


def update_pending_event_clips(camera_key: str, frame) -> None:
    pending = camera_pending_clips.get(camera_key, [])
    if not pending:
        return

    done_indexes = []
    for idx, item in enumerate(pending):
        item["frames"].append(frame.copy())
        item["post_left"] -= 1
        if item["post_left"] <= 0:
            _dispatch_clip_write(camera_key, item)
            done_indexes.append(idx)

    for idx in reversed(done_indexes):
        pending.pop(idx)


def enqueue_event_clip(camera_key: str, frame_idx: int, elder_code: str = "") -> str:
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    code_part = str(elder_code or "unknown").strip() or "unknown"
    file_name = f"fall_{camera_key}_{ts}_{int(frame_idx):06d}_{code_part}.{EVENT_CLIP_EXT}"
    file_path = FACES_DIR / file_name
    url_path = f"/static/faces/{file_name}"

    pre_frames = [f.copy() for f in list(camera_frame_buffers.get(camera_key, []))]
    camera_pending_clips[camera_key].append({
        "file_path": file_path,
        "frames": pre_frames,
        "post_left": max(1, EVENT_CLIP_POST_SECONDS * EVENT_CLIP_FPS),
    })
    return url_path


def try_register_fall_event(bbox, cx, cy, frame_w, frame_h, frame_idx, elder_code="", frame=None, camera_id="cam-0", alert_level="confirmed"):
    # 基于时间 + 空间去重：同一人同一次倒地只记一次
    global fall_count

    level_rank_map = {"suspected": 1, "confirmed": 2, "unrecovered": 3}
    level_rank = int(level_rank_map.get(str(alert_level), 2))

    max_dist = EVENT_DEDUP_CENTER_DIST_RATIO * max(frame_w, frame_h)
    same_event_idx = -1
    for event in recent_fall_events:
        if str(event.get("camera_id") or "") != str(camera_id):
            continue
        if frame_idx - event["frame"] > EVENT_DEDUP_FRAMES:
            continue

        prev_cx, prev_cy = event["center"]
        center_dist = math.hypot(cx - prev_cx, cy - prev_cy)
        near_same = (center_dist <= max_dist) or (bbox_iou(bbox, event["bbox"]) >= EVENT_DEDUP_IOU_TH)
        if near_same:
            prev_rank = int(event.get("level_rank", 0) or 0)
            if level_rank <= prev_rank:
                return False
            same_event_idx = recent_fall_events.index(event)
            break

    if same_event_idx >= 0:
        recent_fall_events[same_event_idx]["frame"] = frame_idx
        recent_fall_events[same_event_idx]["center"] = (float(cx), float(cy))
        recent_fall_events[same_event_idx]["bbox"] = bbox
        recent_fall_events[same_event_idx]["level_rank"] = level_rank
    else:
        recent_fall_events.append({
            "frame": frame_idx,
            "center": (float(cx), float(cy)),
            "bbox": bbox,
            "camera_id": str(camera_id),
            "level_rank": level_rank,
        })

    fall_count += 1
    camera_fall_counts[str(camera_id)] = camera_fall_counts.get(str(camera_id), 0) + 1
    snapshot_path = save_fall_snapshot(frame, frame_idx, str(elder_code or ""))
    video_path = enqueue_event_clip(str(camera_id), frame_idx, str(elder_code or ""))

    if report_fall_event is not None:
        try:
            report_fall_event({
                "event_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "camera_id": str(camera_id),
                "frame_idx": int(frame_idx),
                "cx": float(cx),
                "cy": float(cy),
                "bbox": [float(v) for v in bbox],
                "elder_code": str(elder_code or ""),
                "alert_level": str(alert_level),
                "snapshot_path": snapshot_path,
                "video_path": video_path,
            })
        except Exception as error:
            print(f"[WARN] 外部告警/入库失败: {error}")

    return True


def allocate_elder_code() -> str:
    if ensure_elder is None:
        return ""
    try:
        elder = ensure_elder(DEFAULT_DB_PATH, None)
        return str(elder.get("elder_code") or "")
    except Exception:
        return ""


def prune_recent_identity_pool(current_frame: int, camera_key: str | None = None) -> None:
    recent_identity_pool[:] = [
        item for item in recent_identity_pool
        if current_frame - int(item.get("frame", current_frame)) <= RECENT_IDENTITY_MAX_AGE_FRAMES
    ]


def push_recent_identity(elder_code: str, cx: float, cy: float, current_frame: int, camera_key: str) -> None:
    code = str(elder_code or "").strip()
    if not code:
        return

    prune_recent_identity_pool(current_frame)
    recent_identity_pool.append({
        "elder_code": code,
        "cx": float(cx),
        "cy": float(cy),
        "frame": int(current_frame),
        "camera_key": str(camera_key),
    })
    if len(recent_identity_pool) > RECENT_IDENTITY_POOL_MAX:
        recent_identity_pool[:] = recent_identity_pool[-RECENT_IDENTITY_POOL_MAX:]


def try_reuse_recent_identity(cx: float, cy: float, frame_w: int, frame_h: int, current_frame: int, camera_key: str) -> str:
    prune_recent_identity_pool(current_frame)
    if not recent_identity_pool:
        return ""

    max_dist = RECENT_IDENTITY_MAX_DIST_RATIO * max(float(frame_w), float(frame_h))
    best_idx = -1
    best_dist = 1e12

    for idx, item in enumerate(recent_identity_pool):
        if str(item.get("camera_key") or "") != str(camera_key):
            continue
        dx = float(cx) - float(item.get("cx", 0.0))
        dy = float(cy) - float(item.get("cy", 0.0))
        dist = math.hypot(dx, dy)
        if dist <= max_dist and dist < best_dist:
            best_dist = dist
            best_idx = idx

    if best_idx < 0:
        return ""

    code = str(recent_identity_pool[best_idx].get("elder_code") or "").strip()
    recent_identity_pool.pop(best_idx)
    return code


def extract_face_roi(frame, keypoints, bbox):
    face_points = keypoints[0:5]
    valid_face = face_points[(face_points[:, 0] > 0) & (face_points[:, 1] > 0)]
    valid_count = int(len(valid_face))
    if valid_count < FACE_MIN_KEYPOINTS_FOR_RECOG:
        return None, valid_count

    fx_min, fy_min = valid_face.min(axis=0)
    fx_max, fy_max = valid_face.max(axis=0)
    face_w = max(24.0, fx_max - fx_min)
    face_h = max(24.0, fy_max - fy_min)

    cx = (fx_min + fx_max) / 2.0
    cy = (fy_min + fy_max) / 2.0
    x1 = int(cx - 1.15 * face_w)
    x2 = int(cx + 1.15 * face_w)
    y1 = int(cy - 1.25 * face_h)
    y2 = int(cy + 1.55 * face_h)

    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))

    if x2 - x1 < 36 or y2 - y1 < 36:
        return None, valid_count

    x_min, y_min, x_max, y_max = bbox
    body_h = max(1.0, y_max - y_min)
    roi_bottom_ratio = (float(y2) - float(y_min)) / body_h
    if roi_bottom_ratio > 0.62:
        return None, valid_count

    face_roi = frame[y1:y2, x1:x2]
    if face_roi.size == 0:
        return None, valid_count
    return face_roi, valid_count


def _compute_avatar_score(face_crop, valid_face_points: int) -> float:
    h, w = face_crop.shape[:2]
    if h <= 0 or w <= 0:
        return 0.0

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharp_norm = min(1.0, sharpness / 280.0)
    size_norm = min(1.0, (w * h) / float(140 * 140))
    face_bonus = 1.0 if valid_face_points >= 3 else 0.72
    return (0.58 * size_norm + 0.42 * sharp_norm) * face_bonus


def save_elder_avatar(frame, keypoints, bbox, elder_code: str, state: dict) -> None:
    if not elder_code or update_elder_avatar is None:
        return

    face_crop, valid_count = extract_face_roi(frame, keypoints, bbox)
    if face_crop is None:
        return

    score = _compute_avatar_score(face_crop, valid_count)
    if score < AVATAR_MIN_SCORE:
        return

    prev_score = float(state.get("avatar_best_score", 0.0) or 0.0)
    should_update = (not state.get("avatar_saved")) or (score > prev_score + AVATAR_UPDATE_SCORE_MARGIN)
    if not should_update:
        return

    FACES_DIR.mkdir(parents=True, exist_ok=True)
    file_path = FACES_DIR / f"{elder_code}.jpg"
    ok = cv2.imwrite(str(file_path), face_crop)
    if not ok:
        return

    avatar_path = f"/static/faces/{elder_code}.jpg"
    if update_elder_avatar(DEFAULT_DB_PATH, elder_code, avatar_path):
        state["avatar_saved"] = True
        state["avatar_path"] = avatar_path
        state["avatar_best_score"] = score

while True:
    if interrupt_requested:
        print("[INFO] 收到 Ctrl+C，正在退出...")
        break

    if not camera_entries:
        break

    current_camera_idx = frame_index % len(camera_entries)
    current_entry = camera_entries[current_camera_idx]
    camera_key = str(current_entry["camera_key"])
    cap = current_entry["capture"]

    frame_start_t = time.perf_counter()
    frame_index += 1

    # 清理过期事件，避免列表无限增长
    recent_fall_events[:] = [
        e for e in recent_fall_events if frame_index - e["frame"] <= EVENT_DEDUP_FRAMES
    ]
    prune_recent_identity_pool(frame_index, camera_key)

    with camera_locks[camera_key]:
        latest = camera_latest_frames.get(camera_key)
        frame = latest.copy() if latest is not None else None
    if frame is None:
        time.sleep(0.005)
        continue

    camera_frame_buffers[camera_key].append(frame.copy())
    update_pending_event_clips(camera_key, frame)

    estimated_pitch = CAMERA_PITCH_ANGLE
    stage_after_infer_t = 0.0

    # 跟踪后端：ultralytics(默认) 或 deepsort
    infer_start_t = time.perf_counter()
    used_tracker_this_frame = True
    if TRACKER_BACKEND == "deepsort":
        results = model(frame, verbose=False)[0]
    else:
        if TRACK_FRAME_STRIDE > 1 and (frame_index % TRACK_FRAME_STRIDE != 0):
            results = model(frame, verbose=False)[0]
            used_tracker_this_frame = False
        else:
            results = model.track(frame, persist=TRACK_PERSIST, tracker=TRACKER_CFG, verbose=False)[0]

    if interrupt_requested:
        print("[INFO] 收到 Ctrl+C，正在退出...")
        break

    infer_end_t = time.perf_counter()
    stage_after_infer_t = infer_end_t

    if hasattr(results, "speed") and isinstance(results.speed, dict):
        inference_ms = float(results.speed.get("inference", 0.0))
    else:
        inference_ms = (infer_end_t - infer_start_t) * 1000.0

    if FALL_PERF_INFER_ONLY:
        frame_ms = (time.perf_counter() - frame_start_t) * 1000.0
        instant_fps = 1000.0 / max(1e-6, frame_ms)
        if fps_ema <= 0.0:
            fps_ema = instant_fps
        else:
            fps_ema = FPS_EMA_ALPHA * instant_fps + (1.0 - FPS_EMA_ALPHA) * fps_ema

        cv2.putText(frame, f"PERF-INFER-ONLY FPS: {fps_ema:.1f} Infer: {inference_ms:.1f} ms", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.imshow(f"Smart AI Fall Detection - {camera_key}", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    alert_text = ""

    objects = []  # 存储可支撑物：床/椅/桌

    # 1️⃣ 检测可支撑物（床 / 椅 / 桌）
    for box in results.boxes:
        conf = float(box.conf[0])   # ⭐ 新增
        if conf < DETECTION_CONF_TH:
            continue

        cls = int(box.cls[0])
        label = model.names[cls]

        if is_support_object_label(label):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            objects.append((x1, y1, x2, y2, label))

            if DRAW_OVERLAYS and DRAW_SUPPORT_OBJECTS:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # 2️⃣ 姿态 + 跌倒检测
    person_count = 0

    active_track_ids = set()

    if results.keypoints is None:
        if (not warned_no_keypoints) and str(MODEL_PATH).lower().endswith(".engine"):
            print(
                "[WARN] 当前 .engine 推理结果没有 keypoints，跌倒事件不会触发。"
                "请确认 engine 来自 pose 模型导出，或改用 *.pt pose 模型。"
            )
            warned_no_keypoints = True

    if results.keypoints is not None:
        num_people = len(results.keypoints.xy)
        track_ids = list(range(num_people))
        keypoint_boxes: list[tuple[float, float, float, float]] = []
        keypoint_confs: list[float] = []

        for idx, k in enumerate(results.keypoints.xy):
            keypoints = k.cpu().numpy()
            valid_xy = keypoints[(keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)]
            if len(valid_xy) >= 2:
                x_min, y_min = valid_xy.min(axis=0)
                x_max, y_max = valid_xy.max(axis=0)
                keypoint_boxes.append((float(x_min), float(y_min), float(x_max), float(y_max)))
            else:
                keypoint_boxes.append((0.0, 0.0, 0.0, 0.0))

            if results.boxes is not None and idx < len(results.boxes):
                keypoint_confs.append(float(results.boxes[idx].conf[0]))
            else:
                keypoint_confs.append(0.5)

        if TRACKER_BACKEND == "ultralytics":
            if results.boxes is not None and results.boxes.id is not None:
                track_ids = results.boxes.id.int().cpu().tolist()
            else:
                track_ids = _assign_track_ids_from_previous(camera_key, keypoint_boxes)
        elif TRACKER_BACKEND == "deepsort" and camera_key in camera_trackers:
            detections = []
            det_indices = []
            for idx, bbox in enumerate(keypoint_boxes):
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                if w < 2.0 or h < 2.0:
                    continue
                detections.append(([float(x1), float(y1), float(w), float(h)], float(keypoint_confs[idx]), "person"))
                det_indices.append(idx)

            tracks = camera_trackers[camera_key].update_tracks(detections, frame=frame)
            used_det = set()
            for trk in tracks:
                if not trk.is_confirmed():
                    continue
                l, t, r, b = trk.to_ltrb()
                track_box = (float(l), float(t), float(r), float(b))

                best_idx = -1
                best_iou = 0.0
                for det_idx in det_indices:
                    if det_idx in used_det:
                        continue
                    iou = bbox_iou(track_box, keypoint_boxes[det_idx])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = det_idx

                if best_idx >= 0 and best_iou >= 0.1:
                    track_ids[best_idx] = int(trk.track_id)
                    used_det.add(best_idx)

        _update_track_box_cache(camera_key, [int(tid) for tid in track_ids], keypoint_boxes)

        frame_h, frame_w = frame.shape[:2]
        # 先收集候选人体，再做同帧去重，最后进入状态机
        candidates = []
        frame_max_ratio = 0.0

        for idx, k in enumerate(results.keypoints.xy):
            keypoints = k.cpu().numpy()
            raw_track_id = track_ids[idx] if track_ids is not None and idx < len(track_ids) else idx
            track_id = (camera_key, int(raw_track_id))

            if not valid_person(keypoints):
                continue

            # 过滤仅头脸/局部肢体的人体误检，防止误触发跌倒
            if not has_core_body(keypoints):
                lower_visible = int(np.sum((keypoints[11:17, 0] > 0) & (keypoints[11:17, 1] > 0)))
                if lower_visible < CORE_BODY_FALLBACK_LOWER_POINTS:
                    continue

            valid_xy = keypoints[(keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)]
            if len(valid_xy) < 2:
                continue

            x_min, y_min = valid_xy.min(axis=0)
            x_max, y_max = valid_xy.max(axis=0)
            bbox = (float(x_min), float(y_min), float(x_max), float(y_max))

            # 用于角度估算：收集当前帧最直立人体的宽高比
            box_ratio = (y_max - y_min) / max(1.0, x_max - x_min)
            nose_y = keypoints[0][1]
            hips_y = [keypoints[i][1] for i in [11, 12] if keypoints[i][1] > 0]
            ankles_y = [keypoints[i][1] for i in [15, 16] if keypoints[i][1] > 0]
            if nose_y > 0 and hips_y and ankles_y:
                mean_hip = sum(hips_y) / len(hips_y)
                mean_ankle = sum(ankles_y) / len(ankles_y)
                if nose_y < mean_hip < mean_ankle:
                    frame_max_ratio = max(frame_max_ratio, box_ratio)

            # 质量分：关键点数量 + 下半身完整度 + 框面积
            valid_points = int(np.sum((keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)))
            lower_valid = int(np.sum((keypoints[11:17, 0] > 0) & (keypoints[11:17, 1] > 0)))
            area = max(1.0, (x_max - x_min) * (y_max - y_min))
            quality = valid_points + 2.0 * lower_valid + area / 10000.0

            candidates.append({
                "track_id": track_id,
                "keypoints": keypoints,
                "valid_xy": valid_xy,
                "bbox": bbox,
                "quality": quality,
            })

        # 同帧去重：遮挡导致同一人被拆成两个检测时，只保留质量更高者
        candidates.sort(key=lambda item: item["quality"], reverse=True)
        filtered_people = []
        for candidate in candidates:
            is_dup = False
            for kept in filtered_people:
                if is_duplicate_person_bbox(candidate["bbox"], kept["bbox"], frame_w, frame_h):
                    is_dup = True
                    break
            if not is_dup:
                filtered_people.append(candidate)

        person_count = len(filtered_people)

        # 自动修正：基于人群估计摄像头俯仰角度
        if frame_max_ratio > 1.0:
            pitch_history.append(frame_max_ratio)
            if len(pitch_history) > PITCH_HISTORY_MAX:
                pitch_history.pop(0)

        if AUTO_DETECT_CAMERA_ANGLE and len(pitch_history) >= PITCH_HISTORY_MIN:
            best_ratio = np.percentile(pitch_history, PITCH_RATIO_PERCENTILE)

            if pitch_baseline is None:
                pitch_baseline = max(best_ratio, PITCH_RATIO_BASELINE)
            else:
                pitch_baseline = max(best_ratio, pitch_baseline * (1.0 - PITCH_BASELINE_DECAY))

            baseline_ratio = max(1.0, pitch_baseline)
            cos_theta = max(0.1, min(1.0, best_ratio / baseline_ratio))
            estimated_pitch = math.degrees(math.acos(cos_theta))
        else:
            estimated_pitch = CAMERA_PITCH_ANGLE

        estimated_pitch = max(PITCH_MIN_DEG, min(PITCH_MAX_DEG, estimated_pitch))
        height_comp = max(0.3, math.cos(math.radians(estimated_pitch)))
        inv_comp = 1.0 / max(0.3, height_comp)
        inv_comp = max(THRESH_COMP_MIN, min(THRESH_COMP_MAX, inv_comp))

        dyn_horizontal_th = BASE_HORIZONTAL_RATIO_TH * inv_comp
        dyn_bend_hip_th = BASE_BEND_HIP_ANKLE_RATIO * inv_comp
        dyn_bend_shoulder_th = BASE_BEND_SHOULDER_ANKLE_RATIO * inv_comp
        dyn_ref_drop_ignore = max(2.0, BASE_HEAD_DROP_IGNORE_TH * height_comp)

        for person in filtered_people:
            keypoints = person["keypoints"]
            valid_xy = person["valid_xy"]
            x_min, y_min, x_max, y_max = person["bbox"]
            track_id = person["track_id"]
            cx = int(np.mean(valid_xy[:, 0]))
            cy = int(np.mean(valid_xy[:, 1]))

            active_track_ids.add(track_id)

            if track_id not in person_states:
                initial_elder_code = try_reuse_recent_identity(cx, cy, frame_w, frame_h, frame_index, camera_key)
                if not initial_elder_code:
                    initial_elder_code = allocate_elder_code()
                person_states[track_id] = {
                    "history": [],
                    "fall_state": "NORMAL",
                    "fall_timer": 0,
                    "ground_timer": 0,
                    "low_center_timer": 0,
                    "fall_confirm": 0,
                    "cooldown_timer": 0,
                    "non_fall_frames": 0,
                    "seen_frames": 0,
                    "horizontal_frames": 0,
                    "side_lying_frames": 0,
                    "closeup_frames": 0,
                    "elder_code": initial_elder_code,
                    "face_bind_done": False,
                    "face_bind_tries": 0,
                    "avatar_saved": False,
                    "avatar_path": "",
                    "avatar_best_score": 0.0,
                    "recovery_frames": 0,
                    "last_fall_frame": -1000000,
                    "last_seen_frame": frame_index,
                    "last_center": (float(cx), float(cy)),
                    "last_bbox": [float(x_min), float(y_min), float(x_max), float(y_max)],
                    "suspected_sent": False,
                    "unrecovered_sent": False,
                    "first_seen_ts": time.time(),
                    "entry_logged": False,
                    "dwell_30s_noted": False,
                }

            state = person_states[track_id]
            if state["cooldown_timer"] > 0:
                state["cooldown_timer"] -= 1
            state["seen_frames"] += 1
            state["last_seen_frame"] = frame_index
            state["last_center"] = (float(cx), float(cy))
            state["last_bbox"] = [float(x_min), float(y_min), float(x_max), float(y_max)]

            if (not state.get("entry_logged")) and state["seen_frames"] >= ENTRY_STABLE_FRAMES and insert_person_entry is not None:
                try:
                    insert_person_entry(
                        DEFAULT_DB_PATH,
                        {
                            "entry_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "camera_id": camera_key,
                            "track_token": f"{camera_key}:{track_id[1] if isinstance(track_id, tuple) else track_id}",
                        },
                    )
                    state["entry_logged"] = True
                except Exception:
                    pass

            dwell_seconds = max(0.0, time.time() - float(state.get("first_seen_ts", time.time())))
            if dwell_seconds >= DWELL_ALERT_SECONDS:
                state["dwell_30s_noted"] = True

            if face_service is not None and getattr(face_service, "available", False) and frame_index % FACE_RECOG_INTERVAL == 0:
                try:
                    face_roi, _ = extract_face_roi(frame, keypoints, person["bbox"])
                    if face_roi is not None:
                        matched_code = None
                        if hasattr(face_service, "identify_only"):
                            matched_code = face_service.identify_only(face_roi)

                        if matched_code:
                            matched_code = str(matched_code)
                            if matched_code != str(state.get("elder_code") or ""):
                                state["elder_code"] = matched_code
                                state["avatar_saved"] = False
                                state["avatar_best_score"] = 0.0
                            state["face_bind_done"] = True
                        else:
                            current_code = str(state.get("elder_code") or "")
                            bind_tries = int(state.get("face_bind_tries", 0) or 0)
                            if current_code and (not state.get("face_bind_done")) and bind_tries < FACE_BIND_MAX_TRIES and hasattr(face_service, "attach_face_to_elder"):
                                bound = bool(face_service.attach_face_to_elder(face_roi, current_code))
                                state["face_bind_tries"] = bind_tries + 1
                                if bound:
                                    state["face_bind_done"] = True
                except Exception:
                    pass

            # 1️⃣ 基础信息
            posture = get_posture(keypoints, dyn_horizontal_th)
            ref_y = get_reference_y(keypoints)
            hips_y = []
            for hip_idx in [11, 12]:
                if keypoints[hip_idx][0] > 0 and keypoints[hip_idx][1] > 0:
                    hips_y.append(keypoints[hip_idx][1])
            hip_y = float(np.mean(hips_y)) if hips_y else ref_y
            body_h = float(max(1.0, y_max - y_min))

            state["history"].append((posture, ref_y, hip_y, body_h))
            if len(state["history"]) > MAX_HISTORY:
                state["history"].pop(0)

            if posture == "horizontal":
                state["horizontal_frames"] += 1
            else:
                state["horizontal_frames"] = 0

            # ✅ 改5：避免误判站立
            skip_fall_detection = False
            if posture == "vertical":
                if len(state["history"]) >= 3:
                    if ref_y - state["history"][-2][1] < dyn_ref_drop_ignore:
                        skip_fall_detection = True

            is_fall_event = False if skip_fall_detection else detect_fall_sequence(state["history"])
            on_object = is_supported_by_object(keypoints, cx, cy, objects)
            lower_body_ok = has_enough_lower_body(keypoints)
            sitting_like = is_probable_sitting(keypoints)
            half_squat_like = is_probable_half_squat(keypoints)
            kneeling_like = is_probable_kneeling(keypoints)
            uptilt_like = is_uptilt_suspected(keypoints)
            high_uptilt_mode = estimated_pitch >= HIGH_UPTILT_DEG_TH
            head_feet_close = is_head_feet_height_close(keypoints)
            side_lying_like = is_side_lying_pose(keypoints)
            true_lying_like = is_true_lying_geometry(keypoints)
            lower_body_geom_ok = has_reliable_lower_body_geometry(keypoints)
            head_dominant_closeup = is_head_dominant_closeup(keypoints)

            shoulders_visible = int(sum(1 for i in [5, 6] if keypoints[i][0] > 0 and keypoints[i][1] > 0))
            hips_visible = int(sum(1 for i in [11, 12] if keypoints[i][0] > 0 and keypoints[i][1] > 0))
            knees_visible = int(sum(1 for i in [13, 14] if keypoints[i][0] > 0 and keypoints[i][1] > 0))
            ankles_visible = int(sum(1 for i in [15, 16] if keypoints[i][0] > 0 and keypoints[i][1] > 0))
            nose = keypoints[0]
            head_partial = (nose[0] <= 0 or nose[1] <= 0 or nose[1] < CLOSEUP_HEAD_TOP_EDGE_RATIO_TH * frame_h)

            upper_body_closeup_like = (
                (y_max > CLOSEUP_BOTTOM_RATIO_TH * frame_h)
                and ((y_max - y_min) > CLOSEUP_MIN_HEIGHT_RATIO_TH * frame_h)
                and (shoulders_visible >= 1)
                and (hips_visible >= 1 or head_dominant_closeup)
                and (ankles_visible == 0 or not lower_body_geom_ok)
                and (head_dominant_closeup or head_partial or knees_visible == 0)
            )

            if upper_body_closeup_like:
                state["closeup_frames"] += 1
            else:
                state["closeup_frames"] = 0

            if side_lying_like:
                state["side_lying_frames"] += 1
            else:
                state["side_lying_frames"] = max(0, state["side_lying_frames"] - 1)
            side_lying_ready = state["side_lying_frames"] >= SIDE_LYING_MIN_FRAMES

            standing_support_like = posture == "vertical" and is_standing_bending(keypoints, dyn_bend_hip_th, dyn_bend_shoulder_th) and not head_feet_close
            apply_uptilt_strict = uptilt_like and estimated_pitch >= UPTILT_STRICT_MIN_DEG

            # 防俯视角前倾误判：无论 posture 如何，只要仍是“脚在下方支撑”且头脚不接近，就不触发跌倒
            if standing_support_like:
                is_fall_event = False

            valid_points = int(np.sum((keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)))
            body_long = max(x_max - x_min, y_max - y_min)
            bbox_ratio = (y_max - y_min) / max(1.0, (x_max - x_min))
            relax_ratio_th = HORIZONTAL_RELAX_MAX_RATIO_TH + HORIZONTAL_RELAX_SOFT_RATIO_MARGIN
            horizontal_like = posture == "horizontal" or side_lying_ready
            horizontal_relax_ok = (
                horizontal_like
                and valid_points >= HORIZONTAL_RELAX_MIN_KEYPOINTS
                and body_long >= HORIZONTAL_RELAX_MIN_SIZE
                and bbox_ratio <= relax_ratio_th
            )

            if not ENABLE_HORIZONTAL_RELAX_FALLBACK:
                horizontal_relax_ok = False

            if apply_uptilt_strict:
                # 仰角场景严控：必须是明显横躺 + 头脚接近，才允许触发。
                if bbox_ratio > UPTILT_STRICT_HORIZONTAL_RATIO_TH:
                    is_fall_event = False
                if not head_feet_close:
                    is_fall_event = False
                horizontal_relax_ok = False

            if high_uptilt_mode:
                # 大仰角极限模式：仅当几何上高度符合“躺地”时才允许触发。
                if bbox_ratio > HIGH_UPTILT_STRICT_HORIZONTAL_RATIO_TH:
                    is_fall_event = False
                if not (head_feet_close and true_lying_like and lower_body_geom_ok):
                    is_fall_event = False
                horizontal_relax_ok = False

            # 半蹲仅在竖直姿态生效，避免侧躺抬腿被误当半蹲
            non_fall_posture_like = (posture == "vertical" and (half_squat_like or sitting_like or kneeling_like or standing_support_like or head_dominant_closeup))
            if upper_body_closeup_like:
                non_fall_posture_like = True

            # 高仰角时下半身缺失不允许触发；常规场景避免过度抑制导致漏检
            if high_uptilt_mode and not lower_body_ok:
                is_fall_event = False
                state["fall_confirm"] = 0
                state["ground_timer"] = 0
                state["non_fall_frames"] += 1
                if state["fall_state"] == "FALLEN" and state["non_fall_frames"] >= NON_FALL_RESET_FRAMES and (frame_index - state["last_fall_frame"]) >= FALLEN_MIN_HOLD_FRAMES:
                    state["fall_state"] = "NORMAL"
                    state["fall_timer"] = 0

            if non_fall_posture_like:
                is_fall_event = False
                state["fall_confirm"] = 0
                state["ground_timer"] = 0
                state["non_fall_frames"] += 1
                # 非跌倒姿态持续出现时，解除历史红框残留
                if state["fall_state"] == "FALLEN" and state["non_fall_frames"] >= NON_FALL_RESET_FRAMES and (frame_index - state["last_fall_frame"]) >= FALLEN_MIN_HOLD_FRAMES:
                    state["fall_state"] = "NORMAL"
                    state["fall_timer"] = 0
            else:
                state["non_fall_frames"] = 0

            if (apply_uptilt_strict or high_uptilt_mode) and is_fall_event and not (head_feet_close or horizontal_relax_ok or side_lying_ready):
                is_fall_event = False

            # 近景半身抑制：头部占比过大时，不触发跌倒（存在躺地证据时放行）
            if head_dominant_closeup and not (head_feet_close or horizontal_relax_ok or side_lying_ready):
                is_fall_event = False

            # 近景上半身/截头画面强抑制：直接阻断触发并快速清除历史红框
            if upper_body_closeup_like:
                is_fall_event = False
                state["fall_confirm"] = 0
                state["ground_timer"] = 0
                state["low_center_timer"] = 0
                if state["fall_state"] == "FALLEN" and state["closeup_frames"] >= CLOSEUP_CLEAR_FALL_FRAMES:
                    state["fall_state"] = "NORMAL"
                    state["fall_timer"] = 0
                    state["recovery_frames"] = 0

            color = (0, 255, 0)

            # 2️⃣ 状态机（每个人独立）
            required_confirm_frames = UPTILT_FALL_CONFIRM_FRAMES if apply_uptilt_strict else FALL_CONFIRM_FRAMES
            required_ground_frames = HIGH_UPTILT_GROUND_FRAMES if high_uptilt_mode else GROUND_FALL_FRAMES
            min_stable_frames = UPTILT_MIN_TRACK_STABLE_FRAMES if apply_uptilt_strict else MIN_TRACK_STABLE_FRAMES

            if state["seen_frames"] < min_stable_frames:
                is_fall_event = False
                state["fall_confirm"] = 0
                state["ground_timer"] = 0

            if state["horizontal_frames"] < MIN_HORIZONTAL_FRAMES_FOR_FALL:
                is_fall_event = False

            # 分场景门控：站立/过渡阶段要求下半身几何可靠；已呈现躺地证据时放宽几何门控
            lying_evidence_for_gate = (head_feet_close or horizontal_relax_ok or side_lying_ready)
            require_geom_gate = REQUIRE_LOWER_BODY_GEOMETRY_FOR_FALL and (not lying_evidence_for_gate)
            lower_body_gate_ok = lower_body_ok and (lower_body_geom_ok if require_geom_gate else True)
            can_enter_fall_state = lower_body_gate_ok if REQUIRE_LOWER_BODY_FOR_FALL else (lower_body_gate_ok or horizontal_relax_ok or side_lying_ready)

            if can_enter_fall_state and not non_fall_posture_like and not upper_body_closeup_like:
                # ✅ 2. 多帧确认：连续3帧 fall 才触发
                if is_fall_event and not on_object:
                    state["fall_confirm"] += 1
                else:
                    state["fall_confirm"] = 0

                suspect_threshold = max(1, required_confirm_frames - 1)
                if required_confirm_frames >= 2 and state["fall_confirm"] == suspect_threshold and state["fall_state"] != "FALLEN" and not state.get("suspected_sent"):
                    if state["cooldown_timer"] <= 0:
                        try_register_fall_event(person["bbox"], cx, cy, frame_w, frame_h, frame_index, state.get("elder_code", ""), frame, camera_key, "suspected")
                    state["suspected_sent"] = True

                if state["fall_confirm"] >= required_confirm_frames and state["fall_state"] != "FALLEN":
                    state["fall_state"] = "FALLEN"
                    state["fall_timer"] = FALL_HOLD_TIME
                    state["last_fall_frame"] = frame_index
                    state["recovery_frames"] = 0
                    state["unrecovered_sent"] = False
                    if state["cooldown_timer"] <= 0:
                        try_register_fall_event(person["bbox"], cx, cy, frame_w, frame_h, frame_index, state.get("elder_code", ""), frame, camera_key, "confirmed")
                        state["cooldown_timer"] = FALL_RECOUNT_COOLDOWN_FRAMES

                # ✅ 改4：持续躺地判断（防漏检）
                if posture == "horizontal" and state["horizontal_frames"] >= MIN_HORIZONTAL_FRAMES_FOR_FALL and (head_feet_close or horizontal_relax_ok or side_lying_ready) and not on_object:
                    state["ground_timer"] += 1
                    if state["ground_timer"] > required_ground_frames and state["fall_state"] != "FALLEN":
                        state["fall_state"] = "FALLEN"
                        state["fall_timer"] = FALL_HOLD_TIME
                        state["last_fall_frame"] = frame_index
                        state["recovery_frames"] = 0
                        state["unrecovered_sent"] = False
                        if state["cooldown_timer"] <= 0:
                            try_register_fall_event(person["bbox"], cx, cy, frame_w, frame_h, frame_index, state.get("elder_code", ""), frame, camera_key, "confirmed")
                            state["cooldown_timer"] = FALL_RECOUNT_COOLDOWN_FRAMES
                else:
                    state["ground_timer"] = 0

                # 视频补偿路径：低重心持续达到阈值也触发（针对慢摔、侧后摔）
                low_center_like = (
                    cy > LOW_CENTER_Y_RATIO_TH * frame_h
                    and y_max > LOW_CENTER_BBOX_BOTTOM_RATIO_TH * frame_h
                    and not on_object
                )
                if low_center_like and (posture == "horizontal" or horizontal_relax_ok or side_lying_ready) and lower_body_gate_ok:
                    state["low_center_timer"] += 1
                    if state["low_center_timer"] > LOW_CENTER_HOLD_FRAMES and state["fall_state"] != "FALLEN":
                        state["fall_state"] = "FALLEN"
                        state["fall_timer"] = FALL_HOLD_TIME
                        state["last_fall_frame"] = frame_index
                        state["recovery_frames"] = 0
                        state["unrecovered_sent"] = False
                        if state["cooldown_timer"] <= 0:
                            try_register_fall_event(person["bbox"], cx, cy, frame_w, frame_h, frame_index, state.get("elder_code", ""), frame, camera_key, "confirmed")
                            state["cooldown_timer"] = FALL_RECOUNT_COOLDOWN_FRAMES
                else:
                    state["low_center_timer"] = 0
            else:
                # 下半身不足或坐姿明显：不进行跌倒判定，清空触发计数
                state["fall_confirm"] = 0
                allow_no_lower_body_fallback = (not REQUIRE_LOWER_BODY_FOR_FALL) and (posture == "horizontal") and (horizontal_relax_ok or side_lying_ready)
                if allow_no_lower_body_fallback and not non_fall_posture_like and not on_object:
                    state["ground_timer"] += 1
                    if state["ground_timer"] > required_ground_frames and state["fall_state"] != "FALLEN":
                        state["fall_state"] = "FALLEN"
                        state["fall_timer"] = FALL_HOLD_TIME
                        state["last_fall_frame"] = frame_index
                        state["recovery_frames"] = 0
                        state["unrecovered_sent"] = False
                        if state["cooldown_timer"] <= 0:
                            try_register_fall_event(person["bbox"], cx, cy, frame_w, frame_h, frame_index, state.get("elder_code", ""), frame, camera_key, "confirmed")
                            state["cooldown_timer"] = FALL_RECOUNT_COOLDOWN_FRAMES
                else:
                    state["ground_timer"] = 0
                state["low_center_timer"] = 0

            # 3️⃣ 状态维持
            if state["fall_state"] == "FALLEN":
                state["fall_timer"] -= 1

                # 只有“可信躺倒”才延长红框，避免坐姿长期变红
                fallen_evidence = (head_feet_close or horizontal_relax_ok or side_lying_ready)
                keep_red_allowed = (lower_body_ok and not upper_body_closeup_like) if REQUIRE_LOWER_BODY_FOR_FALL else (not upper_body_closeup_like)
                if keep_red_allowed and (posture == "horizontal" or side_lying_ready) and fallen_evidence and not sitting_like and not on_object:
                    state["fall_timer"] = FALL_HOLD_TIME
                    state["recovery_frames"] = 0
                else:
                    state["recovery_frames"] += 1

                # 4️⃣ 场景语义（床/椅）只影响显示
                if on_object:
                    color = (255, 255, 0)
                    alert_text = "Lying on Bed/Chair"
                else:
                    color = (0, 0, 255)
                    alert_text = "Fall Detected!"

                if state["fall_timer"] <= 0 and state["recovery_frames"] >= RECOVERY_CONFIRM_FRAMES:
                    state["fall_state"] = "NORMAL"
                    state["fall_confirm"] = 0

                if state["fall_state"] == "FALLEN" and (frame_index - state.get("last_fall_frame", frame_index)) >= UNRECOVERED_ALERT_FRAMES and not state.get("unrecovered_sent"):
                    if state["cooldown_timer"] <= 0:
                        try_register_fall_event(person["bbox"], cx, cy, frame_w, frame_h, frame_index, state.get("elder_code", ""), frame, camera_key, "unrecovered")
                    state["unrecovered_sent"] = True

            # 5️⃣ 可视化
            if DRAW_OVERLAYS:
                draw_keypoints = (
                    DRAW_KEYPOINT_MODE == "all"
                    or (DRAW_KEYPOINT_MODE == "fall-only" and state["fall_state"] == "FALLEN")
                )
                if draw_keypoints:
                    for x, y in keypoints:
                        cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)

                if DRAW_NORMAL_BOXES or state["fall_state"] == "FALLEN":
                    cv2.rectangle(frame,
                                (int(x_min), int(y_min)),
                                (int(x_max), int(y_max)),
                                color, 2)

                if DRAW_DWELL_TEXT and state.get("dwell_30s_noted"):
                    cv2.putText(
                        frame,
                        f"Track {track_id[1] if isinstance(track_id, tuple) else track_id}: stay {int(dwell_seconds)}s",
                        (int(x_min), max(20, int(y_min) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 165, 255),
                        2,
                    )

    # 清理已消失目标，防止状态无限增长
    stale_ids = [
        tid for tid, s in person_states.items()
        if tid not in active_track_ids and isinstance(tid, tuple) and str(tid[0]) == camera_key and frame_index - s.get("last_seen_frame", frame_index) > TRACK_STATE_TTL_FRAMES
    ]
    for tid in stale_ids:
        stale_state = person_states.pop(tid, None)
        if stale_state is None:
            continue
        elder_code = str(stale_state.get("elder_code") or "")
        center = stale_state.get("last_center", (None, None))
        if elder_code and isinstance(center, tuple) and len(center) == 2:
            c0, c1 = center
            if c0 is not None and c1 is not None:
                push_recent_identity(elder_code, float(c0), float(c1), frame_index, camera_key)

    frame_ms = (time.perf_counter() - frame_start_t) * 1000.0
    instant_fps = 1000.0 / max(1e-6, frame_ms)
    if fps_ema <= 0.0:
        fps_ema = instant_fps
    else:
        fps_ema = FPS_EMA_ALPHA * instant_fps + (1.0 - FPS_EMA_ALPHA) * fps_ema

    if DRAW_OVERLAYS and (frame_index % DRAW_HUD_EVERY_N_FRAMES == 0):
        # 3️⃣ UI统计信息
        cv2.putText(frame, f"Persons: {person_count}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"Falls: {camera_fall_counts.get(camera_key, 0)}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        mode_text = "(Auto)" if AUTO_DETECT_CAMERA_ANGLE else "(Manual)"
        cv2.putText(frame, f"Cam Pitch: ~{int(estimated_pitch)} deg {mode_text}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

        cv2.putText(frame, f"FPS: {fps_ema:.1f}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Infer: {inference_ms:.1f} ms  Frame: {frame_ms:.1f} ms", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2)

        if alert_text:
            cv2.putText(frame, alert_text, (50, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    if PROFILE_PRINT_INTERVAL > 0 and frame_index % PROFILE_PRINT_INTERVAL == 0:
        post_ms = max(0.0, (time.perf_counter() - stage_after_infer_t) * 1000.0)
        print(
            f"[perf] cam={camera_key} tracker={'on' if used_tracker_this_frame else 'skip'} "
            f"persons={person_count} infer={inference_ms:.1f}ms post={post_ms:.1f}ms frame={frame_ms:.1f}ms"
        )

    cv2.imshow(f"Smart AI Fall Detection - {camera_key}", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

camera_stop_event.set()
for thread in camera_threads:
    thread.join(timeout=0.2)

for entry in camera_entries:
    try:
        entry["capture"].release()
    except Exception:
        pass

if clip_write_queue is not None:
    clip_writer_stop_event.set()
    try:
        clip_write_queue.put_nowait(None)
    except Exception:
        pass
if clip_writer_thread is not None:
    clip_writer_thread.join(timeout=2.0)

cv2.destroyAllWindows()