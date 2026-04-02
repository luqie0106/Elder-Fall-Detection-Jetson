from ultralytics import YOLO
import cv2
import numpy as np
import math

# 加载模型（一个模型同时支持检测+pose）
model = YOLO("yolov8n-pose.pt")


cap = cv2.VideoCapture(0)

# 时间序列缓存（每个人独立）
MAX_HISTORY = 10

# 统计数据
fall_count = 0
person_count = 0
FALL_HOLD_TIME = 30  # 保持红框30帧（约1秒）

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
CAMERA_PITCH_ANGLE = 15.0

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
LOWER_BODY_MIN_POINTS = 2
# 下半身几何可靠性（调大更紧）
LOWER_BODY_MIN_HIP_ANKLE_DY_RATIO = 0.18
LOWER_BODY_KNEE_MID_TOL_RATIO = 0.12
# 核心躯干门控：至少1个肩点+1个髋点，才允许参与跌倒判定
CORE_BODY_MIN_SHOULDERS = 1
CORE_BODY_MIN_HIPS = 1

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
FALL_CONFIRM_FRAMES = 1
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
ENABLE_HORIZONTAL_RELAX_FALLBACK = True

# 侧身躺地门控（用于侧身对摄像头时提升召回）
# 条件：躯干轴近水平 + 人体主轴近水平 + 连续稳定若干帧
SIDE_LYING_TRUNK_DY_RATIO_TH = 0.24
SIDE_LYING_TRUNK_DX_RATIO_TH = 0.10
SIDE_LYING_AXIS_MAX_DEG = 40.0
SIDE_LYING_MIN_FRAMES = 2

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
frame_index = 0

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
    # 恢复兜底：当髋部跟踪抖动时允许触发（仍要求姿态变化+下落+速度+冲击）
    fallback_path = cond1 and cond2 and cond3 and (cond4 or very_large_drop)

    # 侧后摔专用路径：即使“横躺标签”不明显，只要有明显下落+速度+高度塌缩也可触发
    side_back_path = ("vertical" in postures[:FALL_SEQ_WINDOW]) and cond2 and cond3 and cond_shape and (cond4 or very_large_drop)

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

def is_duplicate_person_bbox(box_a, box_b, frame_w, frame_h):
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

def try_register_fall_event(bbox, cx, cy, frame_w, frame_h, frame_idx):
    # 基于时间 + 空间去重：同一人同一次倒地只记一次
    global fall_count

    max_dist = EVENT_DEDUP_CENTER_DIST_RATIO * max(frame_w, frame_h)
    for event in recent_fall_events:
        if frame_idx - event["frame"] > EVENT_DEDUP_FRAMES:
            continue

        prev_cx, prev_cy = event["center"]
        center_dist = math.hypot(cx - prev_cx, cy - prev_cy)
        if center_dist <= max_dist:
            return False

        if bbox_iou(bbox, event["bbox"]) >= EVENT_DEDUP_IOU_TH:
            return False

    recent_fall_events.append({
        "frame": frame_idx,
        "center": (float(cx), float(cy)),
        "bbox": bbox,
    })
    fall_count += 1
    return True

while True:
    frame_index += 1

    # 清理过期事件，避免列表无限增长
    recent_fall_events[:] = [
        e for e in recent_fall_events if frame_index - e["frame"] <= EVENT_DEDUP_FRAMES
    ]

    ret, frame = cap.read()
    if not ret:
        break

    estimated_pitch = CAMERA_PITCH_ANGLE

    # 使用 ByteTrack 进行目标跟踪，保证每个人独立 history
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]

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

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # 2️⃣ 姿态 + 跌倒检测
    person_count = 0

    active_track_ids = set()

    if results.keypoints is not None:
        track_ids = None
        if results.boxes is not None and results.boxes.id is not None:
            track_ids = results.boxes.id.int().cpu().tolist()

        frame_h, frame_w = frame.shape[:2]
        # 先收集候选人体，再做同帧去重，最后进入状态机
        candidates = []
        frame_max_ratio = 0.0

        for idx, k in enumerate(results.keypoints.xy):
            keypoints = k.cpu().numpy()
            track_id = track_ids[idx] if track_ids is not None and idx < len(track_ids) else idx

            if not valid_person(keypoints):
                continue

            # 过滤仅头脸/局部肢体的人体误检，防止误触发跌倒
            if not has_core_body(keypoints):
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

            active_track_ids.add(track_id)

            if track_id not in person_states:
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
                }

            state = person_states[track_id]
            if state["cooldown_timer"] > 0:
                state["cooldown_timer"] -= 1
            state["seen_frames"] += 1

            # 1️⃣ 基础信息
            posture = get_posture(keypoints, dyn_horizontal_th)
            ref_y = get_reference_y(keypoints)
            hips_y = []
            for hip_idx in [11, 12]:
                if keypoints[hip_idx][0] > 0 and keypoints[hip_idx][1] > 0:
                    hips_y.append(keypoints[hip_idx][1])
            hip_y = float(np.mean(hips_y)) if hips_y else ref_y
            body_h = float(max(1.0, y_max - y_min))

            cx = int(np.mean(valid_xy[:, 0]))
            cy = int(np.mean(valid_xy[:, 1]))

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

            if side_lying_like:
                state["side_lying_frames"] += 1
            else:
                state["side_lying_frames"] = 0
            side_lying_ready = state["side_lying_frames"] >= SIDE_LYING_MIN_FRAMES

            standing_support_like = posture == "vertical" and is_standing_bending(keypoints, dyn_bend_hip_th, dyn_bend_shoulder_th) and not head_feet_close
            apply_uptilt_strict = uptilt_like and estimated_pitch >= UPTILT_STRICT_MIN_DEG

            # 防俯视角前倾误判：无论 posture 如何，只要仍是“脚在下方支撑”且头脚不接近，就不触发跌倒
            if standing_support_like:
                is_fall_event = False

            valid_points = int(np.sum((keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)))
            body_long = max(x_max - x_min, y_max - y_min)
            bbox_ratio = (y_max - y_min) / max(1.0, (x_max - x_min))
            horizontal_relax_ok = (
                posture == "horizontal"
                and valid_points >= HORIZONTAL_RELAX_MIN_KEYPOINTS
                and body_long >= HORIZONTAL_RELAX_MIN_SIZE
                and bbox_ratio <= HORIZONTAL_RELAX_MAX_RATIO_TH
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

            # 半蹲门控独立于posture，避免“蹲姿被误分到horizontal”后漏拦截
            non_fall_posture_like = half_squat_like or (posture == "vertical" and (sitting_like or kneeling_like or standing_support_like))

            # 高仰角时下半身缺失不允许触发；常规场景避免过度抑制导致漏检
            if high_uptilt_mode and not lower_body_ok:
                is_fall_event = False
                state["fall_confirm"] = 0
                state["ground_timer"] = 0
                state["non_fall_frames"] += 1
                if state["fall_state"] == "FALLEN" and state["non_fall_frames"] >= NON_FALL_RESET_FRAMES:
                    state["fall_state"] = "NORMAL"
                    state["fall_timer"] = 0

            if non_fall_posture_like:
                is_fall_event = False
                state["fall_confirm"] = 0
                state["ground_timer"] = 0
                state["non_fall_frames"] += 1
                # 非跌倒姿态持续出现时，解除历史红框残留
                if state["fall_state"] == "FALLEN" and state["non_fall_frames"] >= NON_FALL_RESET_FRAMES:
                    state["fall_state"] = "NORMAL"
                    state["fall_timer"] = 0
            else:
                state["non_fall_frames"] = 0

            if (apply_uptilt_strict or high_uptilt_mode) and is_fall_event and not (head_feet_close or horizontal_relax_ok or side_lying_ready):
                is_fall_event = False

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

            if (lower_body_ok or horizontal_relax_ok or side_lying_ready) and not non_fall_posture_like:
                # ✅ 2. 多帧确认：连续3帧 fall 才触发
                if is_fall_event and not on_object:
                    state["fall_confirm"] += 1
                else:
                    state["fall_confirm"] = 0

                if state["fall_confirm"] >= required_confirm_frames and state["fall_state"] != "FALLEN":
                    state["fall_state"] = "FALLEN"
                    state["fall_timer"] = FALL_HOLD_TIME
                    if state["cooldown_timer"] <= 0:
                        try_register_fall_event(person["bbox"], cx, cy, frame_w, frame_h, frame_index)
                        state["cooldown_timer"] = FALL_RECOUNT_COOLDOWN_FRAMES

                # ✅ 改4：持续躺地判断（防漏检）
                if posture == "horizontal" and state["horizontal_frames"] >= MIN_HORIZONTAL_FRAMES_FOR_FALL and (head_feet_close or horizontal_relax_ok or side_lying_ready) and not on_object:
                    state["ground_timer"] += 1
                    if state["ground_timer"] > required_ground_frames and state["fall_state"] != "FALLEN":
                        state["fall_state"] = "FALLEN"
                        state["fall_timer"] = FALL_HOLD_TIME
                        if state["cooldown_timer"] <= 0:
                            try_register_fall_event(person["bbox"], cx, cy, frame_w, frame_h, frame_index)
                            state["cooldown_timer"] = FALL_RECOUNT_COOLDOWN_FRAMES
                else:
                    state["ground_timer"] = 0

                # 视频补偿路径：低重心持续达到阈值也触发（针对慢摔、侧后摔）
                low_center_like = (
                    cy > LOW_CENTER_Y_RATIO_TH * frame_h
                    and y_max > LOW_CENTER_BBOX_BOTTOM_RATIO_TH * frame_h
                    and not on_object
                )
                if low_center_like and (posture == "horizontal" or horizontal_relax_ok or side_lying_ready):
                    state["low_center_timer"] += 1
                    if state["low_center_timer"] > LOW_CENTER_HOLD_FRAMES and state["fall_state"] != "FALLEN":
                        state["fall_state"] = "FALLEN"
                        state["fall_timer"] = FALL_HOLD_TIME
                        if state["cooldown_timer"] <= 0:
                            try_register_fall_event(person["bbox"], cx, cy, frame_w, frame_h, frame_index)
                            state["cooldown_timer"] = FALL_RECOUNT_COOLDOWN_FRAMES
                else:
                    state["low_center_timer"] = 0
            else:
                # 下半身不足或坐姿明显：不进行跌倒判定，清空触发计数
                state["fall_confirm"] = 0
                if posture == "horizontal" and (horizontal_relax_ok or side_lying_ready) and not non_fall_posture_like and not on_object:
                    state["ground_timer"] += 1
                    if state["ground_timer"] > required_ground_frames and state["fall_state"] != "FALLEN":
                        state["fall_state"] = "FALLEN"
                        state["fall_timer"] = FALL_HOLD_TIME
                        if state["cooldown_timer"] <= 0:
                            try_register_fall_event(person["bbox"], cx, cy, frame_w, frame_h, frame_index)
                            state["cooldown_timer"] = FALL_RECOUNT_COOLDOWN_FRAMES
                else:
                    state["ground_timer"] = 0
                state["low_center_timer"] = 0

            # 3️⃣ 状态维持
            if state["fall_state"] == "FALLEN":
                state["fall_timer"] -= 1

                # 只有“可信躺倒”才延长红框，避免坐姿长期变红
                if posture == "horizontal" and (head_feet_close or horizontal_relax_ok or side_lying_ready) and not sitting_like and not on_object:
                    state["fall_timer"] = FALL_HOLD_TIME

                # 4️⃣ 场景语义（床/椅）只影响显示
                if on_object:
                    color = (255, 255, 0)
                    alert_text = "Lying on Bed/Chair"
                else:
                    color = (0, 0, 255)
                    alert_text = "Fall Detected!"

                if state["fall_timer"] <= 0:
                    state["fall_state"] = "NORMAL"
                    state["fall_confirm"] = 0

            # 5️⃣ 可视化
            for x, y in keypoints:
                cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)

            cv2.rectangle(frame,
                        (int(x_min), int(y_min)),
                        (int(x_max), int(y_max)),
                        color, 2)

    # 清理已消失目标，防止状态无限增长
    stale_ids = [tid for tid in person_states.keys() if tid not in active_track_ids]
    for tid in stale_ids:
        person_states.pop(tid, None)

    # 3️⃣ UI统计信息
    cv2.putText(frame, f"Persons: {person_count}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"Falls: {fall_count}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    mode_text = "(Auto)" if AUTO_DETECT_CAMERA_ANGLE else "(Manual)"
    cv2.putText(frame, f"Cam Pitch: ~{int(estimated_pitch)} deg {mode_text}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

    if alert_text:
        cv2.putText(frame, alert_text, (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Smart AI Fall Detection (Final Demo)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()