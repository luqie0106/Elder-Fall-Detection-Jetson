from ultralytics import YOLO
import cv2
import numpy as np
import time

# 加载模型（第一次会自动下载）
model = YOLO("yolov8n.pt")

# 打开摄像头
cap = cv2.VideoCapture(0)

# ROI区域（入侵检测区域）
ROI = (200, 100, 400, 300)  # x1, y1, x2, y2

# 存储每个人停留时间
stay_time = {}
start_time = {}

def is_fall(w, h):
    return w > h * 1.2  # 横着 → 简单判断跌倒

def in_roi(cx, cy):
    x1, y1, x2, y2 = ROI
    return x1 < cx < x2 and y1 < cy < y2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    alert_text = ""

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls != 0:  # 只检测人（COCO中person=0）
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w = x2 - x1
        h = y2 - y1

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        color = (0, 255, 0)

        # 跌倒检测
        if is_fall(w, h):
            color = (0, 0, 255)
            alert_text = "Fall Detected!"

        # 区域入侵检测
        if in_roi(cx, cy):
            color = (0, 165, 255)
            alert_text = "Intrusion!"

        # 停留检测
        person_id = (cx, cy)
        if person_id not in start_time:
            start_time[person_id] = time.time()

        stay_time[person_id] = time.time() - start_time[person_id]

        if stay_time[person_id] > 5:
            alert_text = "Loitering!"

        # 画框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 中心点
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    # 画ROI区域
    cv2.rectangle(frame, (ROI[0], ROI[1]), (ROI[2], ROI[3]), (255, 255, 0), 2)

    # 显示报警
    if alert_text:
        cv2.putText(frame, alert_text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Smart Surveillance (Mac Demo)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()