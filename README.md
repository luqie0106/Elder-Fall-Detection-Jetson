# Elder Fall Detection (AI) - Jetson Orin NX 版

基于 YOLOv8 Pose 的老年人跌倒检测项目，针对 NVIDIA Jetson Orin NX 边缘开发板进行深度优化。  
当前推荐脚本版本：`main.py`。

## 🌟 项目简介

本项目通过摄像头实时检测人体关键点，结合时序规则（姿态变化、重心下落、持续躺地、侧躺几何特征等）判断是否发生跌倒。  
针对边缘计算场景，支持 TensorRT 加速，确保在 Orin NX 上实现高帧率实时检测。

## 🛠️ Jetson Orin NX 环境配置（关键）

在 Orin NX 上运行本项目，必须使用 NVIDIA 预装的 PyTorch 版本，严禁直接使用 `pip install torch` 覆盖。
本机Jetpack版本：# R35 (release), REVISION: 5.0, GCID: 35550185, BOARD: t186ref, EABI: aarch64, DATE: Tue Feb 20 04:46:31 UTC 2024
不同于本机版本的请自行寻找对应的torch版本以及torchvision版本，或者直接使用系统自带的版本。

### 1) 核心依赖对齐

若遇到 `RuntimeError: Couldn't load custom C++ ops`，请按以下步骤对齐版本：

- `PyTorch`：使用系统自带 `2.0.0+nv23.05`
- `Torchvision`：手动编译 `v0.15.1` 以适配 Jetson 专用 Torch

```bash
pip uninstall torchvision -y
git clone --branch v0.15.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.15.1
python3 setup.py install --user
```

### 2) 性能优化（风扇与频率）

在开始检测前，建议开启最高功耗模式并解锁频率，以保证推理 FPS：

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

## 🚀 快速开始

### 运行检测

```bash
python3 "main.py"
```

### TensorRT 加速（推荐）

为了获得最佳性能，建议将 `.pt` 模型导出为 `.engine`（TensorRT）格式:

```python
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")
model.export(format="engine", device=0)
```

生成 `yolov8n-pose.engine` 后，将脚本中的模型加载改为：

```python
model = YOLO("yolov8n-pose.engine")
```

## 🧠 核心算法逻辑

- 多人独立跟踪：基于 `track_id` 维护每个人独立状态机
- 多路径跌倒判定：
  - 动态下落路径：监测头部/髋部下落速度与冲击加速度
  - 静态补偿路径：针对慢速滑落，监测持续低重心与横躺特征
  - 侧身专项判据：通过主轴几何分析补偿侧身遮挡漏检
- 环境自适应：支持 `AUTO_DETECT_CAMERA_ANGLE`，根据人体比例自动补偿俯仰透视畸变

## ⚙️ 关键调参指南

参数位于 `main.py` 顶部，现场可按摄像头高度与视角调整：

| 需求 | 对应参数 | 调整建议 |
|---|---|---|
| 提高灵敏度 | `FALL_CONFIRM_FRAMES` | 调小（如设为 `1`） |
| 减少误报（坐下报跌倒） | `SITTING_TORSO_VERTICAL_RATIO_TH` | 调大 |
| 抑制俯拍误报 | `BASE_BEND_HIP_ANKLE_RATIO` | 调大 |
| 解决红框闪烁 | `FALLEN_MIN_HOLD_FRAMES` | 调大 |

## 📁 目录说明

- `main.py`：当前主版本（含动态透视补偿与侧身优化）
- `yolov8n-pose.pt` / `yolov8n-pose.engine`：姿态模型权重
- `bytetrack.yaml`：跟踪器配置文件（若工程中提供）

## ⚠️ 已知问题与优化建议

- 光照影响：弱光环境关键点抖动可能导致误报，建议调高 `DETECTION_CONF_TH`
- 遮挡处理：当下半身被床体大面积遮挡时稳定性会下降，脚本已通过 `LOWER_BODY_MIN_POINTS` 过滤
- 屏幕翻拍视频与真实监控场景存在域差异，建议分场景建立参数模板

## 🔔 外部接口与告警联动（新）

项目已新增“跌倒事件外部接口”基础能力，采用模块化结构：

- `services/event_pipeline.py`：统一事件管道（发送告警 + 写入数据库）
- `services/notifier.py`：消息通道（控制台、Webhook，微信/电话为占位扩展）
- `storage/events_db.py`：SQLite 事件存储
- `web/app.py` + `web/templates/index.html`：HTML 页面查看历史告警
- `config/alert_config.json`：告警通道配置

### 1) 安装项目依赖（推荐）

#### 通用环境（PC / 非 Jetson）

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements/base.txt
```

#### Jetson 环境（推荐）

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements/jetson.txt
```

Jetson 设备请保持系统预装 `torch/torchvision`，不要通过 pip 覆盖。

### 2) 运行跌倒检测（已接入告警与入库）

```bash
python3 main.py
```

支持通过环境变量配置多摄像头与事件视频片段：

```bash
FALL_CAMERA_SOURCES=0,1 FALL_CLIP_PRE_SECONDS=5 FALL_CLIP_POST_SECONDS=5 python3 main.py
```

- `FALL_CAMERA_SOURCES`：摄像头列表，逗号分隔，支持本地索引和 RTSP/HTTP 地址。
- `FALL_CLIP_PRE_SECONDS`：跌倒前视频秒数，范围 `3~5`。
- `FALL_CLIP_POST_SECONDS`：跌倒后视频秒数，范围 `3~5`。

### 2.1 人脸识别（可选）

你可以借助 `face_recognition` 项目完成老人身份识别：

- 仓库：`https://github.com/ageitgey/face_recognition.git`
- 本项目适配文件：`services/face_recognition_service.py`

安装可选依赖：

```bash
pip install -r requirements/face.txt
```

兼容说明：根目录仍保留 `requirements.txt`、`requirements-jetson.txt`、`requirements-face.txt` 作为入口别名。

启用后，系统会为识别到的人脸自动分配编号（如 `E001`、`E002`），并在告警记录里展示该编号。

### 2.2 多级告警策略（已接入）

系统已支持三级告警，减少一次性强告警误触发：

- `suspected`：疑似跌倒（早期触发，用于预警）；
- `confirmed`：确认跌倒（主告警）；
- `unrecovered`：长时间未恢复（持续高风险提醒）。

网页记录会展示 `alert_level` 字段，便于后续筛查与统计。

### 2.3 事件视频片段（已接入）

除单帧截图外，系统会在跌倒事件发生时保存短视频片段：

- 前置缓冲：事件前 `3~5` 秒；
- 后置补录：事件后 `3~5` 秒；
- 输出位置：`web/static/faces/`（与截图同目录，网页可直接点击查看）。

说明：`suspected/confirmed/unrecovered` 都会记录事件数据；你可在网页按级别复核。

### 2.4 多摄像头并行采集（已接入）

当前已支持多路摄像头并行采集（每路独立抓帧线程，主循环统一推理与告警）：

```bash
FALL_CAMERA_SOURCES=0,1 python3 main.py
```

也支持混合本地与网络摄像头：

```bash
FALL_CAMERA_SOURCES=0,rtsp://user:pass@192.168.1.10:554/stream1 python3 main.py
```

### 3) 启动网页查看告警记录

```bash
python3 web/app.py
```

浏览器访问：`http://127.0.0.1:5000`

若提示端口被占用（`Address already in use`），可改端口启动：

```bash
FALL_WEB_PORT=5001 python3 web/app.py
```

若你使用项目虚拟环境，也可显式指定解释器：

```bash
FALL_WEB_PORT=5001 .venv/bin/python web/app.py
```

### 3.1 推荐启动命令（C）

建议固定使用虚拟环境解释器，避免系统 Python 与依赖不一致：

```bash
cd /path/to/aix_contest
.venv/bin/python web/app.py
```

若 `5000` 被占用：

```bash
cd /path/to/aix_contest
FALL_WEB_PORT=5001 .venv/bin/python web/app.py
```

若仍失败，先检查端口占用再重启：

```bash
lsof -iTCP:5000 -sTCP:LISTEN
lsof -iTCP:5001 -sTCP:LISTEN
pkill -f "web/app.py"
```

改端口后访问：`http://127.0.0.1:5001`

### 4) 配置微信/电话通道建议

- 微信：推荐企业微信机器人 Webhook（可直接配置到 `config/alert_config.json` 的 `webhook.url`）
- 电话：建议接 Twilio/阿里云语音外呼（在 `services/notifier.py` 的 phone_call 适配器中接入）

> 说明：当前仓库默认开启控制台告警；Webhook/微信/电话默认关闭，避免误触发。

### 5) 网页当前能力

当前网页聚焦告警复核：

- 展示最近告警记录（时间、级别、截图、视频链接、摄像头、帧号、中心点、通道状态）；
- 支持一键清空所有告警记录（同时清理运行期截图/视频文件）；
- 端口可通过 `FALL_WEB_PORT` 配置。

### 6) `data/fall_events.db` 云端保持空基线

如果你希望仓库（云端）里保留一个“空数据库基线”，但本地运行产生的数据不参与提交，可使用：

```bash
cd /path/to/aix_contest
git update-index --skip-worktree data/fall_events.db
```

说明：

- 该命令是**本地 Git 配置**，只对当前机器生效；
- 取消该设置时可执行：

```bash
git update-index --no-skip-worktree data/fall_events.db
```

## 📄 第三方许可证说明

本项目包含对第三方库（如 `face_recognition`）的依赖调用。

- 本项目许可证：见根目录 `LICENSE`（Apache-2.0）
- 第三方依赖与许可证信息：见 `THIRD_PARTY_NOTICES.md`

如果你仅“依赖安装并调用 API”，通常无需把第三方项目的 LICENSE 文件复制到本仓库；
若你直接拷贝了第三方源码片段，请按其许可证要求保留对应版权与许可声明。

本项目为 AIX 比赛开发版本，建议结合 `jtop` 实时监控显存、温度与功耗。

## 免责声明

本项目用于辅助预警，不可替代医疗/护理人员的最终判断。建议结合现场告警联动和人工复核。