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
python3 "detect040214(ori:040119).py"
```

### TensorRT 加速（推荐）

为了获得最佳性能，建议将 `.pt` 模型导出为 `.engine`（TensorRT）格式：

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

参数位于 `detect040214(ori:040119).py` 顶部，现场可按摄像头高度与视角调整：

| 需求 | 对应参数 | 调整建议 |
|---|---|---|
| 提高灵敏度 | `FALL_CONFIRM_FRAMES` | 调小（如设为 `1`） |
| 减少误报（坐下报跌倒） | `SITTING_TORSO_VERTICAL_RATIO_TH` | 调大 |
| 抑制俯拍误报 | `BASE_BEND_HIP_ANKLE_RATIO` | 调大 |
| 解决红框闪烁 | `FALLEN_MIN_HOLD_FRAMES` | 调大 |

## 📁 目录说明

- `detect040214(ori:040119).py`：当前主版本（含动态透视补偿与侧身优化）
- `yolov8n-pose.pt` / `yolov8n-pose.engine`：姿态模型权重
- `bytetrack.yaml`：跟踪器配置文件（若工程中提供）

## ⚠️ 已知问题与优化建议

- 光照影响：弱光环境关键点抖动可能导致误报，建议调高 `DETECTION_CONF_TH`
- 遮挡处理：当下半身被床体大面积遮挡时稳定性会下降，脚本已通过 `LOWER_BODY_MIN_POINTS` 过滤
- 屏幕翻拍视频与真实监控场景存在域差异，建议分场景建立参数模板

本项目为 AIX 比赛开发版本，建议结合 `jtop` 实时监控显存、温度与功耗。

## 免责声明

本项目用于辅助预警，不可替代医疗/护理人员的最终判断。建议结合现场告警联动和人工复核。