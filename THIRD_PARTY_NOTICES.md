# Third-Party Notices

本项目使用了第三方开源库。以下信息用于合规披露与追踪。

## face_recognition

- 项目：`face_recognition`
- 仓库：`https://github.com/ageitgey/face_recognition`
- 许可证：MIT License
- 使用方式：作为可选依赖，通过 `requirements-face.txt` 安装，并在 `services/face_recognition_service.py` 中调用其 API 进行人脸编码与比对。
- 说明：本仓库未拷贝该项目源码，仅以依赖方式引用。

## dlib（face_recognition 的关键依赖）

- 项目：`dlib`
- 仓库：`https://github.com/davisking/dlib`
- 许可证：请以其仓库 LICENSE 为准
- 说明：当安装 `face_recognition` 时通常会间接使用 `dlib`。

---

建议：发布镜像、安装包或商用部署前，再次核对所有直接/间接依赖的许可证条款。
