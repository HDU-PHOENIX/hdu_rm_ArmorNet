
# ArmorNet：实时旋转装甲板检测系统

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

**ArmorNet** 是一个面向 RoboMaster 视觉任务的 **实时旋转装甲板检测网络**，  
基于 **YOLOv11-OBB** 架构实现，支持 **Oriented Bounding Box（旋转框）检测**、**多线程推理** 与 **OpenVINO/Triton 部署**，  
适用于机器人嵌入式视觉系统与实时推理场景。

---

## 🚀 项目亮点

- 🧭 **旋转装甲板检测（OBB）** —— 精确识别倾斜与旋转装甲板目标  
- ⚡ **实时推理** —— 支持 OpenVINO、Triton、TensorRT 等多后端加速  
- 🧠 **统一训练引擎** —— 一套 API 实现 Train / Val / Predict / Export  
- 🔧 **模块化架构** —— 可扩展至分割、姿态估计、分类等任务  
- 💡 **轻量化部署** —— 可在 Jetson、工控机等嵌入式设备上高效运行  

---
## 📁 目录结构

项目整体结构如下（简化展示核心模块）👇  
<details>
<summary>点击展开查看完整结构</summary>
hdu_rm_ArmorNet/
├── main.py # 主入口文件
├── readme.md # 项目说明文档
├── requirements.core.txt # 基础依赖列表
└── ultralytics-obb/ # 核心代码目录
├── docs/ # 项目文档
├── examples/ # 推理与部署示例
│ ├── YOLO11-Triton-CPP/ # YOLO11 + Triton C++ 示例
│ └── YOLO-Interactive-Tracking-UI/ # 交互式跟踪 UI 示例
├── pyproject.toml # Python 项目配置
├── readme.md # 子项目说明
├── ultralytics/ # YOLO11 主框架实现
│ ├── assets/ # 样例图片（bus.jpg / zidane.jpg）
│ ├── cfg/ # 配置（模型 / 数据集 / 跟踪器）
│ │ ├── datasets/ # 各类数据集配置（COCO / DOTA / VOC 等）
│ │ ├── models/11/ # YOLO11 系列配置（检测 / 分割 / 姿态）
│ │ └── trackers/ # 跟踪算法配置（BoT-SORT / ByteTrack）
│ ├── data/ # 数据加载与增强模块
│ ├── engine/ # 统一执行引擎（训练 / 验证 / 导出）
│ ├── models/ # 模型定义（检测 / 分割 / 姿态 / OBB）
│ │ ├── yolo/ # YOLO 系列任务（detect / obb / seg / pose）
│ │ ├── fastsam/ nas/ rtdetr/ sam/ # 其他模型族支持
│ │ └── utils/ # 损失函数与算子模块
│ ├── nn/ # 神经网络组件（Conv / Transformer / Head）
│ ├── utils/ # 工具函数（NMS / 绘图 / 指标 / 导出）
│ ├── trackers/ # 多目标跟踪实现（BoT-SORT / ByteTrack）
│ ├── solutions/ # 应用示例（计数 / 安防 / Streamlit Demo）
│ └── hub/ # 云端 Hub 交互（可选）
├── yolo11n.pt # YOLO11（AABB）预训练权重
└── yolo11n-obb.pt # YOLO11（OBB）预训练权重