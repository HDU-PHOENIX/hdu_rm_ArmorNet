.
├── docs/                          # 项目文档
│   └── README.md
├── examples/                      # 部署/推理示例（仅保留与 11/通用相关）
│   ├── README.md
│   ├── YOLO11-Triton-CPP/         # YOLO11 + Triton 的 C++ 部署示例
│   └── YOLO-Interactive-Tracking-UI/ # 交互式跟踪 UI 示例
├── pyproject.toml                 # Python 包构建/依赖
├── yolo11n.pt                     # YOLO11（AABB）预训练权重缓存
├── yolo11n-obb.pt                 # YOLO11（OBB）预训练权重缓存
└── ultralytics/                   # 核心源码
    ├── assets/                    # 样例图片（bus.jpg / zidane.jpg）
    ├── cfg/                       # 配置（模型 / 数据集 / 跟踪器 / 默认超参）
    │   ├── default.yaml           # 训练/推理默认参数（CLI 可覆盖）
    │   ├── datasets/              # 常用数据集配置模板（COCO/DOTA/VisDrone…）
    │   └── models/
    │       ├── 11/                # YOLO11 家族配置（本项目主用）
    │       │   ├── yolo11.yaml        # YOLO11 检测基础结构
    │       │   ├── yolo11-obb.yaml    # YOLO11 旋转框（OBB）结构
    │       │   ├── yolo11-seg.yaml    # YOLO11 分割
    │       │   ├── yolo11-pose.yaml   # YOLO11 姿态
    │       │   ├── yolo11-cls.yaml    # YOLO11 分类
    │       │   ├── yolo11-cls-resnet18.yaml
    │       │   ├── yoloe-11.yaml / yoloe-11-seg.yaml  # YOLOE-11 变体
    │       └── README.md
    │   └── trackers/              # 多目标跟踪器配置（BoT-SORT/ByteTrack）
    ├── data/                      # 数据加载与训练期增强
    │   ├── augment.py             # 训练期图像增强（mosaic/mixup/仿射…）
    │   ├── dataset.py / loaders.py / build.py
    │   ├── split_dota.py          # DOTA 数据切片工具
    │   └── scripts/               # 常用下载脚本
    ├── engine/                    # 统一执行引擎（Train/Val/Predict/Export）
    │   ├── model.py               # YOLO() 封装（train/val/predict/export 入口）
    │   ├── predictor.py           # 推理流程 & preprocess（通用）
    │   ├── trainer.py / validator.py / results.py / exporter.py / tuner.py
    ├── models/                    # 任务脚手架（按任务拆分）
    │   ├── yolo/
    │   │   ├── model.py           # YOLO 任务注册/路由
    │   │   ├── detect/            # AABB 检测：train/val/predict
    │   │   ├── obb/               # 旋转框（OBB）：train/val/predict
    │   │   ├── segment/ pose/ classify/ yoloe/ world/
    │   ├── rtdetr/ sam/ nas/      # 其它模型族入口
    │   └── utils/                 # 局部工具（ops/loss 等，部分任务会用）
    ├── nn/                        # 神经网络组件与任务拼装
    │   ├── modules/               #  模块层（Conv/C2f/SPPF/Head/Transformer…）
    │   │   ├── conv.py / block.py / head.py / transformer.py / utils.py
    │   ├── tasks.py               # 任务级 Model（如 DetectionModel/OBBModel）
    │   ├── autobackend.py         # 后端适配/自动加载
    │   └── text_model.py
    ├── utils/                     # 通用工具（NMS/绘制/日志/导出/指标…）
    │   ├── nms.py / ops.py        # 后处理与几何运算（含 OBB NMS）
    │   ├── loss.py / metrics.py   # 损失与指标（通用）
    │   ├── plotting.py / logger.py / files.py / downloads.py / …
    │   └── export/                # 导出相关（IMX 等）
    ├── trackers/                  # 跟踪实现（BoT-SORT/ByteTrack）
    ├── solutions/                 # 业务示例（计数/安防/测速/Streamlit Demo…）
    └── hub/                       # 云端 Hub 交互（可忽略）
