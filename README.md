# 火焰烟雾检测系统 (Fire-EarlyWarning Detection)

基于YOLOv8的火焰与烟雾实时检测系统，支持四种迁移学习策略的对比训练与评估。

## 项目概述

本项目是一个基于深度学习的火焰和烟雾检测系统，采用YOLOv8作为基础目标检测模型。系统实现了四种主流迁移学习策略的对比训练，包括从头训练、特征提取、微调和全参数微调，旨在为不同应用场景提供最优的模型选择方案。

系统的核心功能包括：目标检测模型训练、多策略对比评估、模型导出部署、预测推理以及详细的性能分析报告生成。通过综合评估模块，系统能够自动计算并可视化四种策略的性能指标，帮助用户快速了解各策略的优劣势，从而做出最优的模型选择决策。

## 技术架构

### 核心模型

项目基于Ultralytics YOLOv8目标检测框架构建，选用yolov8s作为基础模型。YOLOv8是YOLO系列目标检测算法的最新版本，在检测精度和推理速度方面都有显著提升。系统支持检测两个类别：火焰（Fire，类别ID: 0）和烟雾（Smoke，类别ID: 1）。

模型架构采用CSPDarknet作为骨干网络，配合PANet特征金字塔进行多尺度特征融合，最后通过检测头输出边界框坐标、置信度和类别概率。整个模型采用先进的Anchor Free设计理念，简化了模型配置的复杂度，同时保持了优秀的检测性能。

### 依赖环境

项目依赖以下核心Python包：ultralytics>=8.0.0（模型框架）、torch>=2.0.0（深度学习引擎）、torchvision>=0.15.0（数据处理）、opencv-python>=4.7.0（图像处理）、Pillow>=9.5.0（图像加载）、PyYAML>=6.0（配置管理）、pandas>=2.0.0（数据分析）、matplotlib>=3.7.0（可视化）、seaborn>=0.12.0（统计绘图）、tqdm>=4.65.0（进度条）、scikit-learn>=1.2.0（机器学习工具）、onnx>=1.14.0（模型导出）以及onnxruntime>=1.16.0（推理部署）。

详细的依赖列表请参阅 `requirements.txt` 文件。

### 项目结构

```
Fire-EarlyWarning_Detection/
├── configs/                      # 配置文件目录
│   ├── train_config.yaml         # 训练参数配置文件
│   ├── hyp.yaml                  # 超参数配置文件
│   ├── transfer_strategies.yaml  # 迁移学习策略配置
│   └── __init__.py
├── data/                         # 数据集目录
│   ├── dataset.yaml              # 数据集配置文件
│   ├── images/                   # 图像数据
│   │   ├── train/               # 训练集图像
│   │   ├── val/                 # 验证集图像
│   │   └── test/                # 测试集图像
│   └── labels/                   # 标注文件（YOLO格式）
│       ├── train/
│       ├── val/
│       └── test/
├── models/                       # 模型文件目录
│   ├── pretrained/              # 预训练权重
│   │   └── yolov8s.pt
│   ├── yolov8s.yaml            # 模型架构配置
│   └── __init__.py
├── scripts/                      # 训练与推理脚本
│   ├── train.py                 # 基础训练脚本
│   ├── train_multi_strategy.py # 多策略对比训练
│   ├── val.py                  # 模型验证脚本
│   ├── predict.py              # 单模型预测脚本
│   ├── predict_comparison.py   # 多策略预测对比
│   ├── export导出脚本
│.py               # 模型   ├── comprehensive_evaluation.py # 综合评估脚本
│   └── __init__.py
├── utils/                       # 工具函数
│   ├── metrics.py              # 评估指标计算
│   ├── viz.py                  # 可视化工具
│   ├── dataset_tools.py        # 数据集处理工具
│   └── __init__.py
├── runs/                        # 训练与评估结果
│   ├── train/                  # 基础训练输出
│   ├── multi_strategy/         # 多策略训练结果
│   ├── evaluation/             # 评估报告输出
│   ├── experiment_logs/        # 实验日志记录
│   ├── predict/                # 预测结果输出
│   ├── predict_comparison/     # 预测对比结果
│   └── val/                    # 验证结果
├── yolo11n.pt                  # YOLO11n预训练模型（备用）
├── requirements.txt            # 项目依赖
└── README.md                   # 项目说明文档
```

## 功能特性

### 1. 四种迁移学习策略对比

项目实现了四种主流迁移学习策略的完整对比训练，用户可以根据自身数据规模和硬件条件选择最适合的策略。

**策略一：从头训练（Scratch Training）** 不使用任何预训练权重，从随机初始化开始训练整个模型。这种方式适用于有大量标注数据且任务与预训练数据差异较大的场景，可以学习到完全任务特定的知识，但需要较长的训练时间和大量数据支持。

**策略二：特征提取（Feature Extraction）** 冻结预训练模型的骨干网络和颈部网络，仅训练检测头层。这种方式训练速度最快、资源消耗最低，不易过拟合，适用于小数据集或计算资源受限的场景，能够快速验证模型效果。

**策略三：微调（Fine-tuning）** 冻结大部分网络结构，解冻最后几层与检测头一起训练。这种方式在保留预训练知识的同时能够适应新任务的特征，在精度和训练时间之间取得良好平衡，适用于中等规模数据集。

**策略四：全参数微调（Full Fine-tuning）** 解冻所有层对整个网络进行微调训练。这种方式能够最大程度地适应新任务，达到最高精度，但需要较长的训练时间和充足的计算资源，适用于大数据集和追求高精度的应用场景。

### 2. 完整的训练流程

系统提供了完整的模型训练流程，包括参数配置、数据加载、模型训练、验证评估和结果保存。通过配置文件可以灵活调整训练参数，如学习率、批次大小、图像尺寸、训练轮数等。训练过程中支持早停机制，当验证指标不再提升时自动停止训练，避免过拟合。

训练脚本支持从检查点恢复训练，可以在意外中断后继续训练。系统还支持多GPU分布式训练（需要配置），能够充分利用计算资源加速训练过程。所有训练结果都会自动保存到指定目录，包括模型权重、训练日志、验证指标等。

### 3. 综合评估与可视化

项目提供了强大的评估和可视化功能。综合评估脚本能够自动计算多种性能指标，包括mAP50、mAP50-95、精确率、召回率、F1分数等，并生成详细的对比分析报告。

评估模块支持生成多种可视化图表：性能对比柱状图展示各策略在不同指标上的表现；雷达图直观展示各策略的综合性能；混淆矩阵热力图分析预测错误类型分布；训练曲线图追踪训练过程中的指标变化。所有图表都支持中文字体显示，报告以Markdown格式保存，便于阅读和分享。

### 4. 模型导出与部署

训练完成的模型可以导出为多种格式以适应不同的部署场景。支持的导出格式包括：TorchScript（PyTorch原生格式）、ONNX（跨框架通用格式）、OpenVINO（Intel推理引擎）、TensorFlow SavedModel、CoreML（Apple设备）、TFLite（移动端/边缘设备）、TensorRT（NVIDIA GPU加速）以及NCNN（移动端高效推理）等。

导出脚本支持多种高级选项：半精度（FP16）导出可以减少模型体积并加速推理；动态输入形状支持处理不同尺寸的输入图像；模型简化可以减少计算量。导出后的模型可以直接用于生产环境部署，实现实时的火焰烟雾检测。

### 5. 预测推理功能

系统提供了便捷的预测推理功能，支持对单张图像、图像目录和视频文件进行检测。预测脚本支持丰富的参数配置，包括置信度阈值、NMS IoU阈值、输出保存选项等。预测结果可以保存为带标注的图像、标注文本文件或裁剪的目标区域。

预测对比脚本能够同时使用四种策略的模型对测试集进行预测，并详细对比各模型的检测性能。这种对比分析帮助用户了解不同策略在实际推理中的表现差异，为生产环境选择提供数据支持。

### 6. 数据集工具

项目提供了一套完整的数据集处理工具，包括：数据集统计功能，能够统计各划分（训练/验证/测试）的图像和标注数量，分析类别分布；数据集完整性检查，检测缺失标注或缺失图像的问题；数据集划分工具，支持按比例将原始数据划分为训练集、验证集和测试集。

## 快速开始

### 环境配置

首先创建并激活Python环境，然后安装项目依赖：

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 数据准备

将数据集按照YOLO格式组织，目录结构如下：

```
data/
├── images/
│   ├── train/          # 训练图像 (*.jpg, *.png)
│   ├── val/            # 验证图像
│   └── test/           # 测试图像
└── labels/
    ├── train/          # 训练标注 (*.txt, YOLO格式)
    ├── val/            # 验证标注
    └── test/           # 测试标注
```

每个标注文件为YOLO格式，每行表示一个目标：`class_id x_center y_center width height`，坐标均为归一化值（0-1）。

根据实际数据集修改 `data/dataset.yaml` 中的类别名称和数量配置。

### 单策略训练

使用默认配置进行基础训练：

```bash
python scripts/train.py
```

自定义参数训练：

```bash
python scripts/train.py \
    --config configs/train_config.yaml \
    --model yolov8s \
    --weights models/pretrained/yolov8s.pt \
    --epochs 300 \
    --batch 32 \
    --device 0
```

### 多策略对比训练

一次性训练并对比四种迁移学习策略：

```bash
python scripts/train_multi_strategy.py
```

训练完成后，结果将保存在 `runs/multi_strategy/` 目录，综合评估报告生成在 `runs/evaluation/` 目录。

### 模型验证

使用验证集评估训练好的模型：

```bash
python scripts/val.py \
    --weights runs/multi_strategy/strategy_4_full_finetuning/train/weights/best.pt \
    --data data/dataset.yaml
```

### 模型预测

对单张图像或目录进行预测：

```bash
python scripts/predict.py \
    --weights runs/multi_strategy/strategy_4_full_finetuning/train/weights/best.pt \
    --source data/images/test \
    --conf 0.25 \
    --save
```

### 模型导出

将训练好的模型导出为ONNX格式：

```bash
python scripts/export.py \
    --weights runs/multi_strategy/strategy_4_full_finetuning/train/weights/best.pt \
    --format onnx \
    --simplify
```

导出为TensorRT加速格式：

```bash
python scripts/export.py \
    --weights runs/multi_strategy/strategy_4_full_finetuning/train/weights/best.pt \
    --format engine \
    --half
```

### 综合评估

生成详细的性能评估报告和可视化图表：

```bash
python scripts/comprehensive_evaluation.py
```

输出文件包括：`detailed_report.md`（详细评估报告）、`comparison_charts.png`（性能对比图表）、`radar_chart.png`（雷达图）、`evaluation_summary.csv`（指标汇总表）。

## 实验结果

根据最新的实验结果，四种迁移学习策略在火焰烟雾检测任务上的性能对比如下：

| 策略 | mAP50 | mAP50-95 | 精确率 | 召回率 | F1分数 | 训练时间(分钟) | 综合评分 |
|------|-------|----------|--------|--------|--------|---------------|----------|
| 全参数微调 | 0.9475 | 0.6586 | 0.9459 | 0.9145 | 0.9299 | 90.4 | 85.39 |
| 微调 | 0.9479 | 0.6522 | 0.9289 | 0.9157 | 0.9223 | 35.3 | 84.89 |
| 从头训练 | 0.9450 | 0.6281 | 0.9470 | 0.9091 | 0.9277 | 67.5 | 84.32 |
| 特征提取 | 0.9492 | 0.6389 | 0.9321 | 0.8995 | 0.9155 | 18.0 | 84.28 |

**关键发现：**

- **最高mAP50**：特征提取策略达到0.9492，表现最优
- **最高mAP50-95**：全参数微调达到0.6586，整体检测性能最佳
- **最快训练速度**：特征提取仅需18分钟，是最快速的方案
- **综合最优**：全参数微调策略的综合评分最高（85.39分）

**选型建议：**

1. 追求最高精度场景，推荐使用**全参数微调**策略
2. 资源受限或快速验证场景，推荐使用**特征提取**策略
3. 平衡精度与训练时间，推荐使用**微调**策略

详细的实验结果和对比分析请参考 `runs/evaluation/detailed_report.md`。

## 配置说明

### 训练配置 (configs/train_config.yaml)

主要训练参数包括：模型选择（yolov8s）、训练轮数（epochs: 300）、批次大小（batch: 32）、输入图像尺寸（imgsz: 640）、初始学习率（lr0: 0.005）、优化器权重衰减（weight_decay: 0.0005）等。

数据增强参数包括：HSV颜色空间增强、随机翻转、随机缩放、平移、旋转（degrees: 0.0表示不旋转）等。针对火焰检测特点，HSV饱和度增强设置为0.7，有助于模型更好地识别不同颜色的火焰。

### 迁移学习策略配置 (configs/transfer_strategies.yaml)

每种策略都有针对性的参数设计：

- **从头训练**：高学习率（0.01）、强正则化（dropout: 0.5）、标签平滑（0.1）
- **特征提取**：较低学习率（0.001）、较少训练轮数（50）
- **微调**：中等学习率（0.0005）、分层学习率设置
- **全参数微调**：最低学习率（0.0001）、最大训练轮数（150）

## 扩展开发

### 添加新类别

1. 修改 `data/dataset.yaml` 中的 `nc`（类别数量）和 `names`（类别名称）
2. 更新 `utils/viz.py` 中的 `CLASS_NAMES` 和 `CLASS_COLORS`
3. 重新训练模型

### 更换基础模型

项目支持YOLOv8系列所有变体（yolov8n/s/m/l/x），修改训练配置中的模型名称即可：

```bash
python scripts/train.py --model yolov8m  # 使用YOLOv8 Medium模型
```

### 自定义数据增强

在 `configs/train_config.yaml` 中调整数据增强参数，或在训练脚本中通过代码添加自定义增强策略。Ultralytics框架支持Albumentations库，可实现更丰富的数据增强效果。

## 许可证

本项目仅供学习和研究使用。

## 参考资料

- YOLOv8官方文档：https://docs.ultralytics.com/
- Ultralytics GitHub：https://github.com/ultralytics/ultralytics
- YOLO数据集格式：https://docs.ultralytics.com/yolov5/train_custom_data/
