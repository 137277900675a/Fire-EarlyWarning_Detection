#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火焰烟雾检测验证脚本
使用方法: python scripts/val.py --weights WEIGHTS_PATH --data DATA_CONFIG
"""

import argparse
import torch
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='火焰烟雾检测YOLOv8验证脚本')
    parser.add_argument('--weights', type=str, required=True,
                        help='训练权重路径')
    parser.add_argument('--data', type=str, default='data/dataset.yaml',
                        help='数据集配置文件路径')
    parser.add_argument('--batch', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像尺寸')
    parser.add_argument('--conf', type=float, default=0.001,
                        help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.6,
                        help='NMS的IoU阈值')
    parser.add_argument('--device', type=str, default='',
                        help='设备: cuda 或 cpu')
    parser.add_argument('--project', type=str, default='runs/val',
                        help='项目目录')
    parser.add_argument('--name', type=str, default='exp',
                        help='实验名称')
    parser.add_argument('--save_json', action='store_true',
                        help='保存结果到JSON文件')
    parser.add_argument('--save_hybrid', action='store_true',
                        help='保存混合版本的标签')
    parser.add_argument('--verbose', action='store_true',
                        help='详细输出')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='验证数据集划分')
    return parser.parse_args()


def main():
    """主验证函数"""
    args = parse_args()

    # 检查设备
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 检查权重文件是否存在
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"错误: 权重文件不存在: {weights_path}")
        return

    # 加载模型
    print(f"从以下路径加载模型: {weights_path}")
    model = YOLO(str(weights_path))

    # 准备验证参数
    val_params = {
        'data': args.data,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'conf': args.conf,
        'iou': args.iou,
        'device': device,
        'project': args.project,
        'name': args.name,
        'save_json': args.save_json,
        'save_hybrid': args.save_hybrid,
        'verbose': args.verbose,
        'split': args.split,
    }

    # 打印验证配置
    print("\n验证配置:")
    for key, value in val_params.items():
        print(f"  {key}: {value}")

    # 开始验证
    print("\n开始验证...")
    results = model.val(**val_params)

    # 打印验证指标
    print("\n验证结果:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  精确率: {results.box.mp:.4f}")
    print(f"  召回率: {results.box.mr:.4f}")

    return results


if __name__ == "__main__":
    main()
