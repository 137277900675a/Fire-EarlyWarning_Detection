#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火焰烟雾检测训练脚本
使用方法: python scripts/train.py [--config CONFIG_PATH]
"""

import argparse
import torch
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='火焰烟雾检测YOLOv8训练脚本')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='训练配置文件路径')
    parser.add_argument('--model', type=str, default='yolov8s',
                        help='模型变体: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x')
    parser.add_argument('--weights', type=str, default='models/pretrained/yolov8s.pt',
                        help='预训练权重路径')
    parser.add_argument('--data', type=str, default='data/dataset.yaml',
                        help='数据集配置文件路径')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数')
    parser.add_argument('--batch', type=int, default=None,
                        help='批次大小')
    parser.add_argument('--imgsz', type=int, default=None,
                        help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='',
                        help='设备: cuda 或 cpu (留空自动检测)')
    parser.add_argument('--project', type=str, default=None,
                        help='项目目录')
    parser.add_argument('--name', type=str, default=None,
                        help='实验名称')
    parser.add_argument('--resume', action='store_true',
                        help='从上次检查点恢复训练')
    return parser.parse_args()


def load_config(config_path):
    """从YAML文件加载训练配置"""
    import yaml
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


def main():
    """主训练函数"""
    args = parse_args()

    # 检查设备
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    weights_path = Path(args.weights)
    if weights_path.exists():
        print(f"从以下路径加载预训练权重: {weights_path}")
        model = YOLO(str(weights_path))
    else:
        print(f"使用 {args.model} 模型 (预训练)")
        model = YOLO(args.model)

    # 加载配置
    config = load_config(args.config)

    # 准备训练参数
    train_params = {
        'data': args.data,
        'device': device,
        'resume': args.resume,
    }

    # 用命令行参数覆盖配置
    if args.epochs:
        train_params['epochs'] = args.epochs
    elif 'epochs' in config:
        train_params['epochs'] = config['epochs']

    if args.batch:
        train_params['batch'] = args.batch
    elif 'batch' in config:
        train_params['batch'] = config['batch']

    if args.imgsz:
        train_params['imgsz'] = args.imgsz
    elif 'imgsz' in config:
        train_params['imgsz'] = config['imgsz']

    # 添加项目和名称
    if args.project:
        train_params['project'] = args.project
    elif 'project' in config:
        train_params['project'] = config['project']

    if args.name:
        train_params['name'] = args.name
    elif 'name' in config:
        train_params['name'] = config['name']

    # 从配置中添加其他训练参数
    param_keys = [
        'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs',
        'warmup_momentum', 'warmup_bias_lr', 'patience', 'workers',
        'exist_ok', 'verbose', 'seed', 'deterministic', 'single_cls',
        'rect', 'cos_lr', 'close_mosaic', 'amp', 'fraction', 'profile',
        'freeze', 'multi_scale', 'hsv_h', 'hsv_s', 'hsv_v',
        'degrees', 'translate', 'scale', 'shear', 'perspective',
        'flipud', 'fliplr', 'mosaic', 'mixup', 'copy_paste'
    ]

    for key in param_keys:
        if key in config:
            train_params[key] = config[key]

    # 打印训练配置
    print("\n训练配置:")
    for key, value in train_params.items():
        print(f"  {key}: {value}")

    # 开始训练
    print("\n开始训练...")
    results = model.train(**train_params)

    print("\n训练完成!")
    return results


if __name__ == "__main__":
    main()
