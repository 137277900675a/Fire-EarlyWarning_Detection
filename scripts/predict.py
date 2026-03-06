#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火焰烟雾检测预测/推理脚本
使用方法: python scripts/predict.py --weights WEIGHTS_PATH --source SOURCE [--其他选项]
"""

import argparse
import torch
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='火焰烟雾检测YOLOv8预测脚本')
    parser.add_argument('--weights', type=str, required=True,
                        help='训练权重路径')
    parser.add_argument('--source', type=str, required=True,
                        help='图像、视频或目录路径')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS的IoU阈值')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='',
                        help='设备: cuda 或 cpu')
    parser.add_argument('--project', type=str, default='runs/predict',
                        help='项目目录')
    parser.add_argument('--name', type=str, default='exp',
                        help='实验名称')
    parser.add_argument('--save', action='store_true', default=True,
                        help='保存预测结果')
    parser.add_argument('--save_txt', action='store_true',
                        help='保存结果到txt文件')
    parser.add_argument('--save_conf', action='store_true',
                        help='在txt文件中保存置信度')
    parser.add_argument('--save_crop', action='store_true',
                        help='保存裁剪的预测框')
    parser.add_argument('--show', action='store_true',
                        help='显示预测结果')
    parser.add_argument('--show_labels', action='store_true', default=True,
                        help='在图像上显示标签')
    parser.add_argument('--show_conf', action='store_true', default=True,
                        help='在图像上显示置信度')
    parser.add_argument('--vid_stride', type=int, default=1,
                        help='视频帧率步长')
    parser.add_argument('--line_width', type=int, default=3,
                        help='边界框粗细')
    parser.add_argument('--visualize', action='store_true',
                        help='可视化模型特征')
    parser.add_argument('--augment', action='store_true',
                        help='使用增强推理')
    parser.add_argument('--agnostic_nms', action='store_true',
                        help='类别无关的NMS')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                        help='按类别过滤')
    parser.add_argument('--retina_masks', action='store_true',
                        help='高分辨率掩码')
    parser.add_argument('--embed', type=int, nargs='+', default=None,
                        help='提取嵌入向量')
    return parser.parse_args()


def main():
    """主预测函数"""
    args = parse_args()

    # 检查设备
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 验证权重路径
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"错误: 权重文件不存在于 {weights_path}")
        return

    # 确认源路径有效
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"错误: 源路径不存在于 {source_path}")
        return

    # 初始化YOLO模型
    print(f"从以下路径加载模型: {weights_path}")
    model = YOLO(str(weights_path))

    # 设置预测参数
    predict_params = {
        'source': args.source,
        'conf': args.conf,
        'iou': args.iou,
        'imgsz': args.imgsz,
        'device': device,
        'project': args.project,
        'name': args.name,
        'save': args.save,
        'save_txt': args.save_txt,
        'save_conf': args.save_conf,
        'save_crop': args.save_crop,
        'show': args.show,
        'show_labels': args.show_labels,
        'show_conf': args.show_conf,
        'vid_stride': args.vid_stride,
        'line_width': args.line_width,
        'visualize': args.visualize,
        'augment': args.augment,
        'agnostic_nms': args.agnostic_nms,
        'classes': args.classes,
        'retina_masks': args.retina_masks,
        'embed': args.embed,
    }

    # 显示预测参数配置
    print("\n预测设置:")
    for key, value in predict_params.items():
        print(f"  {key}: {value}")

    # 执行推理
    print("\n运行推理...")
    results = model.predict(**predict_params)

    # 汇总结果
    print(f"\n处理了 {len(results)} 张图像")

    return results


if __name__ == "__main__":
    main()
