#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火焰烟雾检测模型导出脚本
使用方法: python scripts/export.py --weights WEIGHTS_PATH --format FORMAT [--其他选项]
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='火焰烟雾检测YOLOv8模型导出脚本')
    parser.add_argument('--weights', type=str, required=True,
                        help='训练权重路径')
    parser.add_argument('--format', type=str, default='onnx',
                        choices=['torchscript', 'onnx', 'openvino', 'tensorflow', 'coreml', 'saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs', 'paddle', 'ncnn'],
                        help='导出格式')
    parser.add_argument('--imgsz', type=int, nargs='+', default=[640],
                        help='输入图像尺寸')
    parser.add_argument('--half', action='store_true',
                        help='使用FP16半精度导出')
    parser.add_argument('--dynamic', action='store_true',
                        help='ONNX/TensorRT动态输入形状')
    parser.add_argument('--simplify', action='store_true',
                        help='简化ONNX模型')
    parser.add_argument('--opset', type=int, default=12,
                        help='ONNX opset版本')
    parser.add_argument('--workspace', type=int, default=4,
                        help='TensorRT工作区大小(GB)')
    parser.add_argument('--int8', action='store_true',
                        help='使用TensorRT INT8量化')
    parser.add_argument('--batch', type=int, default=1,
                        help='导出批次大小')
    parser.add_argument('--device', type=str, default='0',
                        help='设备: cpu 或 0,1,2,3')
    parser.add_argument('--verbose', action='store_true',
                        help='详细输出')
    return parser.parse_args()


def main():
    """主导出函数"""
    args = parse_args()

    # 验证权重路径
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"错误: 权重文件不存在于 {weights_path}")
        return

    # 初始化YOLO模型
    print(f"从以下路径加载模型: {weights_path}")
    model = YOLO(str(weights_path))

    # 设置导出参数
    export_params = {
        'format': args.format,
        'imgsz': args.imgsz,
        'half': args.half,
        'dynamic': args.dynamic,
        'simplify': args.simplify,
        'opset': args.opset,
        'workspace': args.workspace,
        'int8': args.int8,
        'batch': args.batch,
        'device': args.device,
        'verbose': args.verbose,
    }

    # 打印导出配置
    print("\n导出配置:")
    for key, value in export_params.items():
        print(f"  {key}: {value}")

    # 开始导出
    print("\n开始导出...")
    export_path = model.export(**export_params)

    print(f"\n导出完成！保存至: {export_path}")

    return export_path


if __name__ == "__main__":
    main()
