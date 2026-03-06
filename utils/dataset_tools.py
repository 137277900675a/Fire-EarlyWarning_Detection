#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火焰烟雾检测数据集工具
包含数据增强、格式转换和数据集统计功能
"""

import os
import shutil
import yaml
from pathlib import Path
from collections import Counter


def get_dataset_stats(data_dir):
    """获取数据集统计信息

    参数:
        data_dir: 数据集目录路径

    返回:
        包含数据集统计信息的字典
    """
    data_path = Path(data_dir)

    stats = {
        'train_images': 0,
        'train_labels': 0,
        'val_images': 0,
        'val_labels': 0,
        'test_images': 0,
        'class_counts': {},
    }

    # 统计训练图像和标签数量
    train_img_dir = data_path / 'images' / 'train'
    train_lbl_dir = data_path / 'labels' / 'train'

    if train_img_dir.exists():
        stats['train_images'] = len(list(train_img_dir.glob('*.*')))
    if train_lbl_dir.exists():
        stats['train_labels'] = len(list(train_lbl_dir.glob('*.txt')))

    # 统计验证图像和标签数量
    val_img_dir = data_path / 'images' / 'val'
    val_lbl_dir = data_path / 'labels' / 'val'

    if val_img_dir.exists():
        stats['val_images'] = len(list(val_img_dir.glob('*.*')))
    if val_lbl_dir.exists():
        stats['val_labels'] = len(list(val_lbl_dir.glob('*.txt')))

    # 统计测试图像数量
    test_img_dir = data_path / 'images' / 'test'
    if test_img_dir.exists():
        stats['test_images'] = len(list(test_img_dir.glob('*.*')))

    # 从训练标签统计类别分布
    if train_lbl_dir.exists():
        class_counter = Counter()
        for label_file in train_lbl_dir.glob('*.txt'):
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        class_counter[class_id] += 1

        stats['class_counts'] = dict(class_counter)

    return stats


def print_dataset_stats(data_dir):
    """格式化打印数据集统计信息

    参数:
        data_dir: 数据集目录路径
    """
    stats = get_dataset_stats(data_dir)

    print("\n" + "="*50)
    print("数据集统计信息")
    print("="*50)
    print(f"训练图像数量: {stats['train_images']}")
    print(f"训练标签数量: {stats['train_labels']}")
    print(f"验证图像数量: {stats['val_images']}")
    print(f"验证标签数量: {stats['val_labels']}")
    print(f"测试图像数量: {stats['test_images']}")

    if stats['class_counts']:
        print("\n类别分布:")
        total = sum(stats['class_counts'].values())
        for class_id, count in sorted(stats['class_counts'].items()):
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  类别 {class_id}: {count} ({percentage:.2f}%)")

    print("="*50)


def get_class_distribution(data_dir):
    """获取数据集类别分布

    参数:
        data_dir: 数据集目录路径

    返回:
        包含类别分布的字典
    """
    data_path = Path(data_dir)
    train_lbl_dir = data_path / 'labels' / 'train'

    class_counter = Counter()

    if train_lbl_dir.exists():
        for label_file in train_lbl_dir.glob('*.txt'):
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        class_counter[class_id] += 1

    total = sum(class_counter.values())
    distribution = {}
    for class_id, count in class_counter.items():
        distribution[class_id] = {
            'count': count,
            'percentage': (count / total) * 100 if total > 0 else 0
        }

    return distribution


def check_dataset_integrity(data_dir):
    """检查数据集完整性

    参数:
        data_dir: 数据集目录路径

    返回:
        包含完整性检查结果的字典
    """
    data_path = Path(data_dir)
    issues = []

    for split in ['train', 'val', 'test']:
        img_dir = data_path / 'images' / split
        lbl_dir = data_path / 'labels' / split

        if not img_dir.exists():
            continue

        # 获取图像和标签文件
        img_files = set([f.stem for f in img_dir.glob('*.*')])
        lbl_files = set([f.stem for f in lbl_dir.glob('*.txt')]) if lbl_dir.exists() else set()

        # 检查缺失标签
        missing_labels = img_files - lbl_files
        if missing_labels:
            issues.append(f"{split}: {len(missing_labels)} 张图像缺少标签")

        # 检查缺失图像
        missing_images = lbl_files - img_files
        if missing_images:
            issues.append(f"{split}: {len(missing_images)} 个标签缺少图像")

    return {
        'valid': len(issues) == 0,
        'issues': issues
    }


def split_dataset(source_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """将数据集划分为训练/验证/测试集

    参数:
        source_dir: 包含图像的源目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    import random

    source_path = Path(source_dir)

    # 获取所有图像文件
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(list(source_path.glob(ext)))

    random.shuffle(image_files)

    # 计算划分的索引
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }

    return splits
