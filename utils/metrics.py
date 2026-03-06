#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火焰烟雾检测自定义评估指标
包含额外的评估指标和性能分析功能
"""

import numpy as np
from pathlib import Path


def calculate_iou(box1, box2):
    """计算两个边界框的交并比(IoU)

    参数:
        box1: 第一个边界框 [x1, y1, x2, y2]
        box2: 第二个边界框 [x1, y1, x2, y2]

    返回:
        IoU值
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def calculate_ap(recalls, precisions):
    """计算平均精度(AP)

    参数:
        recalls: 召回率列表
        precisions: 精确率列表

    返回:
        AP值
    """
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))

    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

    indices = np.where(recalls[1:] != recalls[:-1])[0]

    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

    return ap


def calculate_confusion_matrix(predictions, targets, num_classes, iou_threshold=0.5):
    """计算混淆矩阵

    参数:
        predictions: 预测结果列表 [(class_id, confidence, box), ...]
        targets: 目标列表 [(class_id, box), ...]
        num_classes: 类别数量
        iou_threshold: 匹配的IoU阈值

    返回:
        混淆矩阵
    """
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)

    matched_targets = set()

    for pred in predictions:
        pred_class, pred_conf, pred_box = pred
        matched = False

        for i, target in enumerate(targets):
            if i in matched_targets:
                continue

            target_class, target_box = target

            iou = calculate_iou(pred_box, target_box)

            if iou >= iou_threshold:
                matrix[target_class][pred_class] += 1
                matched_targets.add(i)
                matched = True
                break

        if not matched:
            matrix[num_classes][pred_class] += 1

    for i, target in enumerate(targets):
        if i not in matched_targets:
            target_class, _ = target
            matrix[target_class][num_classes] += 1

    return matrix


def print_evaluation_metrics(metrics_dict):
    """格式化打印评估指标

    参数:
        metrics_dict: 包含评估指标的字典
    """
    print("\n" + "="*60)
    print("评估指标")
    print("="*60)
    print(f"精确率: {metrics_dict.get('precision', 0):.4f}")
    print(f"召回率: {metrics_dict.get('recall', 0):.4f}")
    print(f"mAP@0.5: {metrics_dict.get('map50', 0):.4f}")
    print(f"mAP@0.5:0.95: {metrics_dict.get('map', 0):.4f}")
    print(f"F1分数: {metrics_dict.get('f1', 0):.4f}")
    print("="*60)
