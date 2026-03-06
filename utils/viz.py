#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火焰烟雾检测可视化工具
包含预测结果可视化、训练结果可视化和数据增强可视化
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


# 定义中文字体文件路径列表（按优先级排序）
CHINESE_FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Noto Sans CJK
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",          # 文泉驿微米黑
    "/usr/share/fonts/truetype/arphic/uming.ttc",             # AR PL UMing
    "/usr/share/fonts/truetype/arphic/ukai.ttc",              # AR PL UKai
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",           # 文泉驿正黑
]

# 创建全局中文字体属性对象
def get_chinese_font(size=10):
    """获取中文字体FontProperties对象

    参数:
        size: 字体大小

    返回:
        FontProperties对象
    """
    for font_path in CHINESE_FONT_PATHS:
        if Path(font_path).exists():
            return font_manager.FontProperties(fname=font_path, size=size)
    # 如果所有字体都不存在，使用默认字体
    return font_manager.FontProperties(size=size)

# 初始化全局中文字体（默认大小）
chinese_font = get_chinese_font()
chinese_font_large = get_chinese_font(16)
chinese_font_small = get_chinese_font(9)


# 不同类别的颜色调色板
CLASS_COLORS = {
    0: (255, 0, 0),     # 红色代表火焰
    1: (128, 128, 128), # 灰色代表烟雾
}

CLASS_NAMES = {
    0: '火焰',
    1: '烟雾',
}


def draw_boxes(image, boxes, class_ids=None, confidences=None, labels=None):
    """在图像上绘制边界框

    参数:
        image: 输入图像 (numpy数组或PIL图像)
        boxes: 边界框列表 [x1, y1, x2, y2]
        class_ids: 类别ID列表
        confidences: 置信度列表
        labels: 标签列表 (可选)

    返回:
        绘制了边界框的图像
    """
    if isinstance(image, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        img = image.copy()

    draw = ImageDraw.Draw(img)

    # 设置字体 - 优先使用支持中文的字体
    font_paths = [
        "/usr/share/fonts/truetype/arphic/uming.ttc",   # AR PL UMing
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", # 文泉驿微米黑
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", # Noto Sans CJK
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", # Noto Serif CJK
        "/usr/share/fonts/truetype/arphic/ukai.ttc",     # AR PL UKai
    ]
    font = None
    for font_path in font_paths:
        if Path(font_path).exists():
            try:
                font = ImageFont.truetype(font_path, 16)
                break
            except:
                continue
    if font is None:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        class_id = class_ids[i] if class_ids is not None else 0
        conf = confidences[i] if confidences is not None else 1.0
        label = labels[i] if labels is not None else CLASS_NAMES.get(class_id, f'类别 {class_id}')

        # 获取颜色
        color = CLASS_COLORS.get(class_id, (255, 255, 255))

        # 绘制矩形框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 绘制标签和置信度文本
        text = f'{label}: {conf:.2f}'
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - text_height - 2), text, fill=(255, 255, 255), font=font)

    return img


def plot_training_results(results_file):
    """绘制训练结果图表

    参数:
        results_file: results.csv文件路径
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    # 加载训练结果数据
    df = pd.read_csv(results_file)

    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('训练结果', fontsize=16, fontproperties=chinese_font_large)

    # 绘制各项指标
    metrics = [
        ('train/box_loss', '边界框损失'),
        ('train/cls_loss', '分类损失'),
        ('train/dfl_loss', 'DFL损失'),
        ('metrics/precision(B)', '精确率'),
        ('metrics/recall(B)', '召回率'),
        ('metrics/mAP50-95(B)', 'mAP50-95'),
    ]

    for idx, (col, title) in enumerate(metrics):
        row = idx // 3
        col_idx = idx % 3

        if col in df.columns:
            axes[row, col_idx].plot(df['epoch'], df[col])
            axes[row, col_idx].set_title(title, fontproperties=chinese_font)
            axes[row, col_idx].set_xlabel('训练轮次', fontproperties=chinese_font)

    plt.tight_layout()
    return fig


def visualize_batch(images, labels, class_names=None):
    """可视化一批图像及其标签

    参数:
        images: 图像列表
        labels: 标签列表 (YOLO格式)
        class_names: 类别名称列表

    返回:
        组合的可视化图像
    """
    import matplotlib.pyplot as plt

    if class_names is None:
        class_names = CLASS_NAMES

    n = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (img, lbl) in enumerate(zip(images, labels)):
        ax = axes[idx]

        # 转换为numpy数组
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        ax.imshow(img)

        # 绘制标签
        h, w = img.shape[:2]
        for label in lbl:
            class_id, x, y, bw, bh = label[:5]
            x1 = int((x - bw/2) * w)
            y1 = int((y - bh/2) * h)
            x2 = int((x + bw/2) * w)
            y2 = int((y + bh/2) * h)

            color = CLASS_COLORS.get(int(class_id), (255, 255, 255))
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                 fill=False, color=color, linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1-5, class_names.get(int(class_id), f'C{int(class_id)}'),
                   color=color, fontsize=10, fontproperties=chinese_font, bbox=dict(facecolor='black', alpha=0.5))

        ax.axis('off')

    # 隐藏空的子图
    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig
