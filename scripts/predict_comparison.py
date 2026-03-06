#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四种迁移学习策略预测对比脚本
使用四个方案的best.pt模型对测试集进行预测，并对比各项指标

使用方法: python scripts/predict_comparison.py

输出:
- 详细的指标对比报告 (predict_comparison_report.md)
- 对比可视化图表 (predict_comparison_charts.png)
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
import pandas as pd
from tqdm import tqdm

# 导入YOLO
from ultralytics import YOLO


# ==================== 字体设置 ====================
CHINESE_FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]


def get_chinese_font(size=10):
    """获取中文字体FontProperties对象"""
    for font_path in CHINESE_FONT_PATHS:
        if Path(font_path).exists():
            return font_manager.FontProperties(fname=font_path, size=size)
    return font_manager.FontProperties(size=size)


chinese_font = get_chinese_font(10)
chinese_font_large = get_chinese_font(16)
chinese_font_small = get_chinese_font(9)


# ==================== 配置 ====================
# 四种策略的模型路径
STRATEGY_MODELS = {
    "从头训练": "runs/multi_strategy/strategy_1_scratch/train/weights/best.pt",
    "特征提取": "runs/multi_strategy/strategy_2_feature_extraction/train/weights/best.pt",
    "微调": "runs/multi_strategy/strategy_3_fine_tuning/train/weights/best.pt",
    "全参数微调": "runs/multi_strategy/strategy_4_full_finetuning/train/weights/best.pt",
}

# 数据集配置
DATA_CONFIG = "data/dataset.yaml"

# 输出目录
OUTPUT_DIR = "runs/predict_comparison"


# ==================== 日志设置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== 指标计算函数 ====================
def calculate_confusion_matrix(
    true_labels: List[int],
    pred_labels: List[int],
    num_classes: int = 2
) -> Dict[Tuple[int, int], int]:
    """计算混淆矩阵
    
    参数:
        true_labels: 真实标签列表
        pred_labels: 预测标签列表
        num_classes: 类别数量
    
    返回:
        混淆矩阵字典 {(真实类别, 预测类别): 数量}
    """
    confusion = defaultdict(int)
    for true, pred in zip(true_labels, pred_labels):
        confusion[(true, pred)] += 1
    return dict(confusion)


def calculate_tp_fp_tn_fn(
    true_labels: List[int],
    pred_labels: List[int],
    num_classes: int = 2
) -> Dict[str, Dict[int, int]]:
    """计算每个类别的TP、FP、TN、FN
    
    参数:
        true_labels: 真实标签列表
        pred_labels: 预测标签列表
        num_classes: 类别数量
    
    返回:
        指标字典 {类别: {TP/FP/TN/FN: 数量}}
    """
    results = {}
    
    for class_id in range(num_classes):
        # 将问题转化为二分类：当前类别 vs 其他类别
        true_binary = [1 if t == class_id else 0 for t in true_labels]
        pred_binary = [1 if p == class_id else 0 for p in pred_labels]
        
        tp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 1)
        tn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 0)
        fn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 0)
        
        results[class_id] = {
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn
        }
    
    return results


def calculate_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    """计算精确率、召回率、F1等指标
    
    参数:
        tp: True Positive
        fp: False Positive
        tn: True Negative
        fn: False Negative
    
    返回:
        指标字典
    """
    # 精确率 (Precision)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # 召回率 (Recall)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1分数
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 准确率
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }


def parse_yolo_labels(label_dir: Path, image_files: List[str]) -> Dict[str, List[int]]:
    """解析YOLO格式的标签文件
    
    参数:
        label_dir: 标签目录路径
        image_files: 图像文件列表
    
    返回:
        {图像文件名: [类别ID列表]}
    """
    labels = {}
    
    for img_file in image_files:
        # 将图像文件名转换为标签文件名
        label_file = label_dir / (Path(img_file).stem + '.txt')
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                classes = []
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        classes.append(class_id)
                labels[img_file] = classes
        else:
            labels[img_file] = []
    
    return labels


# ==================== 预测器类 ====================
class PredictComparison:
    """四种策略预测对比类"""
    
    def __init__(
        self,
        strategy_models: Dict[str, str] = STRATEGY_MODELS,
        data_config: str = DATA_CONFIG,
        output_dir: str = OUTPUT_DIR,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """初始化预测对比器
        
        参数:
            strategy_models: 策略名称到模型路径的映射
            data_config: 数据集配置文件路径
            output_dir: 输出目录
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
        """
        self.strategy_models = strategy_models
        self.data_config = data_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 类别名称
        self.class_names = {0: '火焰', 1: '烟雾'}
        
        # 存储所有模型的预测结果
        self.predictions = {}
        
        # 存储所有模型的指标
        self.metrics = {}
        
        logger.info(f"✓ 预测对比器初始化完成")
        logger.info(f"✓ 输出目录: {self.output_dir}")
        logger.info(f"✓ 策略数量: {len(strategy_models)}")
    
    def load_models(self) -> None:
        """加载所有策略的模型"""
        logger.info("加载模型...")
        
        self.models = {}
        for strategy_name, model_path in self.strategy_models.items():
            path = Path(model_path)
            if path.exists():
                self.models[strategy_name] = YOLO(str(path))
                logger.info(f"✓ 已加载 {strategy_name}: {model_path}")
            else:
                logger.warning(f"⚠ 模型文件不存在: {model_path}")
        
        logger.info(f"成功加载 {len(self.models)} 个模型")
    
    def get_dataset_info(self) -> Tuple[List[str], Dict[str, List[int]]]:
        """获取数据集信息
        
        返回:
            (图像文件列表, 真实标签字典)
        """
        # 尝试使用验证集（如果测试集没有标签）
        test_img_dir = Path("data/images/test")
        test_label_dir = Path("data/labels/test")
        val_img_dir = Path("data/images/val")
        val_label_dir = Path("data/labels/val")
        
        # 优先使用测试集，如果测试集没有标签则使用验证集
        if test_label_dir.exists() and any(test_label_dir.iterdir()):
            img_dir = test_img_dir
            label_dir = test_label_dir
            dataset_type = "测试集"
        elif val_label_dir.exists() and any(val_label_dir.iterdir()):
            img_dir = val_img_dir
            label_dir = val_label_dir
            dataset_type = "验证集"
        else:
            raise ValueError("找不到有效的标签数据集")
        
        # 获取图像文件列表
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [
            f for f in os.listdir(img_dir) 
            if Path(f).suffix.lower() in image_extensions
        ]
        
        # 解析真实标签
        true_labels = parse_yolo_labels(label_dir, image_files)
        
        logger.info(f"使用 {dataset_type}: {len(image_files)} 张图像")
        
        return image_files, true_labels
    
    def predict_single_model(
        self, 
        model: YOLO, 
        image_files: List[str],
        img_dir: Path
    ) -> Dict[str, List[int]]:
        """使用单个模型进行预测
        
        参数:
            model: YOLO模型
            image_files: 图像文件列表
            img_dir: 图像目录
        
        返回:
            {图像文件名: [预测类别ID列表]}
        """
        predictions = {}
        
        # 批量预测
        results = model.predict(
            source=str(img_dir),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            save=False
        )
        
        # 解析预测结果
        for result in results:
            img_name = Path(result.path).name
            pred_classes = []
            
            if result.boxes is not None and len(result.boxes) > 0:
                classes = result.boxes.cls.cpu().numpy()
                pred_classes = [int(c) for c in classes]
            
            predictions[img_name] = pred_classes
        
        return predictions
    
    def run_predictions(self) -> None:
        """运行所有模型的预测"""
        # 获取数据集信息
        image_files, self.true_labels = self.get_dataset_info()
        
        # 确定图像目录
        test_label_dir = Path("data/labels/test")
        val_label_dir = Path("data/labels/val")
        
        if test_label_dir.exists() and any(test_label_dir.iterdir()):
            img_dir = Path("data/images/test")
        else:
            img_dir = Path("data/images/val")
        
        logger.info("开始预测...")
        
        for strategy_name, model in self.models.items():
            logger.info(f"预测: {strategy_name}")
            self.predictions[strategy_name] = self.predict_single_model(
                model, image_files, img_dir
            )
        
        logger.info("预测完成")
    
    def calculate_metrics(self) -> None:
        """计算所有模型的指标"""
        logger.info("计算指标...")
        
        for strategy_name, predictions in self.predictions.items():
            # 收集所有真实标签和预测标签
            all_true = []
            all_pred = []
            
            for img_name, true_labels in self.true_labels.items():
                pred_labels = predictions.get(img_name, [])
                
                # 如果预测为空，认为是背景类（类别0）
                if len(pred_labels) == 0:
                    # 没有预测结果时，假设预测为背景
                    # 对于每个真实标签，添加一个"无预测"标记
                    for true_class in true_labels:
                        all_true.append(true_class)
                        all_pred.append(-1)  # -1 表示无预测
                else:
                    # 如果有预测，取第一个预测结果（YOLO通常会给出多个检测）
                    # 这里简化处理：取置信度最高的预测
                    for true_class in true_labels:
                        all_true.append(true_class)
                        all_pred.append(pred_labels[0])  # 取第一个预测
            
            # 计算每个类别的指标
            class_metrics = calculate_tp_fp_tn_fn(all_true, all_pred, num_classes=2)
            
            # 计算总体指标（加权平均）
            total_tp = sum(m['TP'] for m in class_metrics.values())
            total_fp = sum(m['FP'] for m in class_metrics.values())
            total_tn = sum(m['TN'] for m in class_metrics.values())
            total_fn = sum(m['FN'] for m in class_metrics.values())
            
            overall_metrics = calculate_metrics(total_tp, total_fp, total_tn, total_fn)
            
            # 存储结果
            self.metrics[strategy_name] = {
                'class_metrics': class_metrics,
                'overall': overall_metrics,
                'total_tp': total_tp,
                'total_fp': total_fp,
                'total_tn': total_tn,
                'total_fn': total_fn
            }
            
            logger.info(f"{strategy_name}: Precision={overall_metrics['precision']:.4f}, "
                        f"Recall={overall_metrics['recall']:.4f}, F1={overall_metrics['f1']:.4f}")
        
        logger.info("指标计算完成")
    
    def generate_dataframe(self) -> pd.DataFrame:
        """生成指标DataFrame"""
        data = []
        
        for strategy_name, metrics in self.metrics.items():
            overall = metrics['overall']
            class_metrics = metrics['class_metrics']
            
            row = {
                'strategy_name': strategy_name,
                'precision': overall['precision'],
                'recall': overall['recall'],
                'f1': overall['f1'],
                'accuracy': overall['accuracy'],
                'TP': metrics['total_tp'],
                'FP': metrics['total_fp'],
                'TN': metrics['total_tn'],
                'FN': metrics['total_fn'],
            }
            
            # 添加每个类别的指标
            for class_id, class_name in self.class_names.items():
                if class_id in class_metrics:
                    cm = class_metrics[class_id]
                    cm_metrics = calculate_metrics(cm['TP'], cm['FP'], cm['TN'], cm['FN'])
                    row[f'{class_name}_precision'] = cm_metrics['precision']
                    row[f'{class_name}_recall'] = cm_metrics['recall']
                    row[f'{class_name}_f1'] = cm_metrics['f1']
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # 按F1分数排序
        df = df.sort_values('f1', ascending=False)
        
        return df
    
    def generate_charts(self, df: pd.DataFrame) -> None:
        """生成对比图表"""
        logger.info("生成对比图表...")
        
        # 设置样式
        sns.set_style("white")
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        strategies = df['strategy_name'].tolist()
        x = np.arange(len(strategies))
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('四种迁移学习策略预测指标对比', fontsize=16, fontweight='bold', 
                     fontproperties=chinese_font_large)
        
        # 1. 精确率对比
        ax1 = axes[0, 0]
        bars1 = ax1.bar(x, df['precision'], color=colors[:len(strategies)])
        ax1.set_xlabel('策略', fontproperties=chinese_font)
        ax1.set_ylabel('精确率', fontproperties=chinese_font)
        ax1.set_title('精确率对比', fontproperties=chinese_font)
        ax1.set_xticks(x)
        ax1.set_xticklabels(strategies, rotation=45, ha='right', fontproperties=chinese_font)
        ax1.set_ylim(0, 1)
        for bar, val in zip(bars1, df['precision']):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 2. 召回率对比
        ax2 = axes[0, 1]
        bars2 = ax2.bar(x, df['recall'], color=colors[:len(strategies)])
        ax2.set_xlabel('策略', fontproperties=chinese_font)
        ax2.set_ylabel('召回率', fontproperties=chinese_font)
        ax2.set_title('召回率对比', fontproperties=chinese_font)
        ax2.set_xticks(x)
        ax2.set_xticklabels(strategies, rotation=45, ha='right', fontproperties=chinese_font)
        ax2.set_ylim(0, 1)
        for bar, val in zip(bars2, df['recall']):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 3. F1分数对比
        ax3 = axes[1, 0]
        bars3 = ax3.bar(x, df['f1'], color=colors[:len(strategies)])
        ax3.set_xlabel('策略', fontproperties=chinese_font)
        ax3.set_ylabel('F1分数', fontproperties=chinese_font)
        ax3.set_title('F1分数对比', fontproperties=chinese_font)
        ax3.set_xticks(x)
        ax3.set_xticklabels(strategies, rotation=45, ha='right', fontproperties=chinese_font)
        ax3.set_ylim(0, 1)
        for bar, val in zip(bars3, df['f1']):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 4. TP/FP/TN/FN对比
        ax4 = axes[1, 1]
        width = 0.2
        ax4.bar(x - 1.5*width, df['TP'], width, label='TP', color='#2ecc71')
        ax4.bar(x - 0.5*width, df['FP'], width, label='FP', color='#e74c3c')
        ax4.bar(x + 0.5*width, df['TN'], width, label='TN', color='#3498db')
        ax4.bar(x + 1.5*width, df['FN'], width, label='FN', color='#f39c12')
        ax4.set_xlabel('策略', fontproperties=chinese_font)
        ax4.set_ylabel('数量', fontproperties=chinese_font)
        ax4.set_title('TP/FP/TN/FN 对比', fontproperties=chinese_font)
        ax4.set_xticks(x)
        ax4.set_xticklabels(strategies, rotation=45, ha='right', fontproperties=chinese_font)
        ax4.legend(prop=chinese_font)
        
        plt.tight_layout()
        
        # 保存图表
        chart_file = self.output_dir / "predict_comparison_charts.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        logger.info(f"✓ 对比图表已保存: {chart_file}")
        plt.close()
    
    def generate_report(self, df: pd.DataFrame) -> None:
        """生成详细报告"""
        logger.info("生成详细报告...")
        
        report_lines = []
        report_lines.append("# 火焰烟雾检测 - 四种迁移学习策略预测对比报告\n")
        report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 1. 执行摘要
        report_lines.append("## 一、执行摘要\n\n")
        
        if not df.empty:
            best_f1 = df.loc[df['f1'].idxmax()]
            best_precision = df.loc[df['precision'].idxmax()]
            best_recall = df.loc[df['recall'].idxmax()]
            
            report_lines.append("本报告对比了四种迁移学习策略在测试集上的预测性能。\n\n")
            report_lines.append("**关键发现**:\n")
            report_lines.append(f"- **最高F1分数**: {best_f1['strategy_name']} ({best_f1['f1']:.4f})\n")
            report_lines.append(f"- **最高精确率**: {best_precision['strategy_name']} ({best_precision['precision']:.4f})\n")
            report_lines.append(f"- **最高召回率**: {best_recall['strategy_name']} ({best_recall['recall']:.4f})\n\n")
        
        # 2. 详细指标对比
        report_lines.append("## 二、详细指标对比\n\n")
        
        report_lines.append("### 2.1 总体指标\n\n")
        report_lines.append("| 策略 | 精确率 | 召回率 | F1分数 | 准确率 |\n")
        report_lines.append("|------|--------|--------|--------|--------|\n")
        
        for _, row in df.iterrows():
            report_lines.append(
                f"| {row['strategy_name']} | {row['precision']:.4f} | "
                f"{row['recall']:.4f} | {row['f1']:.4f} | {row['accuracy']:.4f} |\n"
            )
        
        report_lines.append("\n### 2.2 混淆矩阵元素\n\n")
        report_lines.append("| 策略 | TP | FP | TN | FN |\n")
        report_lines.append("|------|----|----|----|----|\n")
        
        for _, row in df.iterrows():
            report_lines.append(
                f"| {row['strategy_name']} | {int(row['TP'])} | {int(row['FP'])} | "
                f"{int(row['TN'])} | {int(row['FN'])} |\n"
            )
        
        # 3. 每个类别的指标
        report_lines.append("\n### 2.3 各类别指标\n\n")
        
        for class_id, class_name in self.class_names.items():
            report_lines.append(f"#### {class_name}\n\n")
            report_lines.append("| 策略 | 精确率 | 召回率 | F1分数 |\n")
            report_lines.append("|------|--------|--------|--------|\n")
            
            for _, row in df.iterrows():
                precision_col = f'{class_name}_precision'
                recall_col = f'{class_name}_recall'
                f1_col = f'{class_name}_f1'
                
                if precision_col in row and not pd.isna(row[precision_col]):
                    report_lines.append(
                        f"| {row['strategy_name']} | {row[precision_col]:.4f} | "
                        f"{row[recall_col]:.4f} | {row[f1_col]:.4f} |\n"
                    )
        
        # 4. 结论
        report_lines.append("\n## 三、结论\n\n")
        
        if not df.empty:
            best = df.iloc[0]
            report_lines.append(f"综合考虑各项指标，**{best['strategy_name']}** 在本次预测对比中表现最优。\n")
            report_lines.append(f"- F1分数: {best['f1']:.4f}\n")
            report_lines.append(f"- 精确率: {best['precision']:.4f}\n")
            report_lines.append(f"- 召回率: {best['recall']:.4f}\n")
        
        # 保存报告
        report_file = self.output_dir / "predict_comparison_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.writelines(report_lines)
        
        logger.info(f"✓ 详细报告已保存: {report_file}")
    
    def run(self) -> None:
        """运行完整的预测对比流程"""
        logger.info("="*70)
        logger.info("开始四种策略预测对比")
        logger.info("="*70)
        
        # 1. 加载模型
        self.load_models()
        
        # 2. 运行预测
        self.run_predictions()
        
        # 3. 计算指标
        self.calculate_metrics()
        
        # 4. 生成DataFrame
        df = self.generate_dataframe()
        
        # 5. 生成图表
        self.generate_charts(df)
        
        # 6. 生成报告
        self.generate_report(df)
        
        # 7. 保存CSV
        csv_file = self.output_dir / "predict_comparison_metrics.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        logger.info(f"✓ 指标已保存: {csv_file}")
        
        # 打印摘要
        print("\n" + "="*70)
        print("预测对比完成!")
        print("="*70)
        print(df.to_string(index=False))
        print("="*70)
        print(f"\n输出目录: {self.output_dir}")
        print("="*70)


def main():
    """主函数"""
    try:
        comparison = PredictComparison()
        comparison.run()
    except Exception as e:
        logger.error(f"✗ 预测对比失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
