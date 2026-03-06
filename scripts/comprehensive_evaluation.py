#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合评估对比脚本
对比四种策略的性能差异,生成详细的评估报告和可视化图表

该脚本用于:
1. 加载各策略的训练结果
2. 计算性能指标对比
3. 生成可视化图表
4. 输出详细的评估报告

作者: 毕业设计项目
版本: 1.0
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
import pandas as pd


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

# 初始化全局中文字体（不同大小）
chinese_font = get_chinese_font(10)
chinese_font_large = get_chinese_font(16)
chinese_font_large_bold = get_chinese_font(14)
chinese_font_small = get_chinese_font(9)
chinese_font_legend = get_chinese_font(12)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """
    综合评估器类
    
    该类用于对四种迁移学习策略进行全面的性能评估和对比分析。
    
    属性:
        results_dir: 训练结果目录
        output_dir: 输出目录
    """
    
    def __init__(
        self,
        results_dir: str = "runs/multi_strategy",
        output_dir: str = "runs/evaluation"
    ):
        """
        初始化评估器
        
        参数:
            results_dir: 训练结果目录
            output_dir: 输出目录
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载训练结果
        self.results_file = self.results_dir / "training_results.json"
        self.results = self._load_results()
        
        logger.info(f"✓ 评估器初始化完成")
        logger.info(f"✓ 结果目录: {self.results_dir}")
        logger.info(f"✓ 输出目录: {self.output_dir}")
    
    def _load_results(self) -> Dict:
        """
        加载训练结果
        
        返回:
            训练结果字典
        """
        try:
            if self.results_file.exists():
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                logger.info(f"✓ 加载了 {len(results)} 个策略的训练结果")
                return results
            else:
                logger.warning(f"⚠ 结果文件不存在: {self.results_file}")
                return {}
        except Exception as e:
            logger.error(f"✗ 加载结果失败: {e}")
            return {}
    
    def calculate_statistics(self) -> pd.DataFrame:
        """
        计算统计数据
        
        返回:
            包含统计数据的DataFrame
        """
        if not self.results:
            logger.warning("没有可用的结果数据")
            return pd.DataFrame()
        
        # 构建数据
        data = []
        for strategy_key, result in self.results.items():
            row = {
                'strategy_key': strategy_key,
                'strategy_name': result.get('strategy_name', strategy_key),
                'mAP50': result.get('best_map50', 0),
                'mAP95': result.get('best_map95', 0),
                'precision': result.get('best_precision', 0),
                'recall': result.get('best_recall', 0),
                'epochs': result.get('epochs_trained', 0),
                'training_time_min': result.get('training_time', 0) / 60,
                'status': result.get('status', 'unknown'),
            }
            # 计算F1分数
            if row['precision'] + row['recall'] > 0:
                row['F1'] = 2 * row['precision'] * row['recall'] / (row['precision'] + row['recall'])
            else:
                row['F1'] = 0
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # 计算综合评分 (可调整权重)
        if not df.empty:
            # 综合评分 = 0.3*mAP50 + 0.3*mAP95 + 0.2*precision + 0.2*recall
            df['overall_score'] = (
                0.3 * df['mAP50'] + 
                0.3 * df['mAP95'] + 
                0.2 * df['precision'] + 
                0.2 * df['recall']
            ) * 100
        
        return df
    
    def generate_comparison_charts(self, df: pd.DataFrame) -> None:
        """
        生成对比图表
        
        参数:
            df: 统计数据DataFrame
        """
        if df.empty:
            logger.warning("没有数据可供可视化")
            return
        
        logger.info("生成对比图表...")
        
        # 设置图表样式（不使用棋盘格）
        sns.set_style("white")
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('四种迁移学习策略性能对比', fontsize=16, fontweight='bold', fontproperties=chinese_font_large)

        strategies = df['strategy_name'].tolist()
        x = np.arange(len(strategies))

        # 1. mAP50对比
        ax1 = axes[0, 0]
        bars1 = ax1.bar(x, df['mAP50'], color=colors[:len(strategies)])
        ax1.set_xlabel('策略', fontproperties=chinese_font)
        ax1.set_ylabel('mAP50', fontproperties=chinese_font)
        ax1.set_title('mAP50 对比', fontproperties=chinese_font)
        ax1.set_xticks(x)
        ax1.set_xticklabels(strategies, rotation=45, ha='right', fontproperties=chinese_font)
        ax1.set_ylim(0, 1)
        for bar, val in zip(bars1, df['mAP50']):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)

        # 2. mAP50-95对比
        ax2 = axes[0, 1]
        bars2 = ax2.bar(x, df['mAP95'], color=colors[:len(strategies)])
        ax2.set_xlabel('策略', fontproperties=chinese_font)
        ax2.set_ylabel('mAP50-95', fontproperties=chinese_font)
        ax2.set_title('mAP50-95 对比', fontproperties=chinese_font)
        ax2.set_xticks(x)
        ax2.set_xticklabels(strategies, rotation=45, ha='right', fontproperties=chinese_font)
        ax2.set_ylim(0, 1)
        for bar, val in zip(bars2, df['mAP95']):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)

        # 3. 精确率和召回率对比
        ax3 = axes[0, 2]
        width = 0.35
        bars3a = ax3.bar(x - width/2, df['precision'], width, label='精确率', color='#3498db', 
                         error_kw={'ecolor': 'gray', 'capsize': 5})
        bars3b = ax3.bar(x + width/2, df['recall'], width, label='召回率', color='#e74c3c',
                         error_kw={'ecolor': 'gray', 'capsize': 5})
        ax3.set_xlabel('策略', fontproperties=chinese_font)
        ax3.set_ylabel('分数', fontproperties=chinese_font)
        ax3.set_title('精确率 vs 召回率', fontproperties=chinese_font)
        ax3.set_xticks(x)
        ax3.set_xticklabels(strategies, rotation=45, ha='right', fontproperties=chinese_font)
        ax3.set_ylim(0, 1)
        ax3.legend(prop=chinese_font)

        # 4. F1分数对比
        ax4 = axes[1, 0]
        bars4 = ax4.bar(x, df['F1'], color=colors[:len(strategies)])
        ax4.set_xlabel('策略', fontproperties=chinese_font)
        ax4.set_ylabel('F1分数', fontproperties=chinese_font)
        ax4.set_title('F1分数对比', fontproperties=chinese_font)
        ax4.set_xticks(x)
        ax4.set_xticklabels(strategies, rotation=45, ha='right', fontproperties=chinese_font)
        ax4.set_ylim(0, 1)
        for bar, val in zip(bars4, df['F1']):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)

        # 5. 训练时间对比
        ax5 = axes[1, 1]
        bars5 = ax5.bar(x, df['training_time_min'], color=colors[:len(strategies)])
        ax5.set_xlabel('策略', fontproperties=chinese_font)
        ax5.set_ylabel('训练时间 (分钟)', fontproperties=chinese_font)
        ax5.set_title('训练时间对比', fontproperties=chinese_font)
        ax5.set_xticks(x)
        ax5.set_xticklabels(strategies, rotation=45, ha='right', fontproperties=chinese_font)
        for bar, val in zip(bars5, df['training_time_min']):
            ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)

        # 6. 综合评分对比
        ax6 = axes[1, 2]
        bars6 = ax6.bar(x, df['overall_score'], color=colors[:len(strategies)])
        ax6.set_xlabel('策略', fontproperties=chinese_font)
        ax6.set_ylabel('综合评分', fontproperties=chinese_font)
        ax6.set_title('综合评分对比 (满分100)', fontproperties=chinese_font)
        ax6.set_xticks(x)
        ax6.set_xticklabels(strategies, rotation=45, ha='right', fontproperties=chinese_font)
        ax6.set_ylim(0, 100)
        for bar, val in zip(bars6, df['overall_score']):
            ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # 保存图表
        chart_file = self.output_dir / "comparison_charts.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        logger.info(f"✓ 对比图表已保存: {chart_file}")
        plt.close()
    
    def generate_radar_chart(self, df: pd.DataFrame) -> None:
        """
        生成雷达图对比
        
        参数:
            df: 统计数据DataFrame
        """
        if df.empty:
            return
        
        logger.info("生成雷达图...")
        
        # 选择用于雷达图的指标
        metrics = ['mAP50', 'precision', 'recall', 'F1']
        
        # 创建角度
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        for idx, row in df.iterrows():
            values = [row[m] for m in metrics]
            values += values[:1]  # 闭合
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['strategy_name'], color=colors[idx])
            ax.fill(angles, values, alpha=0.25, color=colors[idx])
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, size=12, fontproperties=chinese_font_legend)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), prop=chinese_font_legend)
        ax.set_title('四种策略性能雷达图', size=14, fontweight='bold', pad=20, fontproperties=chinese_font_large_bold)
        
        # 保存
        radar_file = self.output_dir / "radar_chart.png"
        plt.savefig(radar_file, dpi=300, bbox_inches='tight')
        logger.info(f"✓ 雷达图已保存: {radar_file}")
        plt.close()
    
    def generate_summary_table(self, df: pd.DataFrame) -> None:
        """
        生成摘要表格
        
        参数:
            df: 统计数据DataFrame
        """
        if df.empty:
            return
        
        logger.info("生成摘要表格...")
        
        # 格式化数据
        summary_df = df.copy()
        summary_df = summary_df.round(4)
        
        # 按综合评分排序
        summary_df = summary_df.sort_values('overall_score', ascending=False)
        
        # 保存为CSV
        csv_file = self.output_dir / "evaluation_summary.csv"
        summary_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        logger.info(f"✓ 摘要表格已保存: {csv_file}")
        
        # 保存为Excel (可选，如果openpyxl可用)
        excel_file = self.output_dir / "evaluation_summary.xlsx"
        try:
            summary_df.to_excel(excel_file, index=False)
            logger.info(f"✓ Excel表格已保存: {excel_file}")
        except ImportError:
            logger.warning("⚠ openpyxl未安装，跳过Excel导出，仅保存CSV格式")
        
        return summary_df
    
    def generate_detailed_report(self, df: pd.DataFrame) -> str:
        """
        生成详细评估报告
        
        参数:
            df: 统计数据DataFrame
            
        返回:
            报告文件路径
        """
        logger.info("生成详细评估报告...")
        
        report_lines = []
        report_lines.append("# 火焰烟雾检测 - 四种迁移学习策略综合评估报告\n")
        report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 1. 执行摘要
        report_lines.append("## 一、执行摘要\n\n")
        
        if not df.empty:
            best_strategy = df.loc[df['overall_score'].idxmax()]
            report_lines.append(f"本次评估共对比了 {len(df)} 种迁移学习策略。\n\n")
            report_lines.append(f"**最佳策略**: {best_strategy['strategy_name']}\n")
            report_lines.append(f"- mAP50: {best_strategy['mAP50']:.4f}\n")
            report_lines.append(f"- mAP50-95: {best_strategy['mAP95']:.4f}\n")
            report_lines.append(f"- 精确率: {best_strategy['precision']:.4f}\n")
            report_lines.append(f"- 召回率: {best_strategy['recall']:.4f}\n")
            report_lines.append(f"- 综合评分: {best_strategy['overall_score']:.2f}\n\n")
        
        # 2. 详细指标对比
        report_lines.append("## 二、详细指标对比\n\n")
        report_lines.append("| 策略 | mAP50 | mAP50-95 | 精确率 | 召回率 | F1 | 综合评分 |\n")
        report_lines.append("|------|-------|----------|--------|--------|-----|----------|\n")
        
        for _, row in df.sort_values('overall_score', ascending=False).iterrows():
            report_lines.append(
                f"| {row['strategy_name']} | {row['mAP50']:.4f} | {row['mAP95']:.4f} | "
                f"{row['precision']:.4f} | {row['recall']:.4f} | {row['F1']:.4f} | {row['overall_score']:.2f} |\n"
            )
        
        # 3. 策略分析 - 根据实际训练的策略动态生成
        report_lines.append("\n## 三、策略分析\n\n")

        # 定义策略描述的映射（根据实际策略名称）
        strategy_descriptions = {
            '从头训练': {
                'description': '不使用预训练权重，从头开始训练整个模型',
                'pros': ['可学习到任务特定知识', '不受预训练偏见影响'],
                'cons': ['训练时间长', '需要大量数据', '容易过拟合'],
                'recommendation': '适用于有大量标注数据且任务与预训练差异较大的场景'
            },
            '特征提取': {
                'description': '冻结预训练模型骨干网络，仅训练检测头',
                'pros': ['训练速度快', '资源消耗低', '不易过拟合'],
                'cons': ['精度受限', '特征提取能力固定'],
                'recommendation': '适用于小数据集或资源受限场景'
            },
            '微调': {
                'description': '解冻预训练模型最后几层进行微调',
                'pros': ['平衡精度和训练时间', '稳定收敛'],
                'cons': ['需要设计解冻策略', '训练周期较长'],
                'recommendation': '适用于中等规模数据集'
            },
            '全参数微调': {
                'description': '全模型微调，使用分层学习率',
                'pros': ['精度最高', '特征适应性强'],
                'cons': ['训练时间长', '资源消耗大'],
                'recommendation': '适用于大数据集和高精度需求'
            }
        }

        # 根据实际训练的策略生成分析
        for _, row in df.iterrows():
            strategy_name = row['strategy_name']
            if strategy_name in strategy_descriptions:
                analysis = strategy_descriptions[strategy_name]
                report_lines.append(f"### {strategy_name}\n")
                report_lines.append(f"**描述**: {analysis['description']}\n\n")
                report_lines.append("**优点**:\n")
                for pro in analysis['pros']:
                    report_lines.append(f"- {pro}\n")
                report_lines.append("\n**缺点**:\n")
                for con in analysis['cons']:
                    report_lines.append(f"- {con}\n")
                report_lines.append(f"\n**推荐场景**: {analysis['recommendation']}\n\n")
        
        # 4. 结论和建议
        report_lines.append("## 四、结论与建议\n\n")
        
        if not df.empty:
            # 精度最优
            best_map = df.loc[df['mAP50'].idxmax()]
            # 速度最快
            fastest = df.loc[df['training_time_min'].idxmin()]
            # 综合最优
            best_overall = df.loc[df['overall_score'].idxmax()]
            
            report_lines.append("### 关键发现\n\n")
            report_lines.append(f"- **最高精度**: {best_map['strategy_name']} (mAP50: {best_map['mAP50']:.4f})\n")
            report_lines.append(f"- **最快训练**: {fastest['strategy_name']} ({fastest['training_time_min']:.1f}分钟)\n")
            report_lines.append(f"- **综合最优**: {best_overall['strategy_name']} (评分: {best_overall['overall_score']:.2f})\n\n")
            
            report_lines.append("### 选型建议\n\n")

            # 根据实际策略动态生成建议
            if len(df) >= 1:
                # 精度最优
                best_map = df.loc[df['mAP50'].idxmax()]
                report_lines.append(f"1. **{'追求最高精度'}**: 推荐 {best_map['strategy_name']}\n")

            if len(df) >= 2:
                # 速度最快
                fastest = df.loc[df['training_time_min'].idxmin()]
                report_lines.append(f"2. **{'追求最快速度'}**: 推荐 {fastest['strategy_name']}\n")

            if len(df) >= 3:
                # 综合最优
                best_overall = df.loc[df['overall_score'].idxmax()]
                report_lines.append(f"3. **{'综合最优'}**: 推荐 {best_overall['strategy_name']}\n")

            if len(df) >= 4:
                # 训练时间最长（通常是最充分训练的）
                longest_training = df.loc[df['training_time_min'].idxmax()]
                report_lines.append(f"4. **{'最充分训练'}**: {longest_training['strategy_name']} ({longest_training['training_time_min']:.1f}分钟)\n")
        
        # 保存报告
        report_file = self.output_dir / "detailed_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.writelines(report_lines)
        
        logger.info(f"✓ 详细报告已保存: {report_file}")
        return str(report_file)
    
    def run_full_evaluation(self) -> None:
        """
        运行完整评估流程
        """
        logger.info("="*70)
        logger.info("开始综合评估")
        logger.info("="*70)
        
        # 1. 计算统计数据
        df = self.calculate_statistics()
        
        if df.empty:
            logger.error("没有可用的训练结果,请先运行训练脚本")
            return
        
        # 2. 生成对比图表
        self.generate_comparison_charts(df)
        
        # 3. 生成雷达图
        self.generate_radar_chart(df)
        
        # 4. 生成摘要表格
        self.generate_summary_table(df)
        
        # 5. 生成详细报告
        report_file = self.generate_detailed_report(df)
        
        # 6. 打印结果摘要
        print("\n" + "="*70)
        print("评估完成! 结果摘要")
        print("="*70)
        print(df.to_string(index=False))
        print("="*70)
        print(f"\n详细报告: {report_file}")
        print("="*70)


def main():
    """
    主函数
    """
    try:
        evaluator = ComprehensiveEvaluator(
            results_dir="runs/multi_strategy",
            output_dir="runs/evaluation"
        )
        
        evaluator.run_full_evaluation()
        
    except Exception as e:
        logger.error(f"✗ 评估失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
