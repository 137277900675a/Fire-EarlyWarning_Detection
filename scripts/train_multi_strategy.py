#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四种迁移学习策略对比训练脚本
参考PPT: 图像分类-迁移学习

本脚本实现四种迁移学习方法的对比训练:
1. 从头训练 (Scratch Training) - 不使用预训练权重
2. 特征提取 (Feature Extraction) - 冻结大部分层，仅训练顶层
3. 微调 (Fine-tuning) - 解冻部分预训练层
4. 全参数微调 (Full Fine-tuning) - 训练所有参数

作者: 毕业设计项目
版本: 3.1
"""

import os
import sys
import yaml
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict, field
import shutil

import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO


# ============================================================================
# 实验日志管理器
# ============================================================================

class ExperimentLogger:
    """
    实验日志管理器
    用于保存每次实验的完整记录，包括终端输出、配置和结果
    """
    
    def __init__(self, base_dir: str = "runs/experiment_logs"):
        """
        初始化实验日志管理器
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建本次实验的目录
        self.experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = self.base_dir / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志文件路径
        self.log_file = self.experiment_dir / "terminal_log.txt"
        self.config_file = self.experiment_dir / "experiment_config.yaml"
        self.results_file = self.experiment_dir / "experiment_results.json"
        
        # 打开日志文件
        self.log_fp = open(self.log_file, 'w', encoding='utf-8')
        
        print(f"实验日志目录: {self.experiment_dir}")
    
    def log(self, message: str):
        """记录日志消息"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        self.log_fp.write(log_line + '\n')
        self.log_fp.flush()
    
    def log_section(self, title: str):
        """记录章节标题"""
        separator = "=" * 70
        self.log(separator)
        self.log(title)
        self.log(separator)
    
    def save_config(self, config: Dict):
        """保存实验配置"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    def save_results(self, results: Dict):
        """保存实验结果"""
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def close(self):
        """关闭日志文件"""
        if self.log_fp:
            self.log_fp.close()
    
    def get_experiment_dir(self) -> Path:
        """获取实验目录路径"""
        return self.experiment_dir


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class TrainingResult:
    """训练结果数据类"""
    strategy_name: str
    epochs_trained: int = 0
    best_map50: float = 0.0
    best_map95: float = 0.0
    best_precision: float = 0.0
    best_recall: float = 0.0
    training_time: float = 0.0
    model_path: str = ""
    status: str = "pending"


# ============================================================================
# 迁移学习训练器类
# ============================================================================

class TransferLearningTrainer:
    """四种迁移学习策略训练器"""
    
    def __init__(
        self,
        config_path: str = "configs/transfer_strategies.yaml",
        dataset_yaml: str = "data/dataset.yaml",
        output_dir: str = "runs/multi_strategy",
        model_dir: str = "models/pretrained"
    ):
        """初始化训练器"""
        # 初始化实验日志管理器
        self.exp_logger = ExperimentLogger("runs/experiment_logs")
        self.exp_logger.log_section("四种迁移学习策略训练器初始化")
        
        self.config_path = Path(config_path)
        self.dataset_yaml = Path(dataset_yaml)
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.strategies_config = self._load_config()
        self.common_params = self.strategies_config.get('common_params', {})
        
        # 保存配置到实验日志
        self.exp_logger.save_config(self.strategies_config)
        
        self.results: Dict[str, TrainingResult] = {}
        self.models: Dict[str, YOLO] = {}
        
        self.exp_logger.log(f"配置加载完成, 输出目录: {self.output_dir}")
    
    def _log(self, message: str):
        """统一日志输出"""
        self.exp_logger.log(message)
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self._log(f"策略配置加载成功: {self.config_path}")
            return config
        except Exception as e:
            self._log(f"配置加载失败: {e}")
            raise
    
    def _get_base_train_params(self, strategy_name: str) -> Dict:
        """获取基础训练参数"""
        aug = self.common_params.get('augmentation', {})
        opt = self.common_params.get('optimizer', {})
        lr_sched = self.common_params.get('lr_scheduler', {})
        train = self.common_params.get('training', {})
        
        return {
            'data': str(self.dataset_yaml),
            'imgsz': self.common_params.get('model', {}).get('imgsz', 640),
            'device': self.common_params.get('model', {}).get('device', 0),
            'workers': train.get('workers', 8),
            'hsv_h': aug.get('hsv_h', 0.015),
            'hsv_s': aug.get('hsv_s', 0.7),
            'hsv_v': aug.get('hsv_v', 0.4),
            'degrees': aug.get('degrees', 10.0),
            'translate': aug.get('translate', 0.1),
            'scale': aug.get('scale', 0.5),
            'flipud': aug.get('flipud', 0.5),
            'fliplr': aug.get('fliplr', 0.5),
            'mosaic': aug.get('mosaic', 1.0),
            'mixup': aug.get('mixup', 0.0),
            'copy_paste': aug.get('copy_paste', 0.0),
            'lr0': opt.get('lr0', 0.01),
            'lrf': opt.get('lrf', 0.01),
            'momentum': opt.get('momentum', 0.937),
            'weight_decay': opt.get('weight_decay', 0.0005),
            'warmup_momentum': lr_sched.get('warmup_momentum', 0.8),
            'warmup_bias_lr': lr_sched.get('warmup_bias_lr', 0.1),
            'close_mosaic': train.get('close_mosaic', 15),
            'amp': train.get('amp', True),
            'verbose': train.get('verbose', True),
            'seed': train.get('seed', 42),
            'cos_lr': True,
            'project': str(self.output_dir / strategy_name),
            'name': 'train',
            'exist_ok': True,
            'save': True,
        }
    
    def freeze_model_layers(self, model: YOLO, freeze_mode: str, unfreeze_layers: str = "") -> None:
        """根据迁移学习策略冻结/解冻模型层"""
        model_obj = model.model
        if hasattr(model_obj, 'model'):
            model_obj = model_obj.model
        
        if freeze_mode == "scratch":
            self._log("【从头训练】随机初始化权重")
        elif freeze_mode == "feature_extraction":
            self._log("【特征提取】冻结骨干网络和颈部，仅训练检测头...")
            if hasattr(model_obj, 'backbone'):
                for param in model_obj.backbone.parameters():
                    param.requires_grad = False
            if hasattr(model_obj, 'neck'):
                for param in model_obj.neck.parameters():
                    param.requires_grad = False
        elif freeze_mode == "partial_unfreeze":
            self._log(f"【微调】解冻部分高层: {unfreeze_layers}...")
            if hasattr(model_obj, 'backbone'):
                for param in model_obj.backbone.parameters():
                    param.requires_grad = False
            if hasattr(model_obj, 'neck'):
                for param in model_obj.neck.parameters():
                    param.requires_grad = False
            if hasattr(model_obj, 'backbone'):
                backbone_params = list(model_obj.backbone.parameters())
                num_to_unfreeze = len(backbone_params) // 3
                for param in backbone_params[-num_to_unfreeze:]:
                    param.requires_grad = True
        elif freeze_mode == "full_unfreeze":
            self._log("【全参数微调】解冻所有层，训练整个网络...")
            for param in model_obj.parameters():
                param.requires_grad = True
        
        total_params = sum(p.numel() for p in model_obj.parameters())
        trainable_params = sum(p.numel() for p in model_obj.parameters() if p.requires_grad)
        self._log(f"可训练参数: {trainable_params:,} / 总参数: {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    def train_strategy_1_scratch(self) -> TrainingResult:
        """方案1: 从头训练 (Scratch Training)"""
        self._log("="*70)
        self._log("方案1: 从头训练 (Scratch Training)")
        self._log("="*70)
        
        config = self.strategies_config.get('strategy_1_scratch', {})
        
        try:
            model_path = self.model_dir / "yolov8s.yaml"
            if not model_path.exists():
                model_path = "yolov8s.yaml"
            
            model = YOLO(str(model_path))
            self._log(f"加载模型架构(无预训练权重): {model_path}")
            self.freeze_model_layers(model, freeze_mode="scratch")
            
            params = self._get_base_train_params("strategy_1_scratch")
            params.update({
                'epochs': config.get('epochs', 200),
                'batch': config.get('batch_size', 32),
                'lr0': config.get('learning_rate', 0.01),
                'patience': 50,
                'warmup_epochs': config.get('warmup_epochs', 10),
                'dropout': config.get('dropout', 0.5),
                'label_smoothing': config.get('label_smoothing', 0.1),
            })
            
            start_time = datetime.now()
            self._log(f"开始训练: epochs={params['epochs']}, batch={params['batch']}, lr={params['lr0']}")
            
            results = model.train(**params)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            result = TrainingResult(
                strategy_name="从头训练",
                epochs_trained=params['epochs'],
                best_map50=results.box.map50 if hasattr(results.box, 'map50') else 0,
                best_map95=results.box.map if hasattr(results.box, 'map') else 0,
                best_precision=results.box.mp if hasattr(results.box, 'mp') else 0,
                best_recall=results.box.mr if hasattr(results.box, 'mr') else 0,
                training_time=training_time,
                model_path=str(self.output_dir / "strategy_1_scratch" / "train" / "weights" / "best.pt"),
                status="success"
            )
            
            self.models['scratch'] = model
            self._log(f"训练完成, mAP50: {result.best_map50:.4f}, 耗时: {training_time/60:.1f}分钟")
            
            return result
            
        except Exception as e:
            self._log(f"训练失败: {e}")
            return TrainingResult(strategy_name="从头训练", status=f"failed: {str(e)}")
    
    def train_strategy_2_feature_extraction(self) -> TrainingResult:
        """方案2: 特征提取 (Feature Extraction)"""
        self._log("="*70)
        self._log("方案2: 特征提取 (Feature Extraction)")
        self._log("="*70)
        
        config = self.strategies_config.get('strategy_2_feature_extraction', {})
        
        try:
            model_path = self.model_dir / "yolov8s.pt"
            if not model_path.exists():
                model_path = "yolov8s.pt"
            
            model = YOLO(str(model_path))
            self._log(f"加载预训练模型: {model_path}")
            self.freeze_model_layers(model, freeze_mode="feature_extraction")
            
            params = self._get_base_train_params("strategy_2_feature_extraction")
            params.update({
                'epochs': config.get('epochs', 50),
                'batch': config.get('batch_size', 32),
                'lr0': config.get('learning_rate', 0.001),
                'patience': 20,
                'warmup_epochs': config.get('warmup_epochs', 3),
            })
            
            start_time = datetime.now()
            self._log(f"开始训练: epochs={params['epochs']}, batch={params['batch']}, lr={params['lr0']}")
            
            results = model.train(**params)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            result = TrainingResult(
                strategy_name="特征提取",
                epochs_trained=params['epochs'],
                best_map50=results.box.map50 if hasattr(results.box, 'map50') else 0,
                best_map95=results.box.map if hasattr(results.box, 'map') else 0,
                best_precision=results.box.mp if hasattr(results.box, 'mp') else 0,
                best_recall=results.box.mr if hasattr(results.box, 'mr') else 0,
                training_time=training_time,
                model_path=str(self.output_dir / "strategy_2_feature_extraction" / "train" / "weights" / "best.pt"),
                status="success"
            )
            
            self.models['feature_extraction'] = model
            self._log(f"训练完成, mAP50: {result.best_map50:.4f}, 耗时: {training_time/60:.1f}分钟")
            
            return result
            
        except Exception as e:
            self._log(f"训练失败: {e}")
            return TrainingResult(strategy_name="特征提取", status=f"failed: {str(e)}")
    
    def train_strategy_3_fine_tuning(self) -> TrainingResult:
        """方案3: 微调 (Fine-tuning)"""
        self._log("="*70)
        self._log("方案3: 微调 (Fine-tuning)")
        self._log("="*70)
        
        config = self.strategies_config.get('strategy_3_fine_tuning', {})
        
        try:
            model_path = self.model_dir / "yolov8s.pt"
            if not model_path.exists():
                model_path = "yolov8s.pt"
            
            model = YOLO(str(model_path))
            self._log(f"加载预训练模型: {model_path}")
            
            unfreeze_layers = config.get('unfreeze_layers', 'last_2_stages')
            self.freeze_model_layers(model, freeze_mode="partial_unfreeze", unfreeze_layers=unfreeze_layers)
            
            params = self._get_base_train_params("strategy_3_fine_tuning")
            params.update({
                'epochs': config.get('epochs', 100),
                'batch': config.get('batch_size', 32),
                'lr0': config.get('learning_rate', 0.0005),
                'patience': 30,
                'warmup_epochs': config.get('warmup_epochs', 5),
            })
            
            start_time = datetime.now()
            self._log(f"开始训练: epochs={params['epochs']}, batch={params['batch']}, lr={params['lr0']}")
            
            results = model.train(**params)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            result = TrainingResult(
                strategy_name="微调",
                epochs_trained=params['epochs'],
                best_map50=results.box.map50 if hasattr(results.box, 'map50') else 0,
                best_map95=results.box.map if hasattr(results.box, 'map') else 0,
                best_precision=results.box.mp if hasattr(results.box, 'mp') else 0,
                best_recall=results.box.mr if hasattr(results.box, 'mr') else 0,
                training_time=training_time,
                model_path=str(self.output_dir / "strategy_3_fine_tuning" / "train" / "weights" / "best.pt"),
                status="success"
            )
            
            self.models['fine_tuning'] = model
            self._log(f"训练完成, mAP50: {result.best_map50:.4f}, 耗时: {training_time/60:.1f}分钟")
            
            return result
            
        except Exception as e:
            self._log(f"训练失败: {e}")
            return TrainingResult(strategy_name="微调", status=f"failed: {str(e)}")
    
    def train_strategy_4_full_finetuning(self) -> TrainingResult:
        """方案4: 全参数微调 (Full Fine-tuning)"""
        self._log("="*70)
        self._log("方案4: 全参数微调 (Full Fine-tuning)")
        self._log("="*70)
        
        config = self.strategies_config.get('strategy_4_full_finetuning', {})
        
        try:
            model_path = self.model_dir / "yolov8s.pt"
            if not model_path.exists():
                model_path = "yolov8s.pt"
            
            model = YOLO(str(model_path))
            self._log(f"加载预训练模型: {model_path}")
            self.freeze_model_layers(model, freeze_mode="full_unfreeze")
            
            params = self._get_base_train_params("strategy_4_full_finetuning")
            params.update({
                'epochs': config.get('epochs', 150),
                'batch': config.get('batch_size', 16),
                'lr0': config.get('learning_rate', 0.0001),
                'patience': 50,
                'warmup_epochs': config.get('warmup_epochs', 10),
                'dropout': config.get('dropout', 0.5),
                'label_smoothing': config.get('label_smoothing', 0.1),
            })
            
            start_time = datetime.now()
            self._log(f"开始训练: epochs={params['epochs']}, batch={params['batch']}, lr={params['lr0']}")
            
            results = model.train(**params)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            result = TrainingResult(
                strategy_name="全参数微调",
                epochs_trained=params['epochs'],
                best_map50=results.box.map50 if hasattr(results.box, 'map50') else 0,
                best_map95=results.box.map if hasattr(results.box, 'map') else 0,
                best_precision=results.box.mp if hasattr(results.box, 'mp') else 0,
                best_recall=results.box.mr if hasattr(results.box, 'mr') else 0,
                training_time=training_time,
                model_path=str(self.output_dir / "strategy_4_full_finetuning" / "train" / "weights" / "best.pt"),
                status="success"
            )
            
            self.models['full_finetuning'] = model
            self._log(f"训练完成, mAP50: {result.best_map50:.4f}, 耗时: {training_time/60:.1f}分钟")
            
            return result
            
        except Exception as e:
            self._log(f"训练失败: {e}")
            return TrainingResult(strategy_name="全参数微调", status=f"failed: {str(e)}")
    
    def train_all_strategies(self) -> Dict[str, TrainingResult]:
        """训练所有四种策略"""
        self.exp_logger.log_section("开始四种迁移学习策略训练")
        
        strategy_map = {
            'scratch': self.train_strategy_1_scratch,
            'feature_extraction': self.train_strategy_2_feature_extraction,
            'fine_tuning': self.train_strategy_3_fine_tuning,
            'full_finetuning': self.train_strategy_4_full_finetuning,
        }
        
        for strategy_key, train_func in strategy_map.items():
            self.exp_logger.log_section(f"开始训练方案: {strategy_key}")
            
            try:
                result = train_func()
                self.results[strategy_key] = result
                self._log(f"{strategy_key} 训练完成")
            except Exception as e:
                self._log(f"{strategy_key} 训练失败: {e}")
                self.results[strategy_key] = TrainingResult(
                    strategy_name=strategy_key,
                    status=f"failed: {str(e)}"
                )
        
        self._save_results()
        self._generate_comparison_report()
        
        # 关闭实验日志
        self.exp_logger.close()
        
        return self.results
    
    def _save_results(self) -> None:
        """保存训练结果"""
        results_data = {}
        for key, result in self.results.items():
            results_data[key] = {
                'strategy_name': result.strategy_name,
                'epochs_trained': result.epochs_trained,
                'best_map50': result.best_map50,
                'best_map95': result.best_map95,
                'best_precision': result.best_precision,
                'best_recall': result.best_recall,
                'training_time': result.training_time,
                'model_path': result.model_path,
                'status': result.status,
            }
        
        results_file = self.output_dir / "training_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        # 保存到实验日志
        self.exp_logger.save_results(results_data)
        self._log(f"结果已保存: {results_file}")
    
    def _generate_comparison_report(self) -> str:
        """生成对比分析报告"""
        self.exp_logger.log_section("生成对比分析报告")
        
        report_lines = []
        report_lines.append("# 火焰烟雾检测 - 四种迁移学习策略对比分析报告\n")
        report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 方案说明
        report_lines.append("## 一、四种方案说明\n\n")
        
        strategies_info = {
            'scratch': {'name': '从头训练 (Scratch)', 'desc': '不使用预训练权重，从随机初始化开始训练'},
            'feature_extraction': {'name': '特征提取', 'desc': '冻结整个骨干网络和颈部，仅训练检测头'},
            'fine_tuning': {'name': '微调', 'desc': '冻结大部分网络，解冻高层与检测头一起训练'},
            'full_finetuning': {'name': '全参数微调', 'desc': '解冻所有层，对整个网络进行微调训练'}
        }
        
        for key, info in strategies_info.items():
            report_lines.append(f"### {info['name']}\n")
            report_lines.append(f"**描述**: {info['desc']}\n\n")
        
        # 实验结果对比
        report_lines.append("## 二、实验结果对比\n\n")
        
        if self.results:
            report_lines.append("| 方案 | mAP50 | mAP50-95 | 精确率 | 召回率 | 训练时间(分) | 状态 |\n")
            report_lines.append("|------|-------|----------|--------|--------|-------------|------|\n")
            
            for key, result in self.results.items():
                strategy_name = result.strategy_name or key
                time_min = result.training_time / 60 if result.training_time > 0 else 0
                report_lines.append(
                    f"| {strategy_name} | {result.best_map50:.4f} | "
                    f"{result.best_map95:.4f} | {result.best_precision:.4f} | "
                    f"{result.best_recall:.4f} | {time_min:.1f} | {result.status} |\n"
                )
        
        # 保存报告
        report_file = self.output_dir / "comparison_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.writelines(report_lines)
        
        self._log(f"对比分析报告已生成: {report_file}")
        return str(report_file)


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    try:
        trainer = TransferLearningTrainer(
            config_path="configs/transfer_strategies.yaml",
            dataset_yaml="data/dataset.yaml",
            output_dir="runs/multi_strategy",
            model_dir="models/pretrained"
        )
        
        results = trainer.train_all_strategies()
        
        print("\n" + "="*70)
        print("训练完成! 结果摘要:")
        print("="*70)
        
        for key, result in results.items():
            print(f"\n方案: {result.strategy_name}")
            print(f"  mAP50: {result.best_map50:.4f}")
            print(f"  mAP50-95: {result.best_map95:.4f}")
            print(f"  精确率: {result.best_precision:.4f}")
            print(f"  召回率: {result.best_recall:.4f}")
            print(f"  状态: {result.status}")
        
        print("\n" + "="*70)
        print(f"实验日志目录: {trainer.exp_logger.get_experiment_dir()}")
        print(f"训练结果目录: {trainer.output_dir}")
        print("="*70)
        
    except Exception as e:
        print(f"程序执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
