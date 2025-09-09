#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   segment_train.py
@Time    :   2025/06/17 15:53:43
@Author  :   angkangyu 
'''

"""
segment_train.py - 分割模型训练主脚本

功能：
提供分割模型的训练、验证和测试流程
支持混合精度训练以提高训练效率和节省显存
支持基于Val_IoU的早停机制避免过拟合
"""

import os
import time
import datetime
import argparse
import numpy as np
from sympy import false, true
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import platform
import matplotlib
import pandas as pd
import random
matplotlib.use('Agg')  # 设置为非交互式后端
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 导入自定义模块
from model.model_interface import MInterface
from data.data_interface import SegmentDataInterface
from utils.metrics_utils import (
    SegmentationMetrics, DiceLoss, CombinedLoss, 
    calculate_model_complexity, visualize_results, 
    visualize_training_history, save_training_history_to_excel,
    save_test_results_to_excel, test_model,
    TverskyLoss, FocalTverskyLoss, EnhancedComboLoss, AdaptiveBalancedLoss,RoadIoULoss,IoUBCELoss,
    AdvancedIoULoss,MultiScaleIoULoss,AdaptiveIoULoss,StructuralIoULoss,HybridIoUBCELoss,UltimateRoadIoULoss
)
from  model.Net_parts.public.lovasz_losses import lovasz_hinge, flatten_binary_scores
warnings.filterwarnings("ignore")

# 设置中文字体
system = platform.system()
if system == "Windows":
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
elif system == "Darwin":  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False



class EarlyStoppingIoU:
    """基于IoU的早停机制类 - IoU越高越好"""
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True, verbose=True):
        """
        Args:
            patience (int): 连续多少个epoch没有改善时停止训练
            min_delta (float): 最小改善幅度，小于此值认为没有改善
            restore_best_weights (bool): 是否恢复最佳权重
            verbose (bool): 是否打印早停信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_iou = 0.0  # 🔥 IoU初始化为0，因为越高越好
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, val_iou, model, epoch):
        """
        检查是否需要早停
        
        Args:
            val_iou (float): 当前验证IoU
            model: 模型对象
            epoch (int): 当前epoch
            
        Returns:
            bool: 是否需要早停
        """
        # 🔥 IoU改善判断：当前IoU > 历史最佳IoU + min_delta
        if val_iou > self.best_iou + self.min_delta:
            # 验证IoU有改善
            self.best_iou = val_iou
            self.counter = 0
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            if self.verbose:
                print(f"Early stopping: validation IoU improved to {val_iou:.6f}")
        else:
            # 验证IoU没有改善
            self.counter += 1
            if self.verbose:
                print(f"Early stopping: no IoU improvement for {self.counter}/{self.patience} epochs (current: {val_iou:.6f}, best: {self.best_iou:.6f})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered! No IoU improvement for {self.patience} consecutive epochs.")
                    print(f"Best validation IoU: {self.best_iou:.6f} at epoch {self.best_epoch + 1}")
                
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        print("Restored best model weights")
        
        return self.early_stop



def set_seed(seed=42):
    """设置随机种子以确保实验可重复性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Main process] torch seed: {torch.initial_seed()}, np seed: {seed}, random seed: {seed}")


# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='Segmentation Model Training Script')
    
    # 数据相关参数
    parser.add_argument('--data_dir', type=str, default="data/Massachusetts/split", 
                        help='Dataset directory')
    parser.add_argument('--batch_size', type=int, default=4, 
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=20, 
                        help='Number of data loader workers')
    parser.add_argument('--img_size', type=int, default=512, 
                        help='Image size')
    
    # 模型相关参数
    parser.add_argument('--in_channels', type=int, default=3, 
                        help='Input channels')
    parser.add_argument('--num_classes', type=int, default=1, 
                        help='Number of classes')
    parser.add_argument('--model_type', type=str, default="DemoNet", 
                        help='Model type')
    
    # 训练相关参数
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, 
                        help='Weight decay')
    parser.add_argument('--epochs', type=int, default=150, 
                        help='Training epochs')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--eta_min', type=float, default=1e-6,
                        help='Minimum learning rate for cosine annealing')
        # 添加调度器选择参数
    parser.add_argument('--scheduler', type=str, default='plateau', 
                        choices=['plateau', 'cosine', 'fixed'],
                        help='Learning rate scheduler type')
    
    # 🔥 基于IoU的早停机制参数
    parser.add_argument('--early_stopping', action='store_true', default=true,
                        help='Enable early stopping based on validation IoU')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--min_delta', type=float, default=0.001,
                        help='Minimum IoU improvement to qualify as an improvement')
    parser.add_argument('--no_early_stopping', action='store_true',
                        help='Disable early stopping')
    
    # 混合精度训练参数
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                        help='Enable mixed precision training')
    parser.add_argument('--no_mixed_precision', action='store_true',
                        help='Disable mixed precision training')
    
    # 其他参数
    parser.add_argument('--save_dir', type=str, default=None, 
                        help='Results save directory')
    parser.add_argument('--no_test', action='store_true', 
                        help='Skip testing after training')
    parser.add_argument('--loss_type', type=str, default='combined', 
                        choices=['bce', 'dice', 'combined', 'tversky', 'focal_tversky','lovasz_hinge','AdaptiveBalancedLoss',
                                 'RoadIoULoss','IoUBCELoss','EnhancedComboLoss',
                                 'AdvancedIoULoss','MultiScaleIoULoss','AdaptiveIoULoss','StructuralIoULoss','HybridIoUBCELoss','UltimateRoadIoULoss'],
                        help='Loss function type')

    args = parser.parse_args()
    
    # 处理早停机制参数
    if args.no_early_stopping:
        args.early_stopping = False
    
    # 处理混合精度训练参数
    if args.no_mixed_precision:
        args.mixed_precision = False
    
    if args.save_dir is None:
        mixed_precision_suffix = "_fp16" if args.mixed_precision else "_fp32"
        early_stop_suffix = "_iou_es" if args.early_stopping else ""
        args.save_dir = f"./runs/segmentation/{args.model_type.lower()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}{mixed_precision_suffix}{early_stop_suffix}"
    
    return args


# 训练时间记录类
class TrainingTimer:
    """训练时间记录器"""
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.epoch_times = []
        self.total_time = 0
        
    def start_training(self):
        """开始训练计时"""
        self.start_time = time.time()
        print(f"Training started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    def start_epoch(self):
        """开始单个epoch计时"""
        return time.time()
        
    def end_epoch(self, epoch_start_time, epoch_num):
        """结束单个epoch计时"""
        epoch_time = time.time() - epoch_start_time
        self.epoch_times.append(epoch_time)
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = len(self.epoch_times)  # 这里应该是总epoch数减去当前epoch
        estimated_remaining = avg_epoch_time * remaining_epochs
        
        print(f"Epoch {epoch_num} time: {epoch_time:.2f}s, Average: {avg_epoch_time:.2f}s/epoch")
        
        return epoch_time, avg_epoch_time
        
    def end_training(self):
        """结束训练计时"""
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time
        
        print(f"Training ended at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total training time: {self.total_time:.2f}s ({self.total_time/60:.2f}min, {self.total_time/3600:.2f}h)")
        print(f"Average epoch time: {np.mean(self.epoch_times):.2f}s")
        print(f"Fastest epoch: {np.min(self.epoch_times):.2f}s")
        print(f"Slowest epoch: {np.max(self.epoch_times):.2f}s")
        
        return self.total_time
        
    def get_time_info(self):
        """获取时间统计信息"""
        return {
            'total_time_seconds': self.total_time,
            'total_time_minutes': self.total_time / 60,
            'total_time_hours': self.total_time / 3600,
            'average_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0,
            'min_epoch_time': np.min(self.epoch_times) if self.epoch_times else 0,
            'max_epoch_time': np.max(self.epoch_times) if self.epoch_times else 0,
            'start_time': datetime.datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S') if self.start_time else None,
            'end_time': datetime.datetime.fromtimestamp(self.end_time).strftime('%Y-%m-%d %H:%M:%S') if self.end_time else None
        }


def save_realtime_training_history(history, filepath, epoch, best_epoch_info=None):
    """实时保存训练历史到Excel文件，包含最佳模型信息"""
    try:
        # 创建DataFrame
        df_data = {
            'Epoch': list(range(1, epoch + 2)),  # epoch从0开始，所以+2
            'Train_Loss': history['train_loss'],
            'Val_Loss': history['val_loss'],
            'Train_Dice': history['train_dice'],
            'Val_Dice': history['val_dice'],
            'Train_IoU': history['train_iou'],
            'Val_IoU': history['val_iou'],
            'Train_mIoU': history['train_miou'],
            'Val_mIoU': history['val_miou'],
            'Train_Accuracy': history['train_accuracy'],
            'Val_Accuracy': history['val_accuracy'],
            'Train_Precision': history['train_precision'],
            'Val_Precision': history['val_precision'],
            'Train_Recall': history['train_recall'],
            'Val_Recall': history['val_recall'],
            'Train_F1': history['train_f1'],
            'Val_F1': history['val_f1'],
            'Train_AUC': history['train_auc'],
            'Val_AUC': history['val_auc'],
            'Learning_Rate': history['learning_rate'],
            'Epoch_Time': history['epoch_times']
        }
        
        df = pd.DataFrame(df_data)
        
        # 保存到Excel
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Training_History', index=False)
            
            # 🔥 添加最佳模型信息到新的工作表
            if best_epoch_info:
                best_model_df = pd.DataFrame([best_epoch_info])
                best_model_df.to_excel(writer, sheet_name='Best_Model_Info', index=False)
            
        print(f"Real-time training history saved to: {filepath}")
        
    except Exception as e:
        print(f"Warning: Failed to save real-time training history: {e}")


def print_and_log_best_model_info(best_epoch_info, save_dir):
    """打印并记录最佳模型信息"""
    print("\n" + "="*60)
    print("🏆 BEST MODEL INFORMATION")
    print("="*60)
    print(f"📍 Best Epoch: {best_epoch_info['Best_Epoch']}")
    print(f"🎯 Best Val IoU: {best_epoch_info['Best_Val_IoU']:.6f}")
    print(f"🎲 Best Val Dice: {best_epoch_info['Best_Val_Dice']:.6f}")
    print(f"📊 Best Val F1: {best_epoch_info['Best_Val_F1']:.6f}")
    print(f"📈 Best Val Accuracy: {best_epoch_info['Best_Val_Accuracy']:.6f}")
    print(f"🔍 Best Val Precision: {best_epoch_info['Best_Val_Precision']:.6f}")
    print(f"🎪 Best Val Recall: {best_epoch_info['Best_Val_Recall']:.6f}")
    print(f"📉 Best Val Loss: {best_epoch_info['Best_Val_Loss']:.6f}")
    print(f"⏱️ Epoch Time: {best_epoch_info['Epoch_Time']:.2f}s")
    print(f"📅 Timestamp: {best_epoch_info['Timestamp']}")
    print("="*60)
    
    # 保存到单独的文件
    best_model_file = os.path.join(save_dir, 'best_model_info.txt')
    with open(best_model_file, 'w') as f:
        f.write("BEST MODEL INFORMATION\n")
        f.write("="*60 + "\n")
        for key, value in best_epoch_info.items():
            f.write(f"{key}: {value}\n")
        f.write("="*60 + "\n")
    
    print(f"📄 Best model info saved to: {best_model_file}")


# 修改训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, save_dir, num_classes=1, mixed_precision=True, 
                early_stopping=True, patience=10, min_delta=0.001):
    """Train segmentation model with optional mixed precision and IoU-based early stopping"""
    os.makedirs(save_dir, exist_ok=True)
    
    metrics = SegmentationMetrics(num_classes=num_classes)
    timer = TrainingTimer()  # 创建计时器
    
    # 🔥 初始化基于IoU的早停机制
    early_stopper = None
    if early_stopping:
        early_stopper = EarlyStoppingIoU(
            patience=patience, 
            min_delta=min_delta, 
            restore_best_weights=True, 
            verbose=True
        )
        print(f"IoU-based early stopping enabled: patience={patience}, min_delta={min_delta}")
    else:
        print("Early stopping disabled")
    
    # 初始化混合精度训练的GradScaler
    scaler = GradScaler() if mixed_precision and device.type == 'cuda' else None
    
    # 检查是否真正启用了混合精度
    use_amp = mixed_precision and device.type == 'cuda' and scaler is not None
    
    print(f"Mixed precision training: {'Enabled' if use_amp else 'Disabled'}")
    if use_amp:
        print(f"Using GradScaler for automatic mixed precision")
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': [],
        'train_iou': [],
        'val_iou': [],
        'train_miou': [],
        'val_miou': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'train_precision': [],
        'val_precision': [],
        'train_recall': [],
        'val_recall': [],
        'train_f1': [],
        'val_f1': [],
        'train_auc': [],
        'val_auc': [],
        'learning_rate': [],
        'epoch_times': []
    }
    
    best_val_dice = 0.0
    best_val_f1 = 0.0
    best_val_iou = 0.0  # 🔥 添加最佳IoU跟踪
    
    # 🔥 新增：最佳模型信息跟踪
    best_epoch_info = {
        'Best_Epoch': 0,
        'Best_Val_IoU': 0.0,
        'Best_Val_Dice': 0.0,
        'Best_Val_F1': 0.0,
        'Best_Val_mIoU': 0.0,
        'Best_Val_Accuracy': 0.0,
        'Best_Val_Precision': 0.0,
        'Best_Val_Recall': 0.0,
        'Best_Val_AUC': 0.0,
        'Best_Val_Loss': float('inf'),
        'Best_Train_IoU': 0.0,
        'Best_Train_Dice': 0.0,
        'Best_Train_Loss': 0.0,
        'Learning_Rate': 0.0,
        'Epoch_Time': 0.0,
        'Timestamp': '',
        'Model_Saved_Path': '',
        'Early_Stopping_Counter': 0,
        'Total_Epochs_So_Far': 0
    }
    
    # 实时训练历史Excel文件路径
    realtime_excel_path = os.path.join(save_dir, 'realtime_training_history.xlsx')
    
    timer.start_training()  # 开始训练计时
    
    # 🔥 训练循环 - 支持基于IoU的早停
    for epoch in range(num_epochs):
        epoch_start_time = timer.start_epoch()  # 开始epoch计时
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        train_iou = 0.0
        train_miou = 0.0
        train_accuracy = 0.0
        train_precision = 0.0
        train_recall = 0.0
        train_f1 = 0.0
        train_auc = 0.0
        
        train_progress = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(train_progress):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Ensure masks are 0-1 float
            masks = masks.float()
            if masks.max() > 1.0:
                masks = masks / 255.0  # Convert 0-255 to 0-1
            
            # Ensure masks have correct shape and data type
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(1)
            
            optimizer.zero_grad()
            
            # 混合精度训练的前向传播
            if use_amp:
                with autocast():
                    outputs = model(images)
                    
                    # Handle model output
                    if isinstance(outputs, list):
                        outputs = outputs[0]
                    
                    # Ensure output and target shapes match
                    if outputs.shape != masks.shape:
                        if len(masks.shape) == 3:
                            masks = masks.unsqueeze(1)
                    
                    loss = criterion(outputs, masks)
            else:
                outputs = model(images)
                
                # Handle model output
                if isinstance(outputs, list):
                    outputs = outputs[0]
                
                # Ensure output and target shapes match
                if outputs.shape != masks.shape:
                    if len(masks.shape) == 3:
                        masks = masks.unsqueeze(1)
                
                loss = criterion(outputs, masks)
            
            # 混合精度训练的反向传播
            if use_amp:
                scaler.scale(loss).backward()
                # Gradient clipping with scaler
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate metrics using the new all-in-one method
            with torch.no_grad():
                # 确保在计算指标时使用float32以保证精度
                if use_amp:
                    outputs_for_metrics = outputs.float()
                    masks_for_metrics = masks.float()
                else:
                    outputs_for_metrics = outputs
                    masks_for_metrics = masks
                    
                batch_metrics = metrics.calculate_all_metrics(
                    outputs_for_metrics, 
                    masks_for_metrics.squeeze(1) if masks_for_metrics.shape[1] == 1 else masks_for_metrics
                )
                
                train_dice += batch_metrics['dice']
                train_iou += batch_metrics['iou']
                train_miou += batch_metrics['miou']
                train_accuracy += batch_metrics['accuracy']
                train_precision += batch_metrics['precision']
                train_recall += batch_metrics['recall']
                train_f1 += batch_metrics['f1']
                train_auc += batch_metrics['auc']
            
            train_progress.set_postfix({
                'loss': loss.item(),
                'dice': batch_metrics['dice'],
                'iou': batch_metrics['iou'],
                'miou': batch_metrics['miou'],
                'acc': batch_metrics['accuracy'],
                'f1': batch_metrics['f1']
            })
        
        # Calculate average metrics
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_iou /= len(train_loader)
        train_miou /= len(train_loader)
        train_accuracy /= len(train_loader)
        train_precision /= len(train_loader)
        train_recall /= len(train_loader)
        train_f1 /= len(train_loader)
        train_auc /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        val_miou = 0.0
        val_accuracy = 0.0
        val_precision = 0.0
        val_recall = 0.0
        val_f1 = 0.0
        val_auc = 0.0

        val_progress = tqdm(val_loader, desc="Validation")
        with torch.no_grad():
            for batch in val_progress:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                # Process masks
                masks = masks.float()
                if masks.max() > 1.0:
                    masks = masks / 255.0
                
                if len(masks.shape) == 3:
                    masks = masks.unsqueeze(1)
                
                # 验证阶段也使用混合精度
                if use_amp:
                    with autocast():
                        outputs = model(images)
                        
                        if isinstance(outputs, list):
                            outputs = outputs[0]
                        
                        loss = criterion(outputs, masks)
                else:
                    outputs = model(images)
                    
                    if isinstance(outputs, list):
                        outputs = outputs[0]
                    
                    loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                
                # Calculate metrics using the new all-in-one method
                # 确保在计算指标时使用float32以保证精度
                if use_amp:
                    outputs_for_metrics = outputs.float()
                    masks_for_metrics = masks.float()
                else:
                    outputs_for_metrics = outputs
                    masks_for_metrics = masks
                    
                batch_metrics = metrics.calculate_all_metrics(
                    outputs_for_metrics, 
                    masks_for_metrics.squeeze(1) if masks_for_metrics.shape[1] == 1 else masks_for_metrics
                )
                
                val_dice += batch_metrics['dice']
                val_iou += batch_metrics['iou']
                val_miou += batch_metrics['miou']
                val_accuracy += batch_metrics['accuracy']
                val_precision += batch_metrics['precision']
                val_recall += batch_metrics['recall']
                val_f1 += batch_metrics['f1']
                val_auc += batch_metrics['auc']
                
                val_progress.set_postfix({
                    'loss': loss.item(),
                    'dice': batch_metrics['dice'],
                    'iou': batch_metrics['iou'],
                    'miou': batch_metrics['miou'],
                    'acc': batch_metrics['accuracy'],
                    'f1': batch_metrics['f1']
                })
        
        # Calculate average validation metrics
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        val_miou /= len(val_loader)
        val_accuracy /= len(val_loader)
        val_precision /= len(val_loader)
        val_recall /= len(val_loader)
        val_f1 /= len(val_loader)
        val_auc /= len(val_loader)

        current_lr = optimizer.param_groups[0]['lr']
                # 🔥 修复：只有在有调度器的情况下才更新学习率
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_iou)   # ReduceLROnPlateau需要传入监控的指标
            else:
                scheduler.step() 

        # 结束epoch计时
        epoch_time, avg_epoch_time = timer.end_epoch(epoch_start_time, epoch+1)

        # Record training history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_miou'].append(train_miou)
        history['val_miou'].append(val_miou)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        history['train_precision'].append(train_precision)
        history['val_precision'].append(val_precision)
        history['train_recall'].append(train_recall)
        history['val_recall'].append(val_recall)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['learning_rate'].append(current_lr)
        history['epoch_times'].append(epoch_time)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}, mIoU: {train_miou:.4f}, Acc: {train_accuracy:.4f}, F1: {train_f1:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}, mIoU: {val_miou:.4f}, Acc: {val_accuracy:.4f}, F1: {val_f1:.4f}")
        print(f"  Learning Rate: {current_lr:.8f}")
        
        # 🔥 基于IoU的早停检查
        if early_stopping and early_stopper is not None:
            if early_stopper(val_iou, model, epoch):
                print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                print(f"Best model was at epoch {early_stopper.best_epoch + 1} with validation IoU: {early_stopper.best_iou:.6f}")
                
                # 保存早停信息到文件
                early_stop_info = {
                    'stopped_at_epoch': epoch + 1,
                    'best_epoch': early_stopper.best_epoch + 1,
                    'best_val_iou': early_stopper.best_iou,
                    'patience': patience,
                    'min_delta': min_delta,
                    'total_planned_epochs': num_epochs,
                    'epochs_saved': num_epochs - (epoch + 1),
                    'metric_monitored': 'Validation_IoU'
                }
                
                with open(os.path.join(save_dir, 'early_stopping_info.txt'), 'w') as f:
                    f.write("Early Stopping Information (IoU-based):\n")
                    f.write("=" * 40 + "\n")
                    for key, value in early_stop_info.items():
                        f.write(f"{key}: {value}\n")
                
                break
        
        # 🔥 更新最佳指标跟踪 (现在主要关注IoU)
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_val_dice = val_dice
            best_val_f1 = val_f1
            
            # 🔥 更新最佳模型信息
            best_epoch_info.update({
                'Best_Epoch': epoch + 1,
                'Best_Val_IoU': val_iou,
                'Best_Val_Dice': val_dice,
                'Best_Val_F1': val_f1,
                'Best_Val_mIoU': val_miou,
                'Best_Val_Accuracy': val_accuracy,
                'Best_Val_Precision': val_precision,
                'Best_Val_Recall': val_recall,
                'Best_Val_AUC': val_auc,
                'Best_Val_Loss': val_loss,
                'Best_Train_IoU': train_iou,
                'Best_Train_Dice': train_dice,
                'Best_Train_Loss': train_loss,
                'Learning_Rate': current_lr,
                'Epoch_Time': epoch_time,
                'Timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Model_Saved_Path': os.path.join(save_dir, 'best_model.pth'),
                'Early_Stopping_Counter': early_stopper.counter if early_stopper else 0,
                'Total_Epochs_So_Far': epoch + 1
            })
            
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f"🏆 Saved NEW BEST model at epoch {epoch+1}!")
            print(f"   📈 IoU: {best_val_iou:.6f} (↑{val_iou - (history['val_iou'][-2] if len(history['val_iou']) > 1 else 0):.6f})")
            print(f"   🎲 Dice: {best_val_dice:.6f}")
            print(f"   📊 F1: {best_val_f1:.6f}")
            
            # 🔥 打印最佳模型信息
            print_and_log_best_model_info(best_epoch_info, save_dir)

        # 🔥 实时保存训练历史到Excel（包含最佳模型信息）
        save_realtime_training_history(history, realtime_excel_path, epoch, best_epoch_info)
        
        # Visualize validation results at epoch end
        if epoch % 30 == 0 or epoch == num_epochs - 1:
            visualize_batch = next(iter(val_loader))
            images = visualize_batch['image'].to(device)
            masks = visualize_batch['mask']
            filenames = visualize_batch['filename']
            
            model.eval()
            with torch.no_grad():
                if use_amp:
                    with autocast():
                        outputs = model(images)
                else:
                    outputs = model(images)
            
            # 确保可视化时使用float32
            if use_amp and hasattr(outputs, 'float'):
                outputs = outputs.float()
            
            for i in range(min(3, len(images))):
                visualize_results(
                    images[i], 
                    masks[i], 
                    outputs[i], 
                    f"epoch_{epoch+1}_{filenames[i]}",
                    os.path.join(save_dir, 'visualizations')
                )
    
    total_training_time = timer.end_training()  # 结束训练计时
    
    # 🔥 训练结束后打印最终的最佳模型信息
    print("\n" + "🎉" * 30)
    print("TRAINING COMPLETED!")
    print("🎉" * 30)
    print_and_log_best_model_info(best_epoch_info, save_dir)
    
    visualize_training_history(history, os.path.join(save_dir, 'training_history.png'))
    
    # 保存训练历史到Excel（最终版本）
    save_training_history_to_excel(history, os.path.join(save_dir, 'training_history.xlsx'))
    
    # 保存训练时间信息
    time_info = timer.get_time_info()
    early_stopped = early_stopping and early_stopper is not None and early_stopper.early_stop
    
    with open(os.path.join(save_dir, 'training_time.txt'), 'w') as f:
        f.write("Training Time Information:\n")
        f.write("=" * 40 + "\n")
        f.write(f"Mixed Precision Training: {'Enabled' if use_amp else 'Disabled'}\n")
        f.write(f"Early Stopping: {'Enabled (IoU-based)' if early_stopping else 'Disabled'}\n")
        if early_stopping:
            f.write(f"Early Stopping Patience: {patience}\n")
            f.write(f"Early Stopping Min Delta: {min_delta}\n")
            f.write(f"Early Stopped: {'Yes' if early_stopped else 'No'}\n")
            if early_stopped:
                f.write(f"Best Validation IoU: {early_stopper.best_iou:.6f}\n")
        f.write(f"Start Time: {time_info['start_time']}\n")
        f.write(f"End Time: {time_info['end_time']}\n")
        f.write(f"Total Training Time: {time_info['total_time_seconds']:.2f} seconds\n")
        f.write(f"Total Training Time: {time_info['total_time_minutes']:.2f} minutes\n")
        f.write(f"Total Training Time: {time_info['total_time_hours']:.2f} hours\n")
        f.write(f"Average Epoch Time: {time_info['average_epoch_time']:.2f} seconds\n")
        f.write(f"Fastest Epoch Time: {time_info['min_epoch_time']:.2f} seconds\n")
        f.write(f"Slowest Epoch Time: {time_info['max_epoch_time']:.2f} seconds\n")
        f.write(f"Total Epochs Completed: {len(history['epoch_times'])}\n")
        f.write(f"Planned Epochs: {num_epochs}\n")
        f.write("\n" + "="*40 + "\n")
        f.write("BEST MODEL INFORMATION:\n")
        f.write("="*40 + "\n")
        for key, value in best_epoch_info.items():
            f.write(f"{key}: {value}\n")
    
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
    
    return model, history, time_info, use_amp
def get_model_architecture_info(model):
    """获取模型架构信息，包括编码器和解码器名称"""
    arch_info = {
        'model_class': model.__class__.__name__,
        'encoder_name': 'Unknown',
        'decoder_name': 'Unknown',
        'encoder_class': 'Unknown',
        'decoder_class': 'Unknown',
        'encoder_params': 0,
        'decoder_params': 0
    }
    
    # 检查模型是否有encoder和decoder属性
    if hasattr(model, 'encoder'):
        arch_info['encoder_name'] = model.encoder.__class__.__name__
        arch_info['encoder_class'] = str(type(model.encoder))
        arch_info['encoder_params'] = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
        
    if hasattr(model, 'decoder'):
        arch_info['decoder_name'] = model.decoder.__class__.__name__
        arch_info['decoder_class'] = str(type(model.decoder))
        arch_info['decoder_params'] = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    
    return arch_info


def save_config_info(args, model, model_info, flops_info, save_dir):
    """增强版配置信息保存，包含编码器解码器信息"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取模型架构信息
    arch_info = get_model_architecture_info(model)
    
    config_file = os.path.join(save_dir, 'config.txt')
    
    with open(config_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TRAINING CONFIGURATION\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 模型架构信息
        f.write("MODEL ARCHITECTURE:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Model Class: {arch_info['model_class']}\n")
        f.write(f"Encoder: {arch_info['encoder_name']}\n")
        f.write(f"Decoder: {arch_info['decoder_name']}\n")
        f.write(f"Encoder Parameters: {arch_info['encoder_params']:,}\n")
        f.write(f"Decoder Parameters: {arch_info['decoder_params']:,}\n")
        f.write(f"Encoder/Decoder Ratio: {arch_info['encoder_params'] / max(arch_info['decoder_params'], 1):.2f}\n")
        
        f.write("\nCOMMAND LINE ARGUMENTS:\n")
        f.write("-" * 30 + "\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nMODEL INFORMATION:\n")
        f.write("-" * 30 + "\n")
        for key, value in model_info.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nMODEL COMPLEXITY:\n")
        f.write("-" * 30 + "\n")
        for key, value in flops_info.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n" + "=" * 60 + "\n")

        f.write(f"Random Seed: {args.seed}\n")
        f.write(f"Random Seed (args.seed): {args.seed}\n")
        f.write(f"Main Process torch.initial_seed: {torch.initial_seed()}\n")   
    # 控制台打印
    print(f"Configuration saved to: {config_file}")
    print(f"\n🏗️ Model Architecture Information:")
    print(f"   Model: {arch_info['model_class']}")
    print(f"   📥 Encoder: {arch_info['encoder_name']} ({arch_info['encoder_params']:,} params)")
    print(f"   📤 Decoder: {arch_info['decoder_name']} ({arch_info['decoder_params']:,} params)")
    print(f"   📊 Encoder/Decoder Ratio: {arch_info['encoder_params'] / max(arch_info['decoder_params'], 1):.2f}")
    
    return arch_info


# 主函数
def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 检查混合精度训练的可用性
    if args.mixed_precision and device.type != 'cuda':
        print("Warning: Mixed precision training is only available on CUDA devices. Disabling mixed precision.")
        args.mixed_precision = False
    
    if args.mixed_precision and device.type == 'cuda':
        # 检查CUDA版本和GPU是否支持混合精度
        if torch.cuda.get_device_capability(device)[0] < 7:
            print("Warning: Mixed precision training works best on GPUs with compute capability 7.0 or higher.")
            print(f"Current GPU compute capability: {torch.cuda.get_device_capability(device)}")
    
    print(f"Mixed precision training: {'Enabled' if args.mixed_precision else 'Disabled'}")
    print(f"Early stopping: {'Enabled (IoU-based)' if args.early_stopping else 'Disabled'}")
    if args.early_stopping:
        print(f"Early stopping patience: {args.patience} epochs")
        print(f"Early stopping min delta: {args.min_delta} (IoU improvement)")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("Loading dataset...")
    data_interface = SegmentDataInterface(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )
    
    train_loader = data_interface.train_dataloader()
    val_loader = data_interface.val_dataloader()
    test_loader = data_interface.test_dataloader()
    
    print(f"Train set size: {len(train_loader.dataset)} samples")
    print(f"Validation set size: {len(val_loader.dataset)} samples")
    print(f"Test set size: {len(test_loader.dataset)} samples")
    
    # Check data
    sample_batch = next(iter(train_loader))
    print(f"Image shape: {sample_batch['image'].shape}")
    print(f"Mask shape: {sample_batch['mask'].shape}")
    print(f"Image range: [{sample_batch['image'].min():.3f}, {sample_batch['image'].max():.3f}]")
    print(f"Mask range: [{sample_batch['mask'].min():.3f}, {sample_batch['mask'].max():.3f}]")
    print(f"Mask unique values: {torch.unique(sample_batch['mask'])}")
    
    print(f"Creating model: {args.model_type}...")
    model_interface = MInterface(
        model_type=args.model_type,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        device=device,
        # encoder="resnet50",
        # pretrain=True
    )
    
    model = model_interface.model
    print(f"{args.model_type} model created successfully, parameters: {model_interface.param_count:,}")
    
    # Calculate model complexity
    total_params, trainable_params, flops_info = calculate_model_complexity(
        model, 
        input_size=(args.in_channels, args.img_size, args.img_size), 
        device=device
    )
    
    # 准备模型信息（在训练开始前）
    model_info = {
        'model_type': args.model_type,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / 1024 / 1024,
        'input_size': args.img_size,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'loss_type': args.loss_type,
        'scheduler': 'ReduceLROnPlateau',
        'eta_min': args.eta_min,
        'mixed_precision': args.mixed_precision,
        'early_stopping': args.early_stopping,
        'early_stopping_metric': 'Validation_IoU' if args.early_stopping else None,
        'patience': args.patience if args.early_stopping else None,
        'min_delta': args.min_delta if args.early_stopping else None,
        'device': str(device),
    }
    
    # 在训练开始前保存配置信息
    # 🔥 使用增强版配置保存（包含编码器解码器信息）
    save_config_info(args, model, model_info, flops_info, args.save_dir)
    
    # Select loss function
    if args.loss_type == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss_type == 'dice':
        criterion = DiceLoss()
    elif args.loss_type == 'combined':
        # Adjust weights, favor Dice loss
        criterion = CombinedLoss(dice_weight=0.7, bce_weight=0.3)
    elif args.loss_type == "EnhancedComboLoss":
        criterion = EnhancedComboLoss(dice_weight=0.3, focal_weight=0.4, tversky_weight=0.3)
    elif args.loss_type == 'tversky':
        criterion = TverskyLoss(alpha=0.7, beta=0.3)
    elif args.loss_type == 'focal_tversky':
        criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)
    elif args.loss_type == 'lovasz':
        criterion = lovasz_hinge()
    elif args.loss_type == 'AdaptiveBalancedLoss':
        criterion = AdaptiveBalancedLoss()
    elif args.loss_type == 'RoadIoULoss':
        criterion = RoadIoULoss()
    elif args.loss_type == 'IoUBCELoss':
        criterion = IoUBCELoss()
    elif args.loss_type == 'AdvancedIoULoss':
        criterion = AdvancedIoULoss()
    elif args.loss_type == 'MultiScaleIoULoss':
        criterion = MultiScaleIoULoss()
    elif args.loss_type == 'AdaptiveIoULoss':
        criterion = AdaptiveIoULoss()
    elif args.loss_type == 'StructuralIoULoss':
        criterion = StructuralIoULoss()
    elif args.loss_type == 'HybridIoUBCELoss':
        criterion = HybridIoUBCELoss()
    elif args.loss_type == 'UltimateRoadIoULoss':
        criterion = UltimateRoadIoULoss()
    print(f"Using loss function: {args.loss_type}")
    

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 🔥 根据参数选择调度器类型
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',          # 监控IoU，越大越好
            factor=0.5,          # 学习率减半
            patience=20,         # 20个epoch没改善就降lr
            min_lr=args.eta_min,
            threshold=0.002      # 与early stopping一致
        )
        print(f"Using ReduceLROnPlateau scheduler with patience={scheduler.patience}, factor={scheduler.factor}")
        
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs//10,  
            eta_min=args.eta_min
        )
        print(f"Using CosineAnnealingLR scheduler with T_max={args.epochs//10}, eta_min={args.eta_min}")
        
    elif args.scheduler == 'fixed':
        # 🔥 固定学习率 - 不进行任何调整
        scheduler = None
        print(f"Using fixed learning rate: {args.lr}")
    
    # 🔥 删除这行有问题的代码，因为scheduler可能是None
    # print(f"Using ReduceLROnPlateau scheduler with patience={scheduler.patience}, factor={scheduler.factor}, min_lr={scheduler.min_lrs}")
    
    print("\nStarting model training...")
    
    # ... 后面的代码保持不变 ...
    
    print("\nStarting model training...")
    
    # 🔥 传递基于IoU的早停参数到训练函数
    model, history, time_info, use_amp = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device,
        save_dir=args.save_dir,
        num_classes=args.num_classes,
        mixed_precision=args.mixed_precision,
        early_stopping=args.early_stopping,
        patience=args.patience,
        min_delta=args.min_delta
    )
    
    if not args.no_test:
        print("\nStarting model testing...")
        test_dice, test_iou, test_miou, test_accuracy, test_precision, test_recall, test_f1, test_auc = test_model(
            model=model,
            test_loader=test_loader,
            device=device,
            save_dir=args.save_dir,
            num_classes=args.num_classes
        )
        
        # 更新模型信息（添加训练时间信息）
        model_info.update(time_info)
        model_info['mixed_precision_actual'] = use_amp  # 实际使用的混合精度状态
        model_info['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_info['actual_epochs'] = len(history['epoch_times'])  # 实际训练的epoch数
        
        test_results = {
            'dice': test_dice,
            'iou': test_iou,
            'miou': test_miou,
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'auc': test_auc
        }
        
        # 保存测试结果到Excel
        save_test_results_to_excel(
            test_results, 
            model_info, 
            os.path.join(args.save_dir, 'test_results.xlsx')
        )
        
        # 更新配置文件（添加最终结果）
        with open(os.path.join(args.save_dir, 'config.txt'), 'a') as f:
            f.write("\nFINAL TEST RESULTS:\n")
            f.write("-" * 30 + "\n")
            for key, value in test_results.items():
                f.write(f"{key}: {value:.4f}\n")
            
            f.write("\nTRAINING TIME INFORMATION:\n")
            f.write("-" * 30 + "\n")
            for key, value in time_info.items():
                f.write(f"{key}: {value}\n")
            
            f.write(f"\nACTUAL EPOCHS COMPLETED: {len(history['epoch_times'])}/{args.epochs}\n")
        
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_model.pth'))
        print(f"Model saved to: {args.save_dir}")
        
        print("\nTraining and testing completed!")
        print(f"Final test results:")
        print(f"  Dice Coefficient: {test_dice:.4f}")
        print(f"  IoU Score: {test_iou:.4f}")
        print(f"  mIoU Score: {test_miou:.4f}")
        print(f"  Accuracy: {test_accuracy:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        print(f"  F1 Score: {test_f1:.4f}")
        print(f"  AUC Score: {test_auc:.4f}")
        print(f"  Mixed Precision: {'Enabled' if use_amp else 'Disabled'}")
        print(f"  Epochs Completed: {len(history['epoch_times'])}/{args.epochs}")


if __name__ == "__main__":
    main()