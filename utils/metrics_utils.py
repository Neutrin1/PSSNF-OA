#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   metrics_utils.py
@Time    :   2025/06/18 10:28:08
@Author  :   angkangyu 
'''
"""
metrics_utils.py - 分割模型评估指标工具模块

功能：
提供分割模型的各种评估指标计算、可视化和结果保存功能
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from torch.nn import functional as F


class AdaptiveBalancedLoss(nn.Module):
    """自适应平衡损失 - 修复混合精度兼容性"""
    def __init__(self, alpha=0.8, gamma=2.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        
    def forward(self, pred, target):
        # 🚀 修复：不要在这里使用sigmoid，直接使用logits
        # pred = torch.sigmoid(pred)  # ❌ 删除这行
        
        # 动态权重计算
        target_sigmoid = target  # target已经是0-1的
        pos_ratio = target_sigmoid.sum() / target_sigmoid.numel()
        neg_ratio = 1 - pos_ratio
        
        # 🚀 修复：使用binary_cross_entropy_with_logits而不是binary_cross_entropy
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # 计算focal weight - 需要先sigmoid
        pred_sigmoid = torch.sigmoid(pred)
        p_t = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Dynamic class weights
        class_weight = torch.where(target == 1, neg_ratio, pos_ratio)
        
        focal_loss = self.alpha * class_weight * focal_weight * ce_loss
        
        # Dice Loss for small objects - 使用sigmoid后的预测
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1 - (2 * intersection + 1) / (union + 1)
        
        return focal_loss.mean() + self.beta * dice_loss.mean()

class CombinedLoss(nn.Module):
    """组合损失 - 修复混合精度兼容性"""
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        # 🚀 修复：使用BCEWithLogitsLoss
        self.bce_loss = nn.BCEWithLogitsLoss()  # 而不是nn.BCELoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce

class DiceLoss(nn.Module):
    """Dice损失 - 确保混合精度兼容性"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        # 🚀 确保在计算Dice时使用sigmoid
        pred = torch.sigmoid(pred)  # 将logits转为概率
        
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice

class TverskyLoss(nn.Module):
    """Tversky损失 - 修复混合精度兼容性"""
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, target):
        # 🚀 使用sigmoid转换logits
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (pred_flat * target_flat).sum()
        FP = ((1 - target_flat) * pred_flat).sum()
        FN = (target_flat * (1 - pred_flat)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - Tversky

class FocalTverskyLoss(nn.Module):
    """Focal Tversky损失 - 修复混合精度兼容性"""
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, pred, target):
        # 🚀 使用sigmoid转换logits
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (pred_flat * target_flat).sum()
        FP = ((1 - target_flat) * pred_flat).sum()
        FN = (target_flat * (1 - pred_flat)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        FocalTversky = (1 - Tversky) ** self.gamma
        
        return FocalTversky

# 🚀 修复ContinuityLoss
class ContinuityLoss(nn.Module):
    """连续性损失 - 修复混合精度兼容性"""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, pred, target, skeleton=None):
        # 🚀 修复：使用binary_cross_entropy_with_logits
        base_loss = F.binary_cross_entropy_with_logits(pred, target)
        
        # 连通性损失
        if skeleton is not None:
            pred_sigmoid = torch.sigmoid(pred)
            skeleton_sigmoid = torch.sigmoid(skeleton)
            # 骨架连通性约束
            connectivity_loss = F.mse_loss(pred_sigmoid * skeleton_sigmoid, target * skeleton_sigmoid)
            return base_loss + self.alpha * connectivity_loss
        
        return base_loss


class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss(alpha=0.7, beta=0.3)
        self.alpha = alpha
    def forward(self, pred, target):
        return self.bce(pred, target) * (1 - self.alpha) + self.tversky(pred, target) * self.alpha

class EnhancedComboLoss(nn.Module):
    def __init__(self, dice_weight=0.3, focal_weight=0.4, tversky_weight=0.3):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.tversky_loss = TverskyLoss(alpha=0.7, beta=0.3)
        
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight

    def forward(self, pred, target):
        if len(target.shape) == 3:
            target = target.unsqueeze(1)
        
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        tversky = self.tversky_loss(pred, target)
        
        return (self.dice_weight * dice + 
                self.focal_weight * focal + 
                self.tversky_weight * tversky)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        if len(target.shape) == 3:
            target = target.unsqueeze(1)
            
        pred = torch.sigmoid(pred)
        target = target.float()
        
        # 计算BCE
        bce = -(target * torch.log(pred + self.smooth) + 
               (1 - target) * torch.log(1 - pred + self.smooth))
        
        # 计算focal weight
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        focal_loss = focal_weight * bce
        return focal_loss.mean()

class RoadIoULoss(nn.Module):
    """
    Soft IoU Loss - 直接针对IoU指标进行优化
    特别适用于道路等细长目标的分割任务
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred是模型的原始输出 (logits)
        # target是真实标签 (0或1)
        
        # 1. 将logits转换为概率
        pred_sigmoid = torch.sigmoid(pred)
        
        # 2. 确保target是浮点型
        target = target.float()
        
        # 3. 展平，以便计算
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        
        # 4. 计算交集和并集 (Soft版本)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        
        # 5. 计算IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # 6. 损失是 1 - IoU
        return 1 - iou

class IoUBCELoss(nn.Module):
    """
    IoU Loss 和 BCE Loss的组合
    结合了区域优化和像素级优化，训练更稳定
    """
    def __init__(self, iou_weight=0.7, bce_weight=0.3):
        super().__init__()
        self.iou_weight = iou_weight
        self.bce_weight = bce_weight
        self.iou_loss = RoadIoULoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        iou = self.iou_loss(pred, target)
        bce = self.bce_loss(pred, target)
        
        # 按权重组合
        combined_loss = self.iou_weight * iou + self.bce_weight * bce
        return combined_loss

# ==================== 专门针对IoU优化的高级损失函数 ====================

class AdvancedIoULoss(nn.Module):
    """
    高级IoU损失 - 包含多种IoU变体的组合
    特别适用于道路等细长目标的分割任务
    """
    def __init__(self, smooth=1e-6, focal_gamma=2.0, boundary_weight=0.3):
        super().__init__()
        self.smooth = smooth
        self.focal_gamma = focal_gamma
        self.boundary_weight = boundary_weight

    def forward(self, pred, target):
        # 1. 将logits转换为概率
        pred_sigmoid = torch.sigmoid(pred)
        target = target.float()
        
        # 2. 标准Soft IoU Loss
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # 3. Focal IoU Loss - 对难分类区域加权
        focal_weight = (1 - iou) ** self.focal_gamma
        focal_iou_loss = focal_weight * (1 - iou)
        
        # 4. 边界增强IoU - 对边界区域特别关注
        # 使用梯度检测边界
        target_grad_x = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        target_grad_y = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        
        # 边界mask
        boundary_mask = torch.zeros_like(target)
        boundary_mask[:, :, :-1, :] += target_grad_x
        boundary_mask[:, :, 1:, :] += target_grad_x
        boundary_mask[:, :, :, :-1] += target_grad_y
        boundary_mask[:, :, :, 1:] += target_grad_y
        boundary_mask = (boundary_mask > 0).float()
        
        # 边界区域的IoU
        if boundary_mask.sum() > 0:
            boundary_pred = pred_sigmoid * boundary_mask
            boundary_target = target * boundary_mask
            
            boundary_pred_flat = boundary_pred.view(-1)
            boundary_target_flat = boundary_target.view(-1)
            
            boundary_intersection = (boundary_pred_flat * boundary_target_flat).sum()
            boundary_union = boundary_pred_flat.sum() + boundary_target_flat.sum() - boundary_intersection
            boundary_iou = (boundary_intersection + self.smooth) / (boundary_union + self.smooth)
            boundary_loss = 1 - boundary_iou
        else:
            boundary_loss = 0
        
        # 5. 组合损失
        total_loss = focal_iou_loss + self.boundary_weight * boundary_loss
        
        return total_loss

class MultiScaleIoULoss(nn.Module):
    """
    多尺度IoU损失 - 在不同尺度上计算IoU
    有效处理不同大小的道路段
    """
    def __init__(self, scales=[1, 2, 4], weights=[0.5, 0.3, 0.2], smooth=1e-6):
        super().__init__()
        self.scales = scales
        self.weights = weights
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        target = target.float()
        
        total_loss = 0
        
        for scale, weight in zip(self.scales, self.weights):
            if scale == 1:
                # 原始尺度
                scale_pred = pred_sigmoid
                scale_target = target
            else:
                # 下采样到不同尺度
                scale_pred = F.avg_pool2d(pred_sigmoid, kernel_size=scale, stride=scale)
                scale_target = F.avg_pool2d(target, kernel_size=scale, stride=scale)
            
            # 计算该尺度的IoU Loss
            pred_flat = scale_pred.view(-1)
            target_flat = scale_target.view(-1)
            
            intersection = (pred_flat * target_flat).sum()
            union = pred_flat.sum() + target_flat.sum() - intersection
            iou = (intersection + self.smooth) / (union + self.smooth)
            
            scale_loss = 1 - iou
            total_loss += weight * scale_loss
        
        return total_loss

class AdaptiveIoULoss(nn.Module):
    """
    自适应IoU损失 - 根据图像内容自动调整权重
    对复杂场景更加鲁棒
    """
    def __init__(self, smooth=1e-6, adaptive_factor=2.0):
        super().__init__()
        self.smooth = smooth
        self.adaptive_factor = adaptive_factor
        
    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        target = target.float()
        
        # 1. 基础IoU计算
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # 2. 自适应权重计算
        # 基于目标密度和预测置信度
        target_density = target_flat.mean()  # 目标密度
        pred_confidence = torch.abs(pred_sigmoid - 0.5).mean()  # 预测置信度
        
        # 自适应因子：目标密度低或置信度低时增加权重
        adaptive_weight = 1.0 + self.adaptive_factor * (1 - target_density) * (1 - pred_confidence)
        
        # 3. Lovász-Softmax启发的IoU损失
        # 对不同的错误类型给予不同权重
        errors = torch.abs(pred_sigmoid - target)
        error_weights = torch.exp(self.adaptive_factor * errors)
        weighted_iou_loss = (1 - iou) * error_weights.mean()
        
        return adaptive_weight * weighted_iou_loss

class StructuralIoULoss(nn.Module):
    """
    结构化IoU损失 - 考虑道路的拓扑结构
    特别适合细长连通目标
    """
    def __init__(self, smooth=1e-6, connectivity_weight=0.4, thinness_weight=0.3):
        super().__init__()
        self.smooth = smooth
        self.connectivity_weight = connectivity_weight
        self.thinness_weight = thinness_weight
        
    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        target = target.float()
        
        # 1. 标准IoU损失
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        base_iou_loss = 1 - iou
        
        # 2. 连通性损失 - 使用形态学操作
        # 腐蚀后的IoU (测试连通性)
        kernel = torch.ones(1, 1, 3, 3).to(pred.device)
        pred_eroded = F.conv2d(pred_sigmoid, kernel, padding=1) / 9.0
        pred_eroded = (pred_eroded > 0.5).float()
        
        target_eroded = F.conv2d(target, kernel, padding=1) / 9.0
        target_eroded = (target_eroded > 0.5).float()
        
        pred_eroded_flat = pred_eroded.view(-1)
        target_eroded_flat = target_eroded.view(-1)
        
        eroded_intersection = (pred_eroded_flat * target_eroded_flat).sum()
        eroded_union = pred_eroded_flat.sum() + target_eroded_flat.sum() - eroded_intersection
        eroded_iou = (eroded_intersection + self.smooth) / (eroded_union + self.smooth)
        connectivity_loss = 1 - eroded_iou
        
        # 3. 细长度损失 - 周长与面积比
        # 计算轮廓长度（使用梯度近似）
        pred_grad_x = torch.abs(pred_sigmoid[:, :, :-1, :] - pred_sigmoid[:, :, 1:, :])
        pred_grad_y = torch.abs(pred_sigmoid[:, :, :, :-1] - pred_sigmoid[:, :, :, 1:])
        pred_perimeter = pred_grad_x.sum() + pred_grad_y.sum()
        
        target_grad_x = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        target_grad_y = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_perimeter = target_grad_x.sum() + target_grad_y.sum()
        
        # 周长比损失
        perimeter_ratio_loss = torch.abs(pred_perimeter - target_perimeter) / (target_perimeter + self.smooth)
        
        # 4. 组合损失
        total_loss = (base_iou_loss + 
                     self.connectivity_weight * connectivity_loss + 
                     self.thinness_weight * perimeter_ratio_loss)
        
        return total_loss

class HybridIoUBCELoss(nn.Module):
    """
    混合IoU-BCE损失 - 结合多种IoU变体和BCE
    最稳定和最有效的组合损失
    """
    def __init__(self, iou_weight=0.6, bce_weight=0.2, advanced_iou_weight=0.2, 
                 smooth=1e-6, focal_gamma=2.0):
        super().__init__()
        self.iou_weight = iou_weight
        self.bce_weight = bce_weight
        self.advanced_iou_weight = advanced_iou_weight
        self.smooth = smooth
        self.focal_gamma = focal_gamma
        
        # 子损失函数
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.advanced_iou = AdvancedIoULoss(smooth=smooth, focal_gamma=focal_gamma)
        
    def forward(self, pred, target):
        # 1. BCE损失 (提供稳定梯度)
        bce_loss = self.bce_loss(pred, target)
        
        # 2. 标准IoU损失
        pred_sigmoid = torch.sigmoid(pred)
        target = target.float()
        
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        iou_loss = 1 - iou
        
        # 3. 高级IoU损失
        advanced_iou_loss = self.advanced_iou(pred, target)
        
        # 4. 组合损失
        total_loss = (self.bce_weight * bce_loss + 
                     self.iou_weight * iou_loss + 
                     self.advanced_iou_weight * advanced_iou_loss)
        
        return total_loss

class UltimateRoadIoULoss(nn.Module):
    """
    终极道路IoU损失 - 集成所有有效技术
    专为遥感道路分割设计的最强损失函数
    """
    def __init__(self, base_iou_weight=0.3, multi_scale_weight=0.25, 
                 adaptive_weight=0.2, structural_weight=0.15, bce_weight=0.1,
                 smooth=1e-6):
        super().__init__()
        self.base_iou_weight = base_iou_weight
        self.multi_scale_weight = multi_scale_weight
        self.adaptive_weight = adaptive_weight
        self.structural_weight = structural_weight
        self.bce_weight = bce_weight
        
        # 组件损失函数
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.multi_scale_iou = MultiScaleIoULoss()
        self.adaptive_iou = AdaptiveIoULoss(smooth=smooth)
        self.structural_iou = StructuralIoULoss(smooth=smooth)
        
    def forward(self, pred, target):
        # 1. 基础IoU损失
        pred_sigmoid = torch.sigmoid(pred)
        target = target.float()
        
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        base_iou_loss = 1 - iou
        
        # 2. 各种专门的IoU损失
        multi_scale_loss = self.multi_scale_iou(pred, target)
        adaptive_loss = self.adaptive_iou(pred, target)
        structural_loss = self.structural_iou(pred, target)
        bce_loss = self.bce_loss(pred, target)
        
        # 3. 加权组合
        total_loss = (self.base_iou_weight * base_iou_loss +
                     self.multi_scale_weight * multi_scale_loss +
                     self.adaptive_weight * adaptive_loss +
                     self.structural_weight * structural_loss +
                     self.bce_weight * bce_loss)
        
        return total_loss


# 改进的分割评估指标
class SegmentationMetrics:
    """Segmentation evaluation metrics class"""
    def __init__(self, num_classes=1, smooth=1e-6):
        self.num_classes = num_classes
        self.smooth = smooth
    
    def dice_coefficient(self, pred, target):
        """Calculate Dice coefficient"""
        if self.num_classes == 1:
            if isinstance(pred, list):
                pred = pred[0]
            
            # Apply sigmoid
            pred_sigmoid = torch.sigmoid(pred)
            
            # Ensure target shape is correct
            if len(target.shape) == 3:
                target = target.unsqueeze(1)
            
            target = target.float()
            
            # Calculate by batch
            intersection = torch.sum(pred_sigmoid * target, dim=(2, 3))
            card_sum = torch.sum(pred_sigmoid, dim=(2, 3)) + torch.sum(target, dim=(2, 3))
            
            # Handle division by zero
            dice = (2.0 * intersection + self.smooth) / (card_sum + self.smooth)
            
            return torch.mean(dice).item()
    
    def road_iou_score(self, pred, target, threshold=0.5):
        """专门计算道路类别的IoU
        
        Args:
            pred: 预测输出，通常是模型输出经过sigmoid的结果
            target: 真实标签
            threshold: 二值化阈值，默认0.5
            
        Returns:
            float: 道路类别的IoU值
        """
        if isinstance(pred, list):
            pred = pred[0]
        
        # 应用sigmoid和二值化
        pred_sigmoid = torch.sigmoid(pred)
        pred_binary = (pred_sigmoid > threshold).float()
        
        # 确保target形状正确
        if len(target.shape) == 3:
            target = target.unsqueeze(1)
        
        target = target.float()
        
        # 只计算道路类别的IoU (前景类)
        # 计算交集和并集
        intersection = torch.sum(pred_binary * target, dim=(1, 2, 3))
        union = torch.sum(pred_binary, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3)) - intersection
        
        # 避免除零错误
        road_iou = intersection / (union + self.smooth)
        
        # 处理union为0的情况
        valid_mask = union > 0
        if valid_mask.sum() > 0:
            return road_iou[valid_mask].mean().item()
        else:
            return 0.0
    

    
    def iou_score(self, pred, target):
        """Calculate IoU score"""
        if self.num_classes == 1:
            if isinstance(pred, list):
                pred = pred[0]
            
            # Fix: binarize first then calculate IoU
            pred_sigmoid = torch.sigmoid(pred)
            pred_binary = (pred_sigmoid > 0.5).float()
            
            # Ensure target shape is correct
            if len(target.shape) == 3:
                target = target.unsqueeze(1)
            
            target = target.float()
            
            # Fix: calculate by batch, avoid numerical underflow
            intersection = torch.sum(pred_binary * target, dim=(2, 3))
            union = torch.sum(pred_binary, dim=(2, 3)) + torch.sum(target, dim=(2, 3)) - intersection
            
            # Fix: handle division by zero
            iou = intersection / (union + self.smooth)
            
            # Fix: only calculate average for valid samples
            valid_mask = union > 0
            if valid_mask.sum() > 0:
                return iou[valid_mask].mean().item()
            else:
                return 0.0

    def miou_score(self, pred, target):
        """Calculate Mean IoU (mIoU) score"""
        if self.num_classes == 1:
            # 二分类情况：计算背景和前景两个类别的平均IoU
            if isinstance(pred, list):
                pred = pred[0]
            
            pred_sigmoid = torch.sigmoid(pred)
            pred_binary = (pred_sigmoid > 0.5).float()
            
            # 确保target形状正确
            if len(target.shape) == 3:
                target = target.unsqueeze(1)
            target = target.float()
            
            ious = []
            
            # 类别0 (背景) 的IoU
            pred_bg = 1 - pred_binary  # 背景预测 
            target_bg = 1 - target     # 背景真值
            
            intersection_bg = torch.sum(pred_bg * target_bg, dim=(2, 3))
            union_bg = torch.sum(pred_bg, dim=(2, 3)) + torch.sum(target_bg, dim=(2, 3)) - intersection_bg
            
            # 类别1 (前景) 的IoU
            intersection_fg = torch.sum(pred_binary * target, dim=(2, 3))
            union_fg = torch.sum(pred_binary, dim=(2, 3)) + torch.sum(target, dim=(2, 3)) - intersection_fg
            
            # 处理除零问题
            iou_bg = intersection_bg / (union_bg + self.smooth)
            iou_fg = intersection_fg / (union_fg + self.smooth)
            
            # 只计算有效样本的平均值
            valid_mask_bg = union_bg > 0
            valid_mask_fg = union_fg > 0
            
            if valid_mask_bg.sum() > 0:
                ious.append(iou_bg[valid_mask_bg].mean().item())
            else:
                ious.append(0.0)
                
            if valid_mask_fg.sum() > 0:
                ious.append(iou_fg[valid_mask_fg].mean().item())
            else:
                ious.append(0.0)
            
            return sum(ious) / len(ious)
            
        else:
            # 多类分割的mIoU计算 (这部分是正确的)
            if isinstance(pred, list):
                pred = pred[0]
            
            pred_softmax = torch.softmax(pred, dim=1)
            pred_argmax = torch.argmax(pred_softmax, dim=1)
            
            # Ensure target shape is correct
            if len(target.shape) == 4 and target.shape[1] == 1:
                target = target.squeeze(1)
            
            target = target.long()
            
            ious = []
            for class_id in range(self.num_classes):
                pred_class = (pred_argmax == class_id).float()
                target_class = (target == class_id).float()
                
                intersection = torch.sum(pred_class * target_class, dim=(1, 2))
                union = torch.sum(pred_class, dim=(1, 2)) + torch.sum(target_class, dim=(1, 2)) - intersection
                
                # Handle division by zero
                iou = intersection / (union + self.smooth)
                valid_mask = union > 0
                if valid_mask.sum() > 0:
                    ious.append(iou[valid_mask].mean().item())
                else:
                    ious.append(0.0)
            
            return np.mean(ious)

    def accuracy(self, pred, target):
        """Calculate Pixel Accuracy"""
        if isinstance(pred, list):
            pred = pred[0]
        
        if self.num_classes == 1:
            # 二分类准确率
            pred_sigmoid = torch.sigmoid(pred)
            pred_binary = (pred_sigmoid > 0.5).float()
            
            # Ensure target shape is correct
            if len(target.shape) == 3:
                target = target.unsqueeze(1)
            
            target = target.float()
            
            # Flatten tensors for calculation
            pred_flat = pred_binary.view(-1)
            target_flat = target.view(-1)
            
            # Calculate accuracy
            correct = (pred_flat == target_flat).sum()
            total = target_flat.numel()
            accuracy = correct / total
            
            return accuracy.item()
        else:
            # 多类准确率
            pred_softmax = torch.softmax(pred, dim=1)
            pred_argmax = torch.argmax(pred_softmax, dim=1)
            
            # Ensure target shape is correct
            if len(target.shape) == 4 and target.shape[1] == 1:
                target = target.squeeze(1)
            
            target = target.long()
            
            correct = (pred_argmax == target).sum()
            total = target.numel()
            accuracy = correct.float() / total
            
            return accuracy.item()

    def precision_recall_f1(self, pred, target):
        """Calculate Precision, Recall, F1"""
        if self.num_classes == 1:
            if isinstance(pred, list):
                pred = pred[0]
            
            # Apply sigmoid and binarize
            pred_sigmoid = torch.sigmoid(pred)
            pred_binary = (pred_sigmoid > 0.5).float()
            
            # Ensure target shape is correct
            if len(target.shape) == 3:
                target = target.unsqueeze(1)
            
            target = target.float()
            
            # Flatten tensors for calculation
            pred_flat = pred_binary.view(-1)
            target_flat = target.view(-1)
            
            # Calculate TP, FP, FN
            tp = (pred_flat * target_flat).sum()
            fp = (pred_flat * (1 - target_flat)).sum()
            fn = ((1 - pred_flat) * target_flat).sum()
            
            # Calculate Precision, Recall, F1
            precision = tp / (tp + fp + self.smooth)
            recall = tp / (tp + fn + self.smooth)
            f1 = 2 * (precision * recall) / (precision + recall + self.smooth)
            
            return precision.item(), recall.item(), f1.item()

    def auc_score(self, pred, target):
        """Calculate AUC score"""
        if self.num_classes == 1:
            if isinstance(pred, list):
                pred = pred[0]
            
            pred = torch.sigmoid(pred)
            
            if len(target.shape) == 3:
                target = target.unsqueeze(1)
            
            pred_flat = pred.reshape(-1).detach().cpu().numpy()
            target_flat = target.reshape(-1).cpu().numpy()
            
            try:
                if len(np.unique(target_flat)) <= 1:
                    return 0.5  # Return random classifier AUC
                return roc_auc_score(target_flat, pred_flat)
            except:
                return 0.5
        
    def precision_recall_f1_acc(self, pred, target):
        """Calculate Precision, Recall, F1, Accuracy"""
        if self.num_classes == 1:
            if isinstance(pred, list):
                pred = pred[0]
            
            # Apply sigmoid and binarize
            pred_sigmoid = torch.sigmoid(pred)
            pred_binary = (pred_sigmoid > 0.5).float()
            
            # Ensure target shape is correct
            if len(target.shape) == 3:
                target = target.unsqueeze(1)
            
            target = target.float()
            
            # Flatten tensors for calculation
            pred_flat = pred_binary.view(-1)
            target_flat = target.view(-1)
            
            # Calculate TP, FP, FN, TN
            tp = (pred_flat * target_flat).sum()
            fp = (pred_flat * (1 - target_flat)).sum()
            fn = ((1 - pred_flat) * target_flat).sum()
            tn = ((1 - pred_flat) * (1 - target_flat)).sum()
            
            # Calculate metrics
            precision = tp / (tp + fp + self.smooth)
            recall = tp / (tp + fn + self.smooth)
            f1 = 2 * (precision * recall) / (precision + recall + self.smooth)
            accuracy = (tp + tn) / (tp + fp + fn + tn + self.smooth)
            
            return precision.item(), recall.item(), f1.item(), accuracy.item()

    def calculate_all_metrics(self, pred, target):
        """Calculate all metrics at once"""
        dice = self.dice_coefficient(pred, target)
        iou = self.iou_score(pred, target)
        miou = self.miou_score(pred, target)
        accuracy = self.accuracy(pred, target)
        precision, recall, f1 = self.precision_recall_f1(pred, target)
        auc = self.auc_score(pred, target)
        
        return {
            'dice': dice,
            'iou': iou,
            'miou': miou,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }

    def calculate_sample_metrics(self, pred, target):
        """Calculate metrics for a single sample"""
        if isinstance(pred, list):
            pred = pred[0]
        
        # Apply sigmoid and binarize
        pred_sigmoid = torch.sigmoid(pred)
        pred_binary = (pred_sigmoid > 0.5).float()
        
        # Ensure target shape is correct
        if len(target.shape) == 3:
            target = target.unsqueeze(1)
        
        target = target.float()
        
        # Calculate metrics
        dice = self.dice_coefficient(pred.unsqueeze(0), target.unsqueeze(0))
        iou = self.iou_score(pred.unsqueeze(0), target.unsqueeze(0))
        miou = self.miou_score(pred.unsqueeze(0), target.unsqueeze(0))
        accuracy = self.accuracy(pred.unsqueeze(0), target.unsqueeze(0))
        precision, recall, f1 = self.precision_recall_f1(pred.unsqueeze(0), target.unsqueeze(0))
        auc = self.auc_score(pred.unsqueeze(0), target.unsqueeze(0))
        
        return {
            'dice': dice,
            'iou': iou,
            'miou': miou,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }


def count_parameters(model):
    """统计模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def calculate_model_complexity(model, input_size=(3, 512, 512), device='cuda'):
    """
    计算模型复杂度，使用简单方法替代thop
    
    Args:
        model: PyTorch模型
        input_size: 输入尺寸，格式为(C, H, W)
        device: 设备类型
        
    Returns:
        tuple: (总参数量, 可训练参数量, FLOPs信息字典)
    """
    batch_size = 1
    total_params, trainable_params = count_parameters(model)
    
    # 计算输入输出尺寸
    input_shape = (batch_size,) + input_size
    dummy_input = torch.randn(input_shape).to(device)
    
    # 预估MACs (FLOPs的一半)
    # 因为thop不兼容，这里使用简单估算，基于参数量和输入尺寸
    # 实际计算MACs需要更精确的分析每个层的操作
    if hasattr(model, 'encoder') and hasattr(model, 'decoder'):
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        decoder_params = sum(p.numel() for p in model.decoder.parameters())
        
        # 粗略估算 - 假设每个参数平均执行2-4次乘加运算
        encoder_flops_estimate = encoder_params * 3 * input_size[1] * input_size[2] / (16 * 16)
        decoder_flops_estimate = decoder_params * 3 * input_size[1] * input_size[2] / (8 * 8)
        estimated_gflops = (encoder_flops_estimate + decoder_flops_estimate) / 1e9
    else:
        # 粗略估算 - 假设每个参数平均执行3次乘加运算
        estimated_gflops = total_params * 3 * input_size[1] * input_size[2] / (32 * 32) / 1e9
    
    # 计算模型大小(MB)
    model_size_mb = total_params * 4 / 1024 / 1024  # 假设每个参数为4字节(float32)
    
    # 输出模型信息
    with torch.no_grad():
        try:
            _ = model(dummy_input)
            inference_success = True
        except Exception as e:
            print(f"推理测试失败: {e}")
            inference_success = False
    
    # 返回复杂度信息
    flops_info = {
        'estimated_gflops': f"{estimated_gflops:.2f} GFLOPs (估算值)",
        'model_size_mb': f"{model_size_mb:.2f} MB",
        'total_params': f"{total_params:,}",
        'trainable_params': f"{trainable_params:,}",
        'input_shape': f"{input_shape}",
        'inference_test': "成功" if inference_success else "失败"
    }
    
    # 控制台输出信息
    print("\n模型复杂度信息:")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"模型大小: {model_size_mb:.2f} MB")
    print(f"估算GFLOPs: {estimated_gflops:.2f}")
    print(f"输入尺寸: {input_shape}")
    
    return total_params, trainable_params, flops_info


# 可视化函数
def visualize_results(image, true_mask, pred_mask, filename, save_dir):
    """Visualize segmentation results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy arrays
    image = image.cpu().numpy().transpose(1, 2, 0)
    # Denormalize image (assuming ImageNet normalization)
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)
    
    true_mask = true_mask.cpu().numpy()
    if len(true_mask.shape) == 3 and true_mask.shape[0] == 1:
        true_mask = true_mask.squeeze(0)
    
    # Process prediction mask
    if isinstance(pred_mask, list):
        pred_mask = pred_mask[0]
    pred_prob = torch.sigmoid(pred_mask).cpu().numpy().squeeze()
    pred_binary = (pred_prob > 0.5).astype(np.float32)
    
    # Create canvas
    plt.figure(figsize=(20, 5))
    
    # Show original image
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Show true mask
    plt.subplot(1, 4, 2)
    plt.imshow(true_mask, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    
    # Show prediction probability
    plt.subplot(1, 4, 3)
    plt.imshow(pred_prob, cmap='gray')
    plt.title('Prediction Probability')
    plt.axis('off')
    
    # Show prediction mask
    plt.subplot(1, 4, 4)
    plt.imshow(pred_binary, cmap='gray')
    plt.title('Prediction Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename}"), dpi=150, bbox_inches='tight')
    plt.close()


# 可视化训练历史
def visualize_training_history(history, save_path):
    """Visualize training history"""
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    
    # Loss curve
    axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red')
    axes[0, 0].set_title('Loss Curve')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Dice coefficient curve
    axes[0, 1].plot(history['train_dice'], label='Train Dice', color='blue')
    axes[0, 1].plot(history['val_dice'], label='Val Dice', color='red')
    axes[0, 1].set_title('Dice Coefficient Curve')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Coefficient')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # IoU curve
    axes[1, 0].plot(history['train_iou'], label='Train IoU', color='blue')
    axes[1, 0].plot(history['val_iou'], label='Val IoU', color='red')
    axes[1, 0].set_title('IoU Curve')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # mIoU curve
    axes[1, 1].plot(history['train_miou'], label='Train mIoU', color='blue')
    axes[1, 1].plot(history['val_miou'], label='Val mIoU', color='red')
    axes[1, 1].set_title('mIoU Curve')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('mIoU')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Accuracy curve
    axes[2, 0].plot(history['train_accuracy'], label='Train Accuracy', color='blue')
    axes[2, 0].plot(history['val_accuracy'], label='Val Accuracy', color='red')
    axes[2, 0].set_title('Accuracy Curve')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Accuracy')
    axes[2, 0].legend()
    axes[2, 0].grid(True)
    
    # F1 curve
    axes[2, 1].plot(history['train_f1'], label='Train F1', color='blue')
    axes[2, 1].plot(history['val_f1'], label='Val F1', color='red')
    axes[2, 1].set_title('F1 Score Curve')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('F1 Score')
    axes[2, 1].legend()
    axes[2, 1].grid(True)
    
    # Precision curve
    axes[3, 0].plot(history['train_precision'], label='Train Precision', color='blue')
    axes[3, 0].plot(history['val_precision'], label='Val Precision', color='red')
    axes[3, 0].set_title('Precision Curve')
    axes[3, 0].set_xlabel('Epoch')
    axes[3, 0].set_ylabel('Precision')
    axes[3, 0].legend()
    axes[3, 0].grid(True)
    
    # Recall curve
    axes[3, 1].plot(history['train_recall'], label='Train Recall', color='blue')
    axes[3, 1].plot(history['val_recall'], label='Val Recall', color='red')
    axes[3, 1].set_title('Recall Curve')
    axes[3, 1].set_xlabel('Epoch')
    axes[3, 1].set_ylabel('Recall')
    axes[3, 1].legend()
    axes[3, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to: {save_path}")


# 保存训练历史到Excel
def save_training_history_to_excel(history, save_path):
    """Save training history to Excel file"""
    try:
        # 创建DataFrame
        df = pd.DataFrame(history)
        
        # 添加epoch列
        df.insert(0, 'epoch', range(1, len(df) + 1))
        
        # 保存到Excel
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Training_History', index=False)
            
            # 添加一个汇总表
            summary_data = {
                'Metric': ['Best Train Loss', 'Best Val Loss', 'Best Train Dice', 'Best Val Dice',
                          'Best Train IoU', 'Best Val IoU', 'Best Train mIoU', 'Best Val mIoU',
                          'Best Train Accuracy', 'Best Val Accuracy', 'Best Train Precision', 'Best Val Precision',
                          'Best Train Recall', 'Best Val Recall', 'Best Train F1', 'Best Val F1',
                          'Best Train AUC', 'Best Val AUC', 'Final Learning Rate'],
                'Value': [min(history['train_loss']), min(history['val_loss']),
                         max(history['train_dice']), max(history['val_dice']),
                         max(history['train_iou']), max(history['val_iou']),
                         max(history['train_miou']), max(history['val_miou']),
                         max(history['train_accuracy']), max(history['val_accuracy']),
                         max(history['train_precision']), max(history['val_precision']),
                         max(history['train_recall']), max(history['val_recall']),
                         max(history['train_f1']), max(history['val_f1']),
                         max(history['train_auc']), max(history['val_auc']),
                         history['learning_rate'][-1]]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"Training history saved to Excel: {save_path}")
        return True
    except Exception as e:
        print(f"Failed to save training history to Excel: {e}")
        return False


# 保存测试结果到Excel
def save_test_results_to_excel(test_results, model_info, save_path):
    """Save test results to Excel file"""
    try:
        # 测试结果数据
        test_data = {
            'Metric': ['Dice Coefficient', 'IoU Score', 'mIoU Score', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC Score'],
            'Value': [test_results['dice'], test_results['iou'], test_results['miou'], test_results['accuracy'],
                     test_results['precision'], test_results['recall'], test_results['f1'], test_results['auc']]
        }
        test_df = pd.DataFrame(test_data)
        
        # 模型信息数据
        model_data = []
        for key, value in model_info.items():
            model_data.append({'Parameter': key, 'Value': value})
        model_df = pd.DataFrame(model_data)
        
        # 保存到Excel
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            test_df.to_excel(writer, sheet_name='Test_Results', index=False)
            model_df.to_excel(writer, sheet_name='Model_Info', index=False)
        
        print(f"Test results saved to Excel: {save_path}")
        return True
    except Exception as e:
        print(f"Failed to save test results to Excel: {e}")
        return False


# 保存详细的每个样本测试结果到Excel
def save_detailed_test_results_to_excel(detailed_results, save_path):
    """Save detailed test results for each sample to Excel file"""
    try:
        df = pd.DataFrame(detailed_results)
        df.to_excel(save_path, index=False)
        print(f"Detailed test results saved to Excel: {save_path}")
        return True
    except Exception as e:
        print(f"Failed to save detailed test results to Excel: {e}")
        return False


# 测试函数
def test_model(model, test_loader, device, save_dir, num_classes=1):
    """Test segmentation model"""
    model.eval()
    metrics = SegmentationMetrics(num_classes=num_classes)
    
    test_dice = 0.0
    test_iou = 0.0
    test_miou = 0.0
    test_accuracy = 0.0
    test_precision = 0.0
    test_recall = 0.0
    test_f1 = 0.0
    test_auc = 0.0
    
    # 用于保存每个样本的详细结果
    detailed_results = []
    
    test_progress = tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_progress):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            filenames = batch['filename']
            
            # Process masks
            masks = masks.float()
            if masks.max() > 1.0:
                masks = masks / 255.0
                
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(1)
            
            outputs = model(images)
            
            if isinstance(outputs, list):
                outputs = outputs[0]
            
            # Calculate batch metrics using the new method
            batch_metrics = metrics.calculate_all_metrics(outputs, masks.squeeze(1) if masks.shape[1] == 1 else masks)
            
            test_dice += batch_metrics['dice']
            test_iou += batch_metrics['iou']
            test_miou += batch_metrics['miou']
            test_accuracy += batch_metrics['accuracy']
            test_precision += batch_metrics['precision']
            test_recall += batch_metrics['recall']
            test_f1 += batch_metrics['f1']
            test_auc += batch_metrics['auc']
            
            # Calculate individual sample metrics
            for i in range(len(images)):
                sample_metrics = metrics.calculate_sample_metrics(
                    outputs[i], 
                    masks[i].squeeze(0) if masks.shape[1] == 1 else masks[i]
                )
                
                detailed_results.append({
                    'filename': filenames[i],
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'dice': sample_metrics['dice'],
                    'iou': sample_metrics['iou'],
                    'miou': sample_metrics['miou'],
                    'accuracy': sample_metrics['accuracy'],
                    'precision': sample_metrics['precision'],
                    'recall': sample_metrics['recall'],
                    'f1': sample_metrics['f1'],
                    'auc': sample_metrics['auc']
                })
            
            test_progress.set_postfix({
                'dice': batch_metrics['dice'],
                'iou': batch_metrics['iou'],
                'miou': batch_metrics['miou'],
                'acc': batch_metrics['accuracy'],
                'f1': batch_metrics['f1']
            })
            
            # # Save some test result visualizations
            # if batch_idx < 10:  # Only save first 10 batches
            for i in range(min(3, len(images))):
                visualize_results(
                    images[i], 
                    masks[i], 
                    outputs[i], 
                    f"test_{filenames[i]}",
                    os.path.join(save_dir, 'test_visualizations')
                )
    
    # Calculate averages
    test_dice /= len(test_loader)
    test_iou /= len(test_loader)
    test_miou /= len(test_loader)
    test_accuracy /= len(test_loader)
    test_precision /= len(test_loader)
    test_recall /= len(test_loader)
    test_f1 /= len(test_loader)
    test_auc /= len(test_loader)
    
    print(f"\nTest Results:")
    print(f"  Dice: {test_dice:.4f}")
    print(f"  IoU: {test_iou:.4f}")
    print(f"  mIoU: {test_miou:.4f}")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1: {test_f1:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    
    # Save test results
    with open(os.path.join(save_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Test Results:\n")
        f.write(f"Dice Coefficient: {test_dice:.4f}\n")
        f.write(f"IoU Score: {test_iou:.4f}\n")
        f.write(f"mIoU Score: {test_miou:.4f}\n")
        f.write(f"Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Precision: {test_precision:.4f}\n")
        f.write(f"Recall: {test_recall:.4f}\n")
        f.write(f"F1 Score: {test_f1:.4f}\n")
        f.write(f"AUC Score: {test_auc:.4f}\n")
    
    # 保存详细的每个样本测试结果到Excel
    save_detailed_test_results_to_excel(
        detailed_results, 
        os.path.join(save_dir, 'detailed_test_results.xlsx')
    )
    
    return test_dice, test_iou, test_miou, test_accuracy, test_precision, test_recall, test_f1, test_auc