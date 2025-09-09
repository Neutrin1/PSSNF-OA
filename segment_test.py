#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_only.py
@Time    :   2025/06/18 15:53:43
@Author  :   neutrin
'''

import argparse
import torch
import os
from datetime import datetime
from model.model_interface import MInterface
from data.data_interface import SegmentDataInterface
from utils.metrics_utils import test_model, SegmentationMetrics

def filter_state_dict(state_dict):
    """过滤掉用于复杂度计算的额外参数"""
    filtered_dict = {}
    for key, value in state_dict.items():
        # 跳过包含 total_ops 和 total_params 的键
        if 'total_ops' not in key and 'total_params' not in key:
            filtered_dict[key] = value
    return filtered_dict

def create_test_save_dir(base_dir, model_type):
    """创建以模型和时间命名的保存目录"""
    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建目录名：模型_时间
    dir_name = f"{model_type}_test_{timestamp}"
    save_dir = os.path.join(base_dir, dir_name)
    
    # 创建目录
    os.makedirs(save_dir, exist_ok=True)
    
    return save_dir

def parse_args():
    parser = argparse.ArgumentParser(description='Test Segmentation Model')
    
    # 基本参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--data_dir', type=str, 
                        default="./data/Massachusetts/split",
                        help='Dataset directory')
    parser.add_argument('--model_type', type=str, default="DemoNet",
                        help='Model type')
    parser.add_argument('--batch_size', type=int, default=4,            
                        help='Batch size')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Image size (None for original size)')
    parser.add_argument('--base_save_dir', type=str, default="./test_results",
                        help='Base directory for saving results')
    
    return parser.parse_args()

def save_test_summary(save_dir, model_type, model_path, test_results):
    """保存测试结果摘要到文件"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    summary_file = os.path.join(save_dir, "test_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("SEGMENTATION MODEL TEST RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Test Time: {timestamp}\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Model Path: {model_path}\n")
        f.write("-"*60 + "\n")
        f.write("METRICS:\n")
        f.write(f"Dice Score:  {test_results[0]:.4f}\n")
        f.write(f"IoU:         {test_results[1]:.4f}\n")
        f.write(f"mIoU:        {test_results[2]:.4f}\n")
        f.write(f"Accuracy:    {test_results[3]:.4f}\n")
        f.write(f"Precision:   {test_results[4]:.4f}\n")
        f.write(f"Recall:      {test_results[5]:.4f}\n")
        f.write(f"F1 Score:    {test_results[6]:.4f}\n")
        f.write(f"AUC:         {test_results[7]:.4f}\n")
        f.write("="*60 + "\n")
    
    print(f"Test summary saved to: {summary_file}")

def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建以模型和时间命名的保存目录
    save_dir = create_test_save_dir(args.base_save_dir, args.model_type)
    print(f"Results will be saved to: {save_dir}")
    
    # 加载数据
    data_interface = SegmentDataInterface(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        img_size=args.img_size
    )
    
    test_loader = data_interface.test_dataloader()
    print(f"Test set size: {len(test_loader.dataset)} samples")
    
    # 创建模型
    model_interface = MInterface(
        model_type=args.model_type,
        in_channels=3,
        num_classes=1,
        device=device
    )
    
    model = model_interface.model
    
    # 加载训练好的权重（过滤掉额外参数）
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # 如果checkpoint是字典且包含'state_dict'键
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 过滤掉复杂度计算相关的参数
        filtered_state_dict = filter_state_dict(state_dict)
        
        # 加载过滤后的权重
        model.load_state_dict(filtered_state_dict, strict=False)
        print(f"Loaded model weights from: {args.model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    # 开始测试
    print("Starting model testing...")
    test_results = test_model(
        model=model,
        test_loader=test_loader,
        device=device,
        save_dir=save_dir,
        num_classes=1
    )
    
    # 打印结果
    print("\nTest Results:")
    print(f"Dice: {test_results[0]:.4f}")
    print(f"IoU: {test_results[1]:.4f}")
    print(f"mIoU: {test_results[2]:.4f}")
    print(f"Accuracy: {test_results[3]:.4f}")
    print(f"Precision: {test_results[4]:.4f}")
    print(f"Recall: {test_results[5]:.4f}")
    print(f"F1: {test_results[6]:.4f}")
    print(f"AUC: {test_results[7]:.4f}")
    
    # 保存测试结果摘要
    save_test_summary(save_dir, args.model_type, args.model_path, test_results)

if __name__ == "__main__":
    main()