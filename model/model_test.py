#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model_test.py
@Time    :   2025/06/18 10:15:34
@Author  :   angkangyu 
'''

"""
test_models.py - 模型测试脚本

功能：
测试各种模型的参数量、GFLOPS、前向传播等
"""

import os
from sympy import false
import torch
import datetime
import sys
from typing import List
from model_interface import MInterface


def get_available_models():
    """获取可用的模型配置"""
    return {
        "UNet": {
            "in_channels": 3,
            "num_classes": 1
        },
        "EMCADNet": {
            "in_channels": 3,
            "num_classes": 1,
            "encoder": "resnet50",
            "pretrain": True
        },
        "R2UNet": {
            "in_channels": 3,
            "num_classes": 1,
            "t": 2
        },
        "AttUNet": {
            "in_channels": 3,
            "num_classes": 1,
        },
        "R2AttUNet":{
            "in_channels": 3,
            "num_classes": 1,
            "t": 2
        },
        "NestedUNet":{
            "img_ch": 3,
            "output_ch": 1,
        },
        "DictUNet":{
            "n_labels": 2,        # 2类分割
            "n_filters": 64,      # 更宽的网络
            "p_dropout": 0.3,     # 较小的dropout
            "batchnorm": True 
        },
        "SegNet": {
            "in_chn": 3,
            "out_chn": 1,
            "BN_momentum": 0.5
        },
        "EffUNet": {
            "in_channels": 3,
            "num_classes": 1,
        },
        "UNetv2": {
            "in_channels": 3,
            "num_classes": 1,
            "deep_supervision": False,
            "pretrained_path": None,
            "auto_download": True,
            "backbone": 'pvt_v2_b2'
        },
        "CBAMUNet": {
            "img_ch": 3,
            "output_ch": 1,
        },
        "SEUNet": {
            "img_ch": 3,
            "output_ch": 1,
            "enhanced": False,  # 是否使用增强的SE模块
            "reduction": 16      # SE模块的压缩率   
        },
        "DEGANet": {
            "in_channels": 3,
            "num_classes": 1,
        },
        "KNet": {
            "in_channels": 3,
            "classes": 1,
        },
        "KNetV2": {
            "in_channels": 3,
            "classes": 1,
        },
        "TGRSv1": {
            "num_classes": 2,  # 背景 + 道路
            "use_enhanced_wavelet": True,  # 是否使用增强的小波变换
            "wavelet_heads": 8  # 小波变换头数
        },
    }


def test_single_model(model_name: str, config: dict, log_file) -> bool:
    """测试单个模型"""
    log_file.write(f"=== 测试 {model_name} 模型 ===\n")
    print(f"正在测试 {model_name} 模型...")
    
    try:
        # 获取模型配置
        config = config.copy()
        in_channels = config.pop("in_channels", 3)
        num_classes = config.pop("num_classes", 1)
        
        # 创建模型接口
        model_interface = MInterface(
            model_type=model_name,
            in_channels=in_channels,
            num_classes=num_classes,
            **config
        )
        
        # 记录模型信息
        log_file.write(f"模型类型: {model_name}\n")
        log_file.write(f"输入通道数: {in_channels}\n")
        log_file.write(f"输出类别数: {num_classes}\n")
        log_file.write(f"总参数量: {model_interface.param_count:,}\n")
        
        # 计算可训练参数量
        trainable_params = sum(p.numel() for p in model_interface.model.parameters() if p.requires_grad)
        log_file.write(f"可训练参数量: {trainable_params:,}\n")
        
        print(f"创建{model_name}模型成功，参数数量: {model_interface.param_count:,}")
        
        # 测试前向传播
        if not test_forward_pass(model_interface, in_channels, log_file, model_name):
            return False
        
        # 计算FPS
        test_fps(model_interface, in_channels, log_file, model_name)
        
        # 计算GFLOPS
        test_gflops(model_interface, in_channels, log_file, model_name)
        
        log_file.write(f"测试状态: 成功\n")
        print(f"✓ {model_name} 模型测试完成")
        return True
        
    except Exception as e:
        log_file.write(f"模型创建失败: {str(e)}\n")
        log_file.write(f"测试状态: 失败\n")
        print(f"✗ {model_name} 模型测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass(model_interface: MInterface, in_channels: int, log_file, model_name: str) -> bool:
    """测试前向传播"""
    try:
        test_input = torch.randn(1, in_channels, 256, 256).to(model_interface.device)
        with torch.no_grad():
            output = model_interface.forward(test_input)
        log_file.write(f"前向传播测试: 成功\n")
        log_file.write(f"输入形状: {test_input.shape}\n")
        log_file.write(f"输出形状: {output.shape}\n")
        print(f"✓ {model_name} 前向传播测试成功")
        return True
    except Exception as e:
        log_file.write(f"前向传播测试: 失败 - {str(e)}\n")
        print(f"✗ {model_name} 前向传播测试失败: {str(e)}")
        return False


def test_fps(model_interface: MInterface, in_channels: int, log_file, model_name: str):
    """测试FPS计算"""
    try:
        print(f"正在计算 {model_name} 的FPS...")
        
        # 测试不同输入尺寸的FPS
        test_sizes = [
            (1, in_channels, 256, 256),
            (1, in_channels, 512, 512),
            (1, in_channels, 1024, 1024)
        ]
        
        log_file.write("FPS测试结果:\n")
        
        for size in test_sizes:
            try:
                fps, avg_time = model_interface.calculate_fps(input_size=size, num_runs=50, warmup_runs=5)
                if isinstance(fps, str):  # 错误信息
                    log_file.write(f"  输入尺寸 {size[2]}x{size[3]}: {fps}\n")
                    print(f"  输入尺寸 {size[2]}x{size[3]}: {fps}")
                else:
                    log_file.write(f"  输入尺寸 {size[2]}x{size[3]}: {fps:.2f} FPS (平均推理时间: {avg_time:.2f}ms)\n")
                    print(f"  输入尺寸 {size[2]}x{size[3]}: {fps:.2f} FPS (平均推理时间: {avg_time:.2f}ms)")
            except Exception as e:
                error_msg = f"尺寸 {size[2]}x{size[3]} 测试失败: {str(e)}"
                log_file.write(f"  {error_msg}\n")
                print(f"  {error_msg}")
                
    except Exception as e:
        log_file.write(f"FPS计算失败: {str(e)}\n")
        print(f"FPS计算失败: {str(e)}")


def test_gflops(model_interface: MInterface, in_channels: int, log_file, model_name: str):
    """测试GFLOPS计算"""
    try:
        print(f"正在计算 {model_name} 的GFLOPS...")
        gflops, params = model_interface.calculate_gflops(input_size=(1, in_channels, 512, 512))
        log_file.write(f"GFLOPS: {gflops}\n")
        log_file.write(f"参数量: {params}\n")
        print(f"  GFLOPS: {gflops}")
        print(f"  参数量: {params}")
    except Exception as e:
        log_file.write(f"GFLOPS计算失败: {str(e)}\n")
        print(f"GFLOPS计算失败: {str(e)}")


def parse_command_line_args(available_models: dict) -> List[str]:
    """解析命令行参数"""
    if len(sys.argv) > 1:
        test_models = [arg for arg in sys.argv[1:] if arg in available_models]
        if not test_models:
            print(f"错误: 指定的模型不存在。可用模型: {list(available_models.keys())}")
            sys.exit(1)
    else:
        test_models = list(available_models.keys())
    
    return test_models


def create_log_file() -> str:
    """创建日志文件名和文件夹"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建测试结果文件夹
    test_dir = "model_test_results"
    os.makedirs(test_dir, exist_ok=True)
    
    # 在文件夹内创建日志文件
    log_file = os.path.join(test_dir, f"model_test_{timestamp}.txt")
    return log_file


def print_usage_info():
    """打印使用说明"""
    print("\n使用方法:")
    print("  python model_test.py                    # 测试所有模型")
    print("  python model_test.py UNet               # 只测试UNet")
    print("  python model_test.py UNet EMCADNet      # 测试UNet和EMCADNet")
    
    try:
        from thop import profile, clever_format
    except ImportError:
        print("\n安装thop库以计算GFLOPS:")
        print("  pip install thop")
    
    print("\n测试包括:")
    print("  - 参数量统计")
    print("  - 前向传播测试")
    print("  - FPS性能测试 (多种输入尺寸)")
    print("  - GFLOPS计算")


def run_model_tests():
    """运行模型测试的主函数"""
    # 获取可用模型和解析命令行参数
    available_models = get_available_models()
    test_models = parse_command_line_args(available_models)
    log_file_name = create_log_file()
    
    print(f"正在测试模型: {test_models}")
    print(f"设备: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # 运行测试
    with open(log_file_name, "w", encoding="utf-8") as f:
        # 写入文件头
        f.write(f"模型测试报告 - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"设备: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}\n")
        f.write("=" * 60 + "\n\n")
        
        # 测试每个模型
        for model_name in test_models:
            test_single_model(model_name, available_models[model_name], f)
            f.write("\n" + "-" * 40 + "\n\n")
        
        f.write("测试完成!\n")
    
    print(f"\n模型测试结果已保存到: {log_file_name}")
    print_usage_info()


if __name__ == "__main__":
    run_model_tests()