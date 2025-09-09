#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   knet_model.py
@Time    :   2025/07/01 
@Author  :   angkangyu 
@Description: 完整的KNet模型
'''

from matplotlib.pylab import f
import torch
import torch.nn as nn

from PSSNF.model.Net_parts.decoder import Decoder5
from PSSNF.model.Net_parts.encoder import Encoder3

# 修改导入方式 - 按照目录结构分开导入编码器和解码器
try:

    
    from .Net_parts.encoder.UEncoder import UNetEncoder
    
    from .Net_parts.encoder.Encoder23 import Encoder23    
   
    #My1
    from .Net_parts.decoder.Decoder1 import Decoder1  # My1解码器

    
except ImportError:
    # 当直接运行时使用绝对导入
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # 编码器导入


    from Net_parts.encoder.UEncoder import UNetEncoder
    #M
    from Net_parts.encoder.Encoder23 import Encoder23 
    # from Net_parts.encoder.Encoder24 import Encoder24  # 新的编码器
     
    


    # 解码器导入
   
    from Net_parts.decoder.Decoder1 import Decoder1


    
    

    
    
class DemoNet(nn.Module):
    """完整的Net模型 - 编码器解码器架构"""
    
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()

        self.encoder = UNetEncoder(in_channels=in_channels)  
        self.decoder = Decoder1(num_classes=num_classes)  
    
    def forward(self, x):
        bottom, skip_connections = self.encoder(x)
        output = self.decoder(bottom, skip_connections)         # ✅ 传递1个参数
        
        return output
    

def create_knet(in_channels=3, num_classes=1):
    """
    创建Net模型
    
    Args:
        in_channels (int): 输入通道数
        num_classes (int): 输出类别数
    
    Returns:
        KNet模型实例
    """
    return DemoNet(in_channels=in_channels, num_classes=num_classes)


# 模型工具函数
def count_parameters(model):
    """统计模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model, input_size=(3, 512, 512)):
    """打印模型摘要信息"""
    device = next(model.parameters()).device
    x = torch.randn(1, *input_size).to(device)
    
    with torch.no_grad():
        output = model(x)
    
    total_params = count_parameters(model)
    
    print("=" * 70)
    print("KNet Encoder-Decoder Model Summary")
    print("=" * 70)
    print(f"Encoder: {type(model.encoder).__name__}")
    print(f"Decoder: {type(model.decoder).__name__}")
    print("-" * 70)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    print("=" * 70)


if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = create_knet(in_channels=3, num_classes=1).to(device)
    
    # 打印模型组件信息
    print("🔍 Model Components:")
    print(f"   Encoder: {type(model.encoder).__name__}")
    print(f"   Decoder: {type(model.decoder).__name__}")
    print()
    
    # 模型摘要
    model_summary(model)
    
    # 测试前向传播
    test_input = torch.randn(2, 3, 512, 512).to(device)
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"✅ Forward pass successful!")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")