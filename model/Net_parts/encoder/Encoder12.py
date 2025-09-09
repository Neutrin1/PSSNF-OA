#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Encoder12.py
@Time    :   2025/07/18 13:40:35
@Author  :   angkangyu 
'''

# here put the import lib


from matplotlib.pylab import f
from torch import nn
from PSSNF.model.Net_parts.public.TUAN import UNetFirstEB,ParallelConvRDLEBEncoderBlock
from PSSNF.model.Net_parts.MyAtt.SA1 import SA1
 
class Encoder12(nn.Module):
    def __init__(self, in_channels, base_channels=32, b1= UNetFirstEB, b2=ParallelConvRDLEBEncoderBlock, use_bn=True):
        super().__init__()
        
        # 四层编码器
        self.enc1 = b1(in_channels, base_channels, use_bn=use_bn)
        self.enc2 = b2(base_channels, base_channels * 2, use_bn=use_bn)
        self.enc3 = b2(base_channels * 2, base_channels * 4,  use_bn=use_bn)
        self.enc4 = b2(base_channels * 4, base_channels * 8,  use_bn=use_bn)      
        self.bottom = b2(base_channels * 8, base_channels * 16, use_bn=use_bn)
        self.sa1 = SA1(base_channels, reduction=8)
        self.sa2 = SA1(base_channels * 2, reduction=8)
        self.sa3 = SA1(base_channels * 4, reduction=8)
        
    # 前向传播
    def forward(self, x):
        x1 = self.enc1(x)  # 第一层输出，跳跃1
        f1 = self.sa1(x1)
        x2 = self.enc2(x1) # 跳跃2
        f2 = self.sa2(x2)
        x3 = self.enc3(x2) # 跳跃3
        f3 = self.sa3(x3)
        x4 = self.enc4(x3) # 跳跃4
        x5 = self.bottom(x4) # 底部连接
        # print(f"Encoder output shapes: x1={x1.shape}, x2={x2.shape}, x3={x3.shape}, x4={x4.shape}, x5={x5.shape}")
        return x5, [x4, f3, f2, f1 ] # [bottom, skip4, skip3, skip2, skip1]