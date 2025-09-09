#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Encoder3.py
@Time    :   2025/07/13 16:56:01
@Author  :   angkangyu 
'''

# here put the import lib

from torch import nn
from PSSNF.model.Net_parts.public.TUAN import UNetFirstEB,ParallelConvRDLEBEncoderBlock, ViTEncoderBlock

"""
结合Encoder1andEncoder3的特性。
Encoder8: 四层编码器
- 输入通道数: in_channels
- 基础通道数: base_channels (默认32)
- 编码器块:
    - b1: UNetFirstEB (默认)
    - b2: ParallelConvRDLEBEncoderBlock (默认)
    - b3: ViTEncoderBlock (默认)
- 是否使用批归一化: use_bn (默认True)
- 前向传播: 返回底部连接和四个跳跃连接的输出 
"""

class Encoder8(nn.Module):
    def __init__(self, in_channels, base_channels=32, b1= UNetFirstEB, b2=ParallelConvRDLEBEncoderBlock, b3=ViTEncoderBlock, use_bn=True):
        super().__init__()
        
        # 四层编码器
        self.enc1 = b1(in_channels, base_channels, use_bn=use_bn)
        self.enc2 = b2(base_channels, base_channels * 2, use_bn=use_bn)
        self.enc3 = b2(base_channels * 2, base_channels * 4,  use_bn=use_bn)
        self.enc4 = b2(base_channels * 4, base_channels * 8,  use_bn=use_bn)      
        self.bottom = b3(base_channels * 8, base_channels * 16, use_bn=use_bn)
        
    # 前向传播
    def forward(self, x):
        x1 = self.enc1(x)  # 第一层输出，跳跃1
        x2 = self.enc2(x1) # 跳跃2
        x3 = self.enc3(x2) # 跳跃3
        x4 = self.enc4(x3) # 跳跃4
        x5 = self.bottom(x4) # 底部连接
        return x5, [x4, x3, x2, x1 ] # [bottom, skip4, skip3, skip2, skip1]