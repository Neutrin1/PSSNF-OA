#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Encoder1.1.py
@Time    :   2025/07/12 21:16:39
@Author  :   angkangyu 
'''

# here put the import lib
# here put the import lib
from traceback import print_tb
import torch
import torch.nn as nn
from PSSNF.model.Net_parts.public.TUAN import UNetRDLEB ,UNetFirstEB, ViTEncoderBlock, SEAttention # 或你想用的编码器块


class Encoder1_1(nn.Module):
    def __init__(self, in_channels, base_channels=32, b1 = UNetFirstEB, b2=UNetRDLEB, b3=ViTEncoderBlock,kernel_size=7, padding=3, use_bn=True):
        super().__init__()
        # 四层编码器
        self.enc1 = b1(in_channels, base_channels, kernel_size=kernel_size, padding=padding, use_bn=use_bn)
        self.enc2 = b2(base_channels, base_channels * 2, kernel_size=kernel_size, padding=padding, use_bn=use_bn)
        self.enc3 = b2(base_channels * 2, base_channels * 4, kernel_size=kernel_size, padding=padding, use_bn=use_bn)
        self.enc4 = b3(base_channels * 4, base_channels * 8,patch_size=4)      # 修改这里
        self.bottom = b3(base_channels * 8, base_channels * 16,patch_size=4)
        self.se2 = SEAttention(base_channels * 2)  # 添加SE注意力机制
        self.se3 = SEAttention(base_channels * 4)
        self.se4 = SEAttention(base_channels * 8)
        self.se5 = SEAttention(base_channels * 16)
        
    # 前向传播
    def forward(self, x):
        x1 = self.enc1(x)  # 第一层输出，跳跃1
        x2 = self.enc2(x1) # 跳跃2
        x2 = self.se2(x2)
        x3 = self.enc3(x2) # 跳跃3
        x3 = self.se3(x3)
        x4 = self.enc4(x3) # 跳跃4
        x4 = self.se4(x4)
        x5 = self.bottom(x4) # 底部连接
        x5 = self.se5(x5)
        
        return x5, [x4, x3, x2, x1 ] # [bottom, skip4, skip3, skip2, skip1]
