#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Encoder2.py
@Time    :   2025/07/13 15:24:28
@Author  :   angkangyu 
'''

# here put the import lib
from torch import nn
from PSSNF.model.Net_parts.public.TUAN import UNetSepConvEB,UNetFirstEB

class Encoder2(nn.Module):
    def __init__(self, in_channels, base_channels=32, b1= UNetFirstEB, b2=UNetSepConvEB,kernel_size=3, padding=1, use_bn=True):
        super().__init__()
        
        # 四层编码器
        self.enc1 = b1(in_channels, base_channels, kernel_size=kernel_size, padding=padding, use_bn=use_bn)
        self.enc2 = b2(base_channels, base_channels * 2, kernel_size=kernel_size, padding=padding, use_bn=use_bn)
        self.enc3 = b2(base_channels * 2, base_channels * 4, kernel_size=kernel_size, padding=padding, use_bn=use_bn)
        self.enc4 = b2(base_channels * 4, base_channels * 8, kernel_size=kernel_size, padding=padding, use_bn=use_bn)      #
        self.bottom = b2(base_channels * 8, base_channels * 16, kernel_size=kernel_size, padding=padding, use_bn=use_bn)
         
    # 前向传播
    def forward(self, x):
        x1 = self.enc1(x)  # 第一层输出，跳跃1
        x2 = self.enc2(x1) # 跳跃2
        x3 = self.enc3(x2) # 跳跃3
        x4 = self.enc4(x3) # 跳跃4
        x5 = self.bottom(x4) # 底部连接
        
        return x5, [x4, x3, x2, x1 ] # [bottom, skip4, skip3, skip2, skip1]
        