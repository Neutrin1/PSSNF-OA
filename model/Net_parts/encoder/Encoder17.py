#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Encoder17.py
@Time    :   2025/07/21 14:16:59
@Author  :   angkangyu 
'''

# here put the import lib
from torch import nn
from PSSNF.model.Net_parts.public.TUAN import UNetFirstEB, UNetRDLEB
"""
    小核条形卷积
"""
class Encoder17(nn.Module):
    def __init__(self, in_channels, base_channels=32, b1=UNetFirstEB, b2=UNetRDLEB, use_bn=True, dropout_p=0.15):
        super().__init__()
        # 四层编码器
        self.enc1 = b1(in_channels, base_channels, use_bn=use_bn)
        self.enc2 = b2(base_channels, base_channels * 2, use_bn=use_bn, kernel_size=3, padding=1)
        self.enc3 = b2(base_channels * 2, base_channels * 4, use_bn=use_bn, kernel_size=3, padding=1)
        self.enc4 = b2(base_channels * 4, base_channels * 8, use_bn=use_bn, kernel_size=3, padding=1)
        self.bottom = b2(base_channels * 8, base_channels * 16, use_bn=use_bn, kernel_size=3, padding=1)
        
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.bottom(x4)
        return x5, [x4, x3, x2, x1]
    
    