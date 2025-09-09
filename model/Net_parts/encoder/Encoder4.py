#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Encoder3.py
@Time    :   2025/07/13 16:56:01
@Author  :   angkangyu 
'''

# here put the import lib

from re import S
from torch import nn
from PSSNF.model.Net_parts.public.TUAN import UNetFirstEB,ParallelConvRDLEBEncoderBlock
from PSSNF.model.Net_parts.public.LSKNet import LSKmodule

class Encoder4(nn.Module):
    def __init__(self, in_channels, base_channels=32, b1= UNetFirstEB, b2=ParallelConvRDLEBEncoderBlock, use_bn=True):
        super().__init__()
        
        # 四层编码器
        self.enc1 = b1(in_channels, base_channels, use_bn=use_bn)
        self.enc2 = b2(base_channels, base_channels * 2, use_bn=use_bn)
        self.enc3 = b2(base_channels * 2, base_channels * 4,  use_bn=use_bn)
        self.enc4 = b2(base_channels * 4, base_channels * 8,  use_bn=use_bn)      
        self.bottom = b2(base_channels * 8, base_channels * 16, use_bn=use_bn)
        # 加入LSKmodule
        self.lsk1 = LSKmodule(base_channels)
        self.lsk2 = LSKmodule(base_channels * 2)
        self.lsk3 = LSKmodule(base_channels * 4)
        self.lsk4 = LSKmodule(base_channels * 8)
    
    # 前向传播
    def forward(self, x):
        x1 = self.enc1(x)
        s1 = self.lsk1(x1)
        x2 = self.enc2(x1)
        s2 = self.lsk2(x2)
        x3 = self.enc3(x2)
        s3 = self.lsk3(x3)
        x4 = self.enc4(x3)
        s4 = self.lsk4(x4)
        x5 = self.bottom(x4)
        return x5, [s4, s3, s2, s1]