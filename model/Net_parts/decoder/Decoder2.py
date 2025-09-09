#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Decoder2.py
@Time    :   2025/07/13 22:12:20
@Author  :   angkangyu 
'''

# here put the import lib
from traceback import print_tb
import torch
import torch.nn as nn
from PSSNF.model.Net_parts.public.TUAN import UNetLDB  # 导入标准UNet解码器块

class Decoder2(nn.Module):
    def __init__(self, num_classes=1, base_channels=48, block=UNetLDB, kernel_size=7, use_bn=True):
        super().__init__()
        self.dec4 = block(base_channels * 16, base_channels * 8, kernel_size=kernel_size, use_bn=use_bn)
        self.dec3 = block(base_channels * 8, base_channels * 4, kernel_size=kernel_size, use_bn=use_bn)
        self.dec2 = block(base_channels * 4, base_channels * 2, kernel_size=kernel_size, use_bn=use_bn)
        self.dec1 = block(base_channels * 2, base_channels, kernel_size=kernel_size, use_bn=use_bn)
        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x5, skip_connections):
        x4, x3, x2, x1 = skip_connections
        # print(f"Decoder input shapes: x5={x5.shape}, x4={x4.shape}, x3={x3.shape}, x2={x2.shape}, x1={x1.shape}")
        d4 = self.dec4(x5, x4)
        d3 = self.dec3(d4, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)
        out = self.final_conv(d1)
        
        return out
