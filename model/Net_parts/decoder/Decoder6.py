#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Decoder6.py
@Time    :   2025/07/26 10:20:19
@Author  :   angkangyu 
'''

# here put the import lib
import torch
import torch.nn as nn
from PSSNF.model.Net_parts.public.CRH380AM import LSCDecoderBlock  # 导入LSC解码器块


class Decoder6(nn.Module):
    def __init__(self, num_classes=1, base_channels=64, BatchNorm=nn.BatchNorm2d):
        super().__init__()
        self.dec4 = LSCDecoderBlock(base_channels * 16, base_channels * 8, base_channels * 8, BatchNorm)
        self.dec3 = LSCDecoderBlock(base_channels * 8, base_channels * 4, base_channels * 4, BatchNorm)
        self.dec2 = LSCDecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2, BatchNorm)
        self.dec1 = LSCDecoderBlock(base_channels * 2, base_channels, base_channels, BatchNorm)
        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x5, skip_connections):
        x4, x3, x2, x1 = skip_connections

        d4 = self.dec4(x5, x4)
        d3 = self.dec3(d4, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)
        out = self.final_conv(d1)

        return out
    
if __name__ == "__main__":
    # 假设输入尺寸为 (1, 1024, 32, 32)，skip分别为 (1, 512, 64, 64)、(1, 256, 128, 128)、(1, 128, 256, 256)、(1, 64, 512, 512)
    num_classes = 1
    base_channels = 64
    model = Decoder6(num_classes=num_classes, base_channels=base_channels)
    model.eval()

    x5 = torch.randn(1, base_channels * 16, 32, 32)
    x4 = torch.randn(1, base_channels * 8, 64, 64)
    x3 = torch.randn(1, base_channels * 4, 128, 128)
    x2 = torch.randn(1, base_channels * 2, 256, 256)
    x1 = torch.randn(1, base_channels, 512, 512)
    skips = [x4, x3, x2, x1]

    out = model(x5, skips)
    print("输出 shape:", out.shape)  # 期望: (1, num_classes, 512, 512)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print("参数总量:", total_params)