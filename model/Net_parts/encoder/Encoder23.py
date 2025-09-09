#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Encoder23.py
@Time    :   2025/07/29 20:22:16
@Author  :   angkangyu 
'''

# here put the import lib


from tkinter import SE
from typing import Self
import torch
import torch.nn as nn
from PSSNF.model.Net_parts.public.CRH380AM import LSCEncoderBlock,AttentionGatedFusion
from PSSNF.model.Net_parts.public.UNetBlock import UNetEncoderBlock,ResidualUNetEncoderBlock,AdvancedResidualUNetEncoderBlock,DenseResidualUNetEncoderBlock

class Encoder23(nn.Module):
    def __init__(self, in_channels, base_channels=32):
        super(Encoder23, self).__init__()

        # UNet 主干编码器块
        self.unet_enc1 = UNetEncoderBlock(in_channels, base_channels, downsample=True)
        self.unet_enc2 = UNetEncoderBlock(base_channels, base_channels*2, downsample=True)
        self.unet_enc3 = UNetEncoderBlock(base_channels*2, base_channels*4, downsample=True)
        self.unet_enc4 = UNetEncoderBlock(base_channels*4, base_channels*8, downsample=True)
        self.unet_enc5 = UNetEncoderBlock(base_channels*8, base_channels*16, downsample=False)

        # 条形卷积分支编码器块（不下采样，仅提取特征）
        self.branch_enc1 = LSCEncoderBlock(in_channels, base_channels, downsample=False)
        self.branch_enc2 = LSCEncoderBlock(base_channels, base_channels*2, downsample=False)
        self.branch_enc3 = LSCEncoderBlock(base_channels*2, base_channels*4, downsample=False)
        self.branch_enc4 = LSCEncoderBlock(base_channels*4, base_channels*8, downsample=False)
        self.branch_enc5 = LSCEncoderBlock(base_channels*8, base_channels*16, downsample=False)
        
        self.AttentionGatedFusion1 = AttentionGatedFusion(base_channels,base_channels,base_channels)
        self.AttentionGatedFusion2 = AttentionGatedFusion(base_channels*2,base_channels*2,base_channels*2)
        self.AttentionGatedFusion3 = AttentionGatedFusion(base_channels*4,base_channels*4,base_channels*4)
        self.AttentionGatedFusion4 = AttentionGatedFusion(base_channels*8,base_channels*8,base_channels*8)
        

    def forward(self, x):
        x1, skip1 = self.unet_enc1(x)
        x2, skip2 = self.unet_enc2(x1)
        x3, skip3 = self.unet_enc3(x2)
        x4, skip4 = self.unet_enc4(x3)
        x5, _ = self.unet_enc5(x4)  # 第五层不做跳跃连接
        
        
        b1 = self.branch_enc1(x)
        b2 = self.branch_enc2(x1)
        b3 = self.branch_enc3(x2)
        b4 = self.branch_enc4(x3)
        
        skip1 = self.AttentionGatedFusion1(skip1, b1)
        skip2 = self.AttentionGatedFusion2(skip2, b2)
        skip3 = self.AttentionGatedFusion3(skip3, b3)
        skip4 = self.AttentionGatedFusion4(skip4, b4)
        

        return x5, [skip4,skip3, skip2, skip1]


if __name__ == "__main__":
    import torch

    # 创建模型实例
    model = Encoder23(in_channels=3, base_channels=64)
    model.eval()

    # 输入张量，假设输入为3通道，大小192x192
    x = torch.randn(1, 3, 512, 512)

    # 前向传播
    x5, skips = model(x)
    print("x5 shape:", x5.shape)
    for i, s in enumerate(skips, 1):
        print(f"skip{i} shape:", s.shape)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)