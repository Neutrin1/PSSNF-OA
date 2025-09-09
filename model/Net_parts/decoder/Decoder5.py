#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   DecoderMask.py
@Time    :   2025/07/22 17:30:00
@Author  :   angkangyu 
'''

import torch
from torch import nn
from PSSNF.model.Net_parts.public.SConv import ConvBlock  # 复用你的ConvBlock

class Decoder5(nn.Module):
    def __init__(self, base_channels=48, num_classes=1, eb=ConvBlock, use_bn=True):
        super().__init__()

        # 反卷积上采样，通道逐步减半，匹配跳跃连接通道数
        self.up5 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec5 = eb(base_channels * 8 * 2, base_channels * 8, use_bn=use_bn)  # 拼接skip4

        self.up4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec4 = eb(base_channels * 4 * 2, base_channels * 4, use_bn=use_bn)  # 拼接skip3

        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec3 = eb(base_channels * 2 * 2, base_channels * 2, use_bn=use_bn)  # 拼接skip2

        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec2 = eb(base_channels * 2, base_channels, use_bn=use_bn)          # 拼接skip1

        # 最后一层卷积，把通道数变成类别数，比如单通道掩码
        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

        # 用于二分类掩码，Sigmoid激活。如果多分类，用Softmax请改写。
        self.activation = nn.Sigmoid()

    def forward(self, x5, skips):
        x4, x3, x2, x1 = skips

        d5 = self.up5(x5)                    # 上采样到 x4 尺寸
        d5 = torch.cat([d5, x4], dim=1)     # 拼接 skip4
        d5 = self.dec5(d5)

        d4 = self.up4(d5)                    # 上采样到 x3 尺寸
        d4 = torch.cat([d4, x3], dim=1)     # 拼接 skip3
        d4 = self.dec4(d4)

        d3 = self.up3(d4)                    # 上采样到 x2 尺寸
        d3 = torch.cat([d3, x2], dim=1)     # 拼接 skip2
        d3 = self.dec3(d3)

        d2 = self.up2(d3)                    # 上采样到 x1 尺寸
        d2 = torch.cat([d2, x1], dim=1)     # 拼接 skip1
        d2 = self.dec2(d2)

        out = self.final_conv(d2)            # 输出类别通道数
        out = self.activation(out)           # 激活输出掩码概率

        return out

if __name__ == "__main__":
    import torch
    from PSSNF.model.Net_parts.public.SConv import ConvBlock

    x5 = torch.randn(1, 768, 16, 16)  # 48*16=768 channels
    skips = [
        torch.randn(1, 384, 32, 32),
        torch.randn(1, 192, 64, 64),
        torch.randn(1, 96, 128, 128),
        torch.randn(1, 48, 256, 256),
    ]

    model = Decoder5(base_channels=48, num_classes=1, eb=ConvBlock, use_bn=True)
    out = model(x5, skips)
    print("DecoderMask output shape:", out.shape)  # 期望 (1, 1, 256, 256)
