#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Unet_decoder.py
@Time    :   2025/07/02 
@Author  :   angkangyu 
@Description: 标准UNet解码器模块 - 匹配ResNet18通道数
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """标准UNet的双卷积块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """标准UNet上采样块"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        # 使用双线性插值或转置卷积进行上采样
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: 来自上一层的特征 (需要上采样)
        x2: 来自编码器的跳跃连接特征
        """
        x1 = self.up(x1)
        
        # 处理尺寸不匹配的情况
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 拼接跳跃连接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetDecoder(nn.Module):
    """标准UNet解码器 - 匹配ResNet18通道数"""
    
    def __init__(self, num_classes=1, bilinear=True):
        super().__init__()
        self.num_classes = num_classes
        self.bilinear = bilinear
        
        # 匹配ResNet18标准通道数: [64, 64, 128, 256, 512]
        # 修正通道数计算，不使用factor
        
        # 解码器上采样层 - 直接使用实际通道数
        self.up1 = Up(512 + 256, 256, bilinear)  # x5(512) + x4(256) = 768 -> 256
        self.up2 = Up(256 + 128, 128, bilinear)  # up1(256) + x3(128) = 384 -> 128  
        self.up3 = Up(128 + 64, 64, bilinear)    # up2(128) + x2(64) = 192 -> 64
        self.up4 = Up(64 + 64, 64, bilinear)     # up3(64) + x1(64) = 128 -> 64
        
        # 最终输出层
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, encoder_features):
        """
        Args:
            encoder_features: ResNet18编码器输出的特征字典
                - x1: [B, 64, H/2, W/2]
                - x2: [B, 64, H/4, W/4] 
                - x3: [B, 128, H/8, W/8]
                - x4: [B, 256, H/16, W/16]
                - x5: [B, 512, H/32, W/32]
        
        Returns:
            output: [B, num_classes, H, W]
        """
        x1 = encoder_features['x1']  # [B, 64, H/2, W/2]
        x2 = encoder_features['x2']  # [B, 64, H/4, W/4]
        x3 = encoder_features['x3']  # [B, 128, H/8, W/8]
        x4 = encoder_features['x4']  # [B, 256, H/16, W/16]
        x5 = encoder_features['x5']  # [B, 512, H/32, W/32]

        # 标准UNet解码器前向传播
        
        # Stage 1: H/32 → H/16, 512+256 -> 256
        x = self.up1(x5, x4)  # [B, 256, H/16, W/16]
        
        # Stage 2: H/16 → H/8, 256+128 -> 128
        x = self.up2(x, x3)   # [B, 128, H/8, W/8]
        
        # Stage 3: H/8 → H/4, 128+64 -> 64
        x = self.up3(x, x2)   # [B, 64, H/4, W/4]
        
        # Stage 4: H/4 → H/2, 64+64 -> 64
        x = self.up4(x, x1)   # [B, 64, H/2, W/2]
        
        # Stage 5: H/2 → H, 最终上采样到原始尺寸
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 64, H, W]
        
        # 最终分类
        output = self.outc(x)  # [B, num_classes, H, W]

        return output


def build_unet_decoder(num_classes=1, bilinear=True):
    """构建UNet解码器"""
    return UNetDecoder(num_classes, bilinear)


# 测试代码
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    decoder = UNetDecoder(num_classes=1, bilinear=True).to(device)
    
    # 模拟ResNet18编码器输出 - 匹配实际的ResNet18通道数
    test_features = {
        'x1': torch.randn(2, 64, 256, 256).to(device),   # [B, 64, H/2, W/2]
        'x2': torch.randn(2, 64, 128, 128).to(device),   # [B, 64, H/4, W/4]
        'x3': torch.randn(2, 128, 64, 64).to(device),    # [B, 128, H/8, W/8]
        'x4': torch.randn(2, 256, 32, 32).to(device),    # [B, 256, H/16, W/16]
        'x5': torch.randn(2, 512, 16, 16).to(device),    # [B, 512, H/32, W/32]
    }
    
    # 前向传播
    with torch.no_grad():
        output = decoder(test_features)
    
    print("标准UNet解码器输出:")
    print(f"Output Shape: {output.shape}")
    print(f"Expected: [2, 1, 512, 512]")
    
    # 计算参数量
    total_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    print("\n通道数匹配检查:")
    print("✅ up1: 512+256=768 -> 256")
    print("✅ up2: 256+128=384 -> 128")  
    print("✅ up3: 128+64=192 -> 64")
    print("✅ up4: 64+64=128 -> 64")
    print("✅ 完全匹配ResNet18标准通道数 [64, 64, 128, 256, 512]")