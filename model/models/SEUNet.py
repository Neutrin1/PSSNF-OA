#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   SEUNet.py
@Time    :   2025/06/22 14:30:00
@Author  :   angkangyu 
'''

# here put the import lib
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from .UNet_parts import *


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SE_block(nn.Module):
    """
    SE Attention Block for skip connections
    """
    def __init__(self, F_g, F_l, F_int=None, reduction=16):
        super(SE_block, self).__init__()
        # SE应用到编码器特征上
        self.se = SEBlock(F_l, reduction)
        
        # 可选的特征对齐卷积
        self.align_conv = nn.Sequential(
            nn.Conv2d(F_l, F_g, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_g)
        ) if F_l != F_g else nn.Identity()

    def forward(self, g, x):
        # 对编码器特征应用SE注意力
        x_att = self.se(x)
        # 特征对齐（如果需要）
        x_aligned = self.align_conv(x_att)
        return x_aligned


class Enhanced_SE_block(nn.Module):
    """
    Enhanced SE Attention Block with cross-layer guidance
    """
    def __init__(self, F_g, F_l, F_int=None, reduction=16):
        super(Enhanced_SE_block, self).__init__()
        
        # 门控指导模块：使用解码器特征指导编码器特征
        self.gate_guide = nn.Sequential(
            nn.Conv2d(F_g, F_l, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_l),
            nn.Sigmoid()
        )
        
        # SE注意力模块
        self.se = SEBlock(F_l, reduction)
        
        # 特征融合权重
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, g, x):
        # 用解码器特征生成门控权重
        gate_weights = self.gate_guide(g)
        
        # 应用门控权重
        x_guided = x * gate_weights
        
        # 应用SE注意力
        x_se = self.se(x)
        
        # 融合门控特征和SE特征
        x_fused = self.fusion_weight * x_guided + (1 - self.fusion_weight) * x_se
        
        return x_fused


class SEUNet(nn.Module):
    """
    SE-UNet: U-Net with Squeeze-and-Excitation attention
    """
    def __init__(self, img_ch=3, output_ch=1, enhanced=False, reduction=16):
        super(SEUNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = DoubleConv(img_ch, filters[0])
        self.Conv2 = DoubleConv(filters[0], filters[1])
        self.Conv3 = DoubleConv(filters[1], filters[2])
        self.Conv4 = DoubleConv(filters[2], filters[3])
        self.Conv5 = DoubleConv(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up4 = up_conv(filters[3], filters[2])
        self.Up3 = up_conv(filters[2], filters[1])
        self.Up2 = up_conv(filters[1], filters[0])

        # 选择使用增强版SE块还是基础SE块
        if enhanced:
            self.Att5 = Enhanced_SE_block(F_g=filters[3], F_l=filters[3], reduction=reduction)
            self.Att4 = Enhanced_SE_block(F_g=filters[2], F_l=filters[2], reduction=reduction)
            self.Att3 = Enhanced_SE_block(F_g=filters[1], F_l=filters[1], reduction=reduction)
            self.Att2 = Enhanced_SE_block(F_g=filters[0], F_l=filters[0], reduction=reduction)
        else:
            self.Att5 = SE_block(F_g=filters[3], F_l=filters[3], reduction=reduction)
            self.Att4 = SE_block(F_g=filters[2], F_l=filters[2], reduction=reduction)
            self.Att3 = SE_block(F_g=filters[1], F_l=filters[1], reduction=reduction)
            self.Att2 = SE_block(F_g=filters[0], F_l=filters[0], reduction=reduction)

        self.Up_conv5 = DoubleConv(filters[4], filters[3])
        self.Up_conv4 = DoubleConv(filters[3], filters[2])
        self.Up_conv3 = DoubleConv(filters[2], filters[1])
        self.Up_conv2 = DoubleConv(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        # 在编码器中也添加SE模块
        self.se1 = SEBlock(filters[0], reduction)
        self.se2 = SEBlock(filters[1], reduction)
        self.se3 = SEBlock(filters[2], reduction)
        self.se4 = SEBlock(filters[3], reduction)
        self.se5 = SEBlock(filters[4], reduction)

        #self.active = torch.nn.Sigmoid()

    def forward(self, x):
        # 编码器路径（添加SE注意力）
        e1 = self.Conv1(x)
        e1 = self.se1(e1)  # SE增强

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        e2 = self.se2(e2)  # SE增强

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e3 = self.se3(e3)  # SE增强

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        e4 = self.se4(e4)  # SE增强

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        e5 = self.se5(e5)  # SE增强

        # 解码器路径（跳跃连接处使用SE注意力）
        d5 = self.Up5(e5)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        #out = self.active(out)

        return out


# 便于使用的工厂函数
def create_se_unet(img_ch=3, output_ch=1, enhanced=False, reduction=16):
    """
    创建SE-UNet模型
    
    Args:
        img_ch: 输入图像通道数
        output_ch: 输出通道数
        enhanced: 是否使用增强版SE块（包含跨层指导）
        reduction: SE块的降维比例
    
    Returns:
        SEUNet模型实例
    """
    return SEUNet(img_ch, output_ch, enhanced, reduction)


# 使用示例
if __name__ == "__main__":
    # 基础版SE-UNet
    model_basic = create_se_unet(img_ch=3, output_ch=1, enhanced=False)
    
    # 增强版SE-UNet
    model_enhanced = create_se_unet(img_ch=3, output_ch=1, enhanced=True)
    
    # 测试
    x = torch.randn(2, 3, 256, 256)
    out_basic = model_basic(x)
    out_enhanced = model_enhanced(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Basic SE-UNet output: {out_basic.shape}")
    print(f"Enhanced SE-UNet output: {out_enhanced.shape}")