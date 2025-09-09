#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   CBAMUNet.py
@Time    :   2025/06/22 10:12:30
@Author  :   angkangyu 
'''

# here put the import lib
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from .UNet_parts import *


class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    """
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class CBAM_block(nn.Module):
    """
    CBAM Attention Block for skip connections
    """
    def __init__(self, F_g, F_l, F_int=None):
        super(CBAM_block, self).__init__()
        # CBAM应用到编码器特征上
        self.cbam = CBAM(F_l)
        
        # 可选的特征对齐卷积
        self.align_conv = nn.Sequential(
            nn.Conv2d(F_l, F_g, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_g)
        ) if F_l != F_g else nn.Identity()

    def forward(self, g, x):
        # 对编码器特征应用CBAM注意力
        x_att = self.cbam(x)
        # 特征对齐（如果需要）
        x_aligned = self.align_conv(x_att)
        return x_aligned


class CBAMUNet(nn.Module):
    """
    CBAM Attention Unet implementation
    """
    def __init__(self, img_ch=3, output_ch=1):
        super(CBAMUNet, self).__init__()

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
        self.Att5 = CBAM_block(F_g=filters[3], F_l=filters[3])
        self.Up_conv5 = DoubleConv(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = CBAM_block(F_g=filters[2], F_l=filters[2])
        self.Up_conv4 = DoubleConv(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = CBAM_block(F_g=filters[1], F_l=filters[1])
        self.Up_conv3 = DoubleConv(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = CBAM_block(F_g=filters[0], F_l=filters[0])
        self.Up_conv2 = DoubleConv(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        #self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        #print(x5.shape)
        d5 = self.Up5(e5)
        #print(d5.shape)
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