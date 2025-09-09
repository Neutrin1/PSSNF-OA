#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   LSC_decoder.py
@Time    :   2025/07/02 
@Author  :   angkangyu 
@Description: LSC解码器模块，通道数对齐ResNet50标准通道数
'''

import math
from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, BatchNorm, inp=False):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn1 = BatchNorm(in_channels // 4)
        self.relu1 = nn.ReLU()
        self.inp = inp

        self.deconv1 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )
        self.deconv2 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv3 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv4 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )

        self.bn2 = BatchNorm(in_channels // 4 + in_channels // 4)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels // 4 + in_channels // 4, n_filters, 1)
        self.bn3 = BatchNorm(n_filters)
        self.relu3 = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)
        if self.inp:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)


class ResN50P_LSC_d(nn.Module):
    def __init__(self, num_classes, BatchNorm=nn.BatchNorm2d):
        super(ResN50P_LSC_d, self).__init__()
        
        # 🔥 匹配ResNet50标准通道数
        # x5: 2048, x4: 1024, x3: 512, x2: 256, x1: 64
        
        # 解码器块设计
        self.decoder4 = DecoderBlock(2048, 1024, BatchNorm)  # 处理x5: 2048 -> 1024
        self.decoder3 = DecoderBlock(2048, 512, BatchNorm)   # 处理decoder4+x4 (1024+1024=2048)
        self.decoder2 = DecoderBlock(1024, 256, BatchNorm, inp=True)  # 处理decoder3+x3 (512+512=1024)
        self.decoder1 = DecoderBlock(512, 128, BatchNorm, inp=True)   # 处理decoder2+x2 (256+256=512)

        # 通道适配层 - 匹配ResNet50标准通道数
        self.conv_x4 = nn.Sequential(
            nn.Conv2d(1024, 1024, 1, bias=False),
            BatchNorm(1024),
            nn.ReLU()
        )

        self.conv_x3 = nn.Sequential(
            nn.Conv2d(512, 512, 1, bias=False),
            BatchNorm(512),
            nn.ReLU()
        )

        self.conv_x2 = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=False),
            BatchNorm(256),
            nn.ReLU()
        )
        
        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(128 + 64, 128, 3, padding=1, bias=False),  # decoder1+x1 (128+64=192)
            BatchNorm(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            BatchNorm(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1)
        )

        self._init_weight()

    def forward(self, encoder_features):
        """
        Args:
            encoder_features: ResNet50编码器输出的特征字典
                - x1: [B, 64, H/2, W/2]     - 初始特征
                - x2: [B, 256, H/4, W/4]    - Layer1输出
                - x3: [B, 512, H/8, W/8]    - Layer2输出
                - x4: [B, 1024, H/16, W/16] - Layer3输出
                - x5: [B, 2048, H/32, W/32] - Layer4输出 + ASPP
        
        Returns:
            output: [B, num_classes, H, W]
        """
        x1 = encoder_features['x1']  # [B, 64, H/2, W/2]
        x2 = encoder_features['x2']  # [B, 256, H/4, W/4]
        x3 = encoder_features['x3']  # [B, 512, H/8, W/8]
        x4 = encoder_features['x4']  # [B, 1024, H/16, W/16]
        x5 = encoder_features['x5']  # [B, 2048, H/32, W/32]
        
        # 🔥 Stage 1: 处理最深层特征 x5
        d4 = self.decoder4(x5)  # [B, 1024, H/32, W/32]
        d4 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False)  # [B, 1024, H/16, W/16]
        
        # 🔥 Stage 2: 合并decoder4输出和x4
        x4_adapted = self.conv_x4(x4)  # [B, 1024, H/16, W/16]
        if d4.shape[2:] != x4_adapted.shape[2:]:
            x4_adapted = F.interpolate(x4_adapted, size=d4.shape[2:], mode='bilinear', align_corners=False)
        d4_cat = torch.cat((d4, x4_adapted), dim=1)  # [B, 2048, H/16, W/16]
        
        d3 = self.decoder3(d4_cat)  # [B, 512, H/16, W/16]
        d3 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)  # [B, 512, H/8, W/8]
        
        # 🔥 Stage 3: 合并decoder3输出和x3
        x3_adapted = self.conv_x3(x3)  # [B, 512, H/8, W/8]
        if d3.shape[2:] != x3_adapted.shape[2:]:
            x3_adapted = F.interpolate(x3_adapted, size=d3.shape[2:], mode='bilinear', align_corners=False)
        d3_cat = torch.cat((d3, x3_adapted), dim=1)  # [B, 1024, H/8, W/8]
        
        d2 = self.decoder2(d3_cat)  # [B, 256, H/4, W/4] (inp=True会自动上采样)
        
        # 🔥 Stage 4: 合并decoder2输出和x2
        x2_adapted = self.conv_x2(x2)  # [B, 256, H/4, W/4]
        if d2.shape[2:] != x2_adapted.shape[2:]:
            x2_adapted = F.interpolate(x2_adapted, size=d2.shape[2:], mode='bilinear', align_corners=False)
        d2_cat = torch.cat((d2, x2_adapted), dim=1)  # [B, 512, H/4, W/4]
        
        d1 = self.decoder1(d2_cat)  # [B, 128, H/2, W/2] (inp=True会自动上采样)
        
        # 🔥 Stage 5: 合并decoder1输出和x1，生成最终输出
        if d1.shape[2:] != x1.shape[2:]:
            x1 = F.interpolate(x1, size=d1.shape[2:], mode='bilinear', align_corners=False)
        final_cat = torch.cat((d1, x1), dim=1)  # [B, 192, H/2, W/2] (128+64)
        
        output = self.final_conv(final_cat)  # [B, num_classes, H/2, W/2]
        output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)  # [B, num_classes, H, W]

        return output

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(num_classes, BatchNorm=nn.BatchNorm2d):
    return ResN50P_LSC_d(num_classes, BatchNorm)


