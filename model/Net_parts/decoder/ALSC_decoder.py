#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   lsc_decoder_adapted.py
@Time    :   2025/07/01 
@Author  :   angkangyu 
@Description: 适配KNet编码器的LSC解码器 - 修复版本
'''

from re import A
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..public.ALSC import AdaptiveDecoderBlock  # 重用原有的AdaptiveDecoderBlock


class ALSCDecoder(nn.Module):
    """适配KNet编码器的自适应条形卷积解码器 - 修复版本"""
    
    def __init__(self, num_classes=1):
        super(ALSCDecoder, self).__init__()
        
        BatchNorm = nn.BatchNorm2d
        
        # 基于KNet编码器的实际输出通道数
        # x5: 320, x4: 112, x3: 40, x2: 24, x1: 32
        
        # 使用自适应解码器块
        self.decoder4 = AdaptiveDecoderBlock(320, 256, BatchNorm)  # 处理x5
        self.decoder3 = AdaptiveDecoderBlock(512, 128, BatchNorm)  # 处理decoder4+x4
        self.decoder2 = AdaptiveDecoderBlock(256, 64, BatchNorm)   # 处理decoder3+x3
        self.decoder1 = AdaptiveDecoderBlock(128, 32, BatchNorm)   # 处理decoder2+x2
        self.decoder0 = AdaptiveDecoderBlock(64, 32, BatchNorm)    # 处理decoder1+x1
        
        # 通道压缩层 - 匹配KNet编码器输出
        self.conv_x4 = nn.Sequential(
            nn.Conv2d(112, 256, 1, bias=False),  # x4: 112->256
            BatchNorm(256),
            nn.ReLU()
        )
        
        self.conv_x3 = nn.Sequential(
            nn.Conv2d(40, 128, 1, bias=False),   # x3: 40->128
            BatchNorm(128),
            nn.ReLU()
        )
        
        self.conv_x2 = nn.Sequential(
            nn.Conv2d(24, 64, 1, bias=False),    # x2: 24->64
            BatchNorm(64),
            nn.ReLU()
        )
        
        self.conv_x1 = nn.Sequential(
            nn.Conv2d(32, 32, 1, bias=False),    # x1: 32->32
            BatchNorm(32),
            nn.ReLU()
        )
        
        # 最终分类头
        self.classifier = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            BatchNorm(16),
            nn.ReLU(),
            nn.Conv2d(16, num_classes, 1)
        )
        
        self._init_weight()
    
    def forward(self, encoder_features):
        """
        Args:
            encoder_features: KNet编码器输出的特征字典
                - x1: [B, 32, H/2, W/2]
                - x2: [B, 24, H/4, W/4] 
                - x3: [B, 40, H/8, W/8]
                - x4: [B, 112, H/16, W/16]
                - x5: [B, 320, H/32, W/32]
        
        Returns:
            output: [B, num_classes, H, W]
        """
        x1 = encoder_features['x1']  # [B, 32, H/2, W/2]
        x2 = encoder_features['x2']  # [B, 24, H/4, W/4]
        x3 = encoder_features['x3']  # [B, 40, H/8, W/8]
        x4 = encoder_features['x4']  # [B, 112, H/16, W/16]
        x5 = encoder_features['x5']  # [B, 320, H/32, W/32]
        
        # Stage 1: H/32 -> H/16
        # Decoder4: 处理x5
        d4_out = self.decoder4(x5)  # 输出: [B, 256, H/32, W/32]
        d4_out = F.interpolate(d4_out, scale_factor=2, mode='bilinear', align_corners=False)  # [B, 256, H/16, W/16]
        
        x4_processed = self.conv_x4(x4)  # 输出: [B, 256, H/16, W/16]
        
        # 确保尺寸匹配后再concat
        if d4_out.shape[2:] != x4_processed.shape[2:]:
            x4_processed = F.interpolate(x4_processed, size=d4_out.shape[2:], 
                                       mode='bilinear', align_corners=False)
        
        d4 = torch.cat((d4_out, x4_processed), dim=1)  # [B, 512, H/16, W/16]
        
        # Stage 2: H/16 -> H/8
        # Decoder3: 处理d4
        d3_out = self.decoder3(d4)  # 输出: [B, 128, H/16, W/16]
        d3_out = F.interpolate(d3_out, scale_factor=2, mode='bilinear', align_corners=False)  # [B, 128, H/8, W/8]
        
        x3_processed = self.conv_x3(x3)  # 输出: [B, 128, H/8, W/8]
        
        # 确保尺寸匹配后再concat
        if d3_out.shape[2:] != x3_processed.shape[2:]:
            x3_processed = F.interpolate(x3_processed, size=d3_out.shape[2:], 
                                       mode='bilinear', align_corners=False)
        
        d3 = torch.cat((d3_out, x3_processed), dim=1)  # [B, 256, H/8, W/8]
        
        # Stage 3: H/8 -> H/4
        # Decoder2: 处理d3
        d2_out = self.decoder2(d3)  # 输出: [B, 64, H/8, W/8]
        d2_out = F.interpolate(d2_out, scale_factor=2, mode='bilinear', align_corners=False)  # [B, 64, H/4, W/4]
        
        x2_processed = self.conv_x2(x2)  # 输出: [B, 64, H/4, W/4]
        
        # 确保尺寸匹配后再concat
        if d2_out.shape[2:] != x2_processed.shape[2:]:
            x2_processed = F.interpolate(x2_processed, size=d2_out.shape[2:], 
                                       mode='bilinear', align_corners=False)
        
        d2 = torch.cat((d2_out, x2_processed), dim=1)  # [B, 128, H/4, W/4]
        
        # Stage 4: H/4 -> H/2
        # Decoder1: 处理d2
        d1_out = self.decoder1(d2)  # 输出: [B, 32, H/4, W/4]
        d1_out = F.interpolate(d1_out, scale_factor=2, mode='bilinear', align_corners=False)  # [B, 32, H/2, W/2]
        
        x1_processed = self.conv_x1(x1)  # 输出: [B, 32, H/2, W/2]
        
        # 确保尺寸匹配后再concat
        if d1_out.shape[2:] != x1_processed.shape[2:]:
            x1_processed = F.interpolate(x1_processed, size=d1_out.shape[2:], 
                                       mode='bilinear', align_corners=False)
        
        d1 = torch.cat((d1_out, x1_processed), dim=1)  # [B, 64, H/2, W/2]
        
        # Stage 5: H/2 -> H
        # Decoder0: 最终处理
        d0 = self.decoder0(d1)  # 输出: [B, 32, H/2, W/2]
        d0 = F.interpolate(d0, scale_factor=2, mode='bilinear', align_corners=False)  # [B, 32, H, W]
        
        # 分类头
        output = self.classifier(d0)  # [B, num_classes, H, W]
        
        return output
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()