#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   aspp.py
@Time    :   2025/07/01 
@Author  :   angkangyu 
@Description: 改进的ASPP模块 - 结合两版本优点
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ASPPModule(nn.Module):
    """ASPP模块的基础组件"""
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                   stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    """改进的ASPP模块 - 适配KNet"""
    def __init__(self, in_channels=320, out_channels=256, output_stride=32, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()
        
        # 根据output_stride设置膨胀率
        if output_stride == 32:
            dilations = [1, 6, 12, 18]
        elif output_stride == 16:
            dilations = [1, 6, 12, 18] 
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            dilations = [1, 6, 12, 18]  # 默认值
        
        # ASPP分支
        self.aspp1 = _ASPPModule(in_channels, out_channels, 1, padding=0, 
                               dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(in_channels, out_channels, 3, padding=dilations[1], 
                               dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(in_channels, out_channels, 3, padding=dilations[2], 
                               dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(in_channels, out_channels, 3, padding=dilations[3], 
                               dilation=dilations[3], BatchNorm=BatchNorm)
        
        # 全局平均池化分支 - 修复BatchNorm问题
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=True),  # 使用bias避免BatchNorm问题
            nn.ReLU()
        )
        
        # 特征融合
        self.conv1 = nn.Conv2d(5 * out_channels, out_channels, 1, bias=False)
        self.bn1 = BatchNorm(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)  # 降低dropout率
        
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(in_channels=320, out_channels=256, output_stride=32):
    """构建ASPP模块"""
    return ASPP(in_channels=in_channels, out_channels=out_channels, output_stride=output_stride)