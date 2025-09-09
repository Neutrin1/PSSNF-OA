#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Encoder15.py
@Time    :   2025/07/20 08:38:10
@Author  :   angkangyu 
'''

from torch import nn
from PSSNF.model.Net_parts.public.TUAN import UNetFirstEB,ParallelConvRDLEBEncoderBlock
from PSSNF.model.Net_parts.MyAtt.SA3 import SA3

class Encoder15(nn.Module):
    def __init__(self, in_channels, base_channels=32, b1= UNetFirstEB, b2=ParallelConvRDLEBEncoderBlock, use_bn=True, dropout_p=0.15):
        super().__init__()
        # 四层编码器
        self.enc1 = b1(in_channels, base_channels, use_bn=use_bn)
        self.dp1 = nn.Dropout2d(p=dropout_p)
        self.enc2 = b2(base_channels, base_channels * 2, use_bn=use_bn)
        self.dp2 = nn.Dropout2d(p=dropout_p)
        self.enc3 = b2(base_channels * 2, base_channels * 4,  use_bn=use_bn)
        self.dp3 = nn.Dropout2d(p=dropout_p)
        self.enc4 = b2(base_channels * 4, base_channels * 8,  use_bn=use_bn)
        self.dp4 = nn.Dropout2d(p=dropout_p)
        self.bottom = b2(base_channels * 8, base_channels * 16, use_bn=use_bn)
        self.dp_bottom = nn.Dropout2d(p=dropout_p)

        self.att1 = SA3(base_channels, reduction=4)
        self.dp_att1 = nn.Dropout2d(p=dropout_p)
        self.att2 = SA3(base_channels * 2)
        self.dp_att2 = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        x1 = self.enc1(x)
        x1 = self.dp1(x1)
        s1 = self.att1(x1)
        s1 = self.dp_att1(s1)

        x2 = self.enc2(x1)
        x2 = self.dp2(x2)
        s2 = self.att2(x2)
        s2 = self.dp_att2(s2)

        x3 = self.enc3(x2)
        x3 = self.dp3(x3)
        x4 = self.enc4(x3)
        x4 = self.dp4(x4)
        x5 = self.bottom(x4)
        x5 = self.dp_bottom(x5)
        return x5, [x4, x3, s2, s1]