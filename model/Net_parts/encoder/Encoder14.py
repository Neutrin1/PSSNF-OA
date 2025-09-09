#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Encoder14.py
@Time    :   2025/07/19 20:16:24
@Author  :   angkangyu 
'''

from torch import nn
from PSSNF.model.Net_parts.public.TUAN import UNetFirstEB, ParallelConvRDLEBEncoderBlock
from PSSNF.model.Net_parts.MyAtt.SA2 import SA2

class Encoder14(nn.Module):
    def __init__(self, in_channels, base_channels=32, b1=UNetFirstEB, b2=ParallelConvRDLEBEncoderBlock, use_bn=True, dropout_p=0.3):
        super().__init__()
        # 四层编码器
        self.enc1 = b1(in_channels, base_channels, use_bn=use_bn)
        self.dp1 = nn.Dropout2d(p=dropout_p)
        self.enc2 = b2(base_channels, base_channels * 2, use_bn=use_bn)
        self.dp2 = nn.Dropout2d(p=dropout_p)
        self.enc3 = b2(base_channels * 2, base_channels * 4, use_bn=use_bn)
        self.dp3 = nn.Dropout2d(p=dropout_p)
        self.enc4 = b2(base_channels * 4, base_channels * 8, use_bn=use_bn)
        self.dp4 = nn.Dropout2d(p=dropout_p)
        self.bottom = b2(base_channels * 8, base_channels * 16, use_bn=use_bn)
        self.dp_bottom = nn.Dropout2d(p=dropout_p)

        self.att1 = SA2(base_channels)
        self.dp_att1 = nn.Dropout2d(p=dropout_p)
        self.att2 = SA2(base_channels * 2)
        self.dp_att2 = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        x1 = self.enc1(x)
        s1 = self.att1(x1)
        s1 = self.dp_att1(s1)

        x2 = self.enc2(x1)
        s2 = self.att2(x2)
        s2 = self.dp_att2(s2)

        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.bottom(x4)

        return x5, [x4, x3, s2, s1]