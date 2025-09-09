#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Encoder20.py
@Time    :   2025/07/22 16:27:34
@Author  :   angkangyu 
'''

# here put the import lib

from turtle import forward
import torch
from torch import nn
from PSSNF.model.Net_parts.public.SConv import ConvBlock, MFConv

class Encoder20(nn.Module):
    def __init__(self, in_channels, base_channels=32, eb1= ConvBlock, eb2= MFConv, use_bn=True, dropout_p=0.15):
        super().__init__()
        
        self.enc1 = eb1(in_channels, base_channels, use_bn=use_bn)
        self.enc2 = eb2(base_channels, base_channels * 2, use_bn=use_bn)
        self.enc3 = eb2(base_channels * 2, base_channels * 4, use_bn=use_bn)
        self.enc4 = eb2(base_channels * 4, base_channels * 8, use_bn=use_bn)
        self.enc5 = eb2(base_channels * 8, base_channels * 16, use_bn=use_bn)
        
    def forward(self,x):
        
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        # print(x5.shape, x4.shape, x3.shape, x2.shape, x1.shape)
        return x5,[x4,x3,x2,x1]
        