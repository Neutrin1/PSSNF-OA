#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Encoder0.py
@Time    :   2025/07/17 14:20:22
@Author  :   angkangyu 
'''

# here put the import lib
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Encoder3.py
@Time    :   2025/07/13 16:56:01
@Author  :   angkangyu 
'''

# here put the import lib

from torch import nn
from PSSNF.model.Net_parts.public.TUAN import UNetFirstEB,ParallelConvRDLEBEncoderBlock
from PSSNF.model.Net_parts.public.Attention import CBAM
# Encoder3 的改型

class Encoder9(nn.Module):
    def __init__(self, in_channels, base_channels=32, b1= UNetFirstEB, b2=ParallelConvRDLEBEncoderBlock, use_bn=True):
        super().__init__()
        
        # 四层编码器
        self.enc1 = b1(in_channels, base_channels, use_bn=use_bn)
        self.enc2 = b2(base_channels, base_channels * 2, use_bn=use_bn)
        self.enc3 = b2(base_channels * 2, base_channels * 4,  use_bn=use_bn)
        self.enc4 = b2(base_channels * 4, base_channels * 8,  use_bn=use_bn)      
        self.bottom = b2(base_channels * 8, base_channels * 16, use_bn=use_bn)
        self.cbam1 = CBAM(base_channels)
        self.cbam2 = CBAM(base_channels * 2)
        self.cbam3 = CBAM(base_channels * 4)
        self.cbam4 = CBAM(base_channels * 8)
        
    # 前向传播
    def forward(self, x):
        x1 = self.enc1(x)  # 第一层输出，跳跃1
        s1 = self.cbam1(x1)
        x2 = self.enc2(x1) # 跳跃2
        s2 = self.cbam2(x2)
        x3 = self.enc3(x2) # 跳跃3
        s3 = self.cbam3(x3)
        x4 = self.enc4(x3) # 跳跃4
        s4 = self.cbam4(x4)
        x5 = self.bottom(x4) # 底部连接
        # print(f"Encoder output shapes: x1={x1.shape}, x2={x2.shape}, x3={x3.shape}, x4={x4.shape}, x5={x5.shape}")
        return x5, [s4, s3, s2, s1 ] # [bottom, skip4, skip3, skip2, skip1]