# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F
# unet组件
from PSSNF.model.Net_parts.public.UNet_parts import *


class UNetDecoder(nn.Module):
    """U-Net解码器部分"""
    
    def __init__(self, num_classes, bilinear=False):
        super(UNetDecoder, self).__init__()
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        
        # 解码器路径（通道数砍半）
        self.up1 = Up(512, 256 // factor, bilinear)  # 上采样层1
        self.up2 = Up(256, 128 // factor, bilinear)  # 上采样层2
        self.up3 = Up(128, 64 // factor, bilinear)   # 上采样层3
        self.up4 = Up(64, 32, bilinear)              # 上采样层4
        self.outc = OutConv(32, num_classes)         # 输出层
    
        
    
    def forward(self, bottom_features, skip_connections):
        # 解包跳跃连接特征
        x4, x3, x2, x1 = skip_connections
        
        # 解码器前向传播
        x = self.up1(bottom_features, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits
    
    