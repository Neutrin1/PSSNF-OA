import torch
import torch.nn as nn
from PSSNF.model.Net_parts.public.UNetBlock import UNetEncoderBlock


class UNetEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(UNetEncoder, self).__init__()
        
        self.enc1 = UNetEncoderBlock(in_channels, base_channels, downsample=False)
        self.enc2 = UNetEncoderBlock(base_channels, base_channels*2)
        self.enc3 = UNetEncoderBlock(base_channels*2, base_channels*4)
        self.enc4 = UNetEncoderBlock(base_channels*4, base_channels*8)
        self.enc5 = UNetEncoderBlock(base_channels*8, base_channels*16)
        
    def forward(self, x):
        x1, skip1 = self.enc1(x)    # [B, 64, H, W]
        x2, skip2 = self.enc2(x1)   # [B, 128, H/2, W/2]
        x3, skip3 = self.enc3(x2)   # [B, 256, H/4, W/4]
        x4, skip4 = self.enc4(x3)   # [B, 512, H/8, W/8]
        x5, skip5 = self.enc5(x4)   # [B, 1024, H/16, W/16]
        
        return skip5, [skip4, skip3, skip2,skip1]