import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# Global Modeling Block via Dilated Conv
# -----------------------
class ConvGlobalBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1, groups=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4, groups=channels)
        self.fuse = nn.Conv2d(channels * 3, channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        out = self.fuse(torch.cat([x1, x2, x3], dim=1))
        return out

# -----------------------
# Up Block with Global Modeling
# -----------------------
class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.global_block = ConvGlobalBlock(out_channels)
        self.norm = nn.GroupNorm(8, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.up(x)
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.norm(self.conv1(x)))
        x = self.relu(self.norm(self.conv2(x)))
        x = self.global_block(x)
        return x

# -----------------------
# Decoder23_ConvGlobal
# -----------------------
class Decoder23(nn.Module):
    def __init__(self, base_channels=64, num_classes=1):
        super().__init__()
        self.dec4 = UpBlock(base_channels * 16, base_channels * 8, base_channels * 8)
        self.dec3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.dec2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.dec1 = UpBlock(base_channels * 2, base_channels, base_channels)

        self.out_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, num_classes, kernel_size=1)
        )

    def forward(self, x5, skips):
        skip4, skip3, skip2, skip1 = skips  # 注意顺序和Encoder返回一致
        d4 = self.dec4(x5, skip4)  # 32 -> 64
        d3 = self.dec3(d4, skip3)  # 64 -> 128
        d2 = self.dec2(d3, skip2)  # 128 -> 256
        d1 = self.dec1(d2, skip1)  # 256 -> 512


        out = self.out_conv(d1)
        
        return out
