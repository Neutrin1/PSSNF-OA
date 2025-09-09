import torch
import torch.nn as nn
import torch.nn.functional as F

class SA3(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SA3, self).__init__()
        
        # 通道压缩注意力
        self.att_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # 多尺度边缘引导（可学习，模拟 Sobel + Laplace）
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, groups=in_channels//4, bias=False),  # group conv 模拟方向性
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # 显式残差调节（可选）
        self.res_weight = nn.Parameter(torch.ones(1))  # 可学习残差权重

    def forward(self, x):
        att = self.att_conv(x)      # [B,1,H,W] 空间注意力
        edge = self.edge_conv(x)    # [B,1,H,W] 学习型边缘特征

        fuse = att * edge + self.res_weight * att  # 残差增强避免注意力衰减

        return x * fuse
