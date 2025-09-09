import torch
import torch.nn as nn
import torch.nn.functional as F

class SA1(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SA1, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # Sobel edge filter卷积核定义（不可训练）
        sobel_kernel = torch.tensor([
            [[[-1, -2, -1],
              [ 0,  0,  0],
              [ 1,  2,  1]]],
            [[[-1,  0,  1],
              [-2,  0,  2],
              [-1,  0,  1]]]
        ], dtype=torch.float32)

        self.sobel_filter = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)
        self.sobel_filter.weight = nn.Parameter(sobel_kernel, requires_grad=False)

    def forward(self, x):
        # 通道压缩注意力图
        att_map = self.reduce_conv(x)  # [B, 1, H, W]

        # 使用Sobel边缘提取（基于灰度图）
        gray = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        edges = self.sobel_filter(gray)     # [B, 2, H, W]
        edge_map = torch.sum(edges ** 2, dim=1, keepdim=True)  # [B, 1, H, W]
        edge_map = torch.sqrt(edge_map + 1e-6)                 # 防止 sqrt(0)
        edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min() + 1e-6)
        edge_map = edge_map.detach()

        # 两者相乘后再加权输入特征
        att = att_map * edge_map  # [B, 1, H, W]
        out = x * att             # 加权原始特征

        return out
