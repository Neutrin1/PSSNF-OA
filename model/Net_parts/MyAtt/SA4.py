import torch
import torch.nn as nn
import torch.nn.functional as F

class SA4(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SA4, self).__init__()

        # 1. 高频增强分支（边缘特征 + 局部纹理）
        self.highfreq = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, groups=in_channels // 4, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # 2. 全局通道注意力（融合 avg 和 max pooling）
        self.global_pool_avg = nn.AdaptiveAvgPool2d(1)
        self.global_pool_max = nn.AdaptiveMaxPool2d(1)
        self.channel_att = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 3. 空间门控机制（控制高频 vs 全局注意力的融合权重）
        self.fuse_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 高频边缘注意力
        edge_att = self.highfreq(x)               # [B, 1, H, W]

        # 通道注意力（全局上下文建模）
        avg_pool = self.global_pool_avg(x)        # [B, C, 1, 1]
        max_pool = self.global_pool_max(x)        # [B, C, 1, 1]
        ch_pool = torch.cat([avg_pool, max_pool], dim=1)  # [B, 2C, 1, 1]
        ch_att = self.channel_att(ch_pool)               # [B, C, 1, 1]
        ch_att = ch_att.expand_as(x)

        # 空间融合门控（平衡边缘增强与通道增强）
        gate = self.fuse_gate(torch.cat([edge_att, edge_att], dim=1))  # [B,1,H,W]

        # 混合增强输出
        enhanced = gate * x * edge_att + (1 - gate) * x * ch_att

        return enhanced
