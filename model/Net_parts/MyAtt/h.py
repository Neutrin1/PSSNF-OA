import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################
# SA3 空间边缘注意力模块（增强型注意力）
##############################################
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
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, groups=in_channels // 4, bias=False),
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


##############################################
# EncEnhanceModule：用于每层编码器后
##############################################
class EncEnhanceModule(nn.Module):
    def __init__(self, in_channels):
        super(EncEnhanceModule, self).__init__()
        self.att = SA3(in_channels)

    def forward(self, x):
        return self.att(x)


##############################################
# BottleNeckFocusModule：用于最底层（瓶颈）
##############################################
class BottleNeckFocusModule(nn.Module):
    def __init__(self, in_channels):
        super(BottleNeckFocusModule, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, dilation=2, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 全局上下文通道权重
        g = self.global_pool(x)
        g = self.global_fc(g)
        x = x * g  # 通道加权

        # 局部空间建模
        x = self.spatial_conv(x)
        return x


##############################################
# Encoder17plus 主干骨架（示意性展示）
##############################################
class Encoder17plus(nn.Module):
    def __init__(self, in_channels, base_channels=32, b1=None, b2=None):
        super(Encoder17plus, self).__init__()
        self.b1 = b1(in_channels, base_channels)  # 第一层 UNetFirstEB
        self.e1_enh = EncEnhanceModule(base_channels)

        self.b2_2 = b2(base_channels, base_channels * 2)
        self.e2_enh = EncEnhanceModule(base_channels * 2)

        self.b2_3 = b2(base_channels * 2, base_channels * 4)
        self.e3_enh = EncEnhanceModule(base_channels * 4)

        self.b2_4 = b2(base_channels * 4, base_channels * 8)
        self.e4_enh = EncEnhanceModule(base_channels * 8)

        self.b2_5 = b2(base_channels * 8, base_channels * 16)
        self.bottle_focus = BottleNeckFocusModule(base_channels * 16)

    def forward(self, x):
        x1 = self.b1(x)
        x1 = self.e1_enh(x1)

        x2 = self.b2_2(x1)
        x2 = self.e2_enh(x2)

        x3 = self.b2_3(x2)
        x3 = self.e3_enh(x3)

        x4 = self.b2_4(x3)
        x4 = self.e4_enh(x4)

        x5 = self.b2_5(x4)
        x5 = self.bottle_focus(x5)

        return [x1, x2, x3, x4, x5]  # 可作为UNet跳跃连接输出
