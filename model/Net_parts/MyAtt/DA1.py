import torch
import torch.nn as nn
import torch.nn.functional as F

class DA1(nn.Module):
    """
    深层特征注意力：SE通道注意力 + 频域高低频融合
    输入: x [B, C, H, W]
    输出: x' [B, C, H, W]
    """
    def __init__(self, in_channels, reduction=16, fusion='add'):
        super(DA1, self).__init__()
        # SE通道注意力
        self.se_fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.se_fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.se_sigmoid = nn.Sigmoid()
        self.fusion = fusion

    def forward(self, x):
        B, C, H, W = x.shape
        # SE通道注意力
        se = F.adaptive_avg_pool2d(x, 1)
        se = F.relu(self.se_fc1(se))
        se = self.se_sigmoid(self.se_fc2(se))  # [B, C, 1, 1]
        x_se = x * se

        # 频域变换（FFT）
        x_fft = torch.fft.fft2(x, norm='ortho')
        x_fft_shift = torch.fft.fftshift(x_fft)
        # 高频增强：抑制低频（中心区域），保留高频
        mask = torch.ones_like(x_fft_shift)
        center_h, center_w = H // 2, W // 2
        radius = min(H, W) // 8  # 可调节高低频分界
        mask[..., center_h-radius:center_h+radius, center_w-radius:center_w+radius] = 0  # 抑制低频
        x_fft_high = x_fft_shift * mask
        x_fft_high = torch.fft.ifftshift(x_fft_high)
        x_high = torch.fft.ifft2(x_fft_high, norm='ortho').real  # 取实部

        # 融合
        if self.fusion == 'add':
            out = x_se + x_high
        elif self.fusion == 'mul':
            out = x_se * x_high
        else:
            out = x_se + x_high  # 默认加性融合

        return out