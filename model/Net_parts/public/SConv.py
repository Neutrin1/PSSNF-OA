'''
@File    :   SConv.py
@Time    :   2025/07/22 19:30:34
@Author  :   angkangyu 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# 第一层特征粗提取
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# 自定义空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size 应该是奇数"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(concat)
        attention = self.sigmoid(attention)
        return x * attention


# 单尺度 7x7 卷积模块（保留四方向卷积）
class FConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, padding=3, use_bn=True):
        super().__init__()
        self.conv_h = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, padding))
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.conv_w = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(padding, 0))
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

        self.deconv_diag1 = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(padding, 0))
        self.deconv_diag2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, padding))

        self.relu = nn.ReLU(inplace=True)
        self.fuse_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)

    def h_transform(self, x):
        shape = x.size()
        x = F.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = F.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = F.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = F.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)

    def forward(self, x):
        out_h = self.relu(self.bn1(self.conv_h(x)))
        out_w = self.relu(self.bn2(self.conv_w(out_h)))

        out_diag1 = self.relu(self.inv_h_transform(self.deconv_diag1(self.h_transform(out_w))))
        out_diag2 = self.relu(self.inv_v_transform(self.deconv_diag2(self.v_transform(out_w))))

        min_h = min(out_h.shape[2], out_w.shape[2], out_diag1.shape[2], out_diag2.shape[2])
        min_w = min(out_h.shape[3], out_w.shape[3], out_diag1.shape[3], out_diag2.shape[3])

        def crop(tensor):
            return tensor[:, :, :min_h, :min_w]

        out = torch.cat([
            crop(out_h), crop(out_w), crop(out_diag1), crop(out_diag2)
        ], dim=1)

        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = self.fuse_conv(out)
        return out


# 修改后的 MFConv：仅使用 FConv（四向卷积）分支
class MFConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()

        self.branch = FConv(in_channels, out_channels, kernel_size=3, padding=1, use_bn=use_bn)

        self.gate_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.fuse_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.fuse_bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.spatial_att = SpatialAttention(kernel_size=9)

    def forward(self, x):
        out = self.branch(x)
        gate = self.gate_fc(out).unsqueeze(-1).unsqueeze(-1)
        out = out * gate

        out = self.fuse_conv(out)
        out = self.fuse_bn(out)
        out = self.relu(out)

        res = F.max_pool2d(x, kernel_size=2, stride=2)
        res = self.res_conv(res)

        out = out + res
        out = self.spatial_att(out)
        return out
