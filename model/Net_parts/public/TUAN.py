#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   TUAN.py
@Time    :   2025/07/11 14:50:29
@Author  :   angkangyu 
'''

# here put the import lib
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .Attention import SEAttention,CBAM

'''
    定义一些基础模块，以后的基础模块大多写入其中，不再单独创建文件
    主要包括：编码器，解码器的主体以及用到的各种注意力机制。
'''
# UNet第一级编码器：扩展通道但不降尺寸
class UNetFirstEB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_bn=True):
        super(UNetFirstEB, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size), padding=padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



# UNet标准编码器块块
class UNetEB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_bn=True):
        super(UNetEB, self).__init__()
        
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.encode(x)
    
# 带残差结构的UNet编码器块
class UNetREB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_bn=True):
        super(UNetREB, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        
        # 如果输入输出通道不一致，用1x1卷积调整
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.encode(x)
        res = self.res_conv(x)
        return out + res     # 残差连接

class UNetRDLEB_no_deform(nn.Module):
    """
    不用可变形卷积的条形卷积+对角卷积编码器块
    """
    def __init__(self, in_channels, out_channels, kernel_size=7, padding=3, use_bn=True):
        super().__init__()
        # 高方向条形卷积 (1, k)
        self.conv_h = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, padding))
        # 宽方向条形卷积 (k, 1)
        self.conv_w = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(padding, 0))
        # 主对角线与反对角线方向（掩码卷积）
        self.conv_diag1 = DiagonalMaskedConv2d(out_channels, out_channels, kernel_size=(kernel_size, kernel_size),
                                               padding=(padding, padding), diag_type='main')
        self.conv_diag2 = DiagonalMaskedConv2d(out_channels, out_channels, kernel_size=(kernel_size, kernel_size),
                                               padding=(padding, padding), diag_type='anti')
        # BatchNorm & 激活
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.bn3 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.bn4 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        # 通道融合与残差
        self.fuse_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.se = SEAttention(out_channels)

    def forward(self, x):
        # 高方向条形卷积
        out_h = self.conv_h(x)
        out_h = self.bn1(out_h)
        out_h = self.relu(out_h)
        # 宽方向条形卷积
        out_w = self.conv_w(out_h)
        out_w = self.bn2(out_w)
        out_w = self.relu(out_w)
        # 主对角线方向
        out_d1 = self.conv_diag1(out_w)
        out_d1 = self.bn3(out_d1)
        out_d1 = self.relu(out_d1)
        # 反对角线方向
        out_d2 = self.conv_diag2(out_w)
        out_d2 = self.bn4(out_d2)
        out_d2 = self.relu(out_d2)
        # 融合
        out = torch.cat([out_h, out_w, out_d1, out_d2], dim=1)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = self.fuse_conv(out)
        # 残差路径
        res = F.max_pool2d(x, kernel_size=2, stride=2)
        res = self.res_conv(res)
        # 加残差 & 注意力
        out = self.se(out + res)
        return out
    
    
    
    
# 带掩码的对角线卷积层（主或反对角线）
class DiagonalMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, diag_type='main', **kwargs):
        super().__init__(*args, **kwargs)
        k = self.kernel_size[0]
        assert self.kernel_size[0] == self.kernel_size[1], 'Only square kernels supported for diagonal masking'

        # 构造掩码
        mask = torch.zeros_like(self.weight)
        for i in range(k):
            if diag_type == 'main':
                mask[:, :, i, i] = 1
            elif diag_type == 'anti':
                mask[:, :, i, k - 1 - i] = 1
            else:
                raise ValueError("diag_type must be 'main' or 'anti'")
        self.register_buffer("mask", mask)

        # 注册前向 hook，防止非对角元素更新
        self.register_forward_pre_hook(self._apply_mask)

    def _apply_mask(self, module, input):
        self.weight.data *= self.mask


# 使用可变形条形卷积 + 对角方向卷积的编码器块
class UNetRDLEB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, padding=3, use_bn=True):
        super().__init__()

        # 高方向可变形卷积 (1, k)
        self.offset_conv_h = nn.Conv2d(in_channels, 2 * 1 * kernel_size, kernel_size=3, padding=1)
        self.deform_conv_h = DeformConv2d(in_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, padding))

        # 宽方向可变形卷积 (k, 1)
        self.offset_conv_w = nn.Conv2d(out_channels, 2 * kernel_size * 1, kernel_size=3, padding=1)
        self.deform_conv_w = DeformConv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(padding, 0))

        # 主对角线与反对角线方向（掩码卷积）
        self.conv_diag1 = DiagonalMaskedConv2d(out_channels, out_channels, kernel_size=(kernel_size, kernel_size),
                                               padding=(padding, padding), diag_type='main')
        self.conv_diag2 = DiagonalMaskedConv2d(out_channels, out_channels, kernel_size=(kernel_size, kernel_size),
                                               padding=(padding, padding), diag_type='anti')

        # BatchNorm & 激活
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.bn3 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.bn4 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        # 通道融合与残差
        self.fuse_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.se = SEAttention(out_channels)  # 可替换为其他注意力模块

    def forward(self, x):
        # 高方向 deform conv
        offset_h = self.offset_conv_h(x)
        out_h = self.deform_conv_h(x, offset_h)
        out_h = self.bn1(out_h)
        out_h = self.relu(out_h)

        # 宽方向 deform conv
        offset_w = self.offset_conv_w(out_h)
        out_w = self.deform_conv_w(out_h, offset_w)
        out_w = self.bn2(out_w)
        out_w = self.relu(out_w)

        # 主对角线方向
        out_d1 = self.conv_diag1(out_w)
        out_d1 = self.bn3(out_d1)
        out_d1 = self.relu(out_d1)

        # 反对角线方向
        out_d2 = self.conv_diag2(out_w)
        out_d2 = self.bn4(out_d2)
        out_d2 = self.relu(out_d2)

        # 融合
        out = torch.cat([out_h, out_w, out_d1, out_d2], dim=1)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = self.fuse_conv(out)

        # 残差路径
        res = F.max_pool2d(x, kernel_size=2, stride=2)
        res = self.res_conv(res)

        # 加残差 & 注意力
        out = self.se(out + res)
        return out

class UNetRDLEB2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, padding=3, use_bn=True):
        super().__init__()

        # 高方向可变形卷积 (1, k)
        self.offset_conv_h = nn.Conv2d(in_channels, 2 * 1 * kernel_size, kernel_size=3, padding=1)
        self.deform_conv_h = DeformConv2d(in_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, padding))

        # 宽方向可变形卷积 (k, 1)
        self.offset_conv_w = nn.Conv2d(out_channels, 2 * kernel_size * 1, kernel_size=3, padding=1)
        self.deform_conv_w = DeformConv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(padding, 0))

        # 主对角线与反对角线方向（掩码卷积）
        self.conv_diag1 = DiagonalMaskedConv2d(out_channels, out_channels, kernel_size=(kernel_size, kernel_size),
                                               padding=(padding, padding), diag_type='main')
        self.conv_diag2 = DiagonalMaskedConv2d(out_channels, out_channels, kernel_size=(kernel_size, kernel_size),
                                               padding=(padding, padding), diag_type='anti')

        # BatchNorm & 激活
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.bn3 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.bn4 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        # 通道融合与残差
        self.fuse_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.se = SEAttention(out_channels)  # 可替换为其他注意力模块
        self.cbam = CBAM(out_channels)  # 可选的CBAM注意力模块
        
    def forward(self, x):
        # 高方向 deform conv
        offset_h = self.offset_conv_h(x)
        out_h = self.deform_conv_h(x, offset_h)
        out_h = self.bn1(out_h)
        out_h = self.relu(out_h)

        # 宽方向 deform conv
        offset_w = self.offset_conv_w(out_h)
        out_w = self.deform_conv_w(out_h, offset_w)
        out_w = self.bn2(out_w)
        out_w = self.relu(out_w)

        # 主对角线方向
        out_d1 = self.conv_diag1(out_w)
        out_d1 = self.bn3(out_d1)
        out_d1 = self.relu(out_d1)

        # 反对角线方向
        out_d2 = self.conv_diag2(out_w)
        out_d2 = self.bn4(out_d2)
        out_d2 = self.relu(out_d2)

        # 融合
        out = torch.cat([out_h, out_w, out_d1, out_d2], dim=1)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = self.fuse_conv(out)

        # 残差路径
        res = F.max_pool2d(x, kernel_size=2, stride=2)
        res = self.res_conv(res)

        # 加残差 & 注意力
        out = self.cbam(out + res)
        return out


"""
    自注意力模块
"""
class SpatialSelfAttention(nn.Module):
    """
    空间自注意力模块（适用于图像特征图）
    输入: (B, C, H, W)
    输出: (B, C, H, W)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C//8)
        proj_key = self.key_conv(x).view(B, -1, H * W)                       # (B, C//8, HW)
        energy = torch.bmm(proj_query, proj_key)                             # (B, HW, HW)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(B, -1, H * W)                   # (B, C, HW)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))              # (B, C, HW)
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out




# 条形卷积解码器
class UNetLDB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, up_kernel=2, up_stride=2, up_padding=0, use_bn=True):
        super().__init__()
        # 上采样
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=up_kernel, stride=up_stride, padding=up_padding
        )
        self.up_bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.up_relu = nn.ReLU(inplace=True)

        # 拼接后通道数
        self.cat_conv_in = out_channels * 2
        # 高方向条形卷积 (1, k)
        self.conv_h = nn.Conv2d(self.cat_conv_in, out_channels, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2))
        # 宽方向条形卷积 (k, 1)
        self.conv_w = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0))
        # 主对角线（掩码卷积）
        self.conv_diag1 = DiagonalMaskedConv2d(
            self.cat_conv_in, out_channels, kernel_size=(kernel_size, kernel_size),
            padding=(kernel_size // 2, kernel_size // 2), diag_type='main'
        )
        # 反对角线（掩码卷积）
        self.conv_diag2 = DiagonalMaskedConv2d(
            self.cat_conv_in, out_channels, kernel_size=(kernel_size, kernel_size),
            padding=(kernel_size // 2, kernel_size // 2), diag_type='anti'
        )

        # BN和激活
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.bn3 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.bn4 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        # 1x1卷积融合
        self.fuse_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
        self.fuse_bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.fuse_relu = nn.ReLU(inplace=True)
        # 残差分支
        self.res_conv = nn.Conv2d(self.cat_conv_in, out_channels, 1)

    def forward(self, x, skip=None):
        # 上采样
        x_up = self.upconv(x)
        x_up = self.up_bn(x_up)
        x_up = self.up_relu(x_up)
        # 跳跃连接
        if skip is not None:
            if x_up.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x_up.shape[2:], mode='bilinear', align_corners=False)
            x_up = torch.cat([x_up, skip], dim=1)
        # 高方向
        out_h = self.conv_h(x_up)
        out_h = self.bn1(out_h)
        out_h = self.relu(out_h)
        # 宽方向
        out_w = self.conv_w(out_h)
        out_w = self.bn2(out_w)
        out_w = self.relu(out_w)
        # 主对角线（掩码卷积）
        out_d1 = self.conv_diag1(x_up)
        out_d1 = self.bn3(out_d1)
        out_d1 = self.relu(out_d1)
        # 反对角线（掩码卷积）
        out_d2 = self.conv_diag2(x_up)
        out_d2 = self.bn4(out_d2)
        out_d2 = self.relu(out_d2)
        # 拼接融合
        out = torch.cat([out_h, out_w, out_d1, out_d2], dim=1)
        out = self.fuse_conv(out)
        out = self.fuse_bn(out)
        out = self.fuse_relu(out)
        # 残差
        res = self.res_conv(x_up)
        out = out + res
        return out
    
    
# 条形卷积解码器去残差
class UNetLDBwr(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, up_kernel=2, up_stride=2, up_padding=0, use_bn=True):
        super().__init__()
        # 上采样
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=up_kernel, stride=up_stride, padding=up_padding
        )
        self.up_bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.up_relu = nn.ReLU(inplace=True)

        # 拼接后通道数
        self.cat_conv_in = out_channels * 2
        # 高方向条形卷积 (1, k)
        self.conv_h = nn.Conv2d(self.cat_conv_in, out_channels, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2))
        # 宽方向条形卷积 (k, 1)
        self.conv_w = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0))
        # 主对角线（掩码卷积）
        self.conv_diag1 = DiagonalMaskedConv2d(
            self.cat_conv_in, out_channels, kernel_size=(kernel_size, kernel_size),
            padding=(kernel_size // 2, kernel_size // 2), diag_type='main'
        )
        # 反对角线（掩码卷积）
        self.conv_diag2 = DiagonalMaskedConv2d(
            self.cat_conv_in, out_channels, kernel_size=(kernel_size, kernel_size),
            padding=(kernel_size // 2, kernel_size // 2), diag_type='anti'
        )

        # BN和激活
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.bn3 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.bn4 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        # 1x1卷积融合
        self.fuse_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
        self.fuse_bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.fuse_relu = nn.ReLU(inplace=True)
        # 残差分支
        self.res_conv = nn.Conv2d(self.cat_conv_in, out_channels, 1)

    def forward(self, x, skip=None):
        # 上采样
        x_up = self.upconv(x)
        x_up = self.up_bn(x_up)
        x_up = self.up_relu(x_up)
        # 跳跃连接
        if skip is not None:
            if x_up.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x_up.shape[2:], mode='bilinear', align_corners=False)
            x_up = torch.cat([x_up, skip], dim=1)
        # 高方向
        out_h = self.conv_h(x_up)
        out_h = self.bn1(out_h)
        out_h = self.relu(out_h)
        # 宽方向
        out_w = self.conv_w(out_h)
        out_w = self.bn2(out_w)
        out_w = self.relu(out_w)
        # 主对角线（掩码卷积）
        out_d1 = self.conv_diag1(x_up)
        out_d1 = self.bn3(out_d1)
        out_d1 = self.relu(out_d1)
        # 反对角线（掩码卷积）
        out_d2 = self.conv_diag2(x_up)
        out_d2 = self.bn4(out_d2)
        out_d2 = self.relu(out_d2)
        # 拼接融合
        out = torch.cat([out_h, out_w, out_d1, out_d2], dim=1)
        out = self.fuse_conv(out)
        out = self.fuse_bn(out)
        out = self.fuse_relu(out)
        
        return out
    
    
# 转置卷积解码器块
class UNetDB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, use_bn=True):
        super(UNetDB, self).__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

    def forward(self, x):
        x = self.upconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


# UNet带跳跃连接的转置卷积解码器块
class UNetDEB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, use_bn=True):
        super(UNetDEB, self).__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bn_up = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)  # 拼接后通道*2
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

    def forward(self, x, skip):
        x = self.upconv(x)
        x = self.bn_up(x)
        x = self.relu(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class MUNetDEB(nn.Module):
    """
    多尺度卷积解码器块，融合多种卷积核尺寸，支持UNet式跳跃连接
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=(3, 5, 7), stride=2, padding=1, output_padding=1, use_bn=True):
        super(MUNetDEB, self).__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bn_up = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        # 多尺度卷积分支
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, kernel_size=k, padding=k // 2),
                nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
                nn.ReLU(inplace=True)
            ) for k in kernel_sizes
        ])

        # 融合卷积
        self.fuse_conv = nn.Conv2d(out_channels * len(kernel_sizes), out_channels, kernel_size=1)
        self.fuse_bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.fuse_relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = self.bn_up(x)
        x = self.relu(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        # 多尺度卷积分支
        feats = [conv(x) for conv in self.convs]
        x = torch.cat(feats, dim=1)
        x = self.fuse_conv(x)
        x = self.fuse_bn(x)
        x = self.fuse_relu(x)
        return x

# 空洞卷积模块
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=(1, 6, 12, 18)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for d in dilations
        ])
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Conv2d(out_channels * (len(dilations) + 1), out_channels, kernel_size=1)

    def forward(self, x):
        out = [conv(x) for conv in self.convs]
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=x.shape[2:], mode='bilinear', align_corners=True)
        out.append(gp)
        out = torch.cat(out, dim=1)
        return self.out(out)
    
    
# Vit的一种标准实现形式
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class ViTEncoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, patch_size=4, depth=2, heads=8, dim_head=64, mlp_ratio=4.0, dropout=0.1
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = None  # 可选：可加可不加

        # Transformer编码器
        dim = out_channels
        mlp_dim = int(dim * mlp_ratio)
        self.transformer = Transformer(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.patch_embed(x)  # (B, out_channels, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.transformer(x)           # (B, N, C)
        x = x.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H', W')
        return x
    
    

#Swin Transformer 的窗口注意力机制

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, attn_dropout=0., proj_dropout=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)
    
    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj_dropout(self.proj(out))
        return out

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio=4., attn_dropout=0., proj_dropout=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.window_size = window_size
        self.shift_size = shift_size

        self.attn = WindowAttention(dim, window_size, num_heads, attn_dropout, proj_dropout)
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(proj_dropout)
        )

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted = x

        # partition windows
        x_windows = rearrange(shifted, 'b (nh ws) (nw ws2) c -> (b nh nw) (ws ws2) c',
                              nh=H // self.window_size, ws=self.window_size,
                              nw=W // self.window_size, ws2=self.window_size)
        attn_windows = self.attn(x_windows)

        # merge windows
        x = rearrange(attn_windows, '(b nh nw) (ws ws2) c -> b (nh ws) (nw ws2) c',
                      b=B, nh=H // self.window_size, nw=W // self.window_size,
                      ws=self.window_size, ws2=self.window_size)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H*W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        # 拼接 2x2 patch
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

# ---- Swin Transformer 编码器组成 ----

class SwinEncoder(nn.Module):
    def __init__(self, *, image_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2,2,6,2], num_heads=[3,6,12,24],
                 window_size=7, mlp_ratio=4., drop_path_rate=0.1):
        super().__init__()
        H, W = pair(image_size)
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

        self.stages = nn.ModuleList()
        self.patch_mergings = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        dim = embed_dim
        for i, d in enumerate(depths):
            blocks = nn.ModuleList()
            for j in range(d):
                blocks.append(SwinBlock(dim, num_heads[i], window_size, shift_size=0 if (j % 2 == 0) else window_size // 2,
                                        mlp_ratio=mlp_ratio, drop_path=dp_rates[cur + j]))
            self.stages.append(blocks)
            cur += d

            if i < len(depths) - 1:
                self.patch_mergings.append(PatchMerging(dim))
                dim *= 2

        self.num_features = [embed_dim * (2**i) for i in range(len(depths))]

    def forward(self, x):
        B, C, H0, W0 = x.shape
        x = self.patch_embed(x)
        H, W = H0 // self.patch_embed.kernel_size[0], W0 // self.patch_embed.kernel_size[1]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        features = []
        for i, blocks in enumerate(self.stages):
            for blk in blocks:
                x = blk(x, H, W)
            B, L, C = x.shape
            features.append(x.transpose(1,2).view(B, C, H, W))
            if i < len(self.patch_mergings):
                x = self.patch_mergings[i](x, H, W)
                H, W = H // 2, W // 2

        return features  # list of 4 tensors, 多尺度特征图

# UNetRDLDB 并联 ViTEncoderBlock
class ParallelUNetViTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, padding=3 ,patch_size=4, vit_depth=2, vit_heads=8, vit_dim_head=64, vit_mlp_ratio=4.0, vit_dropout=0.1, use_bn=True):
        super().__init__()
        # UNetRDLDB分支
        self.unet_branch = UNetRDLEB(
            in_channels, out_channels, kernel_size, padding, use_bn=True
        )
        # ViTEncoderBlock分支
        self.vit_branch = ViTEncoderBlock(
            in_channels, out_channels, patch_size=patch_size, depth=vit_depth,
            heads=vit_heads, dim_head=vit_dim_head, mlp_ratio=vit_mlp_ratio, dropout=vit_dropout
        )
        # 融合1x1卷积
        self.fuse_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        self.fuse_bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.fuse_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out_unet = self.unet_branch(x)
        out_vit = self.vit_branch(x)
        # 尺寸对齐（如果有必要）
        if out_unet.shape[2:] != out_vit.shape[2:]:
            out_vit = F.interpolate(out_vit, size=out_unet.shape[2:], mode='bilinear', align_corners=False)
        out = torch.cat([out_unet, out_vit], dim=1)
        out = self.fuse_conv(out)
        out = self.fuse_bn(out)
        out = self.fuse_relu(out)
        return out


# EFFUnet
class UNetSepConvEB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_bn=True):
        super().__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=False
        )
        # Pointwise convolution
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
    
# 普通卷积并联条形卷积
class ParallelConvRDLEBEncoderBlock(nn.Module):
    """
    并联普通卷积编码器和 UNetRDLEB 编码器，输出融合特征
    """
    def __init__(self, in_channels, out_channels, 
                 conv_kernel_size=3, conv_padding=1, 
                 rdleb_kernel_size=7, rdleb_padding=3, use_bn=True):
        super().__init__()
        # 普通卷积分支
        self.conv_branch = UNetEB(
            in_channels, out_channels, kernel_size=conv_kernel_size, padding=conv_padding, use_bn=use_bn
        )
        # UNetRDLEB分支
        self.rdleb_branch = UNetRDLEB(
            in_channels, out_channels, kernel_size=rdleb_kernel_size, padding=rdleb_padding, use_bn=use_bn
        )
        # 融合1x1卷积
        self.fuse_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        self.fuse_bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.fuse_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out_conv = self.conv_branch(x)
        out_rdleb = self.rdleb_branch(x)
        # 尺寸对齐（UNetRDLEB有下采样）
        if out_conv.shape[2:] != out_rdleb.shape[2:]:
            out_conv = F.max_pool2d(out_conv, kernel_size=2, stride=2)
        out = torch.cat([out_conv, out_rdleb], dim=1)
        out = self.fuse_conv(out)
        out = self.fuse_bn(out)
        out = self.fuse_relu(out)
        return out 
    
    
class ParallelConvRDLEBEncoderBlock_no_deform(nn.Module):
    """
    并联普通卷积编码器和 UNetRDLEB_no_deform 编码器，输出融合特征（无可变形卷积）
    """
    def __init__(self, in_channels, out_channels, 
                 conv_kernel_size=3, conv_padding=1, 
                 rdleb_kernel_size=7, rdleb_padding=3, use_bn=True):
        super().__init__()
        # 普通卷积分支
        self.conv_branch = UNetEB(
            in_channels, out_channels, kernel_size=conv_kernel_size, padding=conv_padding, use_bn=use_bn
        )
        # UNetRDLEB_no_deform分支
        self.rdleb_branch = UNetRDLEB_no_deform(
            in_channels, out_channels, kernel_size=rdleb_kernel_size, padding=rdleb_padding, use_bn=use_bn
        )
        # 融合1x1卷积
        self.fuse_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        self.fuse_bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.fuse_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out_conv = self.conv_branch(x)
        out_rdleb = self.rdleb_branch(x)
        # 尺寸对齐（UNetRDLEB_no_deform有下采样）
        if out_conv.shape[2:] != out_rdleb.shape[2:]:
            out_conv = F.max_pool2d(out_conv, kernel_size=2, stride=2)
        out = torch.cat([out_conv, out_rdleb], dim=1)
        out = self.fuse_conv(out)
        out = self.fuse_bn(out)
        out = self.fuse_relu(out)
        return out


#门控融合双分支编码器块
class GatedParallelConvRDLEBEncoderBlock(nn.Module):
    """
    并联普通卷积编码器和 UNetRDLEB 编码器，门控融合特征
    """
    def __init__(self, in_channels, out_channels, 
                 conv_kernel_size=3, conv_padding=1, 
                 rdleb_kernel_size=7, rdleb_padding=3, use_bn=True):
        super().__init__()
        # 普通卷积分支
        self.conv_branch = UNetEB(
            in_channels, out_channels, kernel_size=conv_kernel_size, padding=conv_padding, use_bn=use_bn
        )
        # UNetRDLEB分支
        self.rdleb_branch = UNetRDLEB(
            in_channels, out_channels, kernel_size=rdleb_kernel_size, padding=rdleb_padding, use_bn=use_bn
        )
        # 门控权重生成器（1x1卷积+sigmoid，输出通道数为 out_channels）
        self.gate_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        self.gate_sigmoid = nn.Sigmoid()
        self.fuse_bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.fuse_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out_conv = self.conv_branch(x)
        out_rdleb = self.rdleb_branch(x)
        # 尺寸对齐（UNetRDLEB有下采样）
        if out_conv.shape[2:] != out_rdleb.shape[2:]:
            out_conv = F.max_pool2d(out_conv, kernel_size=2, stride=2)
        # 拼接两个分支
        feat_cat = torch.cat([out_conv, out_rdleb], dim=1)
        # 生成门控权重
        gate = self.gate_sigmoid(self.gate_conv(feat_cat))
        # 门控融合
        out = gate * out_conv + (1 - gate) * out_rdleb
        out = self.fuse_bn(out)
        out = self.fuse_relu(out)
        return out


# 并联解码器
class ParallelConvRDLDBDecoderBlock(nn.Module):
    """
    并联普通解码器和 UNetRDLDB 解码器，支持跳跃连接，输出融合特征
    """
    def __init__(self, in_channels, out_channels, 
                 conv_kernel_size=3, conv_padding=1, 
                 rdldb_kernel_size=7, up_kernel=2, up_stride=2, up_padding=0, use_bn=True):
        super().__init__()
        # 普通解码分支（转置卷积+两层卷积）
        self.conv_branch = UNetDEB(
            in_channels, out_channels, kernel_size=conv_kernel_size, stride=up_stride, 
            padding=conv_padding, output_padding=up_padding, use_bn=use_bn
        )
        # UNetRDLDB分支
        self.rdldb_branch = UNetLDB(
            in_channels, out_channels, kernel_size=rdldb_kernel_size, up_kernel=up_kernel, 
            up_stride=up_stride, up_padding=up_padding, use_bn=use_bn
        )
        # 融合1x1卷积
        self.fuse_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        self.fuse_bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.fuse_relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        out_conv = self.conv_branch(x, skip)
        out_rdldb = self.rdldb_branch(x)
        # 尺寸对齐（如有必要）
        if out_conv.shape[2:] != out_rdldb.shape[2:]:
            out_rdldb = F.interpolate(out_rdldb, size=out_conv.shape[2:], mode='bilinear', align_corners=False)
        out = torch.cat([out_conv, out_rdldb], dim=1)
        out = self.fuse_conv(out)
        out = self.fuse_bn(out)
        out = self.fuse_relu(out)
        return out
    
    