#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Attention.py
@Time    :   2025/07/02 15:00:43
@Author  :   angkangyu 
@Description: 各种注意力机制的官方标准实现
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention for Efficient Mobile Network Design
    官方论文: https://arxiv.org/abs/2103.02907
    🔥 修复版本：确保输出通道数与输入通道数匹配
    """
    def __init__(self, inp, reduction=32):  # 🔥 修改：移除oup参数，输出通道数等于输入通道数
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))     # 对输入特征图在高度方向进行平均池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))     # 对输入特征图在宽度方向进行平均池化

        mip = max(8, inp // reduction)                    # 中间层通道数，至少为8，通常为输入通道数的1/32  

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)  # 1x1卷积， inp通道数到mip通道数
        self.bn1 = nn.BatchNorm2d(mip)          # 批归一化， 用于稳定训练过程
        self.act = nn.ReLU()                # 激活函数，通常使用ReLU

        # 🔥 修复：输出通道数应该与输入通道数相同
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)   # 对高度方向特征图进行1x1卷积， mip通道数到输入通道数
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)   # 对宽度方向特征图进行1x1卷积， mip通道数到输入通道数

    def forward(self, x):
        identity = x                 # 保留输入特征图                    
        n, c, h, w = x.size()       # 获取输入特征图的尺寸
        
        x_h = self.pool_h(x)        # 对输入特征图在高度方向进行平均池化 [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)        # 对输入特征图在宽度方向进行平均池化，并调整维度顺序 [B, C, W, 1]

        y = torch.cat([x_h, x_w], dim=2)  # 在空间维度上拼接两个池化结果，得到一个新的特征图，尺寸为 [B, C, H+W, 1]
        
        y = self.conv1(y)       # 1x1卷积降维 [B, mip, H+W, 1]
        y = self.bn1(y)         # 批归一化
        y = self.act(y)         # 激活函数

        x_h, x_w = torch.split(y, [h, w], dim=2)    # 将拼接后的特征图分割回高度和宽度方向的特征图
        x_w = x_w.permute(0, 1, 3, 2)               # 调整宽度特征图的维度顺序 

        a_h = self.conv_h(x_h).sigmoid()            # 对高度特征图进行1x1卷积并应用sigmoid激活 [B, C, H, 1]
        a_w = self.conv_w(x_w).sigmoid()            # 对宽度特征图进行1x1卷积并应用sigmoid激活 [B, C, 1, W]

        # 🔥 修复：现在所有张量都是[B, C, H, W]的形状，可以正确相乘
        out = identity * a_h * a_w                  # 将输入特征图与高度和宽度方向的注意力权重相乘

        return out


class SEAttention(nn.Module):
    """
    Squeeze-and-Excitation Networks
    官方论文: https://arxiv.org/abs/1709.01507
    """
    def __init__(self, channel, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    """
    CBAM的Channel Attention部分
    官方论文: https://arxiv.org/abs/1807.06521
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    CBAM的Spatial Attention部分
    官方论文: https://arxiv.org/abs/1807.06521
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    官方论文: https://arxiv.org/abs/1807.06521
    官方代码: https://github.com/Jongchan/attention-module
    """
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class ECAAttention(nn.Module):
    """
    Efficient Channel Attention for Deep Convolutional Neural Networks
    官方论文: https://arxiv.org/abs/1910.03151
    官方代码: https://github.com/BangguWu/ECANet
    """
    def __init__(self, channel, gamma=2, b=1):
        super(ECAAttention, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        
        # Global average pooling
        y = self.avg_pool(x).view(b, 1, c)
        
        # Two different branches of ECA module
        y = self.conv(y)
        
        # Multi-scale information fusion
        y = self.sigmoid(y).view(b, c, 1, 1)
        
        return x * y.expand_as(x)


class SimAM(nn.Module):
    """
    SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks
    官方论文: https://proceedings.mlr.press/v139/yang21o.html
    官方代码: https://github.com/ZjjConan/SimAM
    """
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()
        
        n = w * h - 1
        
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


class SerialAttention(nn.Module):
    """
    串联注意力机制：(a)CBAM → Re-weight → (b)CoordinateAttention → Re-weight → (c)SE → Re-weight → Output
    """
    def __init__(self, channels, reduction_cbam=16, reduction_se=16, reduction_coord=32, kernel_size_cbam=7):
        super(SerialAttention, self).__init__()
        # (a) CBAM
        self.cbam = CBAM(channels, ratio=reduction_cbam, kernel_size=kernel_size_cbam)
        # (b) Coordinate Attention
        self.coord = CoordinateAttention(channels, channels, reduction=reduction_coord)
        # (c) SE
        self.se = SEAttention(channels, reduction=reduction_se)

    def forward(self, x):
        # (a) CBAM
        x1 = x * self.cbam(x)         # Re-weight after CBAM
        # (b) Coordinate Attention
        x2= self.coord(x1)        # Coordinate Attention
        # (c) SE
        x3 = x1 * self.se(x2)           # Re-weight after SE
        return x3
    
    
    


class SelfAttention(nn.Module):
    """
    自注意力机制模块
    """
    def __init__(self, in_dim, reduction=8):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.reduction = reduction
        
        # 生成Query, Key, Value的卷积层
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//reduction, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//reduction, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs:
            x : input feature maps (B X C X H X W)
        returns:
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, height, width = x.size()
        
        # 生成Query, Key, Value
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)  # B X (H*W) X C//reduction
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)  # B X C//reduction X (H*W)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)  # B X C X (H*W)

        # 计算注意力分数
        energy = torch.bmm(proj_query, proj_key)  # B X (H*W) X (H*W)
        attention = self.softmax(energy)  # B X (H*W) X (H*W)
        
        # 应用注意力权重
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B X C X (H*W)
        out = out.view(m_batchsize, C, height, width)  # B X C X H X W
        
        # 残差连接
        out = self.gamma * out + x
        return out


class SelfSEAttention(nn.Module):
    """
    自注意力机制和SE注意力机制结合的复合注意力模块
    先通过自注意力建模空间长程依赖，再通过SE注意力进行通道重要性加权
    """
    def __init__(self, channels, self_reduction=8, se_reduction=16):
        super(SelfSEAttention, self).__init__()
        # 自注意力模块
        self.self_attention = SelfAttention(channels, reduction=self_reduction)
        # SE注意力模块
        self.se_attention = SEAttention(channels, reduction=se_reduction)
        
        # 可选的融合权重参数
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # 先应用自注意力机制
        self_out = self.self_attention(x)
        
        # 再应用SE注意力机制
        se_out = self.se_attention(self_out)
        
        # 加权融合原始输入和两种注意力的输出
        out = self.alpha * self_out + self.beta * se_out
        
        return out


class ParallelSelfSEAttention(nn.Module):
    """
    并行版本的自注意力和SE注意力结合模块
    同时计算自注意力和SE注意力，然后进行特征融合
    """
    def __init__(self, channels, self_reduction=8, se_reduction=16, fusion_mode='add'):
        super(ParallelSelfSEAttention, self).__init__()
        # 自注意力模块
        self.self_attention = SelfAttention(channels, reduction=self_reduction)
        # SE注意力模块  
        self.se_attention = SEAttention(channels, reduction=se_reduction)
        
        self.fusion_mode = fusion_mode
        
        if fusion_mode == 'concat':
            # 如果使用拼接融合，需要额外的1x1卷积降维
            self.fusion_conv = nn.Conv2d(channels * 2, channels, kernel_size=1)
        elif fusion_mode == 'weighted':
            # 可学习的融合权重
            self.self_weight = nn.Parameter(torch.ones(1))
            self.se_weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # 并行计算自注意力和SE注意力
        self_out = self.self_attention(x)
        se_out = self.se_attention(x)
        
        if self.fusion_mode == 'add':
            # 直接相加融合
            out = self_out + se_out
        elif self.fusion_mode == 'multiply':
            # 逐元素相乘融合
            out = self_out * se_out
        elif self.fusion_mode == 'concat':
            # 拼接后通过1x1卷积融合
            concat_out = torch.cat([self_out, se_out], dim=1)
            out = self.fusion_conv(concat_out)
        elif self.fusion_mode == 'weighted':
            # 可学习权重融合
            out = self.self_weight * self_out + self.se_weight * se_out
        else:
            raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")
            
        return out

class CrossFusionAttention(nn.Module):
    """
    自注意力和SE注意力交叉融合模块
    在计算过程中就进行特征交换和融合
    """
    def __init__(self, channels, reduction=8):
        super(CrossFusionAttention, self).__init__()
        self.channels = channels
        self.reduction = reduction
        
        # 共享的特征提取
        self.query_conv = nn.Conv2d(channels, channels//reduction, 1)
        self.key_conv = nn.Conv2d(channels, channels//reduction, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        
        # SE分支的全局池化和FC层
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc = nn.Sequential(
            nn.Linear(channels, channels//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction, channels, bias=False),
        )
        
        # 交叉融合层
        self.cross_conv = nn.Conv2d(channels * 2, channels, 1)
        self.fusion_conv = nn.Conv2d(channels, channels, 3, padding=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()
        
        # 1. 自注意力分支
        query = self.query_conv(x).view(B, -1, H*W).permute(0, 2, 1)  # [B, HW, C//r]
        key = self.key_conv(x).view(B, -1, H*W)                       # [B, C//r, HW] 
        value = self.value_conv(x).view(B, -1, H*W)                   # [B, C, HW]
        
        attention = self.softmax(torch.bmm(query, key))               # [B, HW, HW]
        self_out = torch.bmm(value, attention.permute(0, 2, 1))       # [B, C, HW]
        self_out = self_out.view(B, C, H, W)
        
        # 2. SE分支 - 但使用自注意力的value特征
        se_input = value.view(B, C, H, W)  # 关键：使用自注意力的value特征
        se_weight = self.global_pool(se_input).view(B, C)
        se_weight = self.sigmoid(self.se_fc(se_weight)).view(B, C, 1, 1)
        se_out = se_input * se_weight
        
        # 3. 交叉融合
        cross_feat = torch.cat([self_out, se_out], dim=1)  # [B, 2C, H, W]
        cross_feat = self.cross_conv(cross_feat)           # [B, C, H, W]
        
        # 4. 最终融合
        fused = self.fusion_conv(cross_feat)
        output = x + self.gamma * self_out + self.beta * fused
        
        return output
    
    
    
class GatedSelfSEAttention(nn.Module):
    """
    使用门控机制动态选择自注意力和SE注意力的贡献
    """
    def __init__(self, channels, reduction=8):
        super(GatedSelfSEAttention, self).__init__()
        self.channels = channels
        
        # 共享特征提取层
        self.shared_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 自注意力分支
        self.query_conv = nn.Conv2d(channels, channels//reduction, 1)
        self.key_conv = nn.Conv2d(channels, channels//reduction, 1) 
        self.value_conv = nn.Conv2d(channels, channels, 1)
        
        # SE分支
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc = nn.Sequential(
            nn.Linear(channels, channels//reduction),
            nn.ReLU(),
            nn.Linear(channels//reduction, channels),
        )
        
        # 门控网络 - 决定两个分支的权重
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels, channels//4, 3, padding=1),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(),
            nn.Conv2d(channels//4, 2, 1),  # 输出2个通道：self_weight, se_weight
            nn.Softmax(dim=1)
        )
        
        # 特征增强
        self.enhance_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        
        # 共享特征提取
        shared_feat = self.relu(self.bn(self.shared_conv(x)))
        
        # 自注意力计算
        query = self.query_conv(shared_feat).view(B, -1, H*W).permute(0, 2, 1)
        key = self.key_conv(shared_feat).view(B, -1, H*W)
        value = self.value_conv(shared_feat).view(B, -1, H*W)
        
        attention = self.softmax(torch.bmm(query, key))
        self_out = torch.bmm(value, attention.permute(0, 2, 1)).view(B, C, H, W)
        
        # SE注意力计算
        se_weight = self.global_pool(shared_feat).view(B, C)
        se_weight = torch.sigmoid(self.se_fc(se_weight)).view(B, C, 1, 1)
        se_out = shared_feat * se_weight
        
        # 门控权重计算
        gate_weights = self.gate_conv(shared_feat)  # [B, 2, H, W]
        self_gate = gate_weights[:, 0:1, :, :]      # [B, 1, H, W]
        se_gate = gate_weights[:, 1:2, :, :]        # [B, 1, H, W]
        
        # 门控融合
        gated_out = self_gate * self_out + se_gate * se_out
        
        # 特征增强
        enhanced_out = self.enhance_conv(gated_out)
        
        # 残差连接
        output = x + self.gamma * enhanced_out
        
        return output
    
    
    
class CA2(nn.Module):
    """
    魔改版坐标注意力机制
    改进点：
    1. 条形卷积替代H/W全局平均池化
    2. 多尺度条形卷积
    3. 轻量通道注意力增强
    4. 先CA再门控融合
    """
    def __init__(self, inp, oup, reduction=32, strip_kernels=[3, 5, 7], use_channel_attention=True):
        super(CA2, self).__init__()
        self.inp = inp
        self.oup = oup
        self.strip_kernels = strip_kernels
        self.use_channel_attention = use_channel_attention
        
        # 改进点1&2: 多尺度条形卷积替代全局平均池化
        self.multi_scale_h_convs = nn.ModuleList([
            nn.Conv2d(inp, inp, kernel_size=(1, k), padding=(0, k//2), groups=inp)
            for k in strip_kernels
        ])
        
        self.multi_scale_w_convs = nn.ModuleList([
            nn.Conv2d(inp, inp, kernel_size=(k, 1), padding=(k//2, 0), groups=inp)
            for k in strip_kernels
        ])
        
        # 多尺度特征融合
        self.h_fusion_conv = nn.Conv2d(inp * len(strip_kernels), inp, 1)
        self.w_fusion_conv = nn.Conv2d(inp * len(strip_kernels), inp, 1)
        
        # 原始的池化操作作为补充
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        # 中间层通道数
        mip = max(8, inp // reduction)
        
        # 共享的1x1卷积和归一化
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU()
        
        # 方向特征生成
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
        # 改进点3: 轻量通道注意力
        if use_channel_attention:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(inp, inp // 16, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(inp // 16, inp, 1, bias=False),
                nn.Sigmoid()
            )
        
        # 改进点4: 门控融合机制
        self.gate_conv = nn.Sequential(
            nn.Conv2d(inp, inp // 4, 3, padding=1),
            nn.BatchNorm2d(inp // 4),
            nn.ReLU(),
            nn.Conv2d(inp // 4, 3, 1),  # 3个门控权重：原始、CA_h、CA_w
            nn.Softmax(dim=1)
        )
        
        # 特征增强层
        self.enhance_conv = nn.Sequential(
            nn.Conv2d(inp, inp, 3, padding=1),
            nn.BatchNorm2d(inp),
            nn.ReLU()
        )

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # 改进点3: 先应用通道注意力
        if self.use_channel_attention:
            channel_weight = self.channel_attention(x)
            x_enhanced = x * channel_weight
        else:
            x_enhanced = x
        
        # 改进点1&2: 多尺度条形卷积特征提取
        # H方向多尺度特征
        h_features = []
        for h_conv in self.multi_scale_h_convs:
            h_feat = h_conv(x_enhanced)  # [B, C, H, W]
            h_feat = F.adaptive_avg_pool2d(h_feat, (None, 1))  # [B, C, H, 1]
            h_features.append(h_feat)
        
        # W方向多尺度特征
        w_features = []
        for w_conv in self.multi_scale_w_convs:
            w_feat = w_conv(x_enhanced)  # [B, C, H, W]
            w_feat = F.adaptive_avg_pool2d(w_feat, (1, None))  # [B, C, 1, W]
            w_features.append(w_feat)
        
        # 融合多尺度特征
        x_h = torch.cat(h_features, dim=1)  # [B, C*scales, H, 1]
        x_h = self.h_fusion_conv(x_h)       # [B, C, H, 1]
        
        x_w = torch.cat(w_features, dim=1)  # [B, C*scales, 1, W]
        x_w = self.w_fusion_conv(x_w)       # [B, C, 1, W]
        
        # 结合原始池化特征（保持多样性）
        x_h_pool = self.pool_h(x_enhanced)
        x_w_pool = self.pool_w(x_enhanced)
        
        # 特征融合
        x_h = 0.7 * x_h + 0.3 * x_h_pool
        x_w = 0.7 * x_w + 0.3 * x_w_pool
        
        # 调整维度并拼接
        x_w = x_w.permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)  # [B, C, H+W, 1]
        
        # 共享的特征变换
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # 分离H和W方向特征
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        # 生成注意力权重
        a_h = self.conv_h(x_h).sigmoid()  # [B, C, H, 1]
        a_w = self.conv_w(x_w).sigmoid()  # [B, C, 1, W]
        
        # 改进点4: 门控融合策略
        # 计算CA增强后的特征
        ca_enhanced = identity * a_w * a_h
        
        # 计算门控权重
        gate_weights = self.gate_conv(identity)  # [B, 3, H, W]
        identity_gate = gate_weights[:, 0:1, :, :]    # 原始特征权重
        ca_h_gate = gate_weights[:, 1:2, :, :]        # CA_H权重  
        ca_w_gate = gate_weights[:, 2:3, :, :]        # CA_W权重
        
        # 门控融合
        gated_output = (identity_gate * identity + 
                       ca_h_gate * (identity * a_h) + 
                       ca_w_gate * (identity * a_w))
        
        # 最终特征增强
        enhanced_output = self.enhance_conv(gated_output)
        
        # 残差连接
        final_output = identity + enhanced_output
        
        return final_output

class CA3(nn.Module):
    """
    轻量级版本的增强坐标注意力
    针对移动端和资源受限环境优化
    """
    def __init__(self, inp, oup, reduction=32, strip_kernel=5):
        super(CA3, self).__init__()
        
        # 单一尺度条形卷积（减少参数）
        self.h_conv = nn.Conv2d(inp, inp, kernel_size=(1, strip_kernel), 
                               padding=(0, strip_kernel//2), groups=inp)
        self.w_conv = nn.Conv2d(inp, inp, kernel_size=(strip_kernel, 1), 
                               padding=(strip_kernel//2, 0), groups=inp)
        
        # 池化补充
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        # 轻量化MLP
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, 1, bias=False)
        self.act = nn.ReLU()
        
        self.conv_h = nn.Conv2d(mip, oup, 1, bias=False)
        self.conv_w = nn.Conv2d(mip, oup, 1, bias=False)
        
        # 简化的通道注意力
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inp, inp, 1, groups=inp, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # 轻量通道注意力
        channel_weight = self.channel_gate(x)
        x = x * channel_weight
        
        # 条形卷积 + 池化融合
        x_h = 0.6 * self.h_conv(x) + 0.4 * self.pool_h(x)
        x_w = 0.6 * self.w_conv(x) + 0.4 * self.pool_w(x)
        
        # 维度调整和拼接
        x_h = F.adaptive_avg_pool2d(x_h, (None, 1))
        x_w = F.adaptive_avg_pool2d(x_w, (1, None))
        x_w = x_w.permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        
        # 特征变换
        y = self.act(self.conv1(y))
        
        # 分离并生成权重
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        # 直接应用权重
        out = identity * a_w * a_h
        
        return out