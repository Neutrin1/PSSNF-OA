import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models import resnet34
from .resnet import *

class DynamicSnakeConv(nn.Module):
    """动态蛇形卷积(DSC)模块，用于提取道路形状特征"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DynamicSnakeConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DSBlock(nn.Module):
    """DSC块，包含多个动态蛇形卷积"""
    def __init__(self, in_channels, out_channels, num_convs=2):
        super(DSBlock, self).__init__()
        layers = []
        for i in range(num_convs):
            if i == 0:
                layers.append(DynamicSnakeConv(in_channels, out_channels))
            else:
                layers.append(DynamicSnakeConv(out_channels, out_channels))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 确保reduction不会导致通道数为0
        reduction = min(reduction, in_channels)
        if in_channels // reduction == 0:
            reduction = in_channels
            
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

class MultigatedInformativeSelfAttention(nn.Module):
    """多门控信息自注意力模块(MGSA) - 修复NaN问题"""
    def __init__(self, channels, alpha_init=0.0):
        super(MultigatedInformativeSelfAttention, self).__init__()
        self.alpha_r = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_d = nn.Parameter(torch.tensor(alpha_init))
        self.channel_attn = ChannelAttention(channels)
        self.spatial_attn = SpatialAttention()
        
        # 修复：在__init__中定义卷积层
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(2*channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 添加数值稳定性参数
        self.eps = 1e-8
        
    def forward(self, fr, fd):
        batch_size, channels, height, width = fr.size()
        
        # 重塑特征图为矩阵形式 [B,C,HW]
        fr_flat = fr.view(batch_size, channels, -1)
        fd_flat = fd.view(batch_size, channels, -1)
        
        # 添加L2归一化防止数值爆炸
        fr_flat = F.normalize(fr_flat, p=2, dim=2, eps=self.eps)
        fd_flat = F.normalize(fd_flat, p=2, dim=2, eps=self.eps)
        
        # 计算通道亲和矩阵 - 添加温度缩放
        temperature = math.sqrt(channels)
        Pr = torch.bmm(fr_flat, fr_flat.transpose(1, 2)) / temperature
        Pr = F.softmax(Pr, dim=-1)
        
        Pd = torch.bmm(fd_flat, fd_flat.transpose(1, 2)) / temperature
        Pd = F.softmax(Pd, dim=-1)
        
        # 互导机制: 亲和矩阵逐元素相乘
        Pmix = Pr * Pd
        
        # 限制alpha参数的范围
        alpha_r_clamped = torch.clamp(self.alpha_r, -1.0, 1.0)
        alpha_d_clamped = torch.clamp(self.alpha_d, -1.0, 1.0)
        
        # 应用亲和矩阵增强特征
        fr_enhanced_flat = fr_flat + alpha_r_clamped * torch.bmm(Pmix.transpose(1, 2), fr_flat)
        fd_enhanced_flat = fd_flat + alpha_d_clamped * torch.bmm(Pmix.transpose(1, 2), fd_flat)
        
        fr_enhanced = fr_enhanced_flat.view(batch_size, channels, height, width)
        fd_enhanced = fd_enhanced_flat.view(batch_size, channels, height, width)
        
        # 拼接增强后的特征并通过卷积恢复通道数
        combined = torch.cat([fr_enhanced, fd_enhanced], dim=1)
        combined = self.conv_fusion(combined)
        
        # 空间门控和信息门控
        spa_attn = self.spatial_attn(combined)
        info_attn = self.channel_attn(combined)
        
        spa_gated = spa_attn * combined
        info_gated = info_attn * combined
        
        return spa_gated + info_gated

class CRFEBranch(nn.Module):
    """CRFE模块的分支"""
    def __init__(self, in_channels, out_channels, dilation):
        super(CRFEBranch, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                      dilation=dilation, padding=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

class CascadeReceptiveFieldEnhancement(nn.Module):
    """级联感受野增强模块(CRFE)"""
    def __init__(self, in_channels, out_channels, dilations=[1, 4, 8, 12]):
        super(CascadeReceptiveFieldEnhancement, self).__init__()
        self.dilations = dilations
        self.branches = nn.ModuleList()
        
        # 第一个分支
        self.branches.append(CRFEBranch(in_channels, out_channels, dilations[0]))
        
        # 后续分支
        for i in range(1, len(dilations)):
            self.branches.append(CRFEBranch(in_channels + i*out_channels, out_channels, dilations[i]))
        
        # 修复：输出通道数应该是所有分支的总和
        total_out_channels = len(dilations) * out_channels
        self.final_conv = nn.Sequential(
            nn.Conv2d(total_out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        results = []
        branch_input = x
        
        # 处理每个分支
        for branch in self.branches:
            out = branch(branch_input)
            results.append(out)
            branch_input = torch.cat([branch_input, out], dim=1)
        
        # 拼接所有分支结果并降维
        out = torch.cat(results, dim=1)
        out = self.final_conv(out)
        
        # 残差连接
        residual = self.residual(x)
        out = self.relu(out + residual)
        
        return out

class FeatureFusionUnit(nn.Module):
    """特征融合单元(FFU)"""
    def __init__(self, in_channels_list, out_channels):
        super(FeatureFusionUnit, self).__init__()
        total_channels = sum(in_channels_list)
        self.conv = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.channel_attn = ChannelAttention(out_channels)
        
    def forward(self, *features):
        # 拼接多尺度特征
        combined = torch.cat(features, dim=1)
        # 通道注意力融合
        out = self.conv(combined)
        attn = self.channel_attn(out)
        out = out * attn
        return out

class DEGANet(nn.Module):
    """DEGANet主网络结构"""
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(DEGANet, self).__init__()
        
        # ResNet34编码器 - 使用预训练权重
        resnet = resnet34(pretrained=True)
        
        # 如果输入通道数不是3，需要修改第一层
        if in_channels != 3:
            original_conv1_weight = resnet.conv1.weight.data
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if in_channels < 3:
                resnet.conv1.weight.data = original_conv1_weight[:, :in_channels, :, :]
            else:
                nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        # ResNet编码器各层
        self.resnet_encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.resnet_encoder2 = resnet.layer1  # 64 channels
        self.resnet_encoder3 = resnet.layer2  # 128 channels  
        self.resnet_encoder4 = resnet.layer3  # 256 channels
        self.resnet_encoder5 = resnet.layer4  # 512 channels
        
        # DSC编码器 - 需要处理尺寸匹配
        self.dsc_initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.dsc_encoder1 = DSBlock(64, 64)
        self.dsc_encoder2 = nn.Sequential(
            nn.MaxPool2d(2),  # 下采样匹配ResNet
            DSBlock(64, 128)
        )
        self.dsc_encoder3 = nn.Sequential(
            nn.MaxPool2d(2),  # 下采样匹配ResNet
            DSBlock(128, 256)
        )
        self.dsc_encoder4 = nn.Sequential(
            nn.MaxPool2d(2),  # 下采样匹配ResNet
            DSBlock(256, 512)
        )
        
        # MGSA模块
        self.mgsa1 = MultigatedInformativeSelfAttention(64)
        self.mgsa2 = MultigatedInformativeSelfAttention(128)
        self.mgsa3 = MultigatedInformativeSelfAttention(256)
        self.mgsa4 = MultigatedInformativeSelfAttention(512)
        
        # CRFE模块
        self.crfe1 = CascadeReceptiveFieldEnhancement(64, 64)
        self.crfe2 = CascadeReceptiveFieldEnhancement(128, 128)
        self.crfe3 = CascadeReceptiveFieldEnhancement(256, 256)
        self.crfe4 = CascadeReceptiveFieldEnhancement(512, 512)
        
        # FFU模块 - 需要处理尺寸不匹配问题
        self.ffu = FeatureFusionUnit([64, 128, 256], 64)
        
        # 解码器
        self.decoder4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256+256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128+128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.decoder1 = nn.Conv2d(64+64, out_channels, kernel_size=1)
        
        # 添加权重初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """权重初始化"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # ResNet编码器路径
        x_resnet = self.resnet_encoder1(x)  # [B, 64, H/4, W/4]
        e1_r = self.resnet_encoder2(x_resnet)  # [B, 64, H/4, W/4]
        e2_r = self.resnet_encoder3(e1_r)  # [B, 128, H/8, W/8]
        e3_r = self.resnet_encoder4(e2_r)  # [B, 256, H/16, W/16]
        e4_r = self.resnet_encoder5(e3_r)  # [B, 512, H/32, W/32]
        
        # DSC编码器路径
        x_dsc = self.dsc_initial(x)  # [B, 64, H/4, W/4]
        e1_d = self.dsc_encoder1(x_dsc)  # [B, 64, H/4, W/4]
        e2_d = self.dsc_encoder2(e1_d)  # [B, 128, H/8, W/8]
        e3_d = self.dsc_encoder3(e2_d)  # [B, 256, H/16, W/16]
        e4_d = self.dsc_encoder4(e3_d)  # [B, 512, H/32, W/32]
        
        # MGSA模块处理
        m1 = self.mgsa1(e1_r, e1_d)  
        m2 = self.mgsa2(e2_r, e2_d)  
        m3 = self.mgsa3(e3_r, e3_d)  
        m4 = self.mgsa4(e4_r, e4_d)  
        
        # CRFE模块处理
        c1 = self.crfe1(m1)  
        c2 = self.crfe2(m2)  
        c3 = self.crfe3(m3)  
        c4 = self.crfe4(m4)  
        
        # FFU模块融合 - 需要上采样到相同尺寸
        c2_up = F.interpolate(c2, size=c1.shape[2:], mode='bilinear', align_corners=True)
        c3_up = F.interpolate(c3, size=c1.shape[2:], mode='bilinear', align_corners=True)
        ffu_out = self.ffu(c1, c2_up, c3_up)
        
        # 解码器
        d4 = self.decoder4(c4)  
        d4 = torch.cat([d4, c3], dim=1)  
        
        d3 = self.decoder3(d4)  
        d3 = torch.cat([d3, c2], dim=1)  
        
        d2 = self.decoder2(d3)  
        d2 = torch.cat([d2, ffu_out], dim=1)  
        
        d1 = self.decoder1(d2)  
        
        # 最终上采样到输入尺寸
        out = F.interpolate(d1, scale_factor=4, mode='bilinear', align_corners=True)
        
        return out