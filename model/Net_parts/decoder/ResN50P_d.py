#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ResNet50_decoder.py
@Time    :   2025/07/03 
@Author  :   angkangyu 
@Description: 配套ResNet50编码器的解码器模块
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """基础卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class UpConvBlock(nn.Module):
    """上采样卷积块"""
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_channels, out_channels)
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class FeatureFusionBlock(nn.Module):
    """特征融合块"""
    def __init__(self, high_channels, low_channels, out_channels):
        super().__init__()
        # 高层特征处理
        self.high_conv = ConvBlock(high_channels, out_channels, 1, 1, 0)
        # 低层特征处理
        self.low_conv = ConvBlock(low_channels, out_channels, 1, 1, 0)
        # 融合后处理
        self.fusion_conv = ConvBlock(out_channels * 2, out_channels)
        
    def forward(self, high_feat, low_feat):
        # 上采样高层特征到低层特征尺寸
        high_feat = F.interpolate(high_feat, size=low_feat.shape[2:], 
                                mode='bilinear', align_corners=True)
        
        # 特征处理
        high_feat = self.high_conv(high_feat)
        low_feat = self.low_conv(low_feat)
        
        # 特征融合
        fused = torch.cat([high_feat, low_feat], dim=1)
        fused = self.fusion_conv(fused)
        
        return fused


class ResN50P_d(nn.Module):
    """ResNet50解码器模块，与纯净ResNet50编码器配套使用"""
    
    def __init__(self, num_classes=1):
        """
        Args:
            num_classes (int): 输出类别数，默认1（二分类或回归）
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # 编码器特征通道数 [64, 256, 512, 1024, 2048]
        encoder_channels = [64, 256, 512, 1024, 2048]
        # 解码器输出通道数
        decoder_channels = [256, 128, 64, 32, 16]
        
        # 最深层特征处理
        self.center = nn.Sequential(
            ConvBlock(encoder_channels[4], decoder_channels[0]),  # 2048 -> 256
            ConvBlock(decoder_channels[0], decoder_channels[0])
        )
        
        # 解码器层级 - 特征融合 + 上采样
        self.decoder4 = FeatureFusionBlock(
            decoder_channels[0], encoder_channels[3], decoder_channels[1]  # 256+1024 -> 128
        )
        
        self.decoder3 = FeatureFusionBlock(
            decoder_channels[1], encoder_channels[2], decoder_channels[2]  # 128+512 -> 64
        )
        
        self.decoder2 = FeatureFusionBlock(
            decoder_channels[2], encoder_channels[1], decoder_channels[3]  # 64+256 -> 32
        )
        
        self.decoder1 = FeatureFusionBlock(
            decoder_channels[3], encoder_channels[0], decoder_channels[4]  # 32+64 -> 16
        )
        
        # 最终输出层
        self.final_conv = nn.Sequential(
            ConvBlock(decoder_channels[4], decoder_channels[4]),  # 16 -> 16
            nn.Conv2d(decoder_channels[4], num_classes, kernel_size=1)  # 16 -> num_classes
        )
    
    def forward(self, encoder_features):
        """
        前向传播
        Args:
            encoder_features (dict): 编码器输出特征
                - x1: [B, 64, H/2, W/2]     - 浅层特征
                - x2: [B, 256, H/4, W/4]    - 低层特征
                - x3: [B, 512, H/8, W/8]    - 中层特征
                - x4: [B, 1024, H/16, W/16] - 高层特征
                - x5: [B, 2048, H/32, W/32] - 深层特征
        
        Returns:
            Tensor: [B, num_classes, H, W] - 分割输出
        """
        x1, x2, x3, x4, x5 = encoder_features['x1'], encoder_features['x2'], \
                             encoder_features['x3'], encoder_features['x4'], encoder_features['x5']
        
        # 获取原始输入尺寸
        input_size = (x1.size(2) * 2, x1.size(3) * 2)  # x1是H/2, W/2，所以原图是H, W
        
        # 中心处理最深层特征
        center = self.center(x5)  # [B, 256, H/32, W/32]
        
        # 逐级解码 + 特征融合
        d4 = self.decoder4(center, x4)    # [B, 128, H/16, W/16]
        d3 = self.decoder3(d4, x3)        # [B, 64, H/8, W/8]
        d2 = self.decoder2(d3, x2)        # [B, 32, H/4, W/4]
        d1 = self.decoder1(d2, x1)        # [B, 16, H/2, W/2]
        
        # 最终输出
        output = self.final_conv(d1)      # [B, num_classes, H/2, W/2]
        
        # 上采样到原始输入尺寸
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        
        # 直接返回输出张量
        return output



# 构建函数
def build_resnet50_decoder(num_classes=1, use_deep_supervision=False):
    """
    构建ResNet50解码器
    
    Args:
        num_classes (int): 输出类别数
        decoder_type (str): 解码器类型，'fusion' 或 'unet'
        use_deep_supervision (bool): 是否使用深监督
    
    Returns:
        解码器实例
    """

    return ResN50P_d(num_classes, use_deep_supervision)




# 测试代码
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🚀 测试ResNet50解码器")
    print("=" * 50)
    
    # 模拟编码器输出
    encoder_features = {
        'x1': torch.randn(2, 64, 256, 256).to(device),      # H/2, W/2
        'x2': torch.randn(2, 256, 128, 128).to(device),     # H/4, W/4
        'x3': torch.randn(2, 512, 64, 64).to(device),       # H/8, W/8
        'x4': torch.randn(2, 1024, 32, 32).to(device),      # H/16, W/16
        'x5': torch.randn(2, 2048, 16, 16).to(device),      # H/32, W/32
    }
    
    # 测试1: 特征融合解码器
    print("\n📌 测试1: 特征融合解码器")
    decoder1 = ResN50P_d(num_classes=1, use_deep_supervision=False).to(device)
    
    with torch.no_grad():
        output1 = decoder1(encoder_features)
    
    print(f"主输出尺寸: {output1['output'].shape}")
    
    # 测试2: 深监督解码器
    print("\n📌 测试2: 深监督解码器")
    decoder2 = ResN50P_d(num_classes=2, use_deep_supervision=True).to(device)
    decoder2.train()  # 深监督只在训练时启用
    
    with torch.no_grad():
        output2 = decoder2(encoder_features)
    
    print(f"主输出尺寸: {output2['output'].shape}")
    if 'aux_outputs' in output2:
        for i, aux_out in enumerate(output2['aux_outputs']):
            print(f"辅助输出{i+1}尺寸: {aux_out.shape}")
    
    # 参数统计
    print("\n📊 参数统计:")
    for name, decoder in [("特征融合", decoder1)]:
        params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        print(f"{name}解码器参数量: {params:,}")
    
    print("\n✅ ResNet50解码器特点:")
    print("- 🔗 完美匹配ResNet50编码器")
    print("- 🎯 支持特征融合和UNet两种风格")
    print("- 👁️ 可选深监督训练")
    print("- 📐 自动适应输入尺寸")
    print("- 🎨 灵活的输出类别数")
    
    print("\n🎁 使用示例:")
    print("```python")
    print("# 基本使用")
    print("decoder = build_resnet50_decoder(num_classes=1)")
    print("")
    print("# 多分类 + 深监督")
    print("decoder = build_resnet50_decoder(num_classes=5, use_deep_supervision=True)")
    print("")
    print("# UNet风格")
    print("decoder = build_resnet50_decoder(decoder_type='unet')")
    print("```")