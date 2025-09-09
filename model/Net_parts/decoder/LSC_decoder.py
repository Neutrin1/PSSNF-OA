#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   LSC_decoder.py
@Time    :   2025/07/02 
@Author  :   angkangyu 
@Description: LSC解码器模块，匹配UNet编码器输出格式
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, BatchNorm, inp=False):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn1 = BatchNorm(in_channels // 4)
        self.relu1 = nn.ReLU()
        self.inp = inp

        # 条形卷积 - LSC核心组件
        self.deconv1 = nn.Conv2d(in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4))
        self.deconv2 = nn.Conv2d(in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0))
        self.deconv3 = nn.Conv2d(in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0))
        self.deconv4 = nn.Conv2d(in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4))

        self.bn2 = BatchNorm(in_channels // 2)  # 4个分支拼接后的通道数
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels // 2, n_filters, 1)
        self.bn3 = BatchNorm(n_filters)
        self.relu3 = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # 4个方向的条形卷积
        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)  # 拼接4个分支
        
        if self.inp:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)


class LSCDecoder(nn.Module):
    """
    LSC解码器 - 匹配UNet_LSCA_SS2_e编码器输出格式
    输入格式: (x5, [x4, x3, fused_x2, fused_x1])
    """
    def __init__(self, num_classes, BatchNorm=nn.BatchNorm2d, bilinear=False):
        super(LSCDecoder, self).__init__()
        self.bilinear = bilinear
        
        factor = 2 if bilinear else 1
        
        # 解码器块 - 匹配UNet跳跃连接通道数
        self.decoder4 = DecoderBlock(1024 // factor, 512, BatchNorm)    # x5: 1024->512
        self.decoder3 = DecoderBlock(1024, 256, BatchNorm)              # 512+512->256
        self.decoder2 = DecoderBlock(512, 128, BatchNorm, inp=True)     # 256+256->128  
        self.decoder1 = DecoderBlock(256, 64, BatchNorm, inp=True)      # 128+128->64
        
        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),  # 64+64->64
            BatchNorm(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            BatchNorm(32),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, 1)
        )

        self._init_weight()

    def forward(self, encoder_features):
        """
        Args:
            encoder_features: (x5, [x4, x3, fused_x2, fused_x1])
                - x5: [B, 1024, H/16, W/16] 底部特征
                - x4: [B, 512, H/8, W/8]   深层特征
                - x3: [B, 256, H/4, W/4]   中层特征  
                - fused_x2: [B, 128, H/2, W/2] 融合特征
                - fused_x1: [B, 64, H, W]      融合特征
        
        Returns:
            output: [B, num_classes, H, W]
        """
        # 解析编码器输出
        x5, skip_connections = encoder_features
        x4, x3, fused_x2, fused_x1 = skip_connections
        
        # Stage 1: 处理最深层特征 x5
        d4 = self.decoder4(x5)  # [B, 512, H/16, W/16]
        d4 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False)  # [B, 512, H/8, W/8]
        
        # Stage 2: 融合 d4 + x4
        d4_cat = torch.cat([d4, x4], dim=1)  # [B, 1024, H/8, W/8]
        d3 = self.decoder3(d4_cat)  # [B, 256, H/8, W/8]
        d3 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)  # [B, 256, H/4, W/4]
        
        # Stage 3: 融合 d3 + x3
        d3_cat = torch.cat([d3, x3], dim=1)  # [B, 512, H/4, W/4]
        d2 = self.decoder2(d3_cat)  # [B, 128, H/2, W/2] (inp=True自动上采样)
        
        # Stage 4: 融合 d2 + fused_x2
        d2_cat = torch.cat([d2, fused_x2], dim=1)  # [B, 256, H/2, W/2]
        d1 = self.decoder1(d2_cat)  # [B, 64, H, W] (inp=True自动上采样)
        
        # Stage 5: 最终融合 d1 + fused_x1
        final_cat = torch.cat([d1, fused_x1], dim=1)  # [B, 128, H, W]
        output = self.final_conv(final_cat)  # [B, num_classes, H, W]

        return output

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(num_classes, BatchNorm=nn.BatchNorm2d, bilinear=False):
    """构建LSC解码器"""
    return LSCDecoder(num_classes, BatchNorm, bilinear)


# 测试代码
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试LSC解码器
    decoder = build_decoder(num_classes=1, bilinear=False).to(device)
    
    # 模拟UNet_LSCA_SS2_e编码器输出格式
    batch_size = 2
    H, W = 256, 256
    
    # 编码器输出: (x5, [x4, x3, fused_x2, fused_x1])
    x5 = torch.randn(batch_size, 1024, H//16, W//16).to(device)
    skip_connections = [
        torch.randn(batch_size, 512, H//8, W//8).to(device),    # x4
        torch.randn(batch_size, 256, H//4, W//4).to(device),    # x3
        torch.randn(batch_size, 128, H//2, W//2).to(device),    # fused_x2
        torch.randn(batch_size, 64, H, W).to(device),           # fused_x1
    ]
    
    encoder_features = (x5, skip_connections)
    
    # 前向传播测试
    with torch.no_grad():
        output = decoder(encoder_features)
        print(f"✅ LSC解码器测试成功!")
        print(f"输入x5: {x5.shape}")
        print(f"跳跃连接: {[skip.shape for skip in skip_connections]}")
        print(f"输出: {output.shape}")
        
        total_params = sum(p.numel() for p in decoder.parameters())
        print(f"总参数量: {total_params:,}")
    
    print("\n✅ 测试完成!")