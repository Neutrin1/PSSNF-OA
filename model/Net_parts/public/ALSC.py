import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveDecoderBlock(nn.Module):
    """自适应条形卷积解码器块"""
    def __init__(self, in_channels, n_filters, BatchNorm, inp=False):
        super(AdaptiveDecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn1 = BatchNorm(in_channels // 4)
        self.relu1 = nn.ReLU()
        self.inp = inp
        
        # 自适应权重生成网络
        self.adaptive_weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels // 4, in_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, 4, 1),  # 4个分支的权重
            nn.Sigmoid()
        )
        
        # 多尺度条形卷积
        self.deconv1 = nn.Conv2d(in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4))
        self.deconv2 = nn.Conv2d(in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0))
        self.deconv3 = nn.Conv2d(in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0))
        self.deconv4 = nn.Conv2d(in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4))
        
        # 自适应尺度条形卷积分支
        self.adaptive_deconv1 = nn.ModuleList([
            nn.Conv2d(in_channels // 4, in_channels // 8, (1, k), padding=(0, k//2))
            for k in [3, 5, 7, 9] 
        ])
        self.adaptive_deconv2 = nn.ModuleList([
            nn.Conv2d(in_channels // 4, in_channels // 8, (k, 1), padding=(k//2, 0))
            for k in [3, 5, 7, 9]
        ])
        
        # 计算自适应特征的通道数: 2 * (in_channels // 8) = in_channels // 4
        adaptive_feature_channels = in_channels // 4
        
        # 通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(adaptive_feature_channels, adaptive_feature_channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(adaptive_feature_channels // 2, adaptive_feature_channels, 1),
            nn.Sigmoid()
        )
        
        # 基础特征通道数: 4 * (in_channels // 8) = in_channels // 2
        # 自适应特征通道数: in_channels // 4
        # 总通道数: in_channels // 2 + in_channels // 4 = 3 * in_channels // 4
        total_channels = (in_channels // 2) + (in_channels // 4)
        
        self.bn2 = BatchNorm(total_channels)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(total_channels, n_filters, 1)
        self.bn3 = BatchNorm(n_filters)
        self.relu3 = nn.ReLU()

        self._init_weight()

    def forward(self, x, inp=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # 生成自适应权重
        adaptive_weights = self.adaptive_weight_gen(x)  # [B, 4, 1, 1]
        
        # 基础条形卷积
        x1 = self.deconv1(x)  # [B, in_channels//8, H, W]
        x2 = self.deconv2(x)  # [B, in_channels//8, H, W]
        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))  # [B, in_channels//8, H, W]
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))  # [B, in_channels//8, H, W]
        
        # 自适应多尺度条形卷积
        adaptive_h_features = []
        adaptive_v_features = []
        
        for i, (h_conv, v_conv) in enumerate(zip(self.adaptive_deconv1, self.adaptive_deconv2)):
            weight = adaptive_weights[:, i:i+1, :, :]  # [B, 1, 1, 1]
            h_feat = h_conv(x) * weight  # [B, in_channels//8, H, W]
            v_feat = v_conv(x) * weight  # [B, in_channels//8, H, W]
            adaptive_h_features.append(h_feat)
            adaptive_v_features.append(v_feat)
        
        # 融合自适应特征
        adaptive_h = torch.stack(adaptive_h_features, dim=0).sum(dim=0)  # [B, in_channels//8, H, W]
        adaptive_v = torch.stack(adaptive_v_features, dim=0).sum(dim=0)  # [B, in_channels//8, H, W]
        
        # 合并所有特征
        basic_features = torch.cat((x1, x2, x3, x4), 1)  # [B, in_channels//2, H, W]
        adaptive_features = torch.cat((adaptive_h, adaptive_v), 1)  # [B, in_channels//4, H, W]
        
        # 应用通道注意力
        attention_weights = self.channel_attention(adaptive_features)
        adaptive_features = adaptive_features * attention_weights
        
        # 最终特征融合
        x = torch.cat((basic_features, adaptive_features), dim=1)  # [B, 3*in_channels//4, H, W]
        
        if self.inp:
            x = F.interpolate(x, scale_factor=2)
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
            elif isinstance(m, nn.ConvTranspose2d):
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
