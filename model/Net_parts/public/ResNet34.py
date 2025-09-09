import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """ResNet基础残差块（用于ResNet18/34）"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, BatchNorm=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        
        # 第一个3x3卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm(out_channels)
        
        # 第二个3x3卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
        self._init_weight()
    
    def forward(self, x):
        identity = x
        
        # 第一个卷积块
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 第二个卷积块
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ResNet34EncoderBlock(nn.Module):
    """ResNet34编码器块"""
    
    def __init__(self, in_channels, out_channels, num_blocks, stride=1, BatchNorm=nn.BatchNorm2d):
        super(ResNet34EncoderBlock, self).__init__()
        
        # 下采样层（如果需要）
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                BatchNorm(out_channels)
            )
        
        # 构建残差块序列
        layers = []
        # 第一个块可能需要下采样
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample, BatchNorm))
        
        # 后续块保持相同尺寸
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, BatchNorm=BatchNorm))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)


class ResNet34Encoder(nn.Module):
    """完整的ResNet34编码器 - 5个阶段"""
    
    def __init__(self, in_channels=3, base_channels=64, BatchNorm=nn.BatchNorm2d):
        super(ResNet34Encoder, self).__init__()
        
        # 阶段0: 初始卷积层 (Conv1)
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=7, 
                              stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet34的层配置: [3, 4, 6, 3] - 对应layer1到layer4
        # 阶段1: Conv2_x
        self.layer1 = ResNet34EncoderBlock(base_channels, base_channels, 3, stride=1, BatchNorm=BatchNorm)      # 64->64, /4
        # 阶段2: Conv3_x  
        self.layer2 = ResNet34EncoderBlock(base_channels, base_channels*2, 4, stride=2, BatchNorm=BatchNorm)   # 64->128, /8
        # 阶段3: Conv4_x
        self.layer3 = ResNet34EncoderBlock(base_channels*2, base_channels*4, 6, stride=2, BatchNorm=BatchNorm) # 128->256, /16
        # 阶段4: Conv5_x
        self.layer4 = ResNet34EncoderBlock(base_channels*4, base_channels*8, 3, stride=2, BatchNorm=BatchNorm) # 256->512, /32
        
        self._init_weight()
    
    def forward(self, x):
        """
        输入尺寸变化 (假设输入512x512):
        x: [B, 3, 512, 512]
        """
        # 存储skip connections
        skips = []
        
        # 阶段0: 初始处理
        x = self.conv1(x)      # [B, 64, 256, 256] - /2
        x = self.bn1(x)
        x = self.relu(x)
        skips.append(x)        # Skip0: /2, 64通道
        
        x = self.maxpool(x)    # [B, 64, 128, 128] - /4
        
        # 阶段1: Conv2_x
        x = self.layer1(x)     # [B, 64, 128, 128] - /4
        skips.append(x)        # Skip1: /4, 64通道
        
        # 阶段2: Conv3_x
        x = self.layer2(x)     # [B, 128, 64, 64] - /8  
        skips.append(x)        # Skip2: /8, 128通道
        
        # 阶段3: Conv4_x
        x = self.layer3(x)     # [B, 256, 32, 32] - /16
        skips.append(x)        # Skip3: /16, 256通道
        
        # 阶段4: Conv5_x
        x = self.layer4(x)     # [B, 512, 16, 16] - /32
        
        return x, skips
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# 测试函数
if __name__ == "__main__":
    import torch
    
    # 创建模型实例
    model = ResNet34Encoder(in_channels=3, base_channels=64)
    model.eval()
    
    # 输入张量
    x = torch.randn(2, 3, 512, 512)
    
    # 前向传播
    encoded_feat, skips = model(x)
    
    print("=== ResNet34编码器输出 ===")
    print(f"输入形状: {x.shape}")
    print(f"编码特征形状: {encoded_feat.shape}")
    print("\nSkip连接形状:")
    for i, skip in enumerate(skips):
        print(f"Skip{i}: {skip.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")