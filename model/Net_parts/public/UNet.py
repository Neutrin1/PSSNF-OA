import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """双卷积层：卷积 -> 批归一化 -> ReLU -> 卷积 -> 批归一化 -> ReLU"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetEncoder(nn.Module):
    """UNet编码器"""
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.features = features
        self.encoder_blocks = nn.ModuleList()
        
        # 构建编码器层
        prev_channels = in_channels
        for feature in features:
            self.encoder_blocks.append(DoubleConv(prev_channels, feature))
            prev_channels = feature
        
        # 最底层
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # 下采样层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # 确保输入数据类型与模型权重类型一致
        if x.dtype != next(self.parameters()).dtype:
            x = x.to(dtype=next(self.parameters()).dtype)
        
        skip_connections = []
        
        # 编码路径
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # 瓶颈层
        x = self.bottleneck(x)
        
        return x, skip_connections

class UNetDecoder(nn.Module):
    """UNet解码器"""
    def __init__(self, features=[64, 128, 256, 512], out_channels=1):
        super().__init__()
        self.features = features
        self.out_channels = out_channels
        
        # 构建解码器层
        self.decoder_blocks = nn.ModuleList()
        self.upconv_blocks = nn.ModuleList()
        
        self.final_double_conv = DoubleConv(features[0], features[0])  # 64→64的双卷积
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)  # 64→2的1x1卷积
        # 反向遍历特征通道数
        reversed_features = features[::-1]
        
        for i in range(len(reversed_features)):
            if i == 0:
                # 第一个上采样层：从瓶颈层开始
                in_channels = reversed_features[i] * 2
                out_channels_up = reversed_features[i]
            else:
                in_channels = reversed_features[i-1]
                out_channels_up = reversed_features[i]
            
            # 上采样层
            self.upconv_blocks.append(
                nn.ConvTranspose2d(in_channels, out_channels_up, kernel_size=2, stride=2)
            )
            
            # 解码器块（跳跃连接后的双卷积）
            self.decoder_blocks.append(
                DoubleConv(out_channels_up * 2, out_channels_up)
            )
        
        # 最终输出层
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x, skip_connections):
        # 确保输入数据类型与模型权重类型一致
        if x.dtype != next(self.parameters()).dtype:
            x = x.to(dtype=next(self.parameters()).dtype)
        
        # 确保跳跃连接的数据类型也一致
        target_dtype = next(self.parameters()).dtype
        skip_connections = [skip.to(dtype=target_dtype) if skip.dtype != target_dtype else skip 
                          for skip in skip_connections]
        
        # 反向遍历跳跃连接
        skip_connections = skip_connections[::-1]
        
        for i, (upconv, decoder_block) in enumerate(zip(self.upconv_blocks, self.decoder_blocks)):
            # 上采样
            x = upconv(x)
            
            # 获取对应的跳跃连接
            skip_connection = skip_connections[i]
            
            # 如果尺寸不匹配，调整大小
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)
            
            # 拼接跳跃连接
            x = torch.cat([skip_connection, x], dim=1)
            
            # 双卷积
            x = decoder_block(x)
        
        # 最终输出
        x = self.final_conv(x)
        return x

class UNet(nn.Module):
    """完整的UNet模型"""
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = UNetEncoder(in_channels, features)
        self.decoder = UNetDecoder(features, out_channels)
    
    def forward(self, x):
        # 确保输入数据类型与模型权重类型一致
        if x.dtype != next(self.parameters()).dtype:
            x = x.to(dtype=next(self.parameters()).dtype)
        
        # 编码
        encoded, skip_connections = self.encoder(x)
        
        # 解码
        output = self.decoder(encoded, skip_connections)
        
        return output

# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = UNet(in_channels=3, out_channels=1)
    
    # 测试输入 - 确保使用正确的数据类型
    x = torch.randn(1, 3, 256, 256, dtype=torch.float32)
    
    # 如果使用GPU，确保模型和输入都在同一设备上
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        print(f"Input dtype: {x.dtype}")
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
    
    # 分别使用编码器和解码器
    encoder = UNetEncoder(in_channels=3)
    decoder = UNetDecoder(out_channels=1)
    
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    
    with torch.no_grad():
        encoded, skip_connections = encoder(x)
        decoded = decoder(encoded, skip_connections)
        print(f"Encoded shape: {encoded.shape}")
        print(f"Decoded shape: {decoded.shape}")
        print(f"Encoded dtype: {encoded.dtype}")
        print(f"Decoded dtype: {decoded.dtype}")