import torch.nn as nn
import torch.nn.functional as F
import torch


class UNetEncoderBlock(nn.Module):
    """UNet基础编码器块"""

    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d, downsample=True):
        super(UNetEncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = BatchNorm(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = BatchNorm(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        if self.downsample:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self._init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        before_pool = x
        if self.downsample:
            x = self.pool(x)
        return x, before_pool

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                


class ResidualUNetEncoderBlock(nn.Module):
    """带残差连接的UNet编码器块"""

    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d, downsample=True):
        super(ResidualUNetEncoderBlock, self).__init__()
        
        # 主路径：两个3x3卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = BatchNorm(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        # 残差连接投影（如果输入输出通道不同）
        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                BatchNorm(out_channels)
            )
        
        # 下采样
        self.downsample = downsample
        if self.downsample:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self._init_weight()

    def forward(self, x):
        # 保存输入用于残差连接
        identity = x
        
        # 主路径
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        # 残差连接
        if self.residual_proj is not None:
            identity = self.residual_proj(identity)
        
        x = x + identity  # 残差连接
        x = self.relu2(x)
        
        # 保存用于skip连接的特征
        before_pool = x
        
        # 下采样
        if self.downsample:
            x = self.pool(x)
            
        return x, before_pool

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class AdvancedResidualUNetEncoderBlock(nn.Module):
    """增强版带残差连接的UNet编码器块 - 支持预激活和多种残差形式"""

    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d, 
                 downsample=True, preactivation=False, dropout_rate=0.0):
        super(AdvancedResidualUNetEncoderBlock, self).__init__()
        
        self.preactivation = preactivation
        self.dropout_rate = dropout_rate
        
        if preactivation:
            # Pre-activation ResNet风格
            self.bn1 = BatchNorm(in_channels)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
            
            self.bn2 = BatchNorm(out_channels)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        else:
            # 标准Post-activation
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn1 = BatchNorm(out_channels)
            self.relu1 = nn.ReLU(inplace=True)
            
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = BatchNorm(out_channels)
            self.relu2 = nn.ReLU(inplace=True)
        
        # Dropout层
        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(dropout_rate)
        
        # 残差连接投影
        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                BatchNorm(out_channels)
            )
        
        # 下采样
        self.downsample = downsample
        if self.downsample:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # SE注意力模块（可选）
        self.se_module = SEModule(out_channels, reduction=16)

        self._init_weight()

    def forward(self, x):
        identity = x
        
        if self.preactivation:
            # Pre-activation路径
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.conv1(x)
            
            x = self.bn2(x)
            x = self.relu2(x)
            if hasattr(self, 'dropout') and self.dropout_rate > 0:
                x = self.dropout(x)
            x = self.conv2(x)
        else:
            # Post-activation路径
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            
            if hasattr(self, 'dropout') and self.dropout_rate > 0:
                x = self.dropout(x)
            
            x = self.conv2(x)
            x = self.bn2(x)
        
        # 残差连接
        if self.residual_proj is not None:
            identity = self.residual_proj(identity)
        
        x = x + identity
        
        if not self.preactivation:
            x = self.relu2(x)
        
        # SE注意力
        x = self.se_module(x)
        
        # 保存用于skip连接
        before_pool = x
        
        # 下采样
        if self.downsample:
            x = self.pool(x)
            
        return x, before_pool

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SEModule(nn.Module):
    """Squeeze-and-Excitation模块"""
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DenseResidualUNetEncoderBlock(nn.Module):
    """Dense残差连接的UNet编码器块 - 融合DenseNet思想"""

    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d, 
                 downsample=True, growth_rate=32):
        super(DenseResidualUNetEncoderBlock, self).__init__()
        
        self.growth_rate = growth_rate
        
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.bn1 = BatchNorm(growth_rate)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 第二个卷积层（输入是in_channels + growth_rate）
        self.conv2 = nn.Conv2d(in_channels + growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm(growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        
        # 输出投影层
        self.out_conv = nn.Conv2d(in_channels + 2 * growth_rate, out_channels, kernel_size=1, bias=False)
        self.out_bn = BatchNorm(out_channels)
        self.out_relu = nn.ReLU(inplace=True)
        
        # 残差连接投影
        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                BatchNorm(out_channels)
            )
        
        # 下采样
        self.downsample = downsample
        if self.downsample:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self._init_weight()

    def forward(self, x):
        identity = x
        
        # Dense连接
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        
        # 拼接原始输入和第一个卷积输出
        x_cat = torch.cat([x, x1], dim=1)
        
        x2 = self.conv2(x_cat)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)
        
        # 拼接所有特征
        x_final = torch.cat([x, x1, x2], dim=1)
        
        # 输出投影
        x_out = self.out_conv(x_final)
        x_out = self.out_bn(x_out)
        
        # 残差连接
        if self.residual_proj is not None:
            identity = self.residual_proj(identity)
        
        x_out = x_out + identity
        x_out = self.out_relu(x_out)
        
        # 保存用于skip连接
        before_pool = x_out
        
        # 下采样
        if self.downsample:
            x_out = self.pool(x_out)
            
        return x_out, before_pool

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
                
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


class ImprovedUNetDEB(nn.Module):
    """改进的UNet解码器块"""
    def __init__(self, in_channels, out_channels, skip_channels=None, 
                 kernel_size=3, stride=2, padding=1, output_padding=1, 
                 use_bn=True, activation='relu', use_dropout=False, dropout_rate=0.1):
        super(ImprovedUNetDEB, self).__init__()
        
        # 如果没有指定skip_channels，假设与out_channels相同
        if skip_channels is None:
            skip_channels = out_channels
            
        # 上采样层
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bn_up = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
        # 激活函数选择
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU(inplace=True)
        
        # 拼接后的卷积层 - 注意通道数计算
        concat_channels = out_channels + skip_channels
        self.conv1 = nn.Conv2d(concat_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
        # Dropout层
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout2d(dropout_rate)
        
        self._init_weights()

    def forward(self, x, skip):
        # 上采样
        x = self.upconv(x)
        x = self.bn_up(x)
        x = self.activation(x)
        
        # 尺寸对齐 - 更robust的处理
        if x.shape[2:] != skip.shape[2:]:
            # 使用双线性插值对齐尺寸
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        # 跳跃连接拼接
        x = torch.cat([x, skip], dim=1)
        
        # 双卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        if self.use_dropout:
            x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        return x
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ResidualUNetDEB(nn.Module):
    """带残差连接的UNet解码器块"""
    def __init__(self, in_channels, out_channels, skip_channels=None, 
                 kernel_size=3, stride=2, padding=1, output_padding=1, use_bn=True):
        super(ResidualUNetDEB, self).__init__()
        
        if skip_channels is None:
            skip_channels = out_channels
            
        # 上采样
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bn_up = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        
        # 拼接后的卷积
        concat_channels = out_channels + skip_channels
        self.conv1 = nn.Conv2d(concat_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
        # 残差连接投影（如果需要）
        self.residual_proj = None
        if concat_channels != out_channels:
            self.residual_proj = nn.Sequential(
                nn.Conv2d(concat_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
            )
        
        self._init_weights()

    def forward(self, x, skip):
        # 上采样
        x = self.upconv(x)
        x = self.bn_up(x)
        x = self.relu(x)
        
        # 尺寸对齐
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        # 跳跃连接拼接
        x_concat = torch.cat([x, skip], dim=1)
        identity = x_concat
        
        # 双卷积
        x = self.conv1(x_concat)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        # 残差连接
        if self.residual_proj is not None:
            identity = self.residual_proj(identity)
        
        x = x + identity
        x = self.relu(x)
        
        return x
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AttentionUNetDEB(nn.Module):
    """带注意力机制的UNet解码器块"""
    def __init__(self, in_channels, out_channels, skip_channels=None,
                 kernel_size=3, stride=2, padding=1, output_padding=1, use_bn=True):
        super(AttentionUNetDEB, self).__init__()
        
        if skip_channels is None:
            skip_channels = out_channels
            
        # 上采样
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bn_up = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        
        # 注意力门控 - 对skip连接加权
        self.attention_gate = AttentionGate(skip_channels, out_channels, out_channels // 2)
        
        # 拼接后的卷积
        concat_channels = out_channels + skip_channels
        self.conv1 = nn.Conv2d(concat_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

    def forward(self, x, skip):
        # 上采样
        x = self.upconv(x)
        x = self.bn_up(x)
        x = self.relu(x)
        
        # 尺寸对齐
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        # 注意力门控处理skip连接
        skip_attended = self.attention_gate(skip, x)
        
        # 跳跃连接拼接
        x = torch.cat([x, skip_attended], dim=1)
        
        # 双卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x


class AttentionGate(nn.Module):
    """注意力门控模块"""
    def __init__(self, skip_channels, gating_channels, inter_channels):
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, bias=True)
        self.W_x = nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=True)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # 确保尺寸匹配
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        
        return x * psi