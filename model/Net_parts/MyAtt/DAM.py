import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DAM(nn.Module):
    """
    Deformable Attention Module (DAM) with depthwise separable convolutions
    """
    def __init__(self, in_channels, K=9, hidden_dim=None):
        super(DAM, self).__init__()
        self.in_channels = in_channels
        self.K = K
        self.hidden_dim = hidden_dim if hidden_dim is not None else in_channels

        # 1x1卷积计算偏移量 (Δx, Δy)
        self.offset_conv = nn.Conv2d(
            in_channels, 
            2 * K, 
            kernel_size=1, 
            stride=1, 
            padding=0
        )

        # 使用深度可分离卷积生成Q、K、V
        self.q_conv = DepthwiseSeparableConv(in_channels, self.hidden_dim, kernel_size=1)
        self.k_conv = DepthwiseSeparableConv(in_channels, self.hidden_dim, kernel_size=1)
        self.v_conv = DepthwiseSeparableConv(in_channels, self.hidden_dim, kernel_size=1)

        # 输出卷积层，用于调整通道数与输入一致
        self.out_conv = DepthwiseSeparableConv(self.hidden_dim, in_channels, kernel_size=1)

        # 可学习参数α
        self.alpha = nn.Parameter(torch.zeros(1))

        # 初始化偏移量卷积权重
        init.zeros_(self.offset_conv.weight)
        init.zeros_(self.offset_conv.bias)

    def forward(self, x):
        B, C, H, W = x.shape

        # 步骤1：计算偏移量
        offsets = self.offset_conv(x)  # [B, 2*K, H, W]
        offsets = offsets.view(B, 2, self.K, H, W)  # [B, 2, K, H, W]
        dx = offsets[:, 0, ...]  # [B, K, H, W]
        dy = offsets[:, 1, ...]  # [B, K, H, W]

        # 步骤2：生成采样网格
        y_coord = torch.arange(H, device=x.device, dtype=x.dtype)
        x_coord = torch.arange(W, device=x.device, dtype=x.dtype)
        y_coord, x_coord = torch.meshgrid(y_coord, x_coord, indexing='ij')
        y_coord = y_coord.view(1, 1, H, W).expand(B, self.K, -1, -1)
        x_coord = x_coord.view(1, 1, H, W).expand(B, self.K, -1, -1)
        y_sampled = y_coord + dy
        x_sampled = x_coord + dx
        y_sampled = 2 * y_sampled / (H - 1) - 1
        x_sampled = 2 * x_sampled / (W - 1) - 1
        grid = torch.stack([x_sampled, y_sampled], dim=-1)
        grid = grid.permute(0, 2, 3, 1, 4).reshape(B * self.K, H, W, 2)

        # 步骤3：采样特征
        x_repeated = x.unsqueeze(1).repeat(1, self.K, 1, 1, 1)
        x_repeated = x_repeated.reshape(B * self.K, C, H, W)
        sampled_features = F.grid_sample(
            x_repeated,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        sampled_features = sampled_features.reshape(B, self.K, C, H, W)

        # 步骤4：计算Q、K、V
        K_feat = self.k_conv(sampled_features.reshape(B*self.K, C, H, W))
        K_feat = K_feat.reshape(B, self.K, self.hidden_dim, H, W)
        V_feat = self.v_conv(sampled_features.reshape(B*self.K, C, H, W))
        V_feat = V_feat.reshape(B, self.K, self.hidden_dim, H, W)
        Q_feat = self.q_conv(x)

        # 步骤5：计算注意力权重
        Q_flat = Q_feat.permute(0, 2, 3, 1)
        K_flat = K_feat.permute(0, 3, 4, 1, 2)
        attention_scores = torch.einsum('bhwc, bhwkc -> bhwk', Q_flat, K_flat)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 步骤6：加权求和与残差连接
        V_flat = V_feat.permute(0, 2, 1, 3, 4)
        attn = attention_weights.permute(0, 3, 1, 2).unsqueeze(1)
        weighted_V = (V_flat * attn).sum(dim=2)
        output = self.alpha * weighted_V + Q_feat
        output = self.out_conv(output)

        return output

