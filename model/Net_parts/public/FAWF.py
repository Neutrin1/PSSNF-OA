import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FrequencyAwareMDAF(nn.Module):
    def __init__(self, dim, num_heads, LayerNorm_type):
        super(FrequencyAwareMDAF, self).__init__()
        self.num_heads = num_heads
        self.dim = dim

        # 原有的归一化层
        self.norm1 = LayerNorm(dim, LayerNorm_type)  # 原始特征图
        self.norm2 = LayerNorm(dim, LayerNorm_type)  # 低频分量
        self.norm3 = LayerNorm(dim, LayerNorm_type)  # 高频分量
        
        # 频率自适应权重生成器
        self.freq_weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 3, dim, 1),  # 输入三个特征图
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 3, 1),        # 输出三个频率权重
            nn.Softmax(dim=1)
        )
        
        # 原有的条状卷积 - 用于原始特征图
        self.conv1_1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv1_1_2 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_1_3 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv1_2_1 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_2_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv1_2_3 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        # 低频特征专用卷积（大核，捕获全局结构）
        self.low_freq_conv_h1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.low_freq_conv_h2 = nn.Conv2d(dim, dim, (1, 31), padding=(0, 15), groups=dim)
        self.low_freq_conv_v1 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.low_freq_conv_v2 = nn.Conv2d(dim, dim, (31, 1), padding=(15, 0), groups=dim)

        # 高频特征专用卷积（小核，捕获细节）
        self.high_freq_conv_h1 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
        self.high_freq_conv_h2 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.high_freq_conv_v1 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)
        self.high_freq_conv_v2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        # 投影层
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)
        self.final_project = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x_orig, x_low, x_high):
        """
        Args:
            x_orig: 原始特征图 [B, C, H, W]
            x_low: 小波低频分量 [B, C, H, W] 
            x_high: 小波高频分量（三个分量拼接） [B, C, H, W]
        """
        b, c, h, w = x_orig.shape
        
        # 确保所有输入尺寸一致
        if x_low.shape[-2:] != x_orig.shape[-2:]:
            x_low = F.interpolate(x_low, size=(h, w), mode='bilinear', align_corners=False)
        if x_high.shape[-2:] != x_orig.shape[-2:]:
            x_high = F.interpolate(x_high, size=(h, w), mode='bilinear', align_corners=False)
        
        # 归一化
        x_orig_norm = self.norm1(x_orig)
        x_low_norm = self.norm2(x_low)
        x_high_norm = self.norm3(x_high)
        
        # 生成频率自适应权重
        concat_feat = torch.cat([x_orig, x_low, x_high], dim=1)
        freq_weights = self.freq_weight_gen(concat_feat)  # [B, 3, 1, 1]
        
        # 加权特征
        weighted_orig = x_orig_norm * freq_weights[:, 0:1]
        weighted_low = x_low_norm * freq_weights[:, 1:2]
        weighted_high = x_high_norm * freq_weights[:, 2:3]
        
        # 原始特征的条状卷积
        orig_h1 = self.conv1_1_1(weighted_orig)
        orig_h2 = self.conv1_1_2(weighted_orig)
        orig_h3 = self.conv1_1_3(weighted_orig)
        orig_v1 = self.conv1_2_1(weighted_orig)
        orig_v2 = self.conv1_2_2(weighted_orig)
        orig_v3 = self.conv1_2_3(weighted_orig)
        out_orig = orig_h1 + orig_h2 + orig_h3 + orig_v1 + orig_v2 + orig_v3
        
        # 低频特征的条状卷积（大核）
        low_h1 = self.low_freq_conv_h1(weighted_low)
        low_h2 = self.low_freq_conv_h2(weighted_low)
        low_v1 = self.low_freq_conv_v1(weighted_low)
        low_v2 = self.low_freq_conv_v2(weighted_low)
        out_low = low_h1 + low_h2 + low_v1 + low_v2
        
        # 高频特征的条状卷积（小核）
        high_h1 = self.high_freq_conv_h1(weighted_high)
        high_h2 = self.high_freq_conv_h2(weighted_high)
        high_v1 = self.high_freq_conv_v1(weighted_high)
        high_v2 = self.high_freq_conv_v2(weighted_high)
        out_high = high_h1 + high_h2 + high_v1 + high_v2
        
        # 投影
        out_orig = self.project_out(out_orig)
        out_low = self.project_out(out_low)
        out_high = self.project_out(out_high)
        
        # 交叉注意力计算（原始特征与低频）
        k1 = rearrange(out_orig, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        v1 = rearrange(out_orig, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        q1 = rearrange(out_low, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        
        k2 = rearrange(out_low, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        v2 = rearrange(out_low, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q2 = rearrange(out_orig, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        
        # 归一化
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)
        
        # 注意力计算
        attn1 = (q1 @ k1.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        out1 = (attn1 @ v1) + q1
        
        attn2 = (q2 @ k2.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        out2 = (attn2 @ v2) + q2
        
        # 重构特征图
        out1 = rearrange(out1, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out2 = rearrange(out2, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        
        # 最终融合
        out = self.project_out(out1) + self.project_out(out2) + out_high
        out = self.final_project(out) + x_orig + x_low + x_high
        
        return out


# 保持原有MDAF作为备选
class MDAF(nn.Module):
    def __init__(self, dim, num_heads, LayerNorm_type):
        super(MDAF, self).__init__()
        self.num_heads = num_heads

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)
        
        self.conv1_1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv1_1_2 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_1_3 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv1_2_1 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_2_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv1_2_3 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv2_1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv2_1_2 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv2_1_3 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2_1 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv2_2_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_2_3 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        
        attn_111 = self.conv1_1_1(x1)
        attn_112 = self.conv1_1_2(x1)
        attn_113 = self.conv1_1_3(x1)
        attn_121 = self.conv1_2_1(x1)
        attn_122 = self.conv1_2_2(x1)
        attn_123 = self.conv1_2_3(x1)

        attn_211 = self.conv2_1_1(x2)
        attn_212 = self.conv2_1_2(x2)
        attn_213 = self.conv2_1_3(x2)
        attn_221 = self.conv2_2_1(x2)
        attn_222 = self.conv2_2_2(x2)
        attn_223 = self.conv2_2_3(x2)

        out1 = attn_111 + attn_112 + attn_113 + attn_121 + attn_122 + attn_123
        out2 = attn_211 + attn_212 + attn_213 + attn_221 + attn_222 + attn_223
        out1 = self.project_out(out1)
        out2 = self.project_out(out2)
        
        k1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        v1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        k2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        v2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q2 = rearrange(out1, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q1 = rearrange(out2, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)
        
        attn1 = (q1 @ k1.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        out3 = (attn1 @ v1) + q1
        
        attn2 = (q2 @ k2.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        out4 = (attn2 @ v2) + q2
        
        out3 = rearrange(out3, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out4 = rearrange(out4, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out3) + self.project_out(out4) + x1 + x2

        return out