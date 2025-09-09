import torch
import torch.nn as nn
from Net_parts.public.Attention import CBAM
import torch.nn.functional as F
from einops import rearrange


#解码器块
class LSCDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 1)
        self.bn1 = BatchNorm(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        mid_channels = max(out_channels // 4, 8)
        self.branch1 = nn.Conv2d(out_channels, mid_channels, (1, 9), padding=(0, 4))
        self.branch2 = nn.Conv2d(out_channels, mid_channels, (9, 1), padding=(4, 0))
        self.branch3 = nn.Conv2d(out_channels, mid_channels, (9, 1), padding=(4, 0))
        self.branch4 = nn.Conv2d(out_channels, mid_channels, (1, 9), padding=(0, 4))

        self.bn2 = BatchNorm(mid_channels * 4)
        self.relu2 = nn.ReLU(inplace=True)

        self.out_conv = nn.Conv2d(mid_channels * 4, out_channels, 1)
        self.bn3 = BatchNorm(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

        self._init_weight()

    def forward(self, x_input, skip):
        x = self.upsample(x_input)
        x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.inv_h_transform(self.branch3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.branch4(self.v_transform(x)))

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.out_conv(x)
        x = self.bn3(x)

        # 残差连接（从 upsample 的输入）
        residual = F.interpolate(x_input, size=x.shape[2:], mode='bilinear', align_corners=True)
        if residual.shape[1] != x.shape[1]:
            residual = nn.Conv2d(residual.shape[1], x.shape[1], kernel_size=1).to(x.device)(residual)

        x = x + residual
        x = self.relu3(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def h_transform(self, x):
        shape = x.size()
        x = F.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = F.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = F.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = F.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)

class LSCEncoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, BatchNorm=nn.BatchNorm2d, downsample=True):
        super(LSCEncoderBlock, self).__init__()
        mid_channels = max(in_channels // 4, 8)  # 保证至少为8
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.bn1 = BatchNorm(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.downsample = downsample

        # 四个条形卷积分支
        self.deconv1 = nn.Conv2d(mid_channels, mid_channels // 2, (1, 9), padding=(0, 4))
        self.deconv2 = nn.Conv2d(mid_channels, mid_channels // 2, (9, 1), padding=(4, 0))
        self.deconv3 = nn.Conv2d(mid_channels, mid_channels // 2, (9, 1), padding=(4, 0))
        self.deconv4 = nn.Conv2d(mid_channels, mid_channels // 2, (1, 9), padding=(0, 4))

        # 拼接后的处理
        self.bn2 = BatchNorm(mid_channels * 2)
        self.relu2 = nn.ReLU(inplace=True)
        
        # 输出卷积
        self.conv3 = nn.Conv2d(mid_channels * 2, n_filters, 1)
        self.bn3 = BatchNorm(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

        # 下采样层（如果需要）
        if self.downsample:
            self.downsample_conv = nn.Conv2d(n_filters, n_filters, 3, stride=2, padding=1, bias=False)
            self.downsample_bn = BatchNorm(n_filters)

        # 残差分支：自动对齐维度
        if in_channels != n_filters:
            self.residual_proj = nn.Sequential(
                nn.Conv2d(in_channels, n_filters, kernel_size=1, bias=False),
                BatchNorm(n_filters)
            )
        else:
            self.residual_proj = nn.Identity()

        self._init_weight()

    def forward(self, x):
        residual = self.residual_proj(x)  # 投影残差

        # 降维
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # 四个条形卷积分支
        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))

        # 拼接四个分支
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.bn2(x)
        x = self.relu2(x)

        # 输出卷积
        x = self.conv3(x)
        x = self.bn3(x)

        # 残差连接
        x = x + residual
        x = self.relu3(x)

        # 下采样（如果需要）
        if self.downsample:
            x = self.downsample_conv(x)
            x = self.downsample_bn(x)
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
    
    
                
            

                
#CRH380B              
# 通道空间双门控
class CSGate(nn.Module):
    def __init__(self, in_channels):
        super(CSGate, self).__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, feat1, feat2):
        concat = torch.cat([feat1, feat2], dim=1)
        channel_attention = self.channel_gate(concat)
        spatial_attention = self.spatial_gate(concat)
        attention = channel_attention * spatial_attention
        return attention * feat1 + (1 - attention) * feat2

#CRH380C
class AttentiveGate(nn.Module):
    """
    通道+空间注意力门控机制
    输入：feat1, feat2（shape一致）
    输出：融合后的特征
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # SE通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels * 2, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, feat1, feat2):
        # 通道注意力
        b, c, h, w = feat1.shape
        feats = torch.cat([feat1, feat2], dim=1)  # [B, 2C, H, W]
        # SE通道注意力
        chn_att = self.avg_pool(feats).view(b, -1)  # [B, 2C]
        chn_att = self.channel_fc(chn_att).view(b, 2 * c, 1, 1)  # [B, 2C, 1, 1]
        chn_att1, chn_att2 = torch.split(chn_att, c, dim=1)
        feat1_c = feat1 * chn_att1
        feat2_c = feat2 * chn_att2

        # 空间注意力
        spa_att = torch.cat([
            torch.mean(feats, dim=1, keepdim=True),
            torch.max(feats, dim=1, keepdim=True)[0]
        ], dim=1)  # [B, 2, H, W]
        spa_att = self.spatial_conv(spa_att)  # [B, 1, H, W]
        feat1_s = feat1_c * spa_att
        feat2_s = feat2_c * (1 - spa_att)

        # 融合
        out = feat1_s + feat2_s
        return out
    
 

class GatedFusion(nn.Module):
    """门控融合模块 - 自适应融合UNet特征和LSC特征"""
    def __init__(self, unet_channels, lsc_channels, out_channels):
        super(GatedFusion, self).__init__()
        
        # 特征对齐（如果通道数不同）
        self.unet_align = nn.Conv2d(unet_channels, out_channels, 1) if unet_channels != out_channels else nn.Identity()
        self.lsc_align = nn.Conv2d(lsc_channels, out_channels, 1) if lsc_channels != out_channels else nn.Identity()
        
        # 门控网络 - 学习融合权重
        self.gate_net = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 2, 1),
            nn.Softmax(dim=1)
        )
        
        # 特征增强
        self.enhance_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, unet_feat, lsc_feat):
        """
        Args:
            unet_feat: UNet特征 [B, C1, H, W]
            lsc_feat: LSC特征 [B, C2, H, W]
        Returns:
            融合后的特征 [B, out_channels, H, W]
        """
        # 特征对齐
        unet_aligned = self.unet_align(unet_feat)  # [B, out_channels, H, W]
        lsc_aligned = self.lsc_align(lsc_feat)     # [B, out_channels, H, W]
        
        # 拼接特征用于门控学习
        concat_feat = torch.cat([unet_aligned, lsc_aligned], dim=1)  # [B, out_channels*2, H, W]
        
        # 计算门控权重
        gate_weights = self.gate_net(concat_feat)  # [B, 2, H, W]
        unet_weight = gate_weights[:, 0:1, :, :]   # [B, 1, H, W]
        lsc_weight = gate_weights[:, 1:2, :, :]    # [B, 1, H, W]
        
        # 加权融合
        fused_feat = unet_weight * unet_aligned + lsc_weight * lsc_aligned
        
        # 特征增强
        fused_feat = self.enhance_conv(fused_feat)
        
        return fused_feat


class AttentionGatedFusion(nn.Module):
    """注意力门控融合模块 - 结合CBAM注意力和门控融合"""
    def __init__(self, unet_channels, lsc_channels, out_channels, reduction=16):
        super(AttentionGatedFusion, self).__init__()
        
        # 特征对齐
        self.unet_align = nn.Conv2d(unet_channels, out_channels, 1) if unet_channels != out_channels else nn.Identity()
        self.lsc_align = nn.Conv2d(lsc_channels, out_channels, 1) if lsc_channels != out_channels else nn.Identity()
        
        # 对齐后的特征加上注意力
        self.unet_attention = CBAM(out_channels, reduction)
        self.lsc_attention = CBAM(out_channels, reduction)
        
        # 门控融合
        self.gated_fusion = GatedFusion(out_channels, out_channels, out_channels)
        
        # 最终特征精炼
        self.final_attention = CBAM(out_channels, reduction)
        
    def forward(self, unet_feat, lsc_feat):
        """
        Args:
            unet_feat: UNet特征
            lsc_feat: LSC特征
        Returns:
            注意力门控融合后的特征
        """
        # Step 1: 特征对齐
        unet_aligned = self.unet_align(unet_feat)
        lsc_aligned = self.lsc_align(lsc_feat)
        
        # Step 2: 分别对两个分支应用注意力
        unet_attended = self.unet_attention(unet_aligned)
        lsc_attended = self.lsc_attention(lsc_aligned)
        
        # Step 3: 门控融合
        fused_feat = self.gated_fusion(unet_attended, lsc_attended)
        
        # Step 4: 最终注意力精炼
        final_feat = self.final_attention(fused_feat)
        
        return final_feat



# 长距离建模改进
class NonLocalBlock(nn.Module):
    """Non-Local注意力模块 - 用于长距离依赖建模"""
    def __init__(self, in_channels, reduction=2):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // reduction
        
        # θ, φ, g 变换
        self.theta = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.g = nn.Conv2d(in_channels, self.inter_channels, 1)
        
        # 输出变换
        self.W = nn.Sequential(
            nn.Conv2d(self.inter_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )
        
        # 初始化W为0，确保残差连接的恒等映射
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # 计算theta, phi, g
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)  # [B, C', HW]
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)      # [B, C', HW]
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)          # [B, C', HW]
        
        # 计算注意力权重
        attention = torch.matmul(theta_x.permute(0, 2, 1), phi_x)  # [B, HW, HW]
        attention = F.softmax(attention, dim=-1)
        
        # 应用注意力
        out = torch.matmul(g_x, attention.permute(0, 2, 1))  # [B, C', HW]
        out = out.view(batch_size, self.inter_channels, H, W)
        
        # 输出变换 + 残差连接
        out = self.W(out) + x
        
        return out


class EfficientNonLocalBlock(nn.Module):
    """高效版Non-Local模块 - 降低计算复杂度"""
    def __init__(self, in_channels, reduction=2, sub_sample=True):
        super(EfficientNonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // reduction
        self.sub_sample = sub_sample
        
        # θ, φ, g 变换
        self.theta = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.g = nn.Conv2d(in_channels, self.inter_channels, 1)
        
        # 下采样层降低计算量
        if sub_sample:
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(2))
            self.g = nn.Sequential(self.g, nn.MaxPool2d(2))
        
        # 输出变换
        self.W = nn.Sequential(
            nn.Conv2d(self.inter_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        
        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        else:
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        
        attention = torch.matmul(theta_x.permute(0, 2, 1), phi_x)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.matmul(g_x, attention.permute(0, 2, 1))
        out = out.view(batch_size, self.inter_channels, H, W)
        
        out = self.W(out) + x
        return out
    
    
class AxialAttention(nn.Module):
    """轴向注意力 - 分别在H和W维度建模长距离依赖"""
    def __init__(self, in_channels, heads=8, dim_head=64):
        super(AxialAttention, self).__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        
        self.to_qkv = nn.Conv2d(in_channels, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, in_channels, 1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=self.heads), qkv)
        
        # Height-wise attention
        q_h = rearrange(q, 'b h (x y) d -> b h x (y d)', x=h)
        k_h = rearrange(k, 'b h (x y) d -> b h x (y d)', x=h)
        v_h = rearrange(v, 'b h (x y) d -> b h x (y d)', x=h)
        
        attn_h = torch.softmax(q_h @ k_h.transpose(-2, -1) / (self.dim_head ** 0.5), dim=-1)
        out_h = attn_h @ v_h
        
        # Width-wise attention  
        q_w = rearrange(q, 'b h (x y) d -> b h y (x d)', y=w)
        k_w = rearrange(k, 'b h (x y) d -> b h y (x d)', y=w)
        v_w = rearrange(v, 'b h (x y) d -> b h y (x d)', y=w)
        
        attn_w = torch.softmax(q_w @ k_w.transpose(-2, -1) / (self.dim_head ** 0.5), dim=-1)
        out_w = attn_w @ v_w
        
        # 合并
        out = (rearrange(out_h, 'b h x (y d) -> b (h d) x y', y=w) + 
               rearrange(out_w, 'b h y (x d) -> b (h d) x y', x=h)) / 2
        
        return self.to_out(out) + x


class CrissCrossAttention(nn.Module):
    """十字交叉注意力 - 高效的长距离建模"""
    def __init__(self, in_channels):
        super(CrissCrossAttention, self).__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        
        # 生成query, key, value
        query = self.query(x)  # [B, C/8, H, W]
        key = self.key(x)      # [B, C/8, H, W]
        value = self.value(x)  # [B, C, H, W]
        
        # 十字形注意力
        # Horizontal attention
        query_h = query.permute(0, 3, 1, 2).contiguous().view(B*W, C//8, H)  # [BW, C/8, H]
        key_h = key.permute(0, 3, 1, 2).contiguous().view(B*W, C//8, H)      # [BW, C/8, H]
        value_h = value.permute(0, 3, 1, 2).contiguous().view(B*W, C, H)     # [BW, C, H]
        
        attention_h = torch.bmm(query_h.permute(0, 2, 1), key_h)  # [BW, H, H]
        attention_h = self.softmax(attention_h)
        out_h = torch.bmm(value_h, attention_h.permute(0, 2, 1))  # [BW, C, H]
        out_h = out_h.view(B, W, C, H).permute(0, 2, 3, 1)       # [B, C, H, W]
        
        # Vertical attention
        query_w = query.permute(0, 2, 1, 3).contiguous().view(B*H, C//8, W)  # [BH, C/8, W]
        key_w = key.permute(0, 2, 1, 3).contiguous().view(B*H, C//8, W)      # [BH, C/8, W]
        value_w = value.permute(0, 2, 1, 3).contiguous().view(B*H, C, W)     # [BH, C, W]
        
        attention_w = torch.bmm(query_w.permute(0, 2, 1), key_w)  # [BH, W, W]
        attention_w = self.softmax(attention_w)
        out_w = torch.bmm(value_w, attention_w.permute(0, 2, 1))  # [BH, C, W]
        out_w = out_w.view(B, H, C, W).permute(0, 2, 1, 3)       # [B, C, H, W]
        
        # 融合
        out = self.gamma * (out_h + out_w) + x
        return out
    
    
class EnhancedAttentionGatedFusion(nn.Module):
    """增强版注意力门控融合 - 加入长距离建模"""
    def __init__(self, unet_channels, lsc_channels, out_channels, reduction=16, 
                 use_nonlocal=True, use_criss_cross=False):
        super(EnhancedAttentionGatedFusion, self).__init__()
        
        # 特征对齐
        self.unet_align = nn.Conv2d(unet_channels, out_channels, 1) if unet_channels != out_channels else nn.Identity()
        self.lsc_align = nn.Conv2d(lsc_channels, out_channels, 1) if lsc_channels != out_channels else nn.Identity()
        
        # 局部注意力
        self.unet_attention = CBAM(out_channels, reduction)
        self.lsc_attention = CBAM(out_channels, reduction)
        
        # 长距离建模模块
        if use_nonlocal:
            self.nonlocal_unet = EfficientNonLocalBlock(out_channels, reduction=2)
            self.nonlocal_lsc = EfficientNonLocalBlock(out_channels, reduction=2)
        else:
            self.nonlocal_unet = nn.Identity()
            self.nonlocal_lsc = nn.Identity()
            
        if use_criss_cross:
            self.criss_cross = CrissCrossAttention(out_channels * 2)
        else:
            self.criss_cross = nn.Identity()
        
        # 多尺度上下文聚合
        self.context_block = PyramidPoolingModule(out_channels * 2, out_channels)
        
        # 门控融合
        self.gated_fusion = GatedFusion(out_channels, out_channels, out_channels)
        
        # 最终特征精炼
        self.final_attention = CBAM(out_channels, reduction)
        
        # 残差连接投影
        self.residual_proj = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, unet_feat, lsc_feat):
        # Step 1: 特征对齐
        unet_aligned = self.unet_align(unet_feat)
        lsc_aligned = self.lsc_align(lsc_feat)
        
        # Step 2: 局部注意力
        unet_attended = self.unet_attention(unet_aligned)
        lsc_attended = self.lsc_attention(lsc_aligned)
        
        # Step 3: 长距离建模
        unet_global = self.nonlocal_unet(unet_attended)
        lsc_global = self.nonlocal_lsc(lsc_attended)
        
        # Step 4: 拼接并应用十字交叉注意力
        concat_feat = torch.cat([unet_global, lsc_global], dim=1)
        concat_feat = self.criss_cross(concat_feat)
        
        # Step 5: 多尺度上下文聚合
        context_feat = self.context_block(concat_feat)
        
        # 残差连接
        residual = self.residual_proj(concat_feat)
        context_feat = context_feat + residual
        
        # Step 6: 门控融合
        fused_feat = self.gated_fusion(unet_global, lsc_global)
        
        # Step 7: 特征融合
        final_feat = fused_feat + context_feat
        final_feat = self.final_attention(final_feat)
        
        return final_feat


class PyramidPoolingModule(nn.Module):
    """金字塔池化模块 - 多尺度上下文聚合"""
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6]):
        super(PyramidPoolingModule, self).__init__()
        
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), 1),
                nn.BatchNorm2d(in_channels // len(pool_sizes)),
                nn.ReLU(inplace=True)
            ) for pool_size in pool_sizes
        ])
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramid_feats = [x]
        
        for stage in self.stages:
            pyramid_feats.append(
                F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=True)
            )
        
        return self.conv_out(torch.cat(pyramid_feats, dim=1))

