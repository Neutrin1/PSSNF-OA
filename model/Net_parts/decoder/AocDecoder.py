import math
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

        self.deconv1 = nn.Conv2d(in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4))
        self.deconv2 = nn.Conv2d(in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0))
        self.deconv3 = nn.Conv2d(in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0))
        self.deconv4 = nn.Conv2d(in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4))

        self.bn2 = BatchNorm(in_channels // 2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels // 2, n_filters, 1)
        self.bn3 = BatchNorm(n_filters)
        self.relu3 = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)
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
        x = F.pad(x, (0, shape[-1]))
        x = x.view(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.view(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.view(shape[0], shape[1], -1).contiguous()
        x = F.pad(x, (0, shape[-2]))
        x = x.view(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = F.pad(x, (0, shape[-1]))
        x = x.view(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.view(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.view(shape[0], shape[1], -1)
        x = F.pad(x, (0, shape[-2]))
        x = x.view(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)


class AocDecoder(nn.Module):
    def __init__(self, num_classes=1, base_channels=64, BatchNorm=nn.BatchNorm2d):
        super(AocDecoder, self).__init__()

        # 保证拼接时通道数一致，拼接后作为下一级DecoderBlock输入
        self.decoder4 = DecoderBlock(base_channels * 16, base_channels * 8, BatchNorm)
        self.conv_e4 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, 1, bias=False),
            BatchNorm(base_channels * 8),
            nn.ReLU()
        )

        self.decoder3 = DecoderBlock(base_channels * 16, base_channels * 4, BatchNorm)
        self.conv_e3 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False),
            BatchNorm(base_channels * 4),
            nn.ReLU()
        )

        self.decoder2 = DecoderBlock(base_channels * 8, base_channels * 2, BatchNorm, inp=True)
        self.conv_e2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False),
            BatchNorm(base_channels * 2),
            nn.ReLU()
        )

        self.decoder1 = DecoderBlock(base_channels * 4, base_channels, BatchNorm, inp=True)
        self.conv_e1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 1, bias=False),
            BatchNorm(base_channels),
            nn.ReLU()
        )

        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

        self._init_weight()

    def forward(self, x5, skip_connections):
        """
        Args:
            x5: 最深层特征图 (base_channels * 16)
            skip_connections: List of [x4, x3, x2, x1]，分辨率逐渐增加
        """
        x4, x3, x2, x1 = skip_connections

        d4 = torch.cat((self.decoder4(x5), self.conv_e4(x4)), dim=1)  # [B, base_channels*8 + base_channels*8, H, W]
        d3 = torch.cat((self.decoder3(d4), self.conv_e3(x3)), dim=1)  # [B, base_channels*4 + base_channels*4, H, W]
        d2 = torch.cat((self.decoder2(d3), self.conv_e2(x2)), dim=1)  # [B, base_channels*2 + base_channels*2, H, W]
        d1 = torch.cat((self.decoder1(d2), self.conv_e1(x1)), dim=1)  # [B, base_channels + base_channels, H, W]
        out = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.final_conv(out)

        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, base_channels, BatchNorm):
    return AocDecoder(num_classes, base_channels, BatchNorm)