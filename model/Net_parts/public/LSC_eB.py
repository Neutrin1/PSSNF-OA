import torch
import torch.nn as nn


class LSCEncoderBlock(nn.Module):
    """LSC编码块，用于特征提取和编码"""
    
    def __init__(self, in_channels, n_filters, BatchNorm=nn.BatchNorm2d, downsample=True):
        super(LSCEncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn1 = BatchNorm(in_channels // 4)
        self.relu1 = nn.ReLU()
        self.downsample = downsample

        # 四个条形卷积分支
        self.deconv1 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )
        self.deconv2 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv3 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv4 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )

        # 拼接后的处理
        self.bn2 = BatchNorm(in_channels // 4 + in_channels // 4)  # 4个分支，每个in_channels//8
        self.relu2 = nn.ReLU()
        
        # 输出卷积
        self.conv3 = nn.Conv2d(in_channels // 4 + in_channels // 4, n_filters, 1)
        self.bn3 = BatchNorm(n_filters)
        self.relu3 = nn.ReLU()
        
        # 下采样层（如果需要）
        if self.downsample:
            self.downsample_conv = nn.Conv2d(n_filters, n_filters, 3, stride=2, padding=1, bias=False)
            self.downsample_bn = BatchNorm(n_filters)

        self._init_weight()

    def forward(self, x):
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
        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.bn2(x)
        x = self.relu2(x)
        
        # 输出卷积
        x = self.conv3(x)
        x = self.bn3(x)
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