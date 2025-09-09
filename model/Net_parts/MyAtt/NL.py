import torch
from torch import nn

class NonLocalBlock(nn.Module):
    def __init__(self, channel, size=(32, 32)):
        super(NonLocalBlock, self).__init__()
        self.reduction = 32  # 固定降通道为 32
        self.conv_reduce = nn.Conv2d(channel, self.reduction, kernel_size=1, bias=False)
        self.conv_restore = nn.Conv2d(self.reduction, channel, kernel_size=1, bias=False)

        self.conv_phi = nn.Conv2d(self.reduction, self.reduction, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_theta = nn.Conv2d(self.reduction, self.reduction, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(self.reduction, self.reduction, kernel_size=1, stride=1, padding=0, bias=False)

        self.softmax = nn.Softmax(dim=-1)  # 注意 softmax 应该在最后一个维度上

        self.conv_mask = nn.Conv2d(self.reduction, self.reduction, kernel_size=1, stride=1, padding=0, bias=False)

        self.with_pos = True
        if self.with_pos:
            self.pos_embedding = nn.Parameter(torch.randn(1, size[0] * size[1], size[0] * size[1]))

        self.with_d_pos = True
        if self.with_d_pos:
            self.pos_embedding_decoder = nn.Parameter(torch.randn(1, self.reduction, size[0], size[1]))

    def forward(self, x):
        b, c, h, w = x.size()

        # 降通道
        x_reduced = self.conv_reduce(x)  # [B, 32, H, W]

        # phi / theta / g
        x_phi = self.conv_phi(x_reduced).reshape(b, self.reduction, -1)  # [B, 32, HW]
        x_theta = self.conv_theta(x_reduced).reshape(b, self.reduction, -1).permute(0, 2, 1).contiguous()  # [B, HW, 32]

        if self.with_d_pos:
            x_g = (self.conv_g(x_reduced) + self.pos_embedding_decoder).reshape(b, self.reduction, -1).permute(0, 2, 1).contiguous()
        else:
            x_g = self.conv_g(x_reduced).reshape(b, self.reduction, -1).permute(0, 2, 1).contiguous()  # [B, HW, 32]

        attention = torch.matmul(x_theta, x_phi)  # [B, HW, HW]
        if self.with_pos:
            attention = attention + self.pos_embedding  # [1, HW, HW]
        attention = self.softmax(attention)

        y = torch.matmul(attention, x_g)  # [B, HW, 32]
        y = y.permute(0, 2, 1).contiguous().reshape(b, self.reduction, h, w)  # [B, 32, H, W]
        y = self.conv_mask(y)  # [B, 32, H, W]

        # 升通道 + 残差连接
        y = self.conv_restore(y)  # [B, C, H, W]
        out = y + x  # 残差连接
        return out
