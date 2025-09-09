import torch
import torch.nn as nn
import torch.nn.functional as F



class SA2(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SA2, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, 3, padding=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

        # Trainable edge enhancement（可替代Sobel）
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        att_map = self.sigmoid(self.reduce_conv(x))  # [B,1,H,W]

        gray = x.mean(dim=1, keepdim=True)
        edge_map = self.edge_conv(gray)              # 可训练边缘增强

        att = att_map * edge_map + att_map           # 残差连接避免全零
        return x * att
