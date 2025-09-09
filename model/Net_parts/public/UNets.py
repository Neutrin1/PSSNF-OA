import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """双卷积层：卷积 -> ReLU -> 卷积 -> ReLU，使用padding保持尺寸"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetEncoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        x5 = self.bottleneck(self.pool(x4))
        
        return x5, [x1, x2, x3, x4]


class UNetDecoder(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, skip_connections):
        x1, x2, x3, x4 = skip_connections
        
        x = self.up1(x)
        x = self.dec1(torch.cat([x4, x], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x3, x], dim=1))

        x = self.up3(x)
        x = self.dec3(torch.cat([x2, x], dim=1))

        x = self.up4(x)
        x = self.dec4(torch.cat([x1, x], dim=1))

        return self.final_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        self.encoder = UNetEncoder(in_channels)
        self.decoder = UNetDecoder(out_channels)

    def forward(self, x):
        encoded, skip_connections = self.encoder(x)
        output = self.decoder(encoded, skip_connections)
        return output


if __name__ == "__main__":
    model = UNet(in_channels=1, out_channels=2)
    x = torch.randn(1, 1, 512, 512)

    with torch.no_grad():
        y = model(x)
        print("Input:", x.shape)
        print("Output:", y.shape)