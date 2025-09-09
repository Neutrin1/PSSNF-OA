from torch import nn
from PSSNF.model.Net_parts.public.TUAN import UNetFirstEB, UNetRDLEB, UNetRDLEB2
from PSSNF.model.Net_parts.MyAtt.h import EncEnhanceModule, BottleNeckFocusModule   
"""
    小核条形卷积 + 双注意力机制
    条形卷积2 改CBAM
"""
class Encoder18(nn.Module):
    def __init__(self, in_channels, base_channels=32, b1=UNetFirstEB, b2=UNetRDLEB2, use_bn=True, dropout_p=0.15):
        super().__init__()
        # 四层编码器
        self.enc1 = b1(in_channels, base_channels, use_bn=use_bn)
        self.enc2 = b2(base_channels, base_channels * 2, use_bn=use_bn, kernel_size=3, padding=1)
        self.enc3 = b2(base_channels * 2, base_channels * 4, use_bn=use_bn, kernel_size=3, padding=1)
        self.enc4 = b2(base_channels * 4, base_channels * 8, use_bn=use_bn, kernel_size=3, padding=1)
        self.bottom = b2(base_channels * 8, base_channels * 16, use_bn=use_bn, kernel_size=3, padding=1)
        
        # self.att1 = EncEnhanceModule(base_channels)
        # self.att2 = EncEnhanceModule(base_channels * 2)
        # self.att3 = EncEnhanceModule(base_channels * 4)
        # self.att4 = EncEnhanceModule(base_channels * 8)
        # self.bottleneck_focus = BottleNeckFocusModule(base_channels * 16)
        
        
    def forward(self, x):
        x1 = self.enc1(x)
        # x1 = self.att1(x1)
        x2 = self.enc2(x1)
        # x2 = self.att2(x2)
        x3 = self.enc3(x2)
        # x3 = self.att3(x3)
        x4 = self.enc4(x3)
        # x4 = self.att4(x4)
        x5 = self.bottom(x4)
        # x5 = self.bottleneck_focus(x5)
        # print(f"Encoder18: {x5.shape}, {x4.shape}, {x3.shape}, {x2.shape}, {x1.shape}")
        return x5, [x4, x3, x2, x1]
    
    