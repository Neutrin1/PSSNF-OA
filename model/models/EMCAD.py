import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .EMCAD_parts import EMCAD                   


class EMCADNet(nn.Module):
    def __init__(self, num_classes=1, in_channels=3, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, lgag_ks=3, activation='relu', encoder='resnet50', pretrain=True):
        super(EMCADNet, self).__init__()

        # 如果输入通道数不是3，则添加一个卷积层进行通道转换
        if in_channels != 3:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 3, kernel_size=1),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = None
        
        # backbone network initialization with pretrained weight
        if encoder == 'resnet18':
            self.backbone = resnet18(pretrained=pretrain)
            channels = [512, 256, 128, 64]
        elif encoder == 'resnet34':
            self.backbone = resnet34(pretrained=pretrain)
            channels = [512, 256, 128, 64]
        elif encoder == 'resnet50':
            self.backbone = resnet50(pretrained=pretrain)
            channels = [2048, 1024, 512, 256]
        elif encoder == 'resnet101':
            self.backbone = resnet101(pretrained=pretrain)
            channels = [2048, 1024, 512, 256]
        elif encoder == 'resnet152':
            self.backbone = resnet152(pretrained=pretrain)
            channels = [2048, 1024, 512, 256]
        else:
            raise ValueError('Encoder not implemented! Please choose from resnet18, resnet34, resnet50, resnet101, resnet152.')
        
        print('Model %s created, param count: %d' %
              (encoder + ' backbone: ', sum([m.numel() for m in self.backbone.parameters()])))
        
        # decoder initialization
        self.decoder = EMCAD(channels=channels, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, lgag_ks=lgag_ks, activation=activation)
        
        print('Model %s created, param count: %d' %
              ('EMCAD decoder: ', sum([m.numel() for m in self.decoder.parameters()])))
             
        self.out_head4 = nn.Conv2d(channels[0], num_classes, 1)
        self.out_head3 = nn.Conv2d(channels[1], num_classes, 1)
        self.out_head2 = nn.Conv2d(channels[2], num_classes, 1)
        self.out_head1 = nn.Conv2d(channels[3], num_classes, 1)
        
    def forward(self, x, mode='test'):
        
        # 如果输入通道数不是3，则通过conv转换
        if self.conv is not None:
            x = self.conv(x)
        
        # encoder
        x1, x2, x3, x4 = self.backbone(x)

        # decoder
        dec_outs = self.decoder(x4, [x3, x2, x1])
        
        # prediction heads  
        p4 = self.out_head4(dec_outs[0])
        p3 = self.out_head3(dec_outs[1])
        p2 = self.out_head2(dec_outs[2])
        p1 = self.out_head1(dec_outs[3])

        p4 = F.interpolate(p4, scale_factor=32, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=16, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=8, mode='bilinear')
        p1 = F.interpolate(p1, scale_factor=4, mode='bilinear')

        if mode == 'test':
            # return [p4, p3, p2, p1]
            return p4
        # return [p4, p3, p2, p1]
        return p4
if __name__ == '__main__':
    model = EMCADNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    P = model(input_tensor)
    print(P[0].size(), P[1].size(), P[2].size(), P[3].size())