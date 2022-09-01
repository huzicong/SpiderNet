"""
SpiderNet: Multi-stage Depth Completion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling import GuidedConv


######################################## ConvBlock ##############################################
class ConvLayers(nn.Module):
    def __init__(self, channels):
        super(ConvLayers, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, num_convs, channels):
        super(ConvBlock, self).__init__()
        convs_layers = []
        for i in range(num_convs):
            convs_layers.append(ConvLayers(channels))
        self.conv_block = nn.Sequential(*convs_layers)

        self.alpha = nn.Parameter(torch.FloatTensor([1.0]))

    def forward(self, x):
        return self.conv_block(x) + self.alpha * x


######################################## UNet ###############################################
# DownSample
class DownSample(nn.Module):
    def __init__(self, numconv, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.guidedconv = GuidedConv.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ConvBlock(numconv, out_channels),
            nn.AvgPool2d(kernel_size=2)
        )

    def forward(self, x, kernel_feat):
        x = self.guidedconv(kernel_feat, x)
        return self.down(x)


# UpSample
class UpSample(nn.Module):
    def __init__(self, numconv, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.guidedconv = GuidedConv.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ConvBlock(numconv, out_channels),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

    def forward(self, x, kernel_feat):
        x = self.guidedconv(kernel_feat, x)
        return self.up(x)


class DualUNet(nn.Module):
    """
        Args:
            numconv (int): the number of convs in UNet block
            mid_channels (list): the mid channels of unet, including every encoder layers' output channel
                                the length of list should match the number of encoders
    """
    def __init__(self, numconv, mid_channels):
        super(DualUNet, self).__init__()
        self.rgb_encoder1 = DownSample(numconv, in_channels=mid_channels[0], out_channels=mid_channels[1])
        self.dep_encoder1 = DownSample(numconv, in_channels=mid_channels[0], out_channels=mid_channels[1])

        self.rgb_encoder2 = DownSample(numconv, in_channels=mid_channels[1], out_channels=mid_channels[2])
        self.dep_encoder2 = DownSample(numconv, in_channels=mid_channels[1], out_channels=mid_channels[2])

        self.rgb_encoder3 = DownSample(numconv, in_channels=mid_channels[2], out_channels=mid_channels[3])
        self.dep_encoder3 = DownSample(numconv, in_channels=mid_channels[2], out_channels=mid_channels[3])

        self.rgb_mid = nn.Sequential(
            nn.Conv2d(mid_channels[3], mid_channels[3], 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.dep_mid = nn.Sequential(
            nn.Conv2d(mid_channels[3], mid_channels[3], 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        self.rgb_decoder1 = UpSample(numconv, in_channels=mid_channels[3], out_channels=mid_channels[2])
        self.dep_decoder1 = UpSample(numconv, in_channels=mid_channels[3], out_channels=mid_channels[2])

        self.rgb_decoder2 = UpSample(numconv, in_channels=mid_channels[2], out_channels=mid_channels[1])
        self.dep_decoder2 = UpSample(numconv, in_channels=mid_channels[2], out_channels=mid_channels[1])

        self.rgb_decoder3 = UpSample(numconv, in_channels=mid_channels[1], out_channels=mid_channels[0])
        self.dep_decoder3 = UpSample(numconv, in_channels=mid_channels[1], out_channels=mid_channels[0])

    def forward(self, rgb_feat, dep_feat):
        rgb_feat1 = self.rgb_encoder1(rgb_feat, dep_feat)
        dep_feat1 = self.dep_encoder1(dep_feat, rgb_feat)

        rgb_feat2 = self.rgb_encoder2(rgb_feat1, dep_feat1)
        dep_feat2 = self.dep_encoder2(dep_feat1, rgb_feat1)

        rgb_feat3 = self.rgb_encoder3(rgb_feat2, dep_feat2)
        dep_feat3 = self.dep_encoder3(dep_feat2, rgb_feat2)

        rgb_feat_in = self.rgb_mid(rgb_feat3) + rgb_feat3
        dep_feat_in = self.dep_mid(dep_feat3) + dep_feat3

        rgb_feat = self.rgb_decoder1(rgb_feat_in, dep_feat_in)
        dep_feat = self.dep_decoder1(dep_feat_in, rgb_feat_in)

        rgb_feat_in = rgb_feat + rgb_feat2
        dep_feat_in = dep_feat + dep_feat2

        rgb_feat = self.rgb_decoder2(rgb_feat_in, dep_feat_in)
        dep_feat = self.dep_decoder2(dep_feat_in, rgb_feat_in)

        rgb_feat_in = rgb_feat + rgb_feat1
        dep_feat_in = dep_feat + dep_feat1

        rgb_feat = self.rgb_decoder3(rgb_feat_in, dep_feat_in)
        dep_feat = self.dep_decoder3(dep_feat_in, rgb_feat_in)

        return rgb_feat, dep_feat

############################ Channel Attention ###############################
# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.conv_du(x)
        return att * x


######################################## ORBNet ###############################################
# Scale-Invariant Block (SIB)
class SIB(nn.Module):
    def __init__(self, in_channels, reduction):
        super(SIB, self).__init__()
        self.guidedconv = GuidedConv.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv = ConvBlock(2, in_channels)
        self.ca = CALayer(in_channels, reduction)

    def forward(self, x, kernel_feat):
        res = self.guidedconv(kernel_feat, x)
        res = self.conv(res)
        res = self.ca(res)
        return res + x


# Scale-Invariant Net
class DualSINet(nn.Module):
    def __init__(self, mid_channel, reduction):
        super(DualSINet, self).__init__()
        self.rgb_orb1 = SIB(mid_channel, reduction)
        self.dep_orb1 = SIB(mid_channel, reduction)

        self.rgb_orb2 = SIB(mid_channel, reduction)
        self.dep_orb2 = SIB(mid_channel, reduction)

        self.rgb_orb3 = SIB(mid_channel, reduction)
        self.dep_orb3 = SIB(mid_channel, reduction)

        self.rgb_alpha = nn.Parameter(torch.FloatTensor([1.0]))
        self.dep_alpha = nn.Parameter(torch.FloatTensor([1.0]))

    def forward(self, rgb_feat, dep_feat):
        rgb_feat_out = self.rgb_orb1(rgb_feat, dep_feat)
        dep_feat_out = self.dep_orb1(dep_feat, rgb_feat)

        rgb_feat_in = rgb_feat_out
        dep_feat_in = dep_feat_out

        rgb_feat_out = self.rgb_orb2(rgb_feat_in, dep_feat_in)
        dep_feat_out = self.dep_orb2(dep_feat_in, rgb_feat_in)

        rgb_feat_in = rgb_feat_out
        dep_feat_in = dep_feat_out

        rgb_feat_out = self.rgb_orb3(rgb_feat_in, dep_feat_in) + self.rgb_alpha * rgb_feat
        dep_feat_out = self.dep_orb3(dep_feat_in, rgb_feat_in) + self.dep_alpha * dep_feat

        return rgb_feat_out, dep_feat_out


#################################### Uncertainty Block ###########################################
class UncertBlock(nn.Module):
    def __init__(self, channels):
        super(UncertBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(channels // 2, channels // 4, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(channels // 4, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.block(x)


######################################## MDCNet ###############################################
class SpiderNet(nn.Module):
    def __init__(self, numStage, reduction=4):
        super(SpiderNet, self).__init__()
        self.numStage = numStage

        ########################## stage1 ##########################
        # feat-extraction
        self.rgb_in1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dep_in1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.stage1 = DualUNet(2, [16, 32, 64, 128])

        # uncertainty block
        self.uncert1 = UncertBlock(16)

        # out
        self.out1 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        ########################## stage2 ##########################
        self.stage2 = DualUNet(2, [16, 32, 64, 128])

        # uncertainty block
        self.uncert2 = UncertBlock(16)

        # out
        self.out2 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        ########################## stage3 ##########################
        self.stage3 = DualSINet(16, reduction)

        # uncertainty block
        self.uncert3 = UncertBlock(16)

        # out
        self.out3 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_input, depth_input):
        ########################## stage1 ##########################
        # feat-extraction
        rgb_feat1 = self.rgb_in1(rgb_input)
        dep_feat1 = self.dep_in1(depth_input)

        rgb_feat1, dep_feat1 = self.stage1(rgb_feat1, dep_feat1)

        # uncertainty block
        sigma1 = self.uncert1(rgb_feat1 + dep_feat1)

        # out
        out1 = self.out1(rgb_feat1 + dep_feat1)
        if self.numStage == 1:
            return out1, sigma1

        ########################## stage2 ##########################
        # feat-extraction
        rgb_feat2, dep_feat2 = self.stage2(rgb_feat1, dep_feat1)

        # uncertainty block
        sigma2 = self.uncert2(rgb_feat2 + dep_feat2)

        # out
        out2 = self.out2(rgb_feat2 + dep_feat2)
        if self.numStage == 2:
            return out1, sigma1, out2, sigma2

        ########################## stage3 ##########################
        # feat-extraction
        rgb_feat3, dep_feat3 = self.stage3(rgb_feat2, dep_feat2)

        # uncertainty block
        sigma3 = self.uncert3(rgb_feat3 + dep_feat3)

        # out
        out3 = self.out3(rgb_feat3 + dep_feat3)
        if self.numStage == 3:
            return out1, sigma1, out2, sigma2, out3, sigma3

    def changeStage(self, stage):
        self.numStage = stage


if __name__ == '__main__':
    model = SpiderNet(numStage=3, reduction=4).cuda()
    total_param = sum(p.numel() for p in model.parameters())
    print(total_param)
