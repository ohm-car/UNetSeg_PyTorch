""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .voc_unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        k = 64

        self.inc = InitR(n_channels, k)
        self.down1 = DownR(k, 2*k)
        self.down2 = DownR(2*k, 4*k)
        # factor = 2 if bilinear else 1
        # self.down3 = Down(128, 256 // factor)
        self.down3 = DownR(4*k, 8*k)
        # self.down3 = Down(256, 512 // factor)
        factor = 2 if bilinear else 1
        self.down4 = DownR(8*k, 16*k // factor)
        # self.up1 = Up(1024, 512 // factor, bilinear)
        # self.up2 = Up(512, 256 // factor, bilinear)
        self.up0 = UpR(16*k, 8*k // factor, bilinear)
        self.up1 = UpR(8*k, 4*k // factor, bilinear)
        self.up2 = UpR(4*k, 2*k // factor, bilinear)
        self.up3 = UpR(2*k, k, bilinear)
        self.outc = OutConv(k, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up0(x5, x4)
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits
