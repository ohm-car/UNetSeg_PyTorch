""" Full assembly of the parts to form the complete network 
This is the extended unet model with more hidden layers on the mask branch of the DNN."""

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)   #160, 64
        self.down1 = Down(32, 64)              #80, 128
        # self.down2 = Down(128, 256)             #40, 256
        # self.down3 = Down(256, 512)             #20, 512
        factor = 2 if bilinear else 1
        self.down2 = Down(64, 128 // factor)  #10, 512
        self.up1 = Up(128, 64 // factor, bilinear)    #20, 256
        # self.up2 = Up(512, 256 // factor, bilinear)     #40, 128
        # self.up3 = Up(256, 128 // factor, bilinear)     #80, 64
        self.up2 = Up(64, 32, bilinear)                #160, 64
        self.outI = OutConvRecon(32, n_channels)        #160, 3
        self.ext1 = DoubleConv(32, 64)
        self.ext2 = DoubleConv(64, 64)
        self.ext3 = DoubleConv(64, 32)
        self.outM = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        im_recon = self.outI(x)
        x = self.ext1(x)
        x = self.ext2(x)
        x = self.ext3(x)
        mask_logits = self.outM(x)
        return {'aux' : im_recon, 'out' : mask_logits}