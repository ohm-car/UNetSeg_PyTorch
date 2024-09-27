""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.aaMaxPool import MaxPool2dAA


class DoubleConv75(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if dropout is not None:
            self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=7, padding='same'),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )
        else:    
            self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=7, padding='same'),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if dropout is not None:
            self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )
        else:    
            self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class ResnetMiniBlock(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dropout = None):
        super().__init__()


        self.dropout = dropout
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if dropout:
            self.drop = nn.Dropout(p=dropout)

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.dropout is not None:
            out = self.drop(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        if self.dropout is not None:
            out = self.drop(out)

        return out


class InitR(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            # nn.MaxPool2d(2),
            DoubleConv75(in_channels, out_channels),
            ResnetMiniBlock(out_channels, out_channels),
            ResnetMiniBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.conv(x)

class DownR(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, num_rm=2):
        super().__init__()
        mid_channels = (in_channels + out_channels) // 2
        self.maxpool_conv = nn.Sequential(
            MaxPool2dAA(),
            # nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
            # ,nn.Dropout(p=0.2)
        )

        for i in range(num_rm):
            self.maxpool_conv.append(ResnetMiniBlock(out_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UpR(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, num_rm=2, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        mid_channels = (in_channels + out_channels) // 2
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            self.conv = nn.Sequential(
                DoubleConv(in_channels, out_channels)
                # ,nn.Dropout(p=0.2)
                )
            for i in range(num_rm):
                self.conv.append(ResnetMiniBlock(out_channels, out_channels))
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            # self.conv = DoubleConv(in_channels, out_channels)
            self.conv = nn.Sequential(
                DoubleConv(in_channels, out_channels)
                # ,nn.Dropout(p=0.2)
                )
            for i in range(num_rm):
                self.conv.append(ResnetMiniBlock(out_channels, out_channels))


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1),
#             nn.Sigmoid()
#             )

#     def forward(self, x):
#         return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Dropout(p=0.3)
            # nn.Softmax(dim=0)
            )

    def forward(self, x):
        return self.conv(x)

class OutConvRecon(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConvRecon, self).__init__()
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
            )

    def forward(self, x):
        return self.conv(x)