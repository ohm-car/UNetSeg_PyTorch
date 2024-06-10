import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torchvision.transforms
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

# from tqdm.autonotebook import tqdm

# from idlmam import set_seed

import scipy
import scipy.ndimage

import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import imshow
# %matplotlib inline
# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('png', 'pdf')

import pandas as pd

from sklearn.metrics import accuracy_score

import time

# from idlmam import LastTimeStep, train_network, Flatten, weight_reset, View, LambdaLayer
# from idlmam import AttentionAvg, GeneralScore, DotScore, AdditiveAttentionScore, getMaskByFill

import os

class BlurLayer(nn.Module):
    def __init__(self, kernel_size=5, stride=2, D=2):
        """
        kernel_size: how wide should the blurring be
        stride: how much should the output shrink by
        D: how many dimensions in the input. D=1, D=2, or D=3 for tensors of shapes (B, C, W), (B, C, W, H), (B, C, W, H, Z) respectively.
        """
        super(BlurLayer, self).__init__()
        
        base_1d = scipy.stats.binom.pmf(list(range(kernel_size)), kernel_size, p=0.5)#make a 1d binomial distribution. This computes the normalized filter_i value for all k values.
        #z is a 1d filter
        if D <= 0 or D > 3:
            raise Exception() #invalid option for D!
        if D >= 1:
            z = base_1d #we are good
        if D >= 2:
            z = base_1d[:,None]*z[None,:] #the 2-d filter can be made by multiplying two 1-d filters
        if D >= 3:
            z = base_1d[:,None,None]*z #the 3-d filter can be made by multiplying the 2-d version with a 1-d version
        #Applying the filter is a convolution, so we will save the filter as a parameter in this layer. requires_grad=False because we don't want it to change
        self.weight = nn.Parameter(torch.tensor(z, dtype=torch.float32).unsqueeze(0), requires_grad=False)
        self.stride = stride

    def forward(self, x):
        C = x.size(1) #How many channels are here? 
        ks = self.weight.size(0)#How wide was our internal filter?

        #All three calls are the same, we just need to know which conv function should we call?
        #The groups argument is used to apply the single filter to every channel, since we don't have multipler filters like a normal convolutional layer.
        if len(self.weight.shape)-1 == 1:
            return F.conv1d(x, torch.stack([self.weight]*C), stride=self.stride, groups=C, padding=ks//self.stride)
        elif len(self.weight.shape)-1 == 2:
            return F.conv2d(x, torch.stack([self.weight]*C), stride=self.stride, groups=C, padding=ks//self.stride)
        elif len(self.weight.shape)-1 == 3:
            return F.conv3d(x, torch.stack([self.weight]*C), stride=self.stride, groups=C, padding=ks//self.stride)
        else:
            raise Exception() #We should never reach this code, lets us know we have a bug if it happens!


class MaxPool2dAA(nn.Module):
    def __init__(self, kernel_size=2, ratio=1.7):
        """
        kernel_size: how much to pool by
        ratio: how much larger the bluring filter should be than the pooling size
        """
        super(MaxPool2dAA, self).__init__()

        blur_ks = int(ratio*kernel_size) #make a slightly larger filter for bluring
        self.blur = BlurLayer(kernel_size=blur_ks, stride=kernel_size, D=2) #create the blur kernel
        self.kernel_size = kernel_size #and store the pooling size

    def forward(self, x):
        ks = self.kernel_size 
        tmp = F.max_pool2d(x, ks, stride=1, padding=ks//2) #Apply pooling with a stride=1
        return self.blur(tmp) #blue the result