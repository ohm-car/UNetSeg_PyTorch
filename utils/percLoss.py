import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class percLoss(nn.Module):

    def __init__(self, threshold_prob, num_classes=1):
        super().__init__()
        self.threshold_prob = threshold_prob
        self.num_classes = num_classes
        # self.alpha = alpha
        # self.beta = beta
        # self.sim = F.cosine_similarity()

    def forward(self, target, inputs):

        #Compute percentage and compare

        temp = target
        # print(temp)
        temp2 = (temp > self.threshold_prob) * 1
        # print(temp2)
        perc = torch.sum(temp2)/torch.numel(temp2)
        # print(perc)
        l1loss = nn.L1Loss()
        loss = l1loss(inputs, perc)

        return loss