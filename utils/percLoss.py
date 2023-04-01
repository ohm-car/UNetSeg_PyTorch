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

    def forward(self, pred_mask, target):

        #Compute percentage and compare

        #target: Value between 0 and 1, obtained from CSV
        #pred_mask: A tensor with the shape of the image, and channels equal to number of classes. For now 1 channel.

        # temp = pred_mask
        # print(temp)
        # temp2 = (temp > self.threshold_prob) * 1
        # print(pred_mask.shape)
        pred_perc = torch.mean(torch.squeeze(pred_mask), (1,2))
        pred_perc = torch.unsqueeze(pred_perc, 1)
        # print(temp2)
        # perc = torch.sum(temp2)/torch.numel(temp2)
        # print(perc)
        l1loss = nn.L1Loss()
        loss = l1loss(pred_perc, target)

        return loss