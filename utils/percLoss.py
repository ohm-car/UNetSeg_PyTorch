import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.bernoulli import Bernoulli

class percLoss(nn.Module):

    def __init__(self, threshold_prob=0.9, num_classes=1, regularizer=None, regularizer_weight=0.1, sampler=None):
        super().__init__()
        # threshold_prob hopefully to be used later
        self.threshold_prob = threshold_prob

        self.num_classes = num_classes
        self.rw = regularizer_weight
        self.regularizer = regularizer
        self.sampler = sampler
        # self.alpha = alpha
        # self.beta = beta
        # self.sim = F.cosine_similarity()

    def forward(self, pred_mask, target):

        #Compute percentage and compare

        #target: Value between 0 and 1, obtained from CSV
        #pred_mask: A tensor with the shape of the image, and channels equal to number of classes. For now 1 channel.
        # print(pred_mask.size())

        # Bernoulli differentiable sampler, If required. Not used by default.
        if(self.sampler == 'bernoulli'):
            pred_mask = self.bernoulli_sample(pred_mask)

        # temp = pred_mask
        # print(temp)
        # temp2 = (temp > self.threshold_prob) * 1


        # print("Pred_Mask Shape:", pred_mask.shape)
        # pred_perc = torch.mean(torch.squeeze(pred_mask), (1,2))
        # print("Pred_Perc:", pred_perc)
        # pred_perc = torch.unsqueeze(pred_perc, 1)
        # print("Unsqueezed_Pred_Perc:", pred_perc)

        pred_perc = torch.mean(pred_mask, axis=(2,3))
        # print("Pred_Perc:", pred_perc)
        # print("Target:", target)

        # print("Pred_Perc_Size:", pred_perc.size(), "Target_Size:", target.size())

        # print(temp2)
        # perc = torch.sum(temp2)/torch.numel(temp2)
        # print(perc)
        # l1loss = nn.L1Loss(reduction='sum')
        l1loss = nn.L1Loss()
        # reg_loss = self.omkar_regularize(pred_mask)
        # reg_loss = self.edward_regularize(pred_mask)
        # reg_loss = self.bc_entropy(pred_mask)

        reg_loss = self.regularize(self.regularizer, pred_mask)

        reg = self.rw * reg_loss

        loss = l1loss(pred_perc, target)

        # print("Loss:", loss)

        return loss + reg

    def regularize(self, regularizer, pred_mask):

        match self.regularizer:
            case None:
                return 0
            case 'omkar':
                return self.omkar_regularize(pred_mask)
            case 'edward':
                return self.edward_regularize(pred_mask)
            case 'bce':
                return self.bc_entropy(pred_mask)

    def omkar_regularize(self, pred_mask):

        return torch.mean(1 / (1 + torch.exp(torch.abs(pred_mask - 0.5)))) - (1 / (1 + torch.exp(torch.tensor(0.5))))
        # return torch.mean(1 / (1 + torch.exp(torch.abs(pred_mask - 0.5))))

    def edward_regularize(self, pred_mask):

        return torch.pow(torch.tensor(0.5), 2) - torch.mean(torch.pow(0.5 - pred_mask, 2))

    def bc_entropy(self, pred_mask):

        targets = torch.argmax(pred_mask, dim=0)
        return F.cross_entropy(pred_mask, targets)

        # targets = torch.round(pred_mask)
        # return F.binary_cross_entropy(pred_mask, targets)

    def bernoulli_sample(self, pred_mask):

        sampled_pred_mask = Bernoulli(pred_mask)

        return sampled_pred_mask.rsample()