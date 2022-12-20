'''
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def ComContrastive_Loss(pos_label, sim):
    sim = torch.exp(sim-sim.max())
    return -torch.div(torch.sum(pos_label*sim),torch.sum(sim)).log()/sim.size(0)

def mask_ComContrastive_Loss(pos_label, sim, mask):
    #print(pos_label.size(),sim.size(),mask.size())
    sim = torch.exp(sim-sim.max())*mask
    return -torch.div(torch.sum(pos_label*sim),torch.sum(sim)).log()/sim.size(0)

class SupContrastive_Loss(nn.Module):

    def __init__(self, tau=0.5):
        super(SupContrastive_Loss, self).__init__()
        self.tau = tau

    def similarity(self, x1, x2):
        # Dot Product Kernel
        M = torch.matmul(x1, x2.t())/self.tau
        s = torch.exp(M - torch.max(M, dim=1, keepdim=True)[0])
        return s

    def forward(self, batch_label, *x):
        X = torch.cat(x, 0)  #n*dim
        batch_labels = torch.cat([batch_label for i in range(len(x))], 0)
        len_ = batch_labels.size()[0]

        # computing similarities for each positive and negative pair
        s = self.similarity(X, X)
        mask_i = 1. - torch.from_numpy(np.identity(len_)).to(batch_labels.device) # sum over items in the numerator
        label_matrix = batch_labels.unsqueeze(0).repeat(len_, 1)
        mask_j = (batch_labels.unsqueeze(1) - label_matrix == 0).float()*mask_i # sum over items in the denominator
        #pos_num = torch.sum(mask_j, 1)
        pos_num = torch.clamp(torch.sum(mask_j, 1),1e-10) #change, in case that some classes only have one instance

        # weighted NLL loss
        s_i = torch.clamp(torch.sum(s*mask_i, 1), min=1e-10) 
        s_j = torch.clamp(s*mask_j, min=1e-10)
        log_p = torch.sum(-torch.log(s_j/s_i)*mask_j, 1)/pos_num
        loss = torch.mean(log_p)
        #print(s,s_i,s_j,log_p)

        return loss

class BiSupContrastive_Loss(nn.Module):

    def __init__(self, tau=0.5):
        super(BiSupContrastive_Loss, self).__init__()
        self.tau = tau

    def similarity(self, x1, x2):
        # Dot Product Kernel
        M = torch.matmul(x1, x2.t())/self.tau
        s = torch.exp(M - torch.max(M, dim=1, keepdim=True)[0])
        return s

    def forward(self, batch_label, *x):
        X = torch.cat(x, 0)  #n*dim
        batch_labels = torch.cat([batch_label for i in range(len(x))], 0)
        len_ = batch_labels.size()[0]

        # computing similarities for each positive and negative pair
        s = self.similarity(X, X)
        mask_i = 1. - torch.from_numpy(np.identity(len_)).to(batch_labels.device) # sum over items in the numerator
        label_matrix = batch_labels.unsqueeze(0).repeat(len_, 1)
        mask_j = (batch_labels.unsqueeze(1) - label_matrix == 0).float()*mask_i # sum over items in the denominator
        pos_num = torch.sum(mask_j, 1)
        #pos_num = torch.clamp(torch.sum(mask_j, 1),1e-10) #change, in case that some classes only have one instance

        # weighted NLL loss
        s_i = torch.clamp(torch.sum(s*mask_i, 1), min=1e-10) 
        s_j = torch.clamp(s*mask_j, min=1e-10)
        log_p = torch.sum(-torch.log(s_j/s_i)*mask_j, 1)/pos_num
        loss = torch.mean(log_p)
        #print(s,s_i,s_j,log_p)

        return loss



def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
