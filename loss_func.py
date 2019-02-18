import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pdb
import numpy as np

def MY_BCELoss(input, target, size_average=True):
    eps = 1e-12
    weight = 1
    if (1-target).data.sum() > 0:
        weight = (1-target).data.sum() / target.data.sum()
    loss = - weight * target * input.clamp(min=eps).log() - 1 * (1 - target) * (1 - input).clamp(min=eps).log()
    if size_average:
        loss = loss.mean()
    return loss


def binary_cross_entropy2d(input, target, gpu=None, weight=None, size_average=False):
    # input: (n, c, h, w), target: (n, h, w)

    n, c, h, w = input.size()
    input = nn.Sigmoid()(input)
    input = input.view(n, -1)
    target = target.view(n, -1).float()
    # input = nn.Sigmoid()(input) ---> Applied Sigmoid in the main function, not required here then.
    LOSS = nn.BCELoss()
    if gpu is not None:
        LOSS = LOSS.to(gpu)
    mask = target >= 0
    loss = LOSS(input[mask], target[mask])
    #loss = MY_BCELoss(input[mask], target[mask])
    if size_average:
        loss /= mask.data.sum()
    return loss


def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    mask = mask.float()
    ### either:
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    ### or:
    # target = target.unsqueeze(1)
    # target_onehot = torch.zeros(target.size()[0],c).scatter_(1, target.cpu().data, torch.ones(target.size()[0],1)).cuda()
    # loss = - torch.sum(log_p * Variable(target_onehot))
    if size_average:
        loss /= mask.data.sum()
    return loss
