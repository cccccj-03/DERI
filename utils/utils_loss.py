import os
from typing import Type
import torch
import torch.nn.functional as F
import pandas as pd
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from tqdm import tqdm
import numpy as np

def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous(
                ).view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
    
def clip_loss(x, y, temperature=0.07, device='cuda'):

    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    sim = torch.einsum('i d, j d -> i j', x, y) * 1 / temperature

    labels = torch.arange(x.shape[0]).to(device)

    loss_t = F.cross_entropy(sim, labels) 
    loss_i = F.cross_entropy(sim.T, labels) 

    i2t_acc1, i2t_acc5 = precision_at_k(
        sim, labels, top_k=(1, 5))
    t2i_acc1, t2i_acc5 = precision_at_k(
        sim.T, labels, top_k=(1, 5))
    acc1 = (i2t_acc1 + t2i_acc1) / 2.
    acc5 = (i2t_acc5 + t2i_acc5) / 2.

    return (loss_t + loss_i), acc1, acc5

def weight_loss(x, y, weight, device='cuda', flag=0):
    idx1 = np.argwhere(weight.detach().cpu().numpy() == 1).reshape(-1)
    idx2 = np.argwhere(weight.detach().cpu().numpy() <= 1).reshape(-1)
    if len(idx2) > 0: 
        x1 = x[idx1].to(device)
        y1 = y[idx1].to(device)

        x2 = x[idx2].to(device)
        y2 = x[idx2].to(device)

        loss, acc1, acc5 = clip_loss(x1, y1, device=device)
        loss_code, acc1_code, acc5_code = clip_loss(x2, y2, device=device)

        if flag == 0:
            loss = loss + 0.25 * loss_code
        elif flag == 2:
            loss = loss + 0.05*loss_code#0.06 * loss_code#+ 0.05 * loss_code
        else:#
            loss = loss + 0.05 * loss_code
    else:
        loss, acc1, acc5 = clip_loss(x, y, device=device)

    return loss, acc1, acc5


#### 先Mimic，然后再冻结text部分上code