import scipy.io as sio
import os
import scipy.io as sio
import torch
import matplotlib.pyplot as plt

import numpy as np
import random
import numpy as np


def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    B = (~is_nan).float().sum(*args, **kwargs)
    if len(B[B == 0]) == len(B):
        #B[B == 0] = 1
        vv = v.sum(*args, **kwargs)
        vv[B == 0] = 1
        m = vv
    else:
        m = v.sum(*args, **kwargs)[B != 0] / (B[B != 0])
        m = m[~torch.isnan(m)]
    m[torch.isnan(m)] = 0


    # if torch.isnan(m).sum() == len(m):
    #     m[torch.isnan(m)] = 0.0001
    # else:
    #     m = m[~torch.isnan(m)]
    #     m[torch.isnan(m)] = 0.0001
    return m

class RMSEbasinLosstest(torch.nn.Module):
    # Same as Fredrick 2019
    def __init__(self):
        super(RMSEbasinLosstest, self).__init__()

    def forward(self, output, target):
        Ngage = target.shape[1]
        losssum = 0
        nsample = 0
        target2 = target.clone()
        output2 = output.clone()
        #target2 = tensor.new_tensor(target)
        #output2 = tensor.new_tensor(output)
        is_nan = torch.isnan(target2)
        B = (~is_nan).float().sum(0)
        target2[is_nan] = 0
        output2[is_nan] = 0
        v = (target2 - output2)**2
        if len(B[B == 0]) == len(B):

            vv = v.sum(0)
            vv[B == 0] = 1
            m = vv
        else:
            m = v.sum(0)[B != 0] / (B[B != 0])
            m[torch.isnan(m)] = 0
        #targetMean=nanmean((target2-output2)**2 ,axis=0).T.repeat([target.shape[0],1]).view(target.shape)
        #targetMean = nanmean((v, axis=0)
        #target2[~mask] = 0
        #output2 = output.clone(); output2[~mask]=0
        #targetMean[~mask] = 0
        loss = torch.mean(torch.sqrt((m)))
        #loss = torch.mean(torch.sqrt(targetMean))
        #
        # num = torch.sum((output2 - target2)**2,axis=0)
        # denom = torch.sum((target2 - targetMean)**2,axis=0)
        #
        # maskSum = (denom==0)
        # num[maskSum]=1
        # denom[maskSum]=1
        # NSE = 1-num/denom
        # loss = torch.mean(NSE[~maskSum])
        return loss



