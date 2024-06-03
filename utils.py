import torch, torchvision
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ProgressBar, StochasticWeightAveraging
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from scipy.special import gammaln
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Tuple, Generator, Dict
import os
import sys
import random

import io
import warnings
import time

import math
import functools
import collections
import traceback
from einops import rearrange, repeat

from dotmap import DotMap
from glob import glob

# The following code handles some bug from pytorch lightning when training on Yanke's cluster.
from pytorch_lightning.plugins.environments import SLURMEnvironment
class DisabledSLURMEnvironment(SLURMEnvironment):
    def detect() -> bool:
        return False

    @staticmethod
    def _validate_srun_used() -> None:
        return

    @staticmethod
    def _validate_srun_variables() -> None:
        return
    
def todevice(x, device):
    if isinstance(x,dict):
        for k, v in x.items():
            x[k] = todevice(x[k], device)
    elif isinstance(x,list):
        x = [todevice(i, device) for i in x]
    elif not isinstance(x,str):
        x = x.to(device)
    return x    

def merge_with_mask(event_t_list, T_mask, mesh_t_list):
    '''
    Merge two t lists
    Input:
        event_t_list: (B, n, 1)
        mesh_t_list: (B, m, 1)
        T_mask: (B, n)
    '''

    b, n, _ = event_t_list.shape
    _, m, _ = mesh_t_list.shape
    total_t_list = torch.zeros(b, m+n, 1).to(event_t_list.device)
    total_mask = torch.zeros(b, m+n).to(event_t_list.device)
    for i in range(b):
        nn = torch.sum(T_mask[i,:])    
        total_t_list[i,:(nn+m),0] = torch.sort(torch.cat((event_t_list[i,:nn,0], mesh_t_list[i,:,0])))[0]
        total_mask[i,:(nn+m)] = 1
    return total_t_list, total_mask.bool()  

def interpolate(mesh_rate_list, event_t_list):
    '''
    Input: 
        mesh_rate_list: (B, resolution+1, E_bins)
        event_t_list: (B, n)
    Output:
        event_rate_list: (B, n, E_bins)
    '''
    B, resolutionp1, E_bins = mesh_rate_list.shape
    resolution = resolutionp1 - 1
    B_scaled = event_t_list * resolution
    B_floor = torch.floor(B_scaled).long()
    B_ceil = torch.ceil(B_scaled).long()
    B_remainder = B_scaled - B_floor.float() # (B, n)

    # Gather the values from A
    A_floor = mesh_rate_list.gather(1, B_floor.unsqueeze(2).expand(-1, -1, E_bins))
    A_ceil = mesh_rate_list.gather(1, B_ceil.unsqueeze(2).expand(-1, -1, E_bins))

    # Interpolate
    event_rate_list = A_floor + (A_ceil - A_floor) * B_remainder.unsqueeze(2)

    return event_rate_list
    

def loglikelihood(log_event_rate_list, T_mask, E_mask, log_mesh_rate_list, T):
    '''
    log likelihood of a batch of event list with the same length.
        r(t1) * ... * r(tn) * exp(-integral(r(t)))
    We take the log likelihood for better computational performance
    Input:
        log_event_rate_list: (B, n_event, E_bins)
        T_mask: (B, n_event), if mask == 0 then it's a padding
        E_mask: (B, n_event, E_bins)
        log_mesh_rate_list: (B, n_mesh, E_bins)
        T: (B,)
    '''
    B, n_mesh, E_bins = log_mesh_rate_list.shape
    integral = 0.5 * (
        torch.sum(log_mesh_rate_list[:,1:,:].exp(), dim=(1,2))
        + torch.sum(log_mesh_rate_list[:,:-1,:].exp(), dim=(1,2))
    ) * T / (n_mesh-1)   # (B,)
    return ((log_event_rate_list * T_mask.unsqueeze(-1) * E_mask).sum(dim=(1,2)) - integral).mean()

def total_variation(rate_list, T_mask=None):
    '''
    Calculate total variation for a log rate list. The absolute value of the first entry is calculated twice
    Input:
        rate_list: (B, n, E_bins)
        T_mask: (B, n)
    '''
    if T_mask is not None:
        rate_list = rate_list * T_mask.unsqueeze(-1)
        b, n, e = rate_list.shape
        s = 0
        for i in range(b):
            nn = torch.sum(T_mask[i,:])
            temp = rate_list[i,:nn,:]
            s += torch.sum(torch.abs(torch.diff(temp,dim=0)))
        return s / b
                
    else:
        return (rate_list[:,1:,:] - rate_list[:,:-1,:]).abs().sum(dim=(1,2)).mean()
    

def visualize_hist(times, t_scale):
    times = times / t_scale
    plt.hist(times, bins = torch.arange(torch.ceil(torch.max(times))))