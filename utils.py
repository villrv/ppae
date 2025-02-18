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

# The following code handles some bug from pytorch lightning when training on some specific SLURM cluster.
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

def load_from_less_latents(model, small_dset, big_dset, ckpt_path):
    '''
    Load a pretrained checkpoint that is trained on less latents.
    Latents with the same id are directly loaded. 
    '''
    state = torch.load(ckpt_path)
    state_dict = state['state_dict']
    del state
    small_latents = state_dict['latent']
    del state_dict['latent']
    model.load_state_dict(state_dict, strict=False)
    del state_dict

    # Load latents
    i = 0
    j = 0
    m = len(small_dset)
    assert small_latents.shape[0] == m
    small_latents = small_latents.detach()
    n = len(big_dset)
    with torch.no_grad():
        while True:
            if i == m:
                print('loading complete')
                break
            if j == n and i < m:
                raise ValueError("Not everything is loaded")
            if small_dset[i]['id'] == big_dset[j]['id']:
                model.latent[j].copy_(small_latents[i])
                i += 1
                j += 1
            else:
                j += 1
            
    return model

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
    
def total_variation_normalized(rate_list, T_mask=None):
    '''
    Calculate normalized total variation for a log rate list. The absolute value of the first entry is calculated twice
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
            if nn == 1:
                continue
            temp = rate_list[i,:nn,:]
            s += torch.diff(temp,dim=0).abs().mean()
        return s / b
                
    else:
        return (rate_list[:,1:,:] - rate_list[:,:-1,:]).abs().mean()