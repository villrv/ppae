import torch, torchvision
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ProgressBar
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
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


# def loglikelihood_single(log_event_rate_list, log_mesh_rate_list, T):
#     '''
#     Likelihood of an event list (t1,...,tn) with Poisson rate function r(t) is:
#         r(t1) * ... * r(tn) * exp(-integral(r(t)))
#     We take the log likelihood for better computational performance
#     log likelihood of a single event list. Needed when we hav event lists of different length
#     '''
#     integral = 0.5 * (torch.sum(torch.exp(log_mesh_rate_list[1:])) + torch.sum(torch.exp(log_mesh_rate_list[:-1]))) * T / (len(log_mesh_rate_list)-1)
#     return torch.sum(log_event_rate_list) - integral

def loglikelihood(log_event_rate_list, T_mask, E_mask, log_mesh_rate_list, T):
    '''
    log likelihood of a batch of event list with the same length.
    Input:
        log_event_rate_list: (B, n_event, E_bins)
        T_mask: (B, n_event), if mask == 0 then it's a padding
        E_mask: (B, n_event, E_bins)
        log_mesh_rate_list: (B, n_mesh, E_bins)
        T: (B,)
    '''
    B, n_mesh, E_bins = log_mesh_rate_list.shape
    integral = 0.5 * (torch.sum(torch.exp(log_mesh_rate_list[:,1:,:]), dim=(1,2)) + torch.sum(torch.exp(log_mesh_rate_list[:,:-1,:]), dim=(1,2))) * T / E_bins / (n_mesh-1)   # (B,)
    return torch.mean(torch.sum(log_event_rate_list * T_mask.unsqueeze(-1) * E_mask, dim=(1,2)) - integral)

def loss_TV(log_rate_list):
    '''
    Calculate total variation for a log rate list
    Input:
        log_rate_list: (B, n, E_bins)
    '''
    return torch.mean(torch.sum(torch.abs(log_rate_list[:,1:,:] - log_rate_list[:,:-1,:]), dim=1))