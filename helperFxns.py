"""
Author: Devesh Shah
Project Title: Pytorch Base Format

This file
"""

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict, namedtuple
from itertools import product
import time
import json
import pandas as pd
from networks import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NetworkFactory():
    @staticmethod
    def get_network(name):
        if name == "Network1":
            return Network1()
        elif name == "Network2":
            return Network2()
        elif name == "Network2_DO":
            return Network2(conv_dropout=0.2, fc_dropout=0.5)
        elif name == "Network2withBN":
            return Network2withBN(conv_dropout=0.2, fc_dropout=0.5)


class OptimizerFactory():
    @staticmethod
    def get_optimizer(name, params, lr, momentum=0, dampening=0, betas=(0.9, 0.99), eps=1e-08, weight_decay=0, alpha=0.99):
        if name == "Adam":
            optimizer = torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        elif name == "SGD":
            optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay)
        elif name == "RMSprop":
            optimizer = torch.optim.RMSprop(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum)
        else:
            raise Exception("Invalid Optimizer Type Specified")
        return optimizer


class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs


class RunManager():
    def __init__(self):
        self.epoch_count = 0