import os
import sys
import time
import struct
from datetime import datetime
from copy import deepcopy

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from models import CF10Net
from devices import *
from data_utils import split_data, CustomSubset


class Client(FederatedTrainingDevice):
    def __init__(self, model_fn, optimizer_fn, data, id, parent=None, batch_size=128, train_frac=0.8, l2_lambda=0.01, seed=0, asynch=True):
        super().__init__(model_fn, data)
        self.optimizer = optimizer_fn(self.model.parameters())

        self.data_train = data
        self.train_loader = DataLoader(self.data_train, batch_size=batch_size, shuffle=True)

        self.id = id
        self.parent = parent

        self.dW = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}

        self.W_new = {key : value for key, value in self.model.named_parameters()}
        self.W_new_recv = False

        self.TIME = -1

        self.l2_lambda = l2_lambda

        self.seed = seed
        self.asynch = asynch


    def train(self):
        if self.TIME != -1:
            copy(target=self.W, source=self.W_new)
#        print("[Client - %s - trn] TIME = %s - sync done" % (self.id, self.TIME))
        train_stats = self.compute_weight_update(epochs=1)
#        print("[Client - %s - trn] TIME = %s - train done" % (self.id, self.TIME))
        self.reset()


    def send(self):
        self.parent.obj_q.put(Package(self.TIME, self.dW))


    def compute_weight_update(self, epochs=1, loader=None):
        copy(target=self.W_old, source=self.W)
#        self.optimizer.param_groups[0]["lr"]*=0.99
        train_stats = train_op(self.model, self.train_loader if not loader else loader, self.optimizer, epochs, self.W_old, self.l2_lambda, self.seed, self.asynch)
        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)
        return train_stats

    def reset(self):
        copy(target=self.W, source=self.W_old)


