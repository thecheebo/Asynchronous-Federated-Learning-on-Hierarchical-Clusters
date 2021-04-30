import os
import sys
import time
from datetime import datetime
from copy import deepcopy
from queue import Queue

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

import models
from models import CF10Net
from devices import *


class Leader(FederatedTrainingDevice):
    def __init__(self, model_fn, id, server=None):
        super().__init__(model_fn, None)
        self.id = id
        self.server = server
        self.child_list = []
        self.dW_sum = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.dW_num = 0
        self.dW_avg = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.obj_q = Queue(maxsize=20)
        self.W_new = None
        self.TIME = -1


    def pass_W(self):
        if self.W_new != None:
            for child in self.child_list:
                child.W_new = self.W_new
                child.W_new_recv = True
                child.TIME_new = self.TIME
        self.W_new = None


    def compute_dW(self):
        self.dW_sum = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.dW_num = 0
        T = self.TIME
        while not self.obj_q.empty():
            self.dW_num += 1
            obj = self.obj_q.get()
            self.obj_q.task_done()
            dw = obj.model
            t = obj.time
            for name in dw:
                self.dW_sum[name].data += dw[name].data.clone() / (T - t + 1)


    def send_dW(self):
        self.server.obj_q.put(Package(self.TIME, self.dW_sum, self.dW_num))


