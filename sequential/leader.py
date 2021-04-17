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
        self.client_list = []
        self.dW_sum = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.dW_num = 0
        self.dW_avg = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.obj_q = Queue(maxsize=20)
        self.new_W = None

        self.TIME = 0

    def pass_W(self):
        if self.new_W != None:
            for client in self.client_list:
                client.W_new = self.new_W
                client.W_new_recv = True
        self.new_W = None


    def send_dW(self):
        server.obj_q.put(Package(self.dW_sum, self.dW_num))

    def compute_dW(self):
        self.dW_sum = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        dw_t_list = []
        with self.lock:
            while not self.obj_q.empty():
                dw_t_list.append(self.obj_q.get())
                self.obj_q.task_done()

        T = self.TIME
        for dw_t in dw_t_list:
            dw = dw_t.model
            t = dw_t.time
            for name in dw:
                self.dW_sum[name].data += dw[name].data.clone() / (T - t + 1)
        self.dW_num = len(dw_t_list)
