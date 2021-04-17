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
from data_utils import split_data, CustomSubset


class Server(FederatedTrainingDevice):
    def __init__(self, model_fn, data, testloader, client_list=[]):
        super().__init__(model_fn, data)
        self.testloader = testloader
        self.dw_q = Queue(maxsize=20)
        self.obj_q = Queue(maxsize=20)
        self.client_list = client_list
        self.TIME = 0

    def update(self):
        if not self.obj_q.empty():
            obj = self.obj_q.get()
            self.obj_q.task_done()
            lr = 0.001
            t = obj.time
            N = 4
            alpha = lr / (self.TIME - t + 1) * obj.num / N
            dw = obj.model
            for name in dw:
                self.W[name].data -= dw[name].data.clone() * alpha
            print("[Server - upd]: Updated model with T = %s, t = %s, num = %s, alpha = %s" % (self.TIME, t, obj.num, alpha))


    def eval(self):
        rd = 1
        start_time = time.time()
        try:
            acc = self.evaluate()
            f = open("lol.txt", "a")
            f.write("%s, %s, %s\n" % (rd, acc, int(time.time() - start_time)))
            f.close()
            print("[Server - eval] rd = %s, acc = [ %s ]" % (rd, acc))
            rd += 1
        except:
            print("[Server - eval] - error")
    
    
    def send(self):
        for client in self.client_list:
            client.new_W = self.W
        self.TIME += 1

    def select_clients(self, clients, frac=1.0):
        return random.sample(clients, int(len(clients)*frac))

