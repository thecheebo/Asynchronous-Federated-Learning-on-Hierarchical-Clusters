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
    def __init__(self, model_fn, data, testloader, child_list=[], lr=0.001, N=1, beta=5):
        super().__init__(model_fn, data)
        self.testloader = testloader
        self.obj_q = Queue(maxsize=20)
        self.child_list = child_list
        self.TIME = 0
        self.start_time = time.time()
        self.N = N
        self.lr = lr
        self.beta = beta


    def send(self):
        for child in self.child_list:
            child.W_new = self.W
            child.TIME = self.TIME
        self.TIME += 1


    def update(self):
        while not self.obj_q.empty():
            obj = self.obj_q.get()
            self.obj_q.task_done()
            t = obj.time
            alpha = self.lr * pow((self.TIME - t + 1), self.beta) * obj.num / self.N
            dw = obj.model
            for name in dw:
                self.W[name].data += dw[name].data * alpha
            print("[Server - upd]: Updated model with T = %s, t = %s, num = %s, alpha = %s" % (self.TIME, t, obj.num, alpha))


    def eval(self):
        try:
            acc = self.evaluate()
            f = open("lol.txt", "a")
            f.write("%s, %s, %s\n" % (self.TIME, acc, int(time.time() - self.start_time)))
            f.close()
            print("[Server - eval] TIME = %s, acc = [ %s ]" % (self.TIME, acc))
            rd += 1
        except:
            print("[Server - eval] - error")
    
    
    def select_childs(self, childs, frac=1.0):
        return random.sample(childs, int(len(childs)*frac))

