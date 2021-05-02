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
    def __init__(self, model_fn, data, testloader, child_list=[], lr=0.001, N_CLIENTS=1, res_file="lol.txt", beta=5, seed=0, asynch=True):
        super().__init__(model_fn, data)
        self.res_file = res_file
        self.testloader = testloader
        self.obj_q = Queue(maxsize=20)
        self.child_list = child_list
        self.TIME = 0
        self.start_time = time.time()
        self.N_CLIENTS = N_CLIENTS

        self.lr = lr
        self.beta = beta

        self.seed = seed
        self.asynch = asynch


    def send(self):
        for child in self.child_list:
            child.W_new = self.W
            child.TIME = self.TIME
        self.TIME += 1


    def update(self):
        if self.asynch:
            while not self.obj_q.empty():
                obj = self.obj_q.get()
                self.obj_q.task_done()
                t = obj.time
                alpha = self.lr * pow((self.TIME - t + 1), self.beta) * obj.num / self.N_CLIENTS
                for name in dw:
                    self.W[name].data += dw[name].data * alpha
                dw = obj.model
                print("[Server - upd]: Updated model with T = %s, t = %s, num = %s, alpha = %s" % (self.TIME, t, obj.num, alpha))
        else:
            dws = []
            while not self.obj_q.empty():
                dws.append(self.obj_q.get().model)
                self.obj_q.task_done()
            reduce_add_average(targets=[self.W], sources=dws)
            print("[Server - upd]: Updated model with T = %s, num = %s" % (self.TIME, len(dws)))




    def eval(self):
        try:
            acc = self.evaluate()
            f = open(self.res_file, "a")
            f.write("%s, %s, %s\n" % (self.TIME, acc, int(time.time() - self.start_time)))
            f.close()
            print("[Server - eval] TIME = %s, acc = [ %s ]" % (self.TIME, acc))
        except:
            print("[Server - eval] - error")
    
    
    def select_childs(self, childs, frac=1.0):
        return random.sample(childs, int(len(childs)*frac))


