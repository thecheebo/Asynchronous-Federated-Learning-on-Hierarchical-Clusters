import os
import sys
import time
import socket
import pickle
from datetime import datetime
from copy import deepcopy
from threading import Thread, Lock
from multiprocessing import Process, Pool, get_context, Queue

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from models import CF10Net
from helper import ExperimentLogger, display_train_stats
from devices import *
from data_utils import split_data, CustomSubset






soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
soc.connect(('localhost', 7007))

i = 1
while i < 20:
    try:
        soc.sendall("asdkjfnajkfnweflelkfnekjlfgelifh23214")
        print("[Client - %s - trn] epoch = %s - done, sent to server" % (client.id, epoch))
        resp = soc.recv(1024)
        print("[Client - %s - trn] epoch = %s - recv response from server" % (client.id, epoch))
    except BaseException as e:
        print("[Client - %s - trn]: Server not ready yet. Socket Closed.")
    i += 1
    
soc.close()

