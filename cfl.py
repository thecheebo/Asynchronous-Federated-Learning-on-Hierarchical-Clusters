from IPython.display import clear_output
import os
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models import ConvNet
from helper import ExperimentLogger, display_train_stats
from fl_devices import Server, Client
from data_utils import split_noniid, CustomSubset

import emnist
from emnist import list_datasets


torch.manual_seed(42)
np.random.seed(42)


N_CLIENTS = 10
DIRICHLET_ALPHA = 1.0

print(list_datasets())

data = datasets.EMNIST(root=".", split="byclass", download=True)
