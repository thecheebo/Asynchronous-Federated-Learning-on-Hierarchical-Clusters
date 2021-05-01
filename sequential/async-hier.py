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

sys.path.append('../')
import models
from models import CF10Net
from devices import *
from data_utils import split_data, CustomSubset
from server import *
from client import *
from leader import *


def main(N_CLIENTS, N_LEADERS, lr, l2_lambda, beta, select_rate, ROUNDS, seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    res_file = "exp__async_hier__%s__%s__%s__%s__%s__%s__%s.txt" % (N_CLIENTS, N_LEADERS, lr, l2_lambda, beta, select_rate, seed)
    
    # Server
    test_data, testloader = server_prepare_data()
    server = Server(CF10Net, test_data, testloader, lr=lr, N_CLIENTS=N_CLIENTS, res_file=res_file, beta=beta, seed=seed)
    
    # Client
    client_list = []
    client_datas = client_prepare_data(N_CLIENTS)
    for i, data in enumerate(client_datas):
        leader_id = -1
        if N_LEADERS > 0:
            group_size = int(N_CLIENTS / N_LEADERS)
            leader_id = int(i / group_size)
        client_list.append(Client(CF10Net, lambda x : torch.optim.SGD(x, lr=lr, momentum=0.9), data, id=i, l2_lambda=l2_lambda, seed=seed))
    
    # Leader
    leader_list = []
    for i in range(N_LEADERS):
        leader = Leader(CF10Net, i)
        leader.server = server
        server.child_list.append(leader)
        for j in range(group_size * i, group_size * (i+1)):
            leader.child_list.append(client_list[j])
            client_list[j].parent = leader
        leader_list.append(leader)
    
    for i in range(ROUNDS):
    
        server.send()
    
        for leader in leader_list:
            leader.pass_W()
    
        selected_client_list = random.sample(client_list, int(N_CLIENTS * select_rate))
    
        for client in selected_client_list:
            client.train()
            client.send()
    
        for leader in leader_list:
            leader.compute_dW()
            leader.send_dW()
    
        server.update()
        server.eval()



def server_prepare_data():
    print("--> Preparing data for server...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='~/data', train=False, download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

    test_idcs = np.random.permutation(len(testset))

    return CustomSubset(testset, test_idcs, transforms.Compose([transforms.ToTensor()])), testloader


def client_prepare_data(N_CLIENTS):
    print("--> Preparing and splitting data for clients...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='~/data', train=True, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

    train_idcs = np.random.permutation(len(trainset))

    client_idcs = np.arange(0, len(trainset)).reshape(N_CLIENTS, int(len(trainset) / N_CLIENTS))

    train_labels = []
    for idc in client_idcs:
        for idcc in idc:
            train_labels.append(trainset[idcc][1])
    train_labels = np.array(train_labels)
    DIRICHLET_ALPHA = 10
    client_idcs = split_data(train_idcs, train_labels, alpha=DIRICHLET_ALPHA, n_clients=N_CLIENTS)

    return [CustomSubset(trainset, idcs) for idcs in client_idcs]


if __name__ == "__main__":
    try:
        N_CLIENTS = int(sys.argv[1])
        N_LEADERS = int(sys.argv[2])
        lr = float(sys.argv[3]) 
        l2_lambda = float(sys.argv[4])
        beta = int(sys.argv[5])
        select_rate = float(sys.argv[6])
        ROUNDS = int(sys.argv[7])
        seed = int(sys.argv[8])
    except Exception as e:
        print("args: N_CLIENTS, N_LEADERS, lr, l2_lambda, beta, select_rate, ROUNDS, seed")
        sys.exit()

    main(N_CLIENTS, N_LEADERS, lr, l2_lambda, beta, select_rate, ROUNDS, seed)
