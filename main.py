import os
import sys
from copy import deepcopy

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

from models import ConvNet, CF10Net
from helper import ExperimentLogger, display_train_stats
from fl_devices import Server, Client, Leader
from data_utils import split_data, CustomSubset

torch.manual_seed(42)
np.random.seed(42)

# Global vars
testloader = None
test_data = None
client_datas = None

server = None
clients = []
leaders = []

cfl_stats = ExperimentLogger()
acc_server = 0.0

def main(N_CLIENTS, N_LEADERS, TRAIN_ROUNDS, SELECT_CLIENT_FRAC):
    global server, clients, leaders

    prepare_data(N_CLIENTS)

    ### Create components ###
    print("--> Creating components...")
    server = Server(CF10Net, test_data, testloader)
    clients = [Client(CF10Net, lambda x : torch.optim.SGD(x, lr=0.001, momentum=0.9), dat, idnum=i) for i, dat in enumerate(client_datas)]
    
    if N_LEADERS > 0:
        leaders = [Leader(CF10Net, test_data, i) for i in range(N_LEADERS)]
        for i, client in enumerate(clients):
            client.leader_id=int(i/int((N_CLIENTS/N_LEADERS)))
            leader_id=int(client.id / int((N_CLIENTS/N_LEADERS)))
            leaders[leader_id].client_list.append(client)

    ### Train ###
    print("--> Begin training...")
    for c_round in range(1, TRAIN_ROUNDS + 1):
    
        print("Round: ", c_round)
        for client in clients:
            client.synchronize_with_server(server)

        if N_LEADERS > 0:
            train_with_leader()
        else: 
            train_baseline()

        eval(c_round)

    print("--> Accurary Result: ", acc_server)


def eval(c_round):
    global acc_server
    acc_server = [server.evaluate()]
    cfl_stats.log({"acc_server" : acc_server, "rounds" : c_round})
    display_train_stats(cfl_stats, TRAIN_ROUNDS)


def train_with_leader():
    for leader in leaders:
        print("   leader ", leader.id)
        participating_clients = leader.select_clients(leader.client_list, frac = SELECT_CLIENT_FRAC)
        for client in participating_clients:
            print("      Client ", client.id)
            train_stats = client.compute_weight_update(epochs=1)
            client.send_dW_to_leader(leader)
            client.reset()
        leader.compute_dw_avg()

    server.aggregate_avg_dws(leaders)


def train_baseline():
    participating_clients = server.select_clients(clients, frac = SELECT_CLIENT_FRAC)
    for client in participating_clients:
        print("      Client ", client.id)
        train_stats = client.compute_weight_update(epochs=1)
        client.reset()

    server.aggregate_weight_updates(participating_clients)


def prepare_data(N_CLIENTS):
    print("--> Preparing and splitting data...")

    global testset, testloader, client_datas

    ### Prepare data ###
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='~/data', train=True, download=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='~/data', train=False, download=False, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    ### Split data ###
    DIRICHLET_ALPHA = 10.0
    train_idcs, test_idcs = np.random.permutation(len(trainset)), np.random.permutation(len(testset))
    client_idcs = np.arange(0, len(trainset)).reshape(N_CLIENTS, int(len(trainset) / N_CLIENTS))
    
    train_labels = []
    for idc in client_idcs:
        for idcc in idc:
            train_labels.append(trainset[idcc][1])
    train_labels = np.array(train_labels)
    client_idcs = split_data(train_idcs, train_labels, alpha=DIRICHLET_ALPHA, n_clients=N_CLIENTS)
    
    client_datas = [CustomSubset(trainset, idcs) for idcs in client_idcs]
    test_data = CustomSubset(testset, test_idcs, transforms.Compose([transforms.ToTensor()]))
    

if __name__ == "__main__":
    try:
        N_CLIENTS = int(sys.argv[1])
        N_LEADERS = int(sys.argv[2])
        TRAIN_ROUNDS = int(sys.argv[3])
        SELECT_CLIENT_FRAC = float(sys.argv[4])
    except Exception as e:
        print("args: N_CLIENTS, N_LEADERS, TRAIN_ROUNDS, SELECT_CLIENT_FRAC")
        sys.exit()

    main(N_CLIENTS, N_LEADERS, TRAIN_ROUNDS, SELECT_CLIENT_FRAC)


