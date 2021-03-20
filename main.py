import os
import sys
import time
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

LEADER = False

cfl_stats = ExperimentLogger()
acc_server = 0.0

lock = Lock()

def main(N_CLIENTS, N_LEADERS, EVAL_ROUNDS, SELECT_CLIENT_FRAC, EVAL_INTERVAL):
    global server, clients, leaders

    prepare_data(N_CLIENTS)

    ### Client processes ### 
    print("--> Creating client processes...")
    clients = [Client(CF10Net, lambda x : torch.optim.SGD(x, lr=0.001, momentum=0.9), dat, idnum=i) for i, dat in enumerate(client_datas)]
    client_threads = [Thread(name="clt%s" % client.id, target = train_loop, args=(client, lambda: stop_flag)) for client in clients]
    for thread in client_threads:
        thread.start()
    
    ### Leader processes ### 
    print("--> Creating leader processes...")
    if LEADER:
        leaders = [Leader(CF10Net, test_data, i) for i in range(N_LEADERS)]
        for client in clients:
            leader_id = int(client.id / int((N_CLIENTS/N_LEADERS)))
            client.leader_id = leader_id
            leaders[leader_id].client_list.append(client)
        leader_threads = [Thread(name="led%s" % leader.id, target = leader_loop, args=(leader, lambda: stop_flag)) for leader in leaders]
        for thread in leader_threads:
            thread.start()

    ### Server process ### 
    print("--> Creating server process...")
    server = Server(CF10Net, test_data, testloader)
    ctx = get_context("spawn")
    p = ctx.Process(target = server_loop, args = (EVAL_ROUNDS, EVAL_INTERVAL))
    p.start()


def leader_loop(leader, should_stop):
    rd = 1
    while True:
        if should_stop():
            break
        leader.compute_dw_avg()
        leader.send_dW_to_server(server)
        print("[Leader - %s] rd = %s" % (leader.id, rd))
        rd += 1


def train_loop(client, should_stop):
    epoch = 1
    while True:
        if epoch == 1:
            client.synchronize_with_server(server)
        if should_stop():
            break
        train_stats = client.compute_weight_update(epochs=1)
        client.reset()
        if LEADER:
            client.send_dW_to_leader(leaders[client.leader_id])
        else:
            client.send_dW_to_server(server)
            
        print("[Client - %s] epoch = %s" % (client.id, epoch))
        epoch += 1


def server_loop(eval_rounds, eval_interval):
    global acc_server
    # start_time = datetime.now()

    for rd in range(eval_rounds):
        server.update_model()
        acc_server = [server.evaluate()]
        print("[Server] round = %s, acc = %s" % (rd, acc_server))
        cfl_stats.log({"acc_server" : acc_server, "rounds" : rd})
        display_train_stats(cfl_stats, eval_rounds)
        time.sleep(eval_interval)


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
        EVAL_ROUNDS = int(sys.argv[3])
        SELECT_CLIENT_FRAC = float(sys.argv[4])
        EVAL_INTERVAL = int(sys.argv[5])
    except Exception as e:
        print("args: N_CLIENTS, N_LEADERS, EVAL_ROUNDS, SELECT_CLIENT_FRAC, EVAL_INTERVAL")
        sys.exit()

    LEADER = True if N_LEADERS > 0 else False

    main(N_CLIENTS, N_LEADERS, EVAL_ROUNDS, SELECT_CLIENT_FRAC)


