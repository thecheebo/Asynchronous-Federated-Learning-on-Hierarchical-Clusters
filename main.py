import os
import sys
import time
from datetime import datetime
from copy import deepcopy
from threading import Thread, Lock
from multiprocessing import Process, Pool, get_context, Queue
import yappi

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

clients = []
leaders = []

LEADER = False

cfl_stats = ExperimentLogger()
acc_server = 0.0

lock = Lock()

def main(N_CLIENTS, N_LEADERS, EVAL_ROUNDS, SELECT_CLIENT_FRAC, EVAL_INTERVAL):
    global clients, leaders
    stop_flag = False

    try:
        prepare_data(N_CLIENTS)

        yappi.set_clock_type("wall")
        yappi.start()

        ### Server process ### 
        print("--> Creating server process...")
        server = Server(CF10Net, test_data, testloader)
        server_thread = Thread(name="server", target=server_loop, args=(server, EVAL_ROUNDS, EVAL_INTERVAL,  lambda: stop_flag))
        # server_thread.start()
        # ctx = get_context("spawn")
        # jobs = []
        # for i in range(3):
        #     p = ctx.Process(target=server_loop, args=(server, EVAL_ROUNDS, EVAL_INTERVAL))
        #     jobs.append(p)
        #     p.start()

        ### Client processes ### 
        print("--> Creating client processes...")
        clients = [Client(CF10Net, lambda x : torch.optim.SGD(x, lr=0.001, momentum=0.9), dat, idnum=i) for i, dat in enumerate(client_datas)]
        client_threads = [Thread(name="clt%s" % client.id, target = train_loop, args=(client, server, lambda: stop_flag)) for client in clients]
        for thread in client_threads:
            thread.start()
        
        # ### Leader processes ### 
        # print("--> Creating leader processes...")
        # if LEADER:
        #     leaders = [Leader(CF10Net, test_data, i) for i in range(N_LEADERS)]
        #     for client in clients:
        #         leader_id = int(client.id / int((N_CLIENTS/N_LEADERS)))
        #         client.leader_id = leader_id
        #         leaders[leader_id].client_list.append(client)
        #     leader_threads = [Thread(name="led%s" % leader.id, target = leader_loop, args=(leader, server, lambda: stop_flag)) for leader in leaders]
        #     for thread in leader_threads:
        #         thread.start()

    except (KeyboardInterrupt, SystemExit):
        logger.debug("Gracefully shutting client down...")
    finally:
        stop_flag = True
        # server_thread.join()
        for thread in client_threads:
            thread.join()
        yappi.stop()
        # p.terminate()
        # p.join()

def leader_loop(leader, server, should_stop):
    rd = 1
    while True:
        if should_stop():
            break
        leader.compute_dw_avg()
        leader.send_dW_to_server(server)
        print("[Leader - %s] rd = %s" % (leader.id, rd))
        rd += 1


def train_loop(client, server, should_stop):
    epoch = 1
    while True:
        if epoch == 1:
            client.synchronize_with_server(server)
        if should_stop():
            break
        print("**** here")
        train_stats = client.compute_weight_update(epochs=1)
        client.reset()
        if LEADER:
            print("**** clt to leader")
            client.send_dW_to_leader(leaders[client.leader_id])
        else:
            print("**** clt to server")
            client.send_dW_to_server(server)
            
        print("[Client - %s] epoch = %s" % (client.id, epoch))
        epoch += 1


def server_loop(server, eval_rounds, eval_interval, should_stop):
    global acc_server, cfl_stats
    # start_time = datetime.now()

    rd = 1
    while True:
        server.update_model()
        acc_server = [server.evaluate()]
        print("[Server] round = %s, acc = %s" % (rd, acc_server))
        cfl_stats.log({"acc_server" : acc_server, "rounds" : rd})
        # display_train_stats(cfl_stats, eval_rounds)
        time.sleep(eval_interval)
        rd += 1


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

    main(N_CLIENTS, N_LEADERS, EVAL_ROUNDS, SELECT_CLIENT_FRAC, EVAL_INTERVAL)


