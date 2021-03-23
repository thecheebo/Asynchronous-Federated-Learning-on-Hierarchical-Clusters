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

def main(N_CLIENTS):
    stop_flag = False
    try:
        client_datas = prepare_data(N_CLIENTS)

        print("--> Creating client threads...")
        clients = [Client(CF10Net, lambda x : torch.optim.SGD(x, lr=0.001, momentum=0.9), data, id=i) for i, data in enumerate(client_datas)]
        client_threads = []
        for i, client in enumerate(clients):
            # recv thread
            client_threads.append(Thread(name="clt-recv-%s" % client.id, 
                                 target = client_recv_loop, 
                                 args=(client, lambda: stop_flag)))
            # train thread
            client_threads.append(Thread(name="clt-trn-%s" % client.id, 
                                target = client_trn_loop, 
                                args=(client, lambda: stop_flag)))
        for thread in client_threads:
            thread.start()

    except (KeyboardInterrupt, SystemExit):
        print("Gracefully shutting client down...")
    finally:
        for thread in client_threads:
            thread.join()


def client_recv_loop(client, should_stop):
    soc = socket.socket()
    soc.bind(('', 90 + client.id))
    soc.listen(1)

    while True:
        try:
            print("client listening...",  90 + client.id)
            conn, addr = soc.accept()
            recv_start_time = time.time()
            time_struct = time.gmtime()
            recv_data, status = recv(conn, recv_start_time)
            if status == 0:
                soc.close()
                print("client-%s soc.close()" % client.id)
                break
            print("[Client - %s]: received '%s' from addr %s" % (client.id, len(recv_data), addr))
            client.sync_model(recv_data)
        except:
            soc.close()
            print("(Timeout) Socket Closed Because no Connections Received.\n")
            break


def client_trn_loop(client, should_stop):
    epoch = 1
    soc = socket.socket()
    try:
        soc.connect(("127.0.0.1", 70))
    except BaseException as e:
        print("Server not ready yet. Socket Closed.")
        soc.close()

    while True:
        # if should_stop():
        #     print("stop!")
        #     break

        # train
        train_stats = client.compute_weight_update(epochs=1)
        client.reset()

        # send dw to server/leader
        data = pickle.dumps(client.dW)
        print("train done. sending, len = ", len(data))
        soc.sendall(data)

        print("[Client - %s] epoch = %s" % (client.id, epoch))
        epoch += 1


def prepare_data(N_CLIENTS):
    print("--> Preparing and splitting data...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='~/data', train=True, download=False, transform=transform)

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


class Client(FederatedTrainingDevice):
    def __init__(self, model_fn, optimizer_fn, data, id, leader_id=0, batch_size=128, train_frac=0.8):
        super().__init__(model_fn, data)
        self.optimizer = optimizer_fn(self.model.parameters())

        self.data_train = data
        self.train_loader = DataLoader(self.data_train, batch_size=batch_size, shuffle=True)

        self.id = id
        self.leader_id = leader_id

        self.dW = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}

        self.lock = Lock()

    def sync_model(self, model):
        with self.lock:
            copy(target=self.W, source=model)

    def compute_weight_update(self, epochs=1, loader=None):
        with self.lock:
            copy(target=self.W_old, source=self.W)
#             self.optimizer.param_groups[0]["lr"]*=0.99
            train_stats = train_op(self.model, self.train_loader if not loader else loader, self.optimizer, epochs)
            subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)
        return train_stats

    def send_dW_to_server(self, server):
        print("[Client - %s]: send dw to server" % self.id)
        server.dw_q.put(self.dW)

    def reset(self):
        copy(target=self.W, source=self.W_old)


if __name__ == "__main__":
    try:
        N_CLIENTS = int(sys.argv[1])
    except Exception as e:
        print("args: N_CLIENTS")
        sys.exit()

    main(N_CLIENTS)


