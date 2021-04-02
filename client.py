import os
import sys
import time
import socket
import pickle
import json
import struct
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

def main(N_LEADERS, N_CLIENTS):
    client_datas = prepare_data(N_CLIENTS)

    print("--> Creating clients...", len(client_datas))
    for i, data in enumerate(client_datas):
        leader_id = -1
        if N_LEADERS > 0:
            group_size = int(N_CLIENTS / N_LEADERS)
            leader_id = int(i / group_size)
        client = Client(CF10Net, lambda x : torch.optim.SGD(x, lr=0.001, momentum=0.9), data, id=i, leader_id=leader_id)
#        client = Client(CF10Net, lambda x : torch.optim.Adam(x, lr=0.001), data, id=i)
        

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
    def __init__(self, model_fn, optimizer_fn, data, id, leader_id=-1, batch_size=128, train_frac=0.8):
        super().__init__(model_fn, data)
        self.optimizer = optimizer_fn(self.model.parameters())

        self.data_train = data
        self.train_loader = DataLoader(self.data_train, batch_size=batch_size, shuffle=True)

        self.id = id
        self.leader_id = leader_id

        self.dW = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}

        self.W_new = {key : value for key, value in self.model.named_parameters()}
        self.W_new_recv = False

        self.lock = Lock()

        ###  Threads  ###
        self.stop_flag = False
        try:
            client_threads = []
            # recv thread
            client_threads.append(Thread(name="clt-recv-%s" % self.id, target = self.client_recv_loop))
            # train thread
            client_threads.append(Thread(name="clt-trn-%s" % self.id, target = self.client_trn_loop))

            for thread in client_threads:
                thread.start()
        except (KeyboardInterrupt, SystemExit):
            print("Gracefully shutting client down...")
#        finally:
#            for thread in client_threads:
#                thread.join()


    def client_recv_loop(self):
        HOST = '127.0.0.1'     
        PORT = 9000 + self.id
    
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen()
            while True:
                if self.stop_flag: break
                try:
                    print("[Client - %s - recv]: client listening..." % self.id)
                    conn, addr = s.accept()

                    recv_data = []
                    with conn:
                        print("[Client - %s - recv]: Connected by %s" % (self.id, addr))
                        data = conn.recv(1024)
                        size = pickle.loads(data)
#                        print("<------- len = ", size)
                        conn.sendall(b"start!!!")
                        while True:
                            data = conn.recv(1024)
                            if not data:
                                break
                            recv_data.append(data)
                            size -= len(data)
                            # if str(data)[-2] == '.':
                            if size == 0:
                                print("done!!!")
                                break
                        conn.sendall(b"ACKACKAACK!!!")
                    recv_byte = b"".join(recv_data)
                    print("<------- ", len(recv_byte))
                    recv_data = pickle.loads(recv_byte)
                    print("[Client - %s - recv]: received '%s' from addr %s" % (self.id, len(recv_data), addr))
                    self.W_new = recv_data
                    self.W_new_recv = True
#                    print(3)
                except:
                    print("[Client - %s - recv]: error..." % self.id)
                    continue
        print("[Client - %s - recv]: *** EXIT ***" % self.id)
    

    def client_trn_loop(self):
        rd = 1
        fail_time = None
        while True:
            if fail_time and time.time() - fail_time > 5: self.stop_flag = True
            if self.stop_flag: break
            try:
                if rd > 1 and not self.sync_model():
                    continue
#                self.sync_model()

                print("[Client - %s - trn] rd = %s - sync done" % (self.id, rd))
    
                # train
                train_stats = self.compute_weight_update(epochs=1)
                print("[Client - %s - trn] rd = %s - train done" % (self.id, rd))
                self.reset()
#                print("[Client - %s - trn] rd = %s - reset done" % (self.id, rd))
    
                HOST = '127.0.0.1' 
                PORT = 7007 if self.leader_id == -1 else (8000 + self.leader_id * 2 + 1)        
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((HOST, PORT))
                    print("[Client - %s - trn] rd = %s - connected" % (self.id, rd))
#                    print(self.dW)
                    data = pickle.dumps(self.dW)
                    print("----->", len(data))
                    s.send(pickle.dumps(len(data)))
                    st = s.recv(1024)
#                    print("recv starrttttttttttt")
                    s.sendall(data)
                    print("[Client - %s - trn] rd = %s - done, sent to server/leader %s" % (self.id, rd, PORT))
                    data = s.recv(1024)
                    print("[Client - %s - trn] rd = %s - recv ACK from server/leader" % (self.id, rd))
                rd += 1
                fail_time = None

            except:
                print("[Client - %s - trn] rd = %s - error.." % (self.id, rd))
                if not fail_time: fail_time = time.time()
#            time.sleep(10)
        print("[Client - %s - trn]: *** EXIT ***" % self.id)




    def sync_model(self):
        if (self.W_new_recv):
            # with self.lock:
            copy(target=self.W, source=self.W_new)
            self.W_new_recv = False
            return True
        return False


    def compute_weight_update(self, epochs=1, loader=None):
        # with self.lock:
        copy(target=self.W_old, source=self.W)
#        self.optimizer.param_groups[0]["lr"]*=0.99
        train_stats = train_op(self.model, self.train_loader if not loader else loader, self.optimizer, epochs)
        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)
        return train_stats

    def reset(self):
        # with self.lock:
        copy(target=self.W, source=self.W_old)



if __name__ == "__main__":
    try:
        N_LEADERS = int(sys.argv[1])
        N_CLIENTS = int(sys.argv[2])
    except Exception as e:
        print("args: N_LEADERS, N_CLIENTS")
        sys.exit()

    main(N_LEADERS, N_CLIENTS)


