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
    test_data, testloader = prepare_data()
    print(len(test_data))

    try:
        ### Server process ###
        print("--> Creating server threads...")
        server = Server(CF10Net, test_data, testloader)
        for i in range(N_CLIENTS):
            server.client_list.append(('127.0.0.1', 90 + i))
        server_threads = []
        # recv thread
        server_threads.append(Thread(name="server_recv", 
                                    target=server_recv_loop, 
                                    args=(server, lambda: stop_flag)))
        # update & eval thread
        server_threads.append(Thread(name="server_upd_eval", 
                               target=server_upd_eval_loop, 
                               args=(server, lambda: stop_flag)))
        for thread in server_threads:
            thread.start()
    except (KeyboardInterrupt, SystemExit):
        print("Gracefully shutting server down...")
    finally:
        stop_flag = True
        for thread in server_threads:
            thread.join()


def server_recv_loop(server, should_stop):
    soc = socket.socket()
    soc.bind(('', 70))
    soc.listen(1)

    while True:
        try:
            print("server listening...")
            conn, addr = soc.accept()
            recv_start_time = time.time()
            time_struct = time.gmtime()
            recv_data, status = recv(conn, recv_start_time)
            if status == 0:
                soc.close()
                print("server soc.close()")
                break
            print("[Server]: received '%s' from addr %s" % (len(recv_data), addr))
            server.dw_q.put(recv_data)
        except:
            soc.close()
            print("(Timeout) Socket Closed Because no Connections Received.\n")
            break


def server_upd_eval_loop(server, should_stop):
    global acc_server, cfl_stats

    rd = 1
    while True:
        if rd == 1 or server.update_model():
            print("here")
            acc_server = [server.evaluate()]
            print("[Server] round = %s, acc = %s" % (rd, acc_server))
            # send model to clients/leaders periodically
            for client in server.client_list:
                soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
                try:
                    soc.connect(client)
                    data = pickle.dumps(server.W)
                    soc.sendall(data)
                    soc.close()
                    print("server send to client ", client) 
                except:
                    print("client not up")
                    soc.close()
            rd += 1


def prepare_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='~/data', train=False, download=False, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

    test_idcs = np.random.permutation(len(testset))

    return CustomSubset(testset, test_idcs, transforms.Compose([transforms.ToTensor()])), testloader


class Server(FederatedTrainingDevice):
    def __init__(self, model_fn, data, testloader):
        super().__init__(model_fn, data)
        self.model_cache = []
        self.testloader = testloader
        self.ctx = get_context("spawn")
        self.dw_q = self.ctx.Queue()
        self.lock = Lock()
        self.client_list = []

    def select_clients(self, clients, frac=1.0):
        return random.sample(clients, int(len(clients)*frac))

    def update_model(self):
        # print("qsize = ", self.dw_q.qsize())
        dws = []
        with self.lock:
            while not self.dw_q.empty():
                dws.append(self.dw_q.get())
        if dws:
            reduce_add_average(targets=[self.W], sources=dws)
            print("[Server]: Updated model")
            return True
        else:
            return False


if __name__ == "__main__":
    try:
        N_CLIENTS = int(sys.argv[1])
    except Exception as e:
        print("args: N_CLIENTS")
        sys.exit()

    main(N_CLIENTS)
