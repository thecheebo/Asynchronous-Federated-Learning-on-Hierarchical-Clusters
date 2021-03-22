import os
import sys
import time
import socket
import pickle
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

def main(N_CLIENTS, N_LEADERS, SELECT_CLIENT_FRAC, AGGR_INTERVAL):
    global clients, leaders
    stop_flag = False

    try:
        prepare_data(N_CLIENTS)

        yappi.set_clock_type("wall")
        yappi.start()

        ### Server process ### 
        print("--> Creating server thread...")
        server = Server(CF10Net, test_data, testloader)
        server_recv_thread = Thread(name="server_recv", target=server_recv_loop, args=(server, lambda: stop_flag))
        server_recv_thread.start()
        server_thread = Thread(name="server", target=server_loop, args=(server, lambda: stop_flag))
        server_thread.start()

        ### Client processes ### 
        print("--> Creating client threads...")
        clients = [Client(CF10Net, lambda x : torch.optim.SGD(x, lr=0.001, momentum=0.9), dat, idnum=i) for i, dat in enumerate(client_datas)]
        # client_threads = [Thread(name="clt%s" % client.id, target = train_loop, args=(client, server, lambda: stop_flag)) for client in clients]
        client_threads = []
        for i, client in enumerate(clients):
            server.client_list.append(('127.0.0.1', 90 + client.id))
            recv_thread = Thread(name="clt-recv-%s" % client.id, target = client_recv_loop, args=(client, lambda: stop_flag))
            trn_thread = Thread(name="clt-trn-%s" % client.id, target = client_trn_loop, args=(client, lambda: stop_flag))
            client_threads.append(recv_thread)
            client_threads.append(trn_thread)
        
        for thread in client_threads:
            thread.start()
        
        ### Leader processes ### 
        if LEADER:
            print("--> Creating leader threads...")
            leaders = [Leader(CF10Net, test_data, i) for i in range(N_LEADERS)]
            for client in clients:
                leader_id = int(client.id / int((N_CLIENTS / N_LEADERS)))
                client.leader_id = leader_id
                leaders[leader_id].client_list.append(client)
            leader_threads = [Thread(name="led%s" % leader.id, target = leader_loop, args=(leader, server, AGGR_INTERVAL, lambda: stop_flag)) for leader in leaders]
            for thread in leader_threads:
                thread.start()

    except (KeyboardInterrupt, SystemExit):
        print("Gracefully shutting client down...")
    finally:
        stop_flag = True
        server_thread.join()
        server_recv_thread.join()
        for thread in client_threads:
            thread.join()
        if LEADER:
            for thread in leader_threads:
                thread.join()
        yappi.stop()


def recv(soc, recv_start_time):
    recv_data = b""
    while True:
        try:
            conn, addr = soc.accept()
            data = conn.recv(1024)
            recv_data += data

            if data == b'':
                recv_data = b""
                if (time.time() - recv_start_time) > 5:
                    return None, 0 
            elif str(data)[-2] == '.':
                print("[Client - %s]: received '%s' from addr %s" % (client.id, data, addr))
                if len(recv_data) > 0:
                    try:
                        recv_data = pickle.loads(recv_data)
                        return recv_data, 1
                    except BaseException as e:
                        return None, 0
            else:
                recv_start_time = time.time()
        except BaseException as e:
            return None, 0


def client_recv_loop(client, should_stop):
    soc = socket.socket()
    soc.bind(('', 90 + client.id))
    soc.listen(1)

    while True:
        print("client listening...")
        recv_start_time = time.time()
        time_struct = time.gmtime()
        recv_data, status = recv(soc, recv_start_time)
        if status == 0:
            self.connection.close()
            print("client self.connection.close()")
            break
        print("[Client - %s]: received '%s' from addr %s" % (client.id, recv_data, addr))
        client.sync_model(recv_data)
    

def client_trn_loop(client, should_stop):
    epoch = 1
    soc = socket.socket()
    soc.connect(('127.0.0.1', 70))
    while True:
        print("Training...")
        # if should_stop():
        #     print("stop!")
        #     break

        # train
        train_stats = client.compute_weight_update(epochs=1)
        client.reset()

        # send dw to server/leader
        data = pickle.dumps(client.dW)
        print("train done. sending")
        print(data)
        soc.sendall(data)

        print("[Client - %s] epoch = %s" % (client.id, epoch))
        epoch += 1


def leader_loop(leader, server, aggr_interval, should_stop):
    rd = 1
    while True:
        # if should_stop():
        #     break
        if leader.compute_dw_avg():
            leader.send_dW_to_server(server)
        print("[Leader - %s] aggr avg rd = %s" % (leader.id, rd))
        rd += 1
        time.sleep(aggr_interval)


def server_recv_loop(server, should_stop):
    soc = socket.socket()
    soc.bind(('', 70))
    soc.listen(5)
    while True:
        print("server listening...")
        recv_start_time = time.time()
        time_struct = time.gmtime()
        recv_data, status = recv(soc, recv_start_time)
        if status == 0:
            soc.close()
            print("server self.connection.close()")
            break
        print("[Server]: received '%s' from addr %s" % (recv_data, addr))
        server.dw_q.put(recv_data)


def server_loop(server, should_stop):
    global acc_server, cfl_stats
    soc = socket.socket()

    rd = 1
    while True:
        server.update_model()
        acc_server = [server.evaluate()]
        print("[Server] round = %s, acc = %s" % (rd, acc_server))
        cfl_stats.log({"acc_server" : acc_server, "rounds" : rd})
        # display_train_stats(cfl_stats, eval_rounds)
        # send model to clients/leaders periodically
        for client in server.client_list:
            # soc.connect(client)
            soc.connect(('127.0.0.1', 90))
            data = pickle.dumps(server.W)
            print("len of data = ", len(data))
            soc.sendall(data)
            soc.close()
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
        SELECT_CLIENT_FRAC = float(sys.argv[3])
        AGGR_INTERVAL = int(sys.argv[4])
    except Exception as e:
        print("args: N_CLIENTS, N_LEADERS, SELECT_CLIENT_FRAC, AGGR_INTERVAL")
        sys.exit()

    LEADER = True if N_LEADERS > 0 else False

    main(N_CLIENTS, N_LEADERS, SELECT_CLIENT_FRAC, AGGR_INTERVAL)


