import os
import sys
import time
import socket
import pickle
from datetime import datetime
from copy import deepcopy
from threading import Thread, Lock
# from multiprocessing import Process, Pool, get_context, Queue
from queue import Queue

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
        # send thread
        server_threads.append(Thread(name="server_send",
                               target=server_send_loop,
                               args=(server, lambda: stop_flag)))
        for thread in server_threads:
            thread.start()
    except (KeyboardInterrupt, SystemExit):
        print("Gracefully shutting server down...")
    finally:
        stop_flag = True
        for thread in server_threads:
            thread.join()
        server.dw_q.join()


def server_recv_loop(server, should_stop):
    soc = socket.socket()
    soc.bind(('', 70))
    soc.listen(1)

    i = 1
    while True:
        try:
            print("[Server - recv]: listening...", i)
            conn, addr = soc.accept()
            print("[Server - recv]: accepted", i)
            with conn:
                soc_thread = SocketThread(server = server, conn=conn, client_addr=addr)
                print("[Server - recv]: created", i)
                soc_thread.start()
                print("[Server - recv]: done", i)
        except:
            soc.close()
            print("[Server - recv]: (Timeout) Socket Closed Because no Connections Received.\n")
            break
        i += 1


def server_upd_eval_loop(server, should_stop):
    global acc_server
    rd = 1
    while True:
        if rd == 1 or server.update_model():
            acc_server = [server.evaluate()]
            print("[Server - upd] rd = %s, acc = %s" % (rd, acc_server))
            rd += 1


def server_send_loop(server, should_stop):
    rd = 1
    while True:
        for client in server.client_list:
            soc = socket.socket()
            try:
                soc.connect(client)
                data = pickle.dumps(server.W)
                soc.sendall(data)
                soc.close()
                print("[Server - send]: rd = %s, server send to client %s" % (rd, client)) 
            except:
                print("[Server - send]: client not up")
                soc.close()
        time.sleep(20)
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
        self.testloader = testloader
        # self.ctx = get_context("spawn")
        # self.dw_q = self.ctx.Queue()
        self.dw_q = Queue(maxsize=20)
        self.lock = Lock()
        self.client_list = []

    def select_clients(self, clients, frac=1.0):
        return random.sample(clients, int(len(clients)*frac))

    def update_model(self):
        # print("update_model???")
        if not self.dw_q.empty():
            dws = []
            with self.lock:
                while not self.dw_q.empty():
                    dws.append(self.dw_q.get())
                    self.dw_q.task_done()
            if dws:
                reduce_add_average(targets=[self.W], sources=dws)
                print("[Server - upd]: Updated model")
                return True
        return False


class SocketThread(Thread):

    def __init__(self, server, conn, client_addr, buffer_size=1024, recv_timeout=5):
        Thread.__init__(self)
        self.server = server
        self.conn = conn
        self.client_addr = client_addr
        self.buffer_size = buffer_size
        self.recv_timeout = recv_timeout

    def recv(self):
        received_data = b""
        while True:
            try:
                data = self.conn.recv(self.buffer_size)
                received_data += data
                if data == b'':
                    received_data = b""
                    if (time.time() - self.recv_start_time) > self.recv_timeout:
                        return None, 0
                elif str(data)[-2] == '.':
                    # print("All data ({data_len} bytes) Received from {client_addr}.".format(client_addr=self.client_addr, data_len=len(received_data)))
                    if len(received_data) > 0:
                        try:
                            received_data = pickle.loads(received_data)
                            # self.conn.sendall(pickle.dumps("ACK"))
                            return received_data, 1
                        except BaseException as e:
                            print("[Server - recv]: Error Decoding the Client's Data: {msg}.\n".format(msg=e))
                            return None, 0
                else:
                    self.recv_start_time = time.time()
            except BaseException as e:
                print("[Server - recv]: Error Receiving Data from the Client: {msg}.\n".format(msg=e))
                return None, 0

    def run(self):
        while True:
            print("[Server - recv]: run while loop...")
            self.recv_start_time = time.time()
            recv_data, status = self.recv()
            if status == 0:
                self.conn.close()
                print("[Server - recv]: server self.conn.close()")
                break
            print("[Server - recv]: received '%s' from addr %s" % (len(recv_data), self.client_addr))

            self.server.dw_q.put(recv_data)
            print("[Server - recv]: putted in queue")


if __name__ == "__main__":
    try:
        N_CLIENTS = int(sys.argv[1])
    except Exception as e:
        print("args: N_CLIENTS")
        sys.exit()

    main(N_CLIENTS)
