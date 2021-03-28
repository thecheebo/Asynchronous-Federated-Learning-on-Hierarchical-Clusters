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

import models
from models import CF10Net
from helper import ExperimentLogger, display_train_stats
from devices import *
from data_utils import split_data, CustomSubset

def main(N_CLIENTS):
    test_data, testloader = prepare_data()

    print("--> Creating server...")
    server = Server(CF10Net, test_data, testloader)


def prepare_data():
    print("--> Preparing data...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='~/data', train=False, download=False, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

    test_idcs = np.random.permutation(len(testset))

    return CustomSubset(testset, test_idcs, transforms.Compose([transforms.ToTensor()])), testloader


class Server(FederatedTrainingDevice):
    def __init__(self, model_fn, data, testloader):
        super().__init__(model_fn, data)
        self.testloader = testloader
        self.dw_q = Queue(maxsize=20)
        self.lock = Lock()
        self.client_list = []
        for i in range(N_CLIENTS):
            self.client_list.append(('127.0.0.1', 9000 + i))

        try:
            server_threads = []
            # recv thread
            server_threads.append(Thread(name="server_recv", target=self.server_recv_loop))
            # update & eval thread
            server_threads.append(Thread(name="server_upd_eval", target=self.server_upd_eval_loop))
            # send thread
            server_threads.append(Thread(name="server_send", target=self.server_send_loop))

            for thread in server_threads:
                thread.start()
        except (KeyboardInterrupt, SystemExit):
            print("Gracefully shutting server down...")


    def server_recv_loop(self):
        HOST = '127.0.0.1' 
        PORT = 7007        
    
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen()
            while True:
                try:
                    print("[Server - recv]: listening...")
                    conn, addr = s.accept()
                    recv_data = []
                    with conn:
                        print('[Server - recv]: Connected by', addr)
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
                    print("[Server - recv]: received %s from addr %s" % (len(recv_data), addr))
               #     print(recv_data["conv1.weight"][:50])
                    self.dw_q.put(recv_data)
                    print("[Server - recv]: putted in queue")
                except:
                    print("[Server - recv]: error...")
                    continue
    
    
    def server_upd_eval_loop(self):
        global acc_server
        rd = 1
        while True:
            if rd == 1 or self.update_model():
                acc_server = [self.evaluate()]
                print("[Server - upd] rd = %s, acc = %s" % (rd, acc_server))
                rd += 1
    
    
    def server_send_loop(self):
        rd = 1
        while True:
            for client_addr in self.client_list:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.connect(client_addr)
                        data = pickle.dumps(self.W)
                        print("----->", len(data))
                        s.send(pickle.dumps(len(data)))
                        st = s.recv(1024)
                        print("[Server - send]: rd = %s - recv starttttt from client %s" % (rd, client_addr))
                        s.sendall(data)
                        print("[Server - send]: rd = %s, server send to client %s" % (rd, client_addr))
                        data = s.recv(1024)
                        if data:
                            print("[Server - send]: rd = %s - recv ACK from client %s" % (rd, client_addr))
                        else:
                            print("[Server - send]: rd = %s - recv 00000 from client %s" % (rd, client_addr))
                except:
                    print("[Server - send]: rd = %s - error send model to client %s" % (rd, client_addr))
                    continue
            time.sleep(10)
            rd += 1


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


#class SocketThread(Thread):
#
#        Thread.__init__(self)
#        self.server = server
#        self.conn = conn
#        self.client_addr = client_addr
#        self.buffer_size = buffer_size
#        self.recv_timeout = recv_timeout
#
#    def recv(self):
##        recv_data = b""
#        recv_data = []
#        while True:
#            print("len = ", len(recv_data))
#            try:
#                print(-1)
#                data = self.conn.recv(8)
#                print(0)
##                recv_data += data
#                recv_data.append(data)
#                print(1)
#
#                if data == b'':
#                    print(2)
##                   recv_data = b""
#                    recv_data = []
#                    if (time.time() - self.recv_start_time) > self.recv_timeout:
#                        return None, 0
#                elif str(data)[-2] == '.':
#                    print(3)
#                    if len(recv_data) > 0:
#                        try:
##                            recv_data = pickle.loads(recv_data)
#                            recv_data = pickle.loads(b''.join(recv_data))
#                            return recv_data, 1
#                        except BaseException as e:
#                            print("[Server - recv]: Error Decoding the Client's Data: {msg}.\n".format(msg=e))
#                            return None, 0
#                else:
#                    print(4)
#                    self.recv_start_time = time.time()
#            except BaseException as e:
#                print("[Server - recv]: Error Receiving Data from the Client: {msg}.\n".format(msg=e))
#                return None, 0
#
#    def run(self):
##        while True:
#        print("[Server - recv]: run...")
#        self.recv_start_time = time.time()
#        recv_data, status = recv(self.conn, self.recv_start_time)
#        #recv_data, status = self.recv()
#        if status == 0:
#            self.conn.close()
#            print("[Server - recv]: server self.conn.close()")
#            return
##            break
#        print("[Server - recv]: received '%s' from addr %s" % (recv_data, self.client_addr))
#        try:
#            self.conn.sendall(b'ACKACKACK')
#            print("[Server - recv]: Sent ACK to client")
#        except BaseException as e:
#            print("[Server - recv]: Error Sending ACK to client")
#
#        self.self.dw_q.put(recv_data)
#        print("[Server - recv]: putted in queue")
#        self.conn.close()


if __name__ == "__main__":
    try:
        N_CLIENTS = int(sys.argv[1])
    except Exception as e:
        print("args: N_CLIENTS")
        sys.exit()

    main(N_CLIENTS)
