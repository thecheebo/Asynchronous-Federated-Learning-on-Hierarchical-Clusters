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

    try:
        ### Server process ###
        print("--> Creating server threads...")
        server = Server(CF10Net, test_data, testloader)
        for i in range(N_CLIENTS):
            server.client_list.append(('127.0.0.1', 9000 + i))
        server_threads = []
        # recv thread
        server_threads.append(Thread(name="server_recv", 
                                    target=server_recv_loop, 
                                    args=(server, lambda: stop_flag)))
        # update & eval thread
        server_threads.append(Thread(name="server_upd_eval", 
                               target=server_upd_eval_loop, 
                               args=(server, lambda: stop_flag)))
#        # send thread
#        server_threads.append(Thread(name="server_send",
#                               target=server_send_loop,
#                               args=(server, lambda: stop_flag)))
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
    HOST = '127.0.0.1'  # 标准的回环地址 (localhost)
    PORT = 7007        # 监听的端口 (非系统级的端口: 大于 1023)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        while True:
            print("[Server - recv]: listening...")
            conn, addr = s.accept()
            recv_data = b""
            with conn:
                print('Connected by', addr)
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    recv_data += data
                #    print("================")
                #    print(data)
                    if str(data)[-2] == '.':
                        print("done!!!")
                        break
                #print("sendind ACKACKAACK")
                conn.sendall(b"ACKACKAACK!!!")
            print("-------------------->")
#            print(recv_data)
            print("<--------------------")
            recv_data = pickle.loads(recv_data)
            print(recv_data["conv1.weight"])
            server.dw_q.put(recv_data)
            print("[Server - recv]: putted in queue")




#    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#    soc.bind(('', 7007))
#    soc.listen(1)
#
#    while True:
#        try:
#            print("[Server - recv]: listening...")
#            conn, addr = soc.accept()
#            recv_start_time = time.time()
#            recv_data, status = recv(conn, recv_start_time)
#            if status == 0:
#                soc.close()
#                print("[Server - recv]: conn.close()...")
#            else:
#                print("[Server - recv]: received from addr %s" % (len(recv_data), addr))
#                try:
#                    conn.sendall(b'ACKACKACK')
#                except BaseException as e:
#                    print("ero")
#        except:
#            soc.close()
#            print("timeout")
#            break

#    i = 1
#    while True:
#        try:
#            print("[Server - recv]: listening...", i)
#            conn, addr = soc.accept()
#            print("[Server - recv]: accepted", i)
##            with conn:
#            soc_thread = SocketThread(server = server, conn=conn, client_addr=addr)
#            print("[Server - recv]: created", i)
#            soc_thread.start()
#            print("[Server - recv]: done", i)
#        except:
#            soc.close()
#            print("[Server - recv]: (Timeout) Socket Closed Because no Connections Received.\n")
#            break
##        finally:
##            print("finally")
##            soc_thread.join()
#        i += 1


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
                print("[Server - send]: rd = %s, server send to client %s" % (rd, client)) 
                resp = soc.recv(1024)
                print("[Server - send]: rd = %s - recv response from client %s" % (rd, client))
                # recv_data, status = recv2(soc=soc, buffer_size=1024, recv_timeout=10)
                # if status == 0:
                #     print("[Server - send]: rd = %s - Nothing Received from the client %s" % (rd, client))
                #     break
                # else:
                #     print("[Server - send]: rd = %s - recv response from client %s" % (rd, client))
                soc.close()
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
#        recv_data = b""
        recv_data = []
        while True:
            print("len = ", len(recv_data))
            try:
                print(-1)
                data = self.conn.recv(8)
                print(0)
#                recv_data += data
                recv_data.append(data)
                print(1)

                if data == b'':
                    print(2)
#                   recv_data = b""
                    recv_data = []
                    if (time.time() - self.recv_start_time) > self.recv_timeout:
                        return None, 0
                elif str(data)[-2] == '.':
                    print(3)
                    if len(recv_data) > 0:
                        try:
#                            recv_data = pickle.loads(recv_data)
                            recv_data = pickle.loads(b''.join(recv_data))
                            return recv_data, 1
                        except BaseException as e:
                            print("[Server - recv]: Error Decoding the Client's Data: {msg}.\n".format(msg=e))
                            return None, 0
                else:
                    print(4)
                    self.recv_start_time = time.time()
            except BaseException as e:
                print("[Server - recv]: Error Receiving Data from the Client: {msg}.\n".format(msg=e))
                return None, 0

    def run(self):
#        while True:
        print("[Server - recv]: run...")
        self.recv_start_time = time.time()
        recv_data, status = recv(self.conn, self.recv_start_time)
        #recv_data, status = self.recv()
        if status == 0:
            self.conn.close()
            print("[Server - recv]: server self.conn.close()")
            return
#            break
        print("[Server - recv]: received '%s' from addr %s" % (recv_data, self.client_addr))
        try:
            self.conn.sendall(b'ACKACKACK')
            print("[Server - recv]: Sent ACK to client")
        except BaseException as e:
            print("[Server - recv]: Error Sending ACK to client")

        self.server.dw_q.put(recv_data)
        print("[Server - recv]: putted in queue")
        self.conn.close()


if __name__ == "__main__":
    try:
        N_CLIENTS = int(sys.argv[1])
    except Exception as e:
        print("args: N_CLIENTS")
        sys.exit()

    main(N_CLIENTS)
