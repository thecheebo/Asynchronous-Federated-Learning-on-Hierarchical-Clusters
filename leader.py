import os
import sys
import time
import socket
import pickle
from datetime import datetime
from copy import deepcopy
from threading import Thread, Lock
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

def main(N_LEADERS, N_CLIENTS):
    print("--> Creating leaders...")
    group_size = int(N_CLIENTS / N_LEADERS)
    for i in range(N_LEADERS):
        leader = Leader(CF10Net, i)
        for j in range(group_size * i, group_size * (i+1)):
            leader.client_list.append(('127.0.0.1', 9000 + j))


class Leader(FederatedTrainingDevice):
    def __init__(self, model_fn, id):
        super().__init__(model_fn, None)
        self.id = id
        self.client_list = []
        self.dW_sum = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.dW_num = 0
        self.dW_avg = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}

        try:
            Thread(name="leader_pass_w", target=self.leader_pass_W_loop).start()
            Thread(name="leader_recv_dw", target=self.leader_recv_dW_loop).start()
            Thread(name="leader_send_dw", target=self.leader_send_dW_avg_loop).start()
        except (KeyboardInterrupt, SystemExit):
            print("Gracefully shutting leader down...")


    def leader_pass_W_loop(self):
        HOST = '127.0.0.1' if LOCAL_TEST else '0.0.0.0'
        PORT = 8000 + self.id * 2

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen()
            while True:
                try:
                    print("[Leader - pass - W]: listening...", PORT)
                    conn, addr = s.accept()
                    recv_data = []
                    with conn:
                        print('[Leader - pass - W]: Connected by', addr)
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
                            if size == 0:
                                print("done!!!")
                                break
                        conn.sendall(b"ACKACKAACK!!!")
                    recv_byte = b"".join(recv_data)
                    print("[Leader - pass - W]: received %s from addr %s" % (len(recv_byte), addr))

                    # send recv_byte to client_list
                    for client_addr in self.client_list:
                        self.send(recv_byte, client_addr)
                except:
                    print("[Leader - pass - W]: error...")
                    continue


    def leader_recv_dW_loop(self):
        HOST = '127.0.0.1'
        PORT = 8000 + self.id * 2 + 1

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen()
            while True:
                try:
                    print("[Leader - recv]: listening...", PORT)
                    conn, addr = s.accept()
                    recv_data = []
                    with conn:
                        print('[Leader - recv]: Connected by', addr)
                        data = conn.recv(1024)
                        size = pickle.loads(data)
                        conn.sendall(b"start!!!")
                        while True:
                            data = conn.recv(1024)
                            if not data:
                                break
                            recv_data.append(data)
                            size -= len(data)
                            if size == 0:
                                print("done!!!")
                                break
                        conn.sendall(b"ACKACKAACK!!!")
                    recv_byte = b"".join(recv_data)
                    print("<------- ", len(recv_byte))
                    recv_data = pickle.loads(recv_byte)
                    print("[Leader - recv]: received %s from addr %s" % (len(recv_data), addr))
                    self.add_dw(recv_data)
                    print("[Leader - recv]: dW added")
                except:
                    print("[Leader - recv]: error...")
                    continue


    def leader_send_dW_avg_loop(self):
        rd = 1
        while True:
            if (self.dW_num >= len(self.client_list)):
                HOST = '127.0.0.1' if LOCAL_TEST else 'sp21-cs525-g19-01.cs.illinois.edu'
                PORT = 7007
                data = pickle.dumps(self.cal_dW_avg())
                self.send(data, (HOST, PORT))
                rd += 1
            time.sleep(5)


    def send(self, data_byte, addr):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(addr)
                s.send(pickle.dumps(len(data_byte)))
                st = s.recv(1024)
                s.sendall(data_byte)
                print("[Leader - send]: sent to %s" % repr(addr))
                data = s.recv(1024)
                if data:
                    print("[Leader - send]: recv ACK from", repr(addr))
                else:
                    print("[Leader - send]: recv 000000 from %s" % repr(addr))
        except:
            print("[Leader - send]: error send to %s" % repr(addr))


    def add_dw(self, dW):
        self.dW_num += 1
        for name in self.dW_sum:
            self.dW_sum[name].data += dW[name].data.clone()

    def cal_dW_avg(self):
        for name in self.dW_sum:
            self.dW_avg[name].data = self.dW_sum[name].data / self.dW_num
        # reset
        self.dW_sum = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.dW_num = 0

        return self.dW_avg



if __name__ == "__main__":
    try:
        N_LEADERS = int(sys.argv[1])
        N_CLIENTS = int(sys.argv[2])
    except Exception as e:
        print("args: N_LEADERS, N_CLIENTS")
        sys.exit()

    global LOCAL_TEST = True
    main(N_LEADERS, N_CLIENTS)
