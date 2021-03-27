import random
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import AgglomerativeClustering
#from multiprocessing import Process, Pool, get_context, Queue
from threading import Thread, Lock
import time
import socket
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"

def recv(conn, recv_start_time):
#    recv_data = b""
    recv_data = []
    while True:
#        print("len = ", len(recv_data))
        try:
#            print(-1)
            data = conn.recv(1024)
#            print(0)
#            recv_data += data
            recv_data.append(data)
#            print(1)

            if data == b'':
#                recv_data = b""
                recv_data = []
                if (time.time() - recv_start_time) > 100:
                    return None, 0
            elif str(data)[-2] == '.':
                if len(recv_data) > 0:
                    try:
#                        recv_data = pickle.loads(recv_data)
                        recv_data = pickle.loads(b''.join(recv_data))
                        # conn.sendall(pickle.dumps("ACK"))
                        # print("recv_data = ", recv_data)
                        return recv_data, 1
                    except BaseException as e:
                        return None, 0
            else:
                recv_start_time = time.time()
        except BaseException as e:
            return None, 0


def recv2(soc, buffer_size=1024, recv_timeout=10):
    received_data = b""
    while str(received_data)[-2] != '.':
        try:
            soc.settimeout(recv_timeout)
            received_data += soc.recv(buffer_size)
        except socket.timeout:
            print("A socket.timeout exception occurred because the server did not send any data for {recv_timeout} seconds.".format(recv_timeout=recv_timeout))
            return None, 0
        except BaseException as e:
            return None, 0
            print("An error occurred while receiving data from the server {msg}.".format(msg=e))

    try:
        received_data = pickle.loads(received_data)
    except BaseException as e:
        print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
        return None, 0

    return received_data, 1

def train_op(model, loader, optimizer, epochs=1):
    model.train()  
    for ep in range(epochs):
        running_loss, samples = 0.0, 0
        for x, y in loader: 
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            loss = torch.nn.CrossEntropyLoss()(model(x), y)
            running_loss += loss.item()*y.shape[0]
            samples += y.shape[0]

            loss.backward()
            optimizer.step()  

    return running_loss / samples
      
def eval_op(model, loader):
    model.train()
    samples, correct = 0, 0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            y_ = model(x)
            _, predicted = torch.max(y_.data, 1)

            samples += y.shape[0]
            correct += (predicted == y).sum().item()

    return correct/samples


def copy(target, source):
    for name in target:
        target[name].data = source[name].data.clone()
    
def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone()-subtrahend[name].data.clone()
    
def reduce_add_average(targets, sources):
    for target in targets:
        for name in target:
            tmp = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()
            target[name].data += tmp
        
def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])


def pairwise_angles(sources):
    angles = torch.zeros([len(sources), len(sources)])
    for i, source1 in enumerate(sources):
        for j, source2 in enumerate(sources):
            s1 = flatten(source1)
            s2 = flatten(source2)
            angles[i,j] = torch.sum(s1*s2)/(torch.norm(s1)*torch.norm(s2)+1e-12)

    return angles.numpy()

        
class FederatedTrainingDevice(object):
    def __init__(self, model_fn, data):
        self.model = model_fn().to(device)
        self.data = data
        self.W = {key : value for key, value in self.model.named_parameters()}


    def evaluate(self, loader=None):
        return eval_op(self.model, self.testloader if not loader else loader)
  
 

