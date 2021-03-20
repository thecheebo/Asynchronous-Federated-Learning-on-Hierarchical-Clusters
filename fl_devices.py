import random
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from multiprocessing import Process, Pool, get_context, Queue

device = "cuda" if torch.cuda.is_available() else "cpu"

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
        return eval_op(self.model, self.eval_loader if not loader else loader)
  
class Leader(FederatedTrainingDevice):
    def __init__(self, model_fn, data, id):
        super().__init__(model_fn, data)
        self.id = id
        self.client_list = []
        self.dW_list = []
        self.dW_avg = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.ctx = get_context("spawn")
        self.dw_q = self.ctx.Queue()

    def compute_dw_avg(self):
        # with lock:
        for name in self.dW_avg:
            tmp = torch.mean(torch.stack([dW[name].data for dW in self.dW_list]), dim=0).clone()
            self.dW_avg[name].data += tmp
        self.dW_list = []

    def send_dW_to_server(self, server):
        print("[Leader - %s]: send dw to server" % self.id)
        # server.dW_list.append(self.dW)
        server.dw_q.put(self.dW)

    def select_clients(self, clients, frac=1.0):
        return random.sample(clients, int(len(clients)*frac))
  
class Client(FederatedTrainingDevice):
    def __init__(self, model_fn, optimizer_fn, data, idnum, leader_id=0, batch_size=128, train_frac=0.8):
        super().__init__(model_fn, data)  
        self.optimizer = optimizer_fn(self.model.parameters())
            
        self.data = data
        n_train = int(len(data)*train_frac)
        n_eval = len(data) - n_train 
        data_train, data_eval = torch.utils.data.random_split(self.data, [n_train, n_eval])

        self.train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
        self.eval_loader = DataLoader(data_eval, batch_size=batch_size, shuffle=False)
        
        self.id = idnum
        self.leader_id = leader_id
        
        self.dW = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        
    def synchronize_with_server(self, server):
        copy(target=self.W, source=server.W)
    
    def compute_weight_update(self, epochs=1, loader=None):
        print("**** here - 1")
        copy(target=self.W_old, source=self.W)
#         self.optimizer.param_groups[0]["lr"]*=0.99
        print("**** here - 2")
        train_stats = train_op(self.model, self.train_loader if not loader else loader, self.optimizer, epochs)
        print("**** here - 3")
        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)
        print("**** here - 4")
        return train_stats  
    
    def send_dW_to_leader(self, leader):
        print("[Client - %s]: send dw to leader %s" % (self.id, leader.id))
        # print("leader.dW_list.len = ", len(leader.dW_list))
        # leader.dW_list.append(self.dW)
        leader.dw_q.put(self.dW)

    def send_dW_to_server(self, server):
        print("[Client - %s]: send dw to server" % self.id)
        # server.dW_list.append(self.dW)
        server.dw_q.put(self.dW)

    def reset(self): 
        copy(target=self.W, source=self.W_old)

    
class Server(FederatedTrainingDevice):
    def __init__(self, model_fn, data, testloader):
        super().__init__(model_fn, data)
        self.loader = DataLoader(self.data, batch_size=128, shuffle=False)
        self.model_cache = []
        self.eval_loader = testloader
        self.dW_list = []
        self.ctx = get_context("spawn")
        self.dw_q = self.ctx.Queue()
    
    def select_clients(self, clients, frac=1.0):
        return random.sample(clients, int(len(clients)*frac)) 
    
    def update_model(self):
        if not self.dw_q.empty():
            dw = self.dw_q.get()
            reduce_add_average(targets=[self.W], sources=[dw])
            print("[Server]: updated model")

    def aggregate_avg_dws(self, leaders):
        reduce_add_average(targets=[self.W], sources=[leader.dW_avg for leader in leaders])

    def aggregate_weight_updates(self, clients):
        reduce_add_average(targets=[self.W], sources=[client.dW for client in clients])
        
    def aggregate_clusterwise(self, client_clusters):
        for cluster in client_clusters:
            reduce_add_average(targets=[client.W for client in cluster], 
                               sources=[client.dW for client in cluster])
            
            
    def compute_max_update_norm(self, cluster):
        return np.max([torch.norm(flatten(client.dW)).item() for client in cluster])

    
    def compute_mean_update_norm(self, cluster):
        return torch.norm(torch.mean(torch.stack([flatten(client.dW) for client in cluster]), 
                                     dim=0)).item()

    def cache_model(self, idcs, params, accuracies):
        self.model_cache += [(idcs, 
                            {name : params[name].data.clone() for name in params}, 
                            [accuracies[i] for i in idcs])]


