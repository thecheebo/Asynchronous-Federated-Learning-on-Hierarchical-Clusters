import random
import torch
from torch.utils.data import DataLoader
import numpy as np
import time
import struct

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"


def train_op(model, loader, optimizer, epochs=1, W_old=None, l2_lambda=0.01, seed=0, asynch=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    model.train()
    for ep in range(epochs):
        running_loss, samples = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            loss = torch.nn.CrossEntropyLoss()(model(x), y)

            if asynch:
                # Add regularization term for async learning
                l2_reg = torch.tensor(0.)
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        l2_reg += torch.nn.MSELoss(reduction='sum')(param, W_old[name])
                loss += l2_lambda * l2_reg

            running_loss += loss.item()*y.shape[0]
            samples += y.shape[0]

            loss.backward()
            optimizer.step()

    return running_loss / samples


      
def eval_op(model, loader, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    model.train()
    samples, correct = 0.0, 0.0

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


def pairwise_angles(sources, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

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
        return eval_op(self.model, self.testloader if not loader else loader, self.seed)
  
 
class Package:
     def __init__(self, time=-1, model=None, num=1):
         self.time = time
         self.model = model
         self.num = num

