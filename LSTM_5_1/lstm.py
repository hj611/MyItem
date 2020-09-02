import collections
import math
import os
import random
import sys
import tarfile
import time
import json
from collections import namedtuple
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchtext
import torchtext.vocab as Vocab
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim import lr_scheduler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = pd.read_csv('E:/LSTM_5_1/train_2018Q1_model_2.csv')

def classifyRes(arr):
    #arr = arr.reshape(arr.size,)
    for i in range(arr.shape[0]):
        arr[i] = 1 if arr[i] > 0.5 else 0
    return arr

def mySample(data):
    num = 0
    total = len(data)
    sample_list = []

    for X, y in data:
        num = num + y
    

    class_sample_count = np.array([total - num, num])
    weight = 1. / class_sample_count

    for X, y in data:
        sample_list.append(weight[int(y)])

    sample_weight = torch.FloatTensor(sample_list)
    sampler = WeightedRandomSampler(sample_weight, total)
    return sampler


class MyData(Dataset):
    def __init__(self, data, numstep, length):

        self.serial_number = data['serial_number'].value_counts()

        self.input = []
        self.label = []

        for diskname in self.serial_number.index.sort_values():

            #根据disk得到数据
            self.data_diskname = data[data.serial_number == diskname]

            #归一化
            self.value = self.data_diskname.iloc[:, length:-1].values
            self.max = np.max(self.value)
            self.min = np.min(self.value)
            self.scalar = self.max - self.min 
            self.data_norm = list(map(lambda x: x / self.scalar, self.value))

            #根据disk得到label
            self.data_label = self.data_diskname.loc[:, 'label'].values.tolist()

            #封装X:0，1，2 Y: 3
            for i in range(len(self.data_norm) - numstep - 1):
                self.input.append(torch.Tensor(self.data_norm[i: i + numstep]))
                if(self.data_label[i + numstep] == 1):
                    self.label.append(torch.ones(1))
                elif(self.data_label[i + numstep] == 0):
                    self.label.append(torch.zeros(1))

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx], self.label[idx]

data_train = MyData(data, 7, 6)
sampler = mySample(data_train)
train_iter = DataLoader(MyData(data, 1, 6), batch_size = 32, sampler=sampler, drop_last=True)

class MyLSTM(nn.Module):

    def __init__(self): 
        super(MyLSTM, self).__init__()
        self.lstmlayer = nn.LSTM(input_size=13, hidden_size=128, num_layers=2, batch_first=True)
        self.linear = nn.Linear(128, 2)
    def forward(self, X, state):
        X, (H, C) = self.lstmlayer(X, state)
        X = X[:, X.size(1) - 1 : X.size(1), :]
        X = X.reshape(X.shape[0], X.shape[2])
        X = F.sigmoid(X)
        X = self.linear(X)
        return X, (H, C)


def train(net, train_iter, optimizer, scheduler, num_epochs):
    i = 10
    print("training start")
    loss = torch.nn.CrossEntropyLoss()
    state = None
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.float()
            y = y.to(device)
            
            if state is not None:
                if isinstance (state, tuple): # LSTM, state:(h, c)  
                    state = (state[0].detach(), state[1].detach())
                else:   
                    state = state.detach()

            y_hat, state = net(X, state)
            optimizer.zero_grad()
            l = loss(y_hat, y.long().reshape(-1))
            l.backward()
            optimizer.step()
            scheduler.step( )
     
            train_l_sum += l.item()
            #y_hat = F.softmax(y_hat, dim=1)
            y_hat = y_hat.argmax(dim=1)
            y_hat = torch.squeeze(y_hat)
            y = torch.squeeze(y)
            train_acc_sum += (y_hat == y).sum().item()
            n += y.shape[0]
            batch_count += 1
        

            
        print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
        % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start))


net = MyLSTM()
net = net.to(device)
net = torch.nn.DataParallel(net)
lr, num_epochs = 0.05, 10
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

train(net, train_iter, optimizer, scheduler, num_epochs)
'''
for X, y in train_iter:
    y_hat, stata = net(X, None)
    print(y_hat)  
    print(classifyRes(y_hat))
    print(y_hat)
    print(y.shape)
    print(F.softmax(y_hat))
    break
'''