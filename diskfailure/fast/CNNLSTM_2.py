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

#时间步模式为【1-7】【2-8】

def mySample(data):
    num = 0
    total = len(train_set)
    sample_list = []

    for X, y in train_set:
        num = num + y
    

    class_sample_count = np.array([total - num, num])
    weight = 1. / class_sample_count

    for X, y in train_set:
        sample_list.append(weight[int(y)])

    sample_weight = torch.FloatTensor(sample_list)
    sampler = WeightedRandomSampler(sample_weight, total)
    return sampler

class MyData(Dataset):
    def __init__(self, traindata, numstep, length):
        self.serial_number = traindata['serial_number'].value_counts()
        self.value = traindata.iloc[:, length:-1].values
        max = np.max(self.value)
        min = np.min(self.value)
        scalar = max - min 
        self.datas = list(map(lambda x: x / scalar, self.value))
        self.datalabel = train_data.loc[:, 'label'].values.tolist()
        self.input = []
        self.label = []

        for diskname in self.serial_number.index.sort_values():
            traindata_name = traindata[traindata.serial_number == diskname]
            self.datalabel = traindata_name.loc[:, 'label'].values.tolist()
            for i in range(len(traindata_name) - numstep):
                self.input.append(torch.Tensor(self.datas[i: i + numstep]))
                if(torch.Tensor(self.datalabel[i: i + numstep]).sum() != 0):
                    self.label.append(torch.ones(1))
                else:
                    self.label.append(torch.zeros(1))
    def __len__(self):
        return len(self.input)
    def __getitem__(self, idx):
        #data = self.datas[idx].reshape(1, -1)
        return self.input[idx], self.label[idx]

train_data = pd.read_csv('F:/data/train_2018Q1_model_2.csv')
train_set = MyData(train_data, 7, 6)
sampler = mySample(train_set)
train_iter = DataLoader(MyData(train_data, 7, 6), batch_size = 10, drop_last=True)

class CNNLSTM(nn.Module):

    def __init__(self): 
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 13, out_channels = 128, kernel_size = 4) # 要求转置
        self.lstmlayer = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True)
        self.linear = nn.Linear(128, 2)
    def forward(self, X, state):
        pad = nn.ZeroPad2d(padding=(2, 1, 0, 0))
        X = X.permute(0, 2, 1)
        X = pad(X)
        X = self.conv1(X)
        X =  X.permute(0, 2, 1) 
        #X = X.reshape(7, 10, -1)
        X, (H, C) = self.lstmlayer(X, state)
        X = X[:, X.size(1) - 1, :]
        x = torch.sigmoid(X)
        X = self.linear(X)
        return X, (H, C)

def train(net, train_iter, optimizer, scheduler, num_epochs):
    print("training start")
    loss = torch.nn.CrossEntropyLoss()
    state = None
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            if state is not None:
                if isinstance (state, tuple): # LSTM, state:(h, c)  
                    state = (state[0].detach(), state[1].detach())
                else:   
                    state = state.detach()
            #X = X.to(device) 
            #y = y.to(device)
            #y = y.long()
            y_hat, state = net(X.float().cuda(), state)
            #y_hat = y_hat.to(device)
            #y_hat = y_hat.reshape(-1, 2)
            y = y.reshape(-1).cuda( ).long()
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            scheduler.step( )
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
        % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start))
        

net = CNNLSTM()
net = net.to(device)
net = torch.nn.DataParallel(net)
lr, num_epochs = 0.1, 20
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

train(net, train_iter, optimizer, scheduler, num_epochs)

