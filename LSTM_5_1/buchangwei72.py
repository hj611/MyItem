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

train_data = pd.read_csv('E:/LSTM_5_1/train_2018_1_model_2.csv')
train_set = MyData(train_data, 7, 5)
sampler = mySample(train_set)
train_iter = DataLoader(MyData(train_data, 7, 5), batch_size = 10, drop_last=True)

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

'''
lr 0.001
sampler 

epoch 1, loss 0.6938, train acc 0.506, time 9.0 sec
epoch 2, loss 0.6935, train acc 0.499, time 5.8 sec
epoch 3, loss 0.6933, train acc 0.501, time 5.8 sec
epoch 4, loss 0.6933, train acc 0.505, time 5.8 sec
epoch 5, loss 0.6933, train acc 0.501, time 5.8 sec
epoch 6, loss 0.6934, train acc 0.499, time 5.8 sec
epoch 7, loss 0.6932, train acc 0.506, time 5.8 sec
epoch 8, loss 0.6931, train acc 0.511, time 5.8 sec
epoch 9, loss 0.6931, train acc 0.512, time 5.8 sec
epoch 10, loss 0.6933, train acc 0.507, time 5.9 sec
epoch 12, loss 0.6933, train acc 0.504, time 5.9 sec
epoch 13, loss 0.6934, train acc 0.500, time 5.9 sec
epoch 14, loss 0.6933, train acc 0.500, time 6.0 sec
epoch 15, loss 0.6935, train acc 0.494, time 5.9 sec
epoch 16, loss 0.6927, train acc 0.509, time 5.9 sec
epoch 17, loss 0.6933, train acc 0.505, time 5.8 sec
epoch 18, loss 0.6933, train acc 0.496, time 5.8 sec
epoch 19, loss 0.6933, train acc 0.498, time 5.8 sec
epoch 20, loss 0.6934, train acc 0.498, time 5.8 sec
'''
'''
lr 0.01

epoch 1, loss 0.5643, train acc 0.783, time 6.0 sec
epoch 2, loss 0.5377, train acc 0.783, time 5.8 sec
epoch 3, loss 0.5385, train acc 0.783, time 5.8 sec
epoch 4, loss 0.5350, train acc 0.783, time 5.8 sec
epoch 5, loss 0.5331, train acc 0.783, time 6.1 sec
epoch 6, loss 0.5322, train acc 0.783, time 5.8 sec
epoch 7, loss 0.5322, train acc 0.783, time 5.8 sec
epoch 8, loss 0.5311, train acc 0.783, time 5.8 sec
epoch 9, loss 0.5306, train acc 0.783, time 5.8 sec
epoch 10, loss 0.5310, train acc 0.783, time 5.8 sec
epoch 12, loss 0.5300, train acc 0.783, time 5.8 sec
epoch 13, loss 0.5297, train acc 0.783, time 5.8 sec
epoch 14, loss 0.5293, train acc 0.783, time 5.8 sec
epoch 15, loss 0.5291, train acc 0.783, time 5.8 sec
epoch 16, loss 0.5291, train acc 0.783, time 5.8 sec
epoch 17, loss 0.5327, train acc 0.783, time 5.8 sec
epoch 18, loss 0.5296, train acc 0.783, time 5.8 sec
epoch 19, loss 0.5282, train acc 0.783, time 5.8 sec
epoch 20, loss 0.5275, train acc 0.783, time 6.0 sec
'''

'''
lr 0.1

epoch 1, loss 4.0400, train acc 0.687, time 6.1 sec
epoch 2, loss 4.2492, train acc 0.697, time 5.8 sec
epoch 3, loss 4.4604, train acc 0.691, time 5.9 sec
epoch 4, loss 4.5277, train acc 0.689, time 5.9 sec
epoch 5, loss 4.5477, train acc 0.688, time 5.9 sec
epoch 6, loss 4.4714, train acc 0.696, time 5.9 sec
epoch 7, loss 4.4803, train acc 0.684, time 5.9 sec
epoch 8, loss 4.3915, train acc 0.685, time 5.9 sec
epoch 9, loss 4.4387, train acc 0.690, time 5.9 sec
epoch 10, loss 4.4701, train acc 0.693, time 6.1 sec
epoch 11, loss 4.6568, train acc 0.691, time 6.2 sec
epoch 12, loss 4.4448, train acc 0.683, time 6.0 sec
epoch 13, loss 4.5057, train acc 0.690, time 6.2 sec
epoch 14, loss 4.6124, train acc 0.693, time 6.2 sec
epoch 15, loss 4.6510, train acc 0.694, time 6.3 sec
epoch 16, loss 4.6420, train acc 0.696, time 6.0 sec
epoch 17, loss 4.6375, train acc 0.692, time 6.5 sec
epoch 18, loss 4.5246, train acc 0.695, time 6.0 sec
epoch 19, loss 4.4367, train acc 0.687, time 6.0 sec
epoch 20, loss 4.4593, train acc 0.698, time 6.0 sec
'''

'''
lr 0.1
sampler 

epoch 1, loss 0.9214, train acc 0.491, time 6.0 sec
epoch 2, loss 0.8795, train acc 0.502, time 5.9 sec
epoch 3, loss 0.8444, train acc 0.498, time 6.0 sec
epoch 4, loss 0.8496, train acc 0.508, time 5.9 sec
epoch 5, loss 0.8707, train acc 0.496, time 5.9 sec
epoch 6, loss 0.8525, train acc 0.493, time 6.1 sec
epoch 7, loss 0.8387, train acc 0.506, time 6.1 sec
epoch 8, loss 0.8767, train acc 0.496, time 5.8 sec
epoch 9, loss 0.8488, train acc 0.494, time 5.8 sec
epoch 10, loss 0.8326, train acc 0.495, time 5.9 sec
epoch 11, loss 0.8453, train acc 0.490, time 5.9 sec
epoch 12, loss 0.8573, train acc 0.493, time 5.9 sec
epoch 13, loss 0.8062, train acc 0.507, time 5.8 sec
epoch 14, loss 0.8352, train acc 0.503, time 5.8 sec
epoch 15, loss 0.8134, train acc 0.500, time 5.9 sec
epoch 16, loss 0.8559, train acc 0.496, time 5.9 sec
epoch 17, loss 0.8373, train acc 0.505, time 5.9 sec
epoch 18, loss 0.8676, train acc 0.504, time 5.9 sec
epoch 19, loss 0.8875, train acc 0.511, time 5.9 sec
epoch 20, loss 0.8301, train acc 0.497, time 5.8 sec
'''

'''
lr 0.01
sampler 

epoch 1, loss 0.6968, train acc 0.502, time 6.0 sec
epoch 2, loss 0.6941, train acc 0.494, time 5.8 sec
epoch 3, loss 0.6939, train acc 0.502, time 5.8 sec
epoch 4, loss 0.6940, train acc 0.504, time 5.8 sec
epoch 5, loss 0.6937, train acc 0.506, time 5.8 sec
epoch 6, loss 0.6937, train acc 0.500, time 5.8 sec
epoch 7, loss 0.6938, train acc 0.497, time 5.8 sec
epoch 8, loss 0.6941, train acc 0.498, time 5.8 sec
epoch 9, loss 0.6940, train acc 0.492, time 5.9 sec
epoch 10, loss 0.6937, train acc 0.505, time 5.9 sec
epoch 11, loss 0.6936, train acc 0.503, time 5.8 sec
epoch 12, loss 0.6933, train acc 0.513, time 5.8 sec
epoch 13, loss 0.6934, train acc 0.506, time 5.8 sec
epoch 14, loss 0.6938, train acc 0.500, time 5.8 sec
epoch 15, loss 0.6943, train acc 0.490, time 5.9 sec
epoch 16, loss 0.6937, train acc 0.504, time 5.8 sec
epoch 17, loss 0.6940, train acc 0.496, time 5.8 sec
epoch 18, loss 0.6939, train acc 0.506, time 5.8 sec
epoch 19, loss 0.6937, train acc 0.497, time 5.8 sec
epoch 20, loss 0.6936, train acc 0.498, time 5.9 sec
'''