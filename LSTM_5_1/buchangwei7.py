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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train = pd.read_csv('E:/LSTM_5_1/train_2018_1_model_2.csv')

dt = train['dt'].values

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
        self.input = []
        self.label = []
        for diskname in self.serial_number.index.sort_values():
            traindata_name = traindata[traindata.serial_number == diskname]
            self.datalabel = traindata_name.loc[:, 'label'].values.tolist()
            j = 0
            while(j + numstep <= len(traindata_name)):
                self.input.append(torch.Tensor(self.datas[j:j + numstep]))
                if(torch.Tensor(self.datalabel[j:j + numstep]).sum() != 0):
                    self.label.append(torch.ones(1))
                else:
                    self.label.append(torch.zeros(1))
                j = j + 7  
    def __len__(self):
        return len(self.input)
    def __getitem__(self, idx):
        #data = self.datas[idx].reshape(1, -1)
        return self.input[idx], self.label[idx]


train_data = pd.read_csv('E:/LSTM_5_1/train_2018_1_model_2.csv')
train_set = MyData(train_data, 7, 5)
sampler = mySample(train_set)
train_iter = DataLoader(MyData(train_data, 7, 5), batch_size = 10, sampler=sampler)

class CNNLSTM(nn.Module):

    def __init__(self): 
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 13, out_channels = 128, kernel_size = 4) # 要求转置
        self.lstmlayer = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True)
        self.linear = nn.Linear(128, 2)
    def forward(self, X):
        pad = nn.ZeroPad2d(padding=(2, 1, 0, 0))
        X = X.permute(0, 2, 1)
        X = pad(X)
        X = self.conv1(X)
        X =  X.permute(0, 2, 1) 
        X, (H, C) = self.lstmlayer(X)
        X = X[:, X.size(1) - 1, :]
        x = torch.sigmoid(X)
        X = self.linear(X)
        return X


def train(net, train_iter, optimizer, num_epochs):
    print("training start")
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            #X = X.to(device) 
            #y = y.to(device)
            #y = y.long()
            y_hat = net(X.float().cuda())
            #y_hat = y_hat.to(device)
            #y_hat = y_hat.reshape(-1, 2)
            y = y.reshape(-1).cuda( ).long()
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
        % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start))

net = CNNLSTM()
net = net.to(device)
net = torch.nn.DataParallel(net)
lr, num_epochs = 0.01, 20
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

train(net, train_iter, optimizer, num_epochs)

'''
想着把lr改下 改成动态的先大后小
没加sample
lr = 0.1

epoch 1, loss 1.4176, train acc 0.665, time 1.2 sec
epoch 2, loss 0.9477, train acc 0.697, time 1.0 sec
epoch 3, loss 0.9734, train acc 0.693, time 1.0 sec
epoch 4, loss 0.9874, train acc 0.702, time 1.0 sec
epoch 5, loss 0.9754, train acc 0.690, time 1.0 sec
epoch 6, loss 1.0206, train acc 0.702, time 1.0 sec
epoch 7, loss 0.9791, train acc 0.689, time 1.0 sec
epoch 8, loss 1.0209, train acc 0.708, time 1.0 sec
epoch 9, loss 0.9786, train acc 0.681, time 1.0 sec
epoch 10, loss 1.0222, train acc 0.693, time 1.0 sec
epoch 11, loss 0.9810, train acc 0.689, time 1.0 sec
epoch 12, loss 1.0190, train acc 0.708, time 1.0 sec
epoch 13, loss 0.9810, train acc 0.689, time 1.0 sec
epoch 14, loss 1.0219, train acc 0.693, time 1.0 sec
epoch 15, loss 0.9802, train acc 0.689, time 1.0 sec
epoch 16, loss 1.0215, train acc 0.702, time 1.0 sec
epoch 17, loss 0.9790, train acc 0.681, time 1.0 sec
epoch 18, loss 1.0227, train acc 0.693, time 1.0 sec
epoch 19, loss 0.9811, train acc 0.689, time 1.0 sec
epoch 20, loss 1.0204, train acc 0.708, time 1.0 sec
'''

'''
有sampler
lr = 0.1

epoch 1, loss 0.8559, train acc 0.494, time 1.3 sec
epoch 2, loss 0.7647, train acc 0.488, time 1.0 sec
epoch 3, loss 0.7512, train acc 0.489, time 1.0 sec
epoch 4, loss 0.7710, train acc 0.494, time 1.0 sec
epoch 5, loss 0.7881, train acc 0.480, time 1.0 sec
epoch 6, loss 0.7319, train acc 0.486, time 1.1 sec
epoch 7, loss 0.7211, train acc 0.533, time 1.0 sec
epoch 8, loss 0.7575, train acc 0.500, time 1.0 sec
epoch 9, loss 0.7808, train acc 0.514, time 1.0 sec
epoch 10, loss 0.7439, train acc 0.516, time 1.0 sec
epoch 11, loss 0.7556, train acc 0.500, time 1.0 sec
epoch 12, loss 0.7685, train acc 0.486, time 1.0 sec
epoch 13, loss 0.7558, train acc 0.492, time 1.0 sec
epoch 14, loss 0.7480, train acc 0.515, time 1.0 sec
epoch 15, loss 0.7659, train acc 0.479, time 1.0 sec
epoch 16, loss 0.7547, train acc 0.490, time 1.0 sec
epoch 17, loss 0.7704, train acc 0.497, time 1.0 sec
epoch 18, loss 0.7923, train acc 0.479, time 1.0 sec
epoch 19, loss 0.8775, train acc 0.498, time 1.0 sec
epoch 20, loss 0.7798, train acc 0.529, time 1.0 sec
'''

'''
有sampler
lr = 0.01


epoch 1, loss 0.6995, train acc 0.521, time 1.2 sec
epoch 2, loss 0.6958, train acc 0.471, time 1.0 sec
epoch 3, loss 0.6931, train acc 0.525, time 1.0 sec
epoch 4, loss 0.6915, train acc 0.533, time 1.0 sec
epoch 5, loss 0.6946, train acc 0.490, time 1.0 sec
epoch 6, loss 0.6938, train acc 0.489, time 1.0 sec
epoch 7, loss 0.6932, train acc 0.504, time 1.0 sec
epoch 8, loss 0.6950, train acc 0.492, time 1.0 sec
epoch 9, loss 0.6921, train acc 0.529, time 1.0 sec
epoch 10, loss 0.6939, train acc 0.507, time 1.0 sec
epoch 11, loss 0.6950, train acc 0.500, time 1.0 sec
epoch 12, loss 0.6935, train acc 0.508, time 1.0 sec
epoch 13, loss 0.6942, train acc 0.502, time 1.0 sec
epoch 14, loss 0.6934, train acc 0.506, time 1.0 sec
epoch 15, loss 0.6946, train acc 0.484, time 1.0 sec
epoch 16, loss 0.6931, train acc 0.515, time 1.0 sec
epoch 17, loss 0.6948, train acc 0.491, time 1.0 sec
epoch 18, loss 0.6940, train acc 0.494, time 1.0 sec
epoch 19, loss 0.6945, train acc 0.493, time 1.0 sec
epoch 20, loss 0.6944, train acc 0.496, time 1.0 sec
'''