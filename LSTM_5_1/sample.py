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
from torch.utils.data import DataLoader, Dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train = pd.read_csv('E:/LSTM_5_1/train_2018_1_model_2.csv')

dt = train['dt'].values

class MyData(Dataset):
    def __init__(self, traindata, numstep):
        self.serial_number = traindata['serial_number'].value_counts()
        self.value = traindata.iloc[:, 6:-1].values
        max = np.max(self.value)
        min = np.min(self.value)
        scalar = max - min 
        self.datas = list(map(lambda x: x / scalar, self.value))
        self.datalabel = train_data.loc[:, 'label'].values.tolist()
        self.input = []
        self.label = []
        for i in self.serial_number.values:
            for j in range(i - numstep):
                self.input.append(torch.Tensor(self.datas[j: j + numstep]))
                self.label.append(torch.Tensor(self.datalabel[j: j + numstep]))
    def __len__(self):
        return len(self.input)
    def __getitem__(self, idx):
        #data = self.datas[idx].reshape(1, -1)
        return self.input[idx], self.label[idx]
train_data = pd.read_csv('E:/LSTM_5_1/train1.csv')
train_iter = DataLoader(MyData(train_data, 7), batch_size = 10)

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
        x = torch.sigmoid(X)
        X = self.linear(X)
        return X

def train(net, train_iter, optimizer, num_epochs):
    print("training start")
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device) 
            #y = y.to(device)
            y_hat = net(X.float().cuda())
            #y_hat = y_hat.to(device)
            y_hat = y_hat.reshape(-1, 2)
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
lr, num_epochs = 0.001, 20
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

train(net, train_iter, optimizer, num_epochs)