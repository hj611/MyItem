import pandas as pd 
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
import numpy as np 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class diskLSTM(nn.Module):
    def __init__(self):
        super(diskLSTM, self).__init__()

        self.LSTM_1 = nn.LSTM(input_size=6, hidden_size=32, batch_first=True)
        self.LSTM_2 = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        self.LSTM_3 = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
 
        self.linear_1 = nn.Linear(128, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, 1)
    
    def forward(self, X, state):
        X, _ = self.LSTM_1(X, state)
        X, _ = self.LSTM_2(X, state)
        X, _ = self.LSTM_3(X, state)

        X = X[:, X.size(1) - 1, :]
        X.squeeze_(1)

        X = self.linear_1(X)
        X = self.linear_2(X)
        X = self.linear_3(X)

        return X


class Mydata(Dataset):
    def __init__(self, PositiveSample, NegativeSample, numstep):
        self.serial_numberOfPositive = list(PositiveSample['serial_number'].value_counts().index)
        self.serial_numberOfNegative = list(NegativeSample['serial_number'].value_counts().index)

        self.PositiveSample_fea = PositiveSample.iloc[:, 6:12]
        self.NegativeSample_fea = NegativeSample.iloc[:, 6:12]

        self.colOfPositive = self.PositiveSample_fea.columns
        self.colOfNegative = self.PositiveSample_fea.columns

        for i in self.colOfPositive:
            max = np.max(PositiveSample[i])
            min = np.min(PositiveSample[i])
            scalar = max - min 
            PositiveSample[i] = PositiveSample[i] / scalar


        print('PositiveSample')
        print(PositiveSample.head(5))


        for i in self.colOfNegative:
            max = np.max(NegativeSample[i])
            min = np.min(NegativeSample[i])
            scalar = max - min 
            NegativeSample[i] = NegativeSample[i] / scalar

        print('NegativeSample')
        print(NegativeSample.head(5))

        self.inputs = []
        self.labels = []

        for i in self.serial_numberOfPositive:
            Positive_bySerial = PositiveSample[PositiveSample['serial_number'] == i].iloc[:, 6: 12]
            for j in range(len(Positive_bySerial)):
                self.inputs.append(torch.Tensor(np.array(Positive_bySerial[j: j + numstep])))
                self.labels.append(torch.ones(1))

        for i in self.serial_numberOfNegative:
            Negetive_bySerial = NegativeSample[NegativeSample['serial_number'] == i].iloc[:, 6: 12]
            for j in range(len(Positive_bySerial)):
                self.inputs.append(torch.Tensor(np.array(Positive_bySerial[j: j + numstep])))
                self.labels.append(torch.ones(1))
        
    def __len__(self):
        print('len')
        print(len(self.inputs))
        return len(self.inputs)

    def __getitem__(self, idx):
        print('labels')
        print(len(inputs))
        return self.inputs[idx], self.labels[idx]


def train(net, train_iter, optimizer, num_epochs):
    print("training start")
    loss = torch.nn.BCELoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        state = None
        for X, y in train_iter:
            optimizer.zero_grad()
            if state is not None:
                if isinstance (state, tuple): # LSTM, state:(h, c)  
                    state = (state[0].detach(), state[1].detach())
                else:   
                    state = state.detach()
            y_hat, state = net(X.float(), state)
            y_hat.squeeze_(1)
            l = loss(y_hat, y.long())
            l.backward()
            optimizer.step()
            train_l_sum += l.item()
            train_acc_sum += (y_hat == y).sum().item()
            n += y.size(0)
            batch_count += 1
        print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
        % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start))


        
PositiveSample = pd.read_csv('G:/BaiduNetdiskDownload/positiveSample.csv')
NegativeSample = pd.read_csv('G:/BaiduNetdiskDownload/NegativeSample.csv')

train_set = Mydata(PositiveSample, NegativeSample, 5)
train_iter = DataLoader(train_set, batch_size = 32, shuffle = True)


net = diskLSTM()
net = net.to(device)
lr, num_epochs = 0.01, 30
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

train(net, train_iter, optimizer, num_epochs)


















'''
model = nn.Sequential(
    nn.Linear(10, 1),
    nn.Sigmoid()
)
criterion = nn.BCELoss()

x = torch.randn(16, 10)  # (16, 10)
y = torch.empty(16).random_(2)  # shape=(16, ) 其中每个元素值为0或1

out = model(x)  # (16, 1)
out = out.squeeze(dim=-1)  # (16, )

loss = criterion(out, y)
'''