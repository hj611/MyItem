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
train_data = pd.read_csv('F:/data/train_2018Q1_model_2.csv')

#只有LSTM 时间步为1

class MyData(Dataset):
    def __init__(self, train_data):
        self.value = train_data.iloc[:, 6:-1].values
        max = np.max(self.value)
        min = np.min(self.value) 
        scalar = max - min 
        self.datas = list(map(lambda x: x / scalar, self.value))
        self.label = train_data.loc[:, 'label'].values.tolist()
    def __len__(self):
        return len(self.datas)
    def __getitem__(self, idx):
        data = self.datas[idx].reshape(1, -1)
        return data, self.label[idx]


train_iter = DataLoader(MyData(train_data), batch_size = 10)

class myLSTM(nn.Module):

    def __init__(self):
        super(myLSTM, self).__init__()

        self.lstmlayer = nn.LSTM(input_size=13, hidden_size=128, num_layers=1, batch_first=True)
        self.linear = nn.Linear(128, 2)
    def forward(self, X):
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
            y = y.to(device)
            y_hat = net(X.float())
            y_hat = y_hat.reshape(-1, 2)
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
    

net = myLSTM()
net = torch.nn.DataParallel(net)
lr, num_epochs = 0.001, 20
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

train(net, train_iter, optimizer, num_epochs)