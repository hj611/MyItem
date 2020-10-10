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
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim import lr_scheduler
from imblearn.over_sampling import SMOTE
from collections import Counter

device = torch.device('cuda')


inputs = torch.load('E:/WJ/data/inputs.dat')
labels = torch.load('E:/WJ/data/labels.dat')

print(device)

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


        X = self.linear_1(X)

        X = F.tanh(X)
        X = self.linear_2(X)

        X = F.tanh(X)
        X = self.linear_3(X)

        X = torch.sigmoid(X)

        return X

class Mydata(Dataset):
    def __init__(self, inputs, labels, numstep):

        self.fea_num = 6

        self.X_som = []
        self.y_smo = []

        
        self.inputs_som = []
        self.labels_som = []

  
        for i in range(len(inputs)):
            inputs[i] = np.array(inputs[i].reshape(numstep * self.fea_num))
        
        self.X_som, self.y_smo = SMOTE(random_state=42).fit_sample(inputs, labels)

        for i in self.X_som:
            i = torch.Tensor(i)
            self.inputs_som.append(i.reshape(numstep, self.fea_num))

        for i in self.y_smo:
            if(i == 1):
                self.labels_som.append(torch.ones(1))
            else:
                self.labels_som.append(torch.zeros(1))

    def __len__(self):
        return len(self.inputs_som)

    def __getitem__(self, idx):
        return self.inputs_som[idx], self.labels_som[idx]

    def get(self):
        return self.inputs_som, self.labels_som

def classifyRes(y_hat):
    for i in range(y_hat.size(0)):
        if(y_hat[i] > 0.5):
            y_hat[i] = 1
        else:
            y_hat[i] = 0
    return y_hat


def train(net, train_iter, optimizer, num_epochs):
    print("training start")
    loss = torch.nn.BCELoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        state = None
        for X, y in train_iter:
            X = X.to(device)
            y = y.squeeze(1)
            y = y.to(device)
            y_hat = net(X.float(), state)
            y_hat = y_hat.to(device)
            l = loss(y_hat, y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.item()
            y_hat = classifyRes(y_hat)
            train_acc_sum += (y_hat.cuda() == y).sum().item()
            n += y.size(0)
            batch_count += 1
        print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
        % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start))





train_set = Mydata(inputs, labels, 5)
train_iter = DataLoader(train_set, batch_size = 64, shuffle = True)


net = diskLSTM()
net = net.to(device)
lr, num_epochs = 0.0001, 20
optimizer = torch.optim.Adamax(net.parameters(), lr=lr)
print('lr', lr)
train(net, train_iter, optimizer, num_epochs)

'''
cuda
lr 0.001
training start
epoch 1, loss 0.6932, train acc 31.935, time 59.3 sec
epoch 2, loss 0.6931, train acc 31.991, time 58.6 sec
epoch 3, loss 0.6923, train acc 31.990, time 59.7 sec
epoch 4, loss 0.6911, train acc 32.084, time 59.6 sec
epoch 5, loss 0.6909, train acc 32.095, time 59.0 sec
epoch 6, loss 0.6907, train acc 32.107, time 59.0 sec
epoch 7, loss 0.6906, train acc 32.057, time 59.6 sec
epoch 8, loss 0.6904, train acc 32.108, time 59.4 sec
epoch 9, loss 0.6900, train acc 32.144, time 58.8 sec
epoch 10, loss 0.6897, train acc 32.183, time 59.4 sec
epoch 11, loss 0.6888, train acc 32.219, time 59.6 sec
epoch 12, loss 0.6867, train acc 32.237, time 59.2 sec
epoch 13, loss 0.6835, train acc 32.363, time 58.8 sec
epoch 14, loss 0.6815, train acc 32.405, time 59.6 sec
epoch 15, loss 0.6799, train acc 32.448, time 59.5 sec
epoch 16, loss 0.6791, train acc 32.478, time 58.9 sec
epoch 17, loss 0.6792, train acc 32.464, time 58.9 sec
epoch 18, loss 0.6793, train acc 32.469, time 59.6 sec
epoch 19, loss 0.6798, train acc 32.463, time 59.3 sec
epoch 20, loss 0.6798, train acc 32.474, time 58.6 sec
'''

'''
lr 0.0001
training start
epoch 1, loss 0.6932, train acc 32.015, time 58.5 sec
epoch 2, loss 0.6931, train acc 31.945, time 58.1 sec
epoch 3, loss 0.6931, train acc 31.926, time 59.0 sec
epoch 4, loss 0.6930, train acc 31.985, time 58.6 sec
epoch 5, loss 0.6929, train acc 31.886, time 57.9 sec
epoch 6, loss 0.6927, train acc 31.975, time 58.9 sec
epoch 7, loss 0.6923, train acc 31.992, time 58.7 sec
epoch 8, loss 0.6919, train acc 32.027, time 57.8 sec
epoch 9, loss 0.6917, train acc 32.047, time 58.9 sec
epoch 10, loss 0.6916, train acc 32.054, time 58.9 sec
epoch 11, loss 0.6913, train acc 32.102, time 58.1 sec
epoch 12, loss 0.6913, train acc 32.067, time 58.6 sec
epoch 13, loss 0.6913, train acc 32.039, time 58.9 sec
epoch 14, loss 0.6912, train acc 32.085, time 58.1 sec
epoch 15, loss 0.6911, train acc 32.149, time 58.5 sec
epoch 16, loss 0.6911, train acc 32.108, time 59.0 sec
epoch 17, loss 0.6910, train acc 32.147, time 58.5 sec
epoch 18, loss 0.6909, train acc 32.117, time 58.0 sec
epoch 19, loss 0.6908, train acc 32.135, time 59.1 sec
epoch 20, loss 0.6909, train acc 32.057, time 58.7 sec
'''