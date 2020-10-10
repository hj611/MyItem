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
from torch.autograd import Variable
device = torch.device('cuda')


inputs = torch.load('E:/WJ/data/inputs.dat')
labels = torch.load('E:/WJ/data/labels.dat')

print(device)

class diskLSTM(nn.Module):
    def __init__(self, batch_size):
        super(diskLSTM, self).__init__()

        self.LSTM_1 = nn.LSTM(input_size=6, hidden_size=32, batch_first=True)
        self.LSTM_2 = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        self.LSTM_3 = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
 
        self.linear_1 = nn.Linear(128, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, 1)

        self.state_1 = (Variable(torch.randn(1, batch_size, 32).cuda()), Variable(torch.randn(1, batch_size, 32).cuda()))
        self.state_2 = (Variable(torch.randn(1, batch_size, 64).cuda()), Variable(torch.randn(1, batch_size, 64).cuda()))
        self.state_3 = (Variable(torch.randn(1, batch_size, 128).cuda()), Variable(torch.randn(1, batch_size, 128).cuda()))
    
    def forward(self, X):
        X, _ = self.LSTM_1(X, self.state_1)
        X, _ = self.LSTM_2(X, self.state_2)
        X, _ = self.LSTM_3(X, self.state_3)

        X = X[:, X.size(1) - 1, :]


        X = self.linear_1(X)

        X = torch.tanh(X)
        X = self.linear_2(X)

        X = torch.tanh(X)
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
        train_l_sum, train_acc_sum, n, batch_count, start, sum_fp, fp = 0.0, 0.0, 0, 0, time.time(), 0.0, 0.0
        for X, y in train_iter:
            X = X.to(device)
            y = y.squeeze(1)
            y = y.to(device)
            y_hat = net(X.float())
            y_hat = y_hat.reshape(y_hat.size(0))
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
            for i in range(len(y_hat)):
                if(y[i] == 1 and y[i] == y_hat[i]):
                    fp = fp + 1
            sum_fp = sum_fp + y.sum().item()
        print('epoch %d, loss %.4f, train acc %.3f, FP acc %.3f,time %.1f sec'
        % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, fp / sum_fp, time.time() - start))





train_set = Mydata(inputs, labels, 5)
train_iter = DataLoader(train_set, batch_size = 64, shuffle = True, drop_last=True)


net = diskLSTM(64)
net = net.to(device)
lr, num_epochs = 0.001, 100
optimizer = torch.optim.Adamax(net.parameters(), lr=lr)
print('lr', lr)
train(net, train_iter, optimizer, num_epochs)



'''
cuda
lr 0.001
training start
epoch 1, loss 0.6932, train acc 31.991, FP acc 0.527,time 91.2 sec
epoch 2, loss 0.6930, train acc 32.056, FP acc 0.547,time 92.3 sec
epoch 3, loss 0.6925, train acc 31.886, FP acc 0.499,time 93.2 sec
epoch 4, loss 0.6916, train acc 32.025, FP acc 0.621,time 100.0 sec
epoch 5, loss 0.6910, train acc 32.082, FP acc 0.613,time 94.4 sec
epoch 6, loss 0.6906, train acc 32.119, FP acc 0.704,time 100.1 sec
epoch 7, loss 0.6907, train acc 32.060, FP acc 0.603,time 94.5 sec
epoch 8, loss 0.6905, train acc 32.118, FP acc 0.649,time 97.4 sec
epoch 9, loss 0.6903, train acc 32.114, FP acc 0.701,time 96.2 sec
epoch 10, loss 0.6897, train acc 32.174, FP acc 0.658,time 95.1 sec
epoch 11, loss 0.6894, train acc 32.182, FP acc 0.726,time 83.4 sec
epoch 12, loss 0.6889, train acc 32.170, FP acc 0.743,time 96.9 sec
epoch 13, loss 0.6872, train acc 32.252, FP acc 0.781,time 94.6 sec
epoch 14, loss 0.6856, train acc 32.284, FP acc 0.811,time 92.3 sec
epoch 15, loss 0.6836, train acc 32.344, FP acc 0.853,time 91.1 sec
epoch 16, loss 0.6824, train acc 32.373, FP acc 0.905,time 92.6 sec
epoch 17, loss 0.6811, train acc 32.439, FP acc 0.863,time 91.6 sec
epoch 18, loss 0.6806, train acc 32.429, FP acc 0.910,time 91.2 sec
epoch 19, loss 0.6799, train acc 32.442, FP acc 0.937,time 92.6 sec
epoch 20, loss 0.6794, train acc 32.459, FP acc 0.967,time 91.4 sec
epoch 21, loss 0.6793, train acc 32.452, FP acc 0.961,time 92.3 sec
epoch 22, loss 0.6787, train acc 32.484, FP acc 0.953,time 90.5 sec
epoch 23, loss 0.6788, train acc 32.472, FP acc 0.954,time 89.7 sec
epoch 24, loss 0.6786, train acc 32.480, FP acc 0.971,time 90.3 sec
epoch 25, loss 0.6786, train acc 32.478, FP acc 0.977,time 90.8 sec
epoch 26, loss 0.6784, train acc 32.488, FP acc 0.977,time 89.9 sec
epoch 27, loss 0.6791, train acc 32.464, FP acc 0.969,time 90.0 sec
epoch 28, loss 0.6792, train acc 32.458, FP acc 0.985,time 91.1 sec
epoch 29, loss 0.6786, train acc 32.491, FP acc 0.987,time 90.1 sec
epoch 30, loss 0.6781, train acc 32.502, FP acc 0.981,time 90.0 sec
epoch 31, loss 0.6787, train acc 32.475, FP acc 0.975,time 91.2 sec
epoch 32, loss 0.6787, train acc 32.481, FP acc 0.982,time 90.3 sec
epoch 33, loss 0.6788, train acc 32.472, FP acc 0.973,time 90.1 sec
epoch 34, loss 0.6782, train acc 32.500, FP acc 0.984,time 91.0 sec
epoch 35, loss 0.6781, train acc 32.496, FP acc 0.984,time 90.1 sec
epoch 36, loss 0.6787, train acc 32.479, FP acc 0.945,time 89.7 sec
epoch 37, loss 0.6784, train acc 32.483, FP acc 0.987,time 90.6 sec
epoch 38, loss 0.6791, train acc 32.467, FP acc 0.985,time 90.5 sec
epoch 39, loss 0.6788, train acc 32.478, FP acc 0.981,time 89.7 sec
epoch 40, loss 0.6782, train acc 32.491, FP acc 0.990,time 90.1 sec
epoch 41, loss 0.6789, train acc 32.495, FP acc 0.989,time 90.8 sec
epoch 42, loss 0.6787, train acc 32.470, FP acc 0.982,time 89.8 sec
epoch 43, loss 0.6783, train acc 32.488, FP acc 0.981,time 89.6 sec
epoch 44, loss 0.6787, train acc 32.485, FP acc 0.985,time 91.2 sec
epoch 45, loss 0.6786, train acc 32.476, FP acc 0.986,time 90.2 sec
epoch 46, loss 0.6790, train acc 32.468, FP acc 0.978,time 89.7 sec
epoch 47, loss 0.6787, train acc 32.486, FP acc 0.980,time 90.8 sec
epoch 48, loss 0.6791, train acc 32.470, FP acc 0.974,time 90.2 sec
epoch 49, loss 0.6782, train acc 32.500, FP acc 0.972,time 89.7 sec
epoch 50, loss 0.6785, train acc 32.491, FP acc 0.983,time 90.4 sec
epoch 51, loss 0.6785, train acc 32.480, FP acc 0.988,time 90.6 sec
epoch 52, loss 0.6788, train acc 32.483, FP acc 0.980,time 89.8 sec
epoch 53, loss 0.6787, train acc 32.472, FP acc 0.987,time 89.9 sec
epoch 54, loss 0.6791, train acc 32.465, FP acc 0.971,time 90.9 sec
epoch 55, loss 0.6785, train acc 32.480, FP acc 0.990,time 89.9 sec
epoch 56, loss 0.6784, train acc 32.491, FP acc 0.987,time 89.7 sec
epoch 57, loss 0.6786, train acc 32.474, FP acc 0.971,time 91.1 sec
epoch 58, loss 0.6781, train acc 32.500, FP acc 0.991,time 90.1 sec
epoch 59, loss 0.6785, train acc 32.482, FP acc 0.988,time 89.6 sec
epoch 60, loss 0.6792, train acc 32.460, FP acc 0.990,time 90.7 sec
epoch 61, loss 0.6783, train acc 32.488, FP acc 0.992,time 90.4 sec
epoch 62, loss 0.6780, train acc 32.500, FP acc 0.989,time 89.7 sec
epoch 63, loss 0.6787, train acc 32.493, FP acc 0.991,time 90.2 sec
epoch 64, loss 0.6781, train acc 32.499, FP acc 0.992,time 90.7 sec
epoch 65, loss 0.6781, train acc 32.495, FP acc 0.988,time 89.7 sec
epoch 66, loss 0.6785, train acc 32.481, FP acc 0.985,time 89.8 sec
epoch 67, loss 0.6788, train acc 32.487, FP acc 0.985,time 91.0 sec
epoch 68, loss 0.6779, train acc 32.499, FP acc 0.982,time 89.9 sec
epoch 69, loss 0.6784, train acc 32.489, FP acc 0.977,time 89.7 sec
epoch 70, loss 0.6779, train acc 32.502, FP acc 0.991,time 91.0 sec
epoch 71, loss 0.6787, train acc 32.480, FP acc 0.992,time 90.1 sec
epoch 72, loss 0.6784, train acc 32.482, FP acc 0.991,time 89.7 sec
epoch 73, loss 0.6786, train acc 32.480, FP acc 0.993,time 90.6 sec
epoch 74, loss 0.6777, train acc 32.507, FP acc 0.980,time 90.5 sec
epoch 75, loss 0.6783, train acc 32.485, FP acc 0.992,time 89.6 sec
epoch 76, loss 0.6780, train acc 32.505, FP acc 0.991,time 90.1 sec
epoch 77, loss 0.6789, train acc 32.477, FP acc 0.989,time 90.8 sec
epoch 78, loss 0.6788, train acc 32.476, FP acc 0.978,time 89.9 sec
epoch 79, loss 0.6785, train acc 32.484, FP acc 0.990,time 89.8 sec
epoch 80, loss 0.6780, train acc 32.499, FP acc 0.992,time 91.1 sec
epoch 81, loss 0.6780, train acc 32.508, FP acc 0.994,time 90.0 sec
epoch 82, loss 0.6778, train acc 32.502, FP acc 0.993,time 89.6 sec
epoch 83, loss 0.6779, train acc 32.494, FP acc 0.994,time 90.9 sec
epoch 84, loss 0.6779, train acc 32.505, FP acc 0.993,time 90.2 sec
epoch 85, loss 0.6781, train acc 32.499, FP acc 0.993,time 89.7 sec
epoch 86, loss 0.6784, train acc 32.482, FP acc 0.993,time 90.4 sec
epoch 87, loss 0.6785, train acc 32.484, FP acc 0.984,time 90.6 sec
epoch 88, loss 0.6781, train acc 32.497, FP acc 0.993,time 89.7 sec
epoch 89, loss 0.6781, train acc 32.496, FP acc 0.994,time 90.0 sec
epoch 90, loss 0.6789, train acc 32.471, FP acc 0.992,time 90.9 sec
epoch 91, loss 0.6782, train acc 32.495, FP acc 0.991,time 90.0 sec
epoch 92, loss 0.6782, train acc 32.493, FP acc 0.994,time 89.7 sec
epoch 94, loss 0.6786, train acc 32.475, FP acc 0.964,time 93.5 sec
epoch 95, loss 0.6786, train acc 32.484, FP acc 0.993,time 96.5 sec
epoch 96, loss 0.6783, train acc 32.483, FP acc 0.978,time 99.7 sec
epoch 97, loss 0.6780, train acc 32.493, FP acc 0.990,time 98.5 sec
epoch 98, loss 0.6782, train acc 32.485, FP acc 0.993,time 94.4 sec
epoch 99, loss 0.6793, train acc 32.459, FP acc 0.992,time 92.2 sec
epoch 100, loss 0.6785, train acc 32.485, FP acc 0.994,time 92.0 sec
'''