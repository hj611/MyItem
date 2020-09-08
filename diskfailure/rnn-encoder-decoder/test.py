from Encoder import *
from Decoder import *
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch
import pandas as pd
import MyData
import torch.nn as nn


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

class MyData(Dataset):
    def __init__(self, traindata, numstep, length):
        self.serial_number = traindata['serial_number'].value_counts()
        self.value = traindata.iloc[:, length:-1].values
        max = np.max(self.value)
        min = np.min(self.value)
        scalar = max - min 
        self.datas = list(map(lambda x: x / scalar, self.value))
        self.datalabel = traindata.loc[:, 'label'].values.tolist()
        self.input = []
        self.label = []

        for diskname in self.serial_number.index.sort_values():
            traindata_name = traindata[traindata.serial_number == diskname]
            self.datalabel = traindata_name.loc[:, 'label'].values.tolist()
            for i in range(len(traindata_name) - numstep):
                self.input.append(torch.Tensor(self.datas[i: i + numstep]))
                if(torch.Tensor(self.datalabel[i + numstep: i + numstep + numstep]).sum() != 0):
                    self.label.append(torch.ones(1))
                else:
                    self.label.append(torch.zeros(1))
    def __len__(self):
        return len(self.input)
    def __getitem__(self, idx):
        #data = self.datas[idx].reshape(1, -1)
        return self.input[idx], self.label[idx]


train_data = pd.read_csv('F:/data/train_2018Q1_model_2.csv')
data = MyData(train_data, 7, 6)
train_iter = DataLoader(data, batch_size = 10, drop_last=True)

encoder = Encoder(13)

lstm = nn.LSTM(input_size=13, hidden_size=128, num_layers=2, batch_first=True)

for X, y in train_iter:
    print(lstm(X.float(), None))
    break