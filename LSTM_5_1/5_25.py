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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    def _dataX(pos, corpus_indices_1):
        tensor_corpus = np.array(corpus_indices_1.iloc[:, 5: -1][pos: pos + num_steps])
        tensor_corpus = torch.tensor(tensor_corpus, dtype=torch.float32)
        return tensor_corpus
    def _dataY(pos, corpus_indices_1):
        tensor_corpus = np.array(corpus_indices_1.loc[:, ['label']][pos: pos + num_steps])
        tensor_corpus = torch.tensor(tensor_corpus, dtype=torch.float32)
        return tensor_corpus

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    serial_number = corpus_indices['serial_number'].value_counts()
    for Serial_number in serial_number.index:
        corpus_indices_1 = corpus_indices[corpus_indices.serial_number == Serial_number]
        numberOfSerial_number = len(corpus_indices_1) #31
        num_examples = numberOfSerial_number // num_steps #31 // 7 = 4
        epoch_size = num_examples // batch_size # 4 // 2 = 2
        example_indices = list(range(numberOfSerial_number)) #0-31
        random.shuffle(example_indices)
        print("LenOfcorpus_indices_1", len(corpus_indices_1))

        for i in range(epoch_size): #0,1
            i = i * batch_size # 0 * 2 = 0, 1 * 2 = 2
            batch_indices = example_indices[i: i + batch_size] #0:1|1:2 0, 1
            X = [_dataX(j * num_steps, corpus_indices_1) for j in batch_indices]
            Y = [_dataY(j * num_steps, corpus_indices_1) for j in batch_indices]
            yield X, Y
           


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, data_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.data_size = data_size
        self.dense = nn.Linear(self.hidden_size, 2)
        self.state = None

    def forward(self, inputs, state): # inputs: (batch, seq_len)
        # 获取one-hot向量表示
        #  # X是个list
        Y, self.state = self.rnn(torch.stack(inputs), state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state

num_hiddens = 64
data_size = 13

state = None
train = pd.read_csv('train_2018_1_model_2.csv')
lstm_layer = nn.LSTM(input_size=data_size, hidden_size=num_hiddens)
model = RNNModel(lstm_layer, data_size)
i = 10
for X, Y in data_iter_random(train, 2, 7):
    print("*************************************************") 
    (output, state) = lstm_layer(torch.stack(X), state)
