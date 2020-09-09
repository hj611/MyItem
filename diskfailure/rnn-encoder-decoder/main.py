import RNN
from MyData import *
import utils
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

train_data = pd.read_csv('F:/data/train_2018Q1_model_2.csv')

num_epochs = 10
batch_size = 10
numstep = 7
length = 6

def main():
    data = MyData(train_data, numstep, length)
    sampler = utils.mySample(data)
    train_iter = DataLoader(data, batch_size = 10, drop_last=True, sampler=sampler)
    rnn = RNN(13)

    losses = []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        #print("=" * 50 + (" EPOCH %i " % epoch) + "=" * 50)
        for i, batch in enumerate(train_iter):
            input, target = batch

            loss, outputs, acc_sum = rnn.train(input, target)
            train_l_sum += loss
            train_acc_sum += acc_sum
            n += target.shape[0]
            batch_count += 1

        print(batch_count)
        print(train_l_sum)
        print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
        % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start))

main()