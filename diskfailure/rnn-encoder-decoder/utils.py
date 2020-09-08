from RNN import *
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
    total = len(data)
    sample_list = []

    for X, y in data:
        num = num + y
    

    class_sample_count = np.array([total - num, num])
    weight = 1. / class_sample_count

    for X, y in data:
        sample_list.append(weight[int(y)])

    sample_weight = torch.FloatTensor(sample_list)
    sampler = WeightedRandomSampler(sample_weight, total)
    return sampler