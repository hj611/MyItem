from Encoder import *
from Decoder import *
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch
import pandas as pd
import MyData
import torch.nn as nn

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



for i, batch in enumerate(train_iter):
    print(batch.shape)
    break