import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset

pad = nn.ZeroPad2d(padding=(2, 1, 0, 0))
X = torch.randn(2, 10, 48) 
final_output = 2
import d2lzh_pytorch as d2l
class CNNLSTM(nn.Module):

    def __init__(self):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 48, out_channels = 128, kernel_size = 4) # 要求转置
        self.maxpool1d = nn.MaxPool1d(kernel_size = 2)  
        self.lstmlayer = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True)
        # 输入数据x的向量维数10, 设定lstm隐藏层的特征维度20, 此model用2个lstm层。如果是1，可以省略，默认为1)
        # 注意最后一个lstm只输出每个batch的最后一个时间步的y    
        self.linear = nn.Linear(128, 2)
    def forward(self, X):
        pad = nn.ZeroPad2d(padding=(2, 1, 0, 0))
        X = X.permute(0, 2, 1)
        X = pad(X)
        X = self.conv1(X)
        X = self.maxpool1d(X)
        X =  X.permute(0, 2, 1) 
        X, (H, C) = self.lstmlayer(X)
        X = X[:, X.size(1) - 1, :]
        x = torch.sigmoid(X)
        X = self.linear(X)
        return X
       
class MyData(Dataset):
    def __init__(self):
        self.input = []
        self.lable = []
        for i in range(100):
            self.input.append(torch.randn(10, 48))
            self.lable.append(1)
        a = range(1, 100, 1)
        b = random.sample(a, 50)
        for i in b:
            self.lable[i] = 0

    def __len__(self):
        return 100
    def __getitem__(self, idx):
        return self.input[idx], self.lable[idx]

data = DataLoader(MyData(), batch_size = 2)

def train(net, train_iter, batch_size, optimizer, num_epochs):
    print("training start")
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        #test_acc = evaluate_accuracy(test_iter, net)
        #print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              #% (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        print('end')

net = CNNLSTM()
lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

train(net, data, 2, optimizer, num_epochs)
