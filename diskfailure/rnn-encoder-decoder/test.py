import torch 
import torch.nn as nn
from torch.autograd import Variable

x = torch.ones(10, 7)
liner = nn.Linear(7, 2)
print(liner(x))
print(nn.Softmax(liner(x)))