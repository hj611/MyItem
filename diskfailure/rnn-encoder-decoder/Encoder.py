import torch 
import torch.nn as nn
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers= 1, batch_first=True)

    def forward(self, input, hidden):
        output, hidden_state = self.LSTM(input, hidden)
    
    def first_hidden(self):
        return Variable(torch.FloatTensor(1, 1, self.hidden_size).zero_())

