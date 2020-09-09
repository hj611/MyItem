import torch 
import torch.nn as nn
from torch.autograd import Variable

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size = 64):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers= 1, batch_first=True)
        self.liner = nn.Linear(hidden_size, 2)
        self.softmax = nn.Softmax()
    
    def forward(self, input, hidden):
        output, hidden_state = self.LSTM(input.float())
        liner = self.liner(hidden_state[0].squeeze(0))
        softmax = self.softmax(liner)
        return output, softmax, hidden_state