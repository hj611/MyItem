import torch 
import torch.nn as nn
from torch.autograd import Variable

class Decoder(nn.Module):
    def __init__(self, input_size = 64, hidden_size = 128):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers= 1, batch_first=True)
        self.liner = nn.Linear(hidden_size, input_size)
        self.softmax = nn.Softmax()
    
    def forward(self, input, hidden):
        output, hidden_state = self.LSTM(input, hidden)
        output = output.view(output.size(0), output.size(2))
        liner = self.liner(output)
        softmax = self.softmax(liner)
        return output, softmax, hidden_state