from Encoder import *
from Decoder import *
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch

class RNN(object):
    def __init__(self, input_size):
        super(RNN, self).__init__()

        self.encoder = Encoder(input_size)
        self.decoder = Decoder(input_size)

        self.loss = nn.CrossEntropyLoss()
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr = 0.1)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr = 0.1)

    def train(self, input, target):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        hidden_state = None
        #Encoder
        #hidden_state = self.encoder.first_hidden()
        _, hidden_state = self.encoder.forward(input.long(), hidden_state)


        #Decoder
        total_loss, outputs = 0, []  

        _, softmax, hidden_state = self.decoder.forward(input, hidden_state)
        total_loss = self.loss(softmax.long(), target.long())
        total_loss.backward()
        

        self.decoder_optimizer.step()
        self.encoder_optimizer.step()

        return total_loss.data[0], outputs

def eval(self, input):
        hidden_state = self.encoder.first_hidden()

        # Encoder
        for ivec in input:
            _, hidden_state = self.encoder.forward(Variable(ivec), hidden_state)

        sentence = []
        input = self.sos
        # Decoder
        while input.data[0, 0] != 1:
            output, _, hidden_state = self.decoder.forward(input, hidden_state)
            word = np.argmax(output.data.numpy()).reshape((1, 1))
            input = Variable(torch.LongTensor(word))
            sentence.append(word)

        return sentence