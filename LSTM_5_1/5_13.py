import pandas as pd 
import numpy as np 
import torch
import random

def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    pos = 0
    indexofserial_number = 0
    def _dataX(pos):
        return corpus_indices.iloc[pos: pos + num_steps, 5: -1] 
    def _dataY(pos):
        return corpus_indices['label'][pos: pos + num_steps]

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    serial_number = corpus_indices['serial_number'].value_counts() 
    for numberOfSerial_number in serial_number:
        print('numberOfSerial_number', numberOfSerial_number)
        corpus_indices = corpus_indices[corpus_indices.serial_number == serial_number.index[indexofserial_number]]
        print('len', len(corpus_indices))
        indexofserial_number = indexofserial_number + 1
        num_examples = numberOfSerial_number // num_steps
        epoch_size = num_examples // batch_size
        example_indices = list(range(epoch_size))
        random.shuffle(example_indices)
        for i in range(epoch_size):
            i = i * batch_size
            batch_indices = example_indices[i: i + batch_size]
            X = []
            Y = []
            for j in batch_indices:
                pos = pos + j * num_steps
                X.append(_dataX(pos))
                Y.append(_dataY(pos))
            pos = pos + serial_number
            yield X, Y

train_2018_2_model_2 = pd.read_csv('train_2018_2_model_2.csv')

for X, Y in data_iter_random(train_2018_2_model_2, 2, 7):
    print(X)
    print(Y)