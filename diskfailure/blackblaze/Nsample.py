import os
import pandas as pd
import numpy as np

dataOfBlack_2020_Q1 = []

data = pd.read_hdf('E:/data/final_BlackBlaze_Q1_Q2_Q3.h5')
date = list(data['date'].value_counts().index)
date.sort()
print(date)

for i in date:
    dataOfBlack_2020_Q1.append(data[data['date'] == i])

for i in range(5):
    dataOfBlack_2020_Q1[i] = dataOfBlack_2020_Q1[i].drop(labels = list(dataOfBlack_2020_Q1[i][dataOfBlack_2020_Q1[i].failure == 1].index))



listOfnegativeSample = []
for i in range(5, len(dataOfBlack_2020_Q1)):
    dataOferrordisk = dataOfBlack_2020_Q1[i][dataOfBlack_2020_Q1[i].failure == 1]
    serial = list(dataOferrordisk.serial_number.values)
    k = 0
    index = []
    print(i, '@@@@@@@@@@@@@@@@@@@@@@')
    while(k <= 5):
        print(i - k)
        for j in serial:
            index = index + list(dataOfBlack_2020_Q1[i - k][dataOfBlack_2020_Q1[i - k].serial_number == j].index)
        print(index, '*********************')
        dataOfBlack_2020_Q1[i - k] = dataOfBlack_2020_Q1[i - k].drop(labels = index)
        index = []
        print(dataOfBlack_2020_Q1[i - k][dataOfBlack_2020_Q1[i - k]['failure'] == 1].index)
        k = k + 1
   
Negativesample = pd.concat(dataOfBlack_2020_Q1, ignore_index=True)
Negativesample.to_csv('E:/data/NegativeSample.csv')