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



listOfpositiveSample = []
for i in range(5, len(dataOfBlack_2020_Q1)):
    dataOferrordisk = dataOfBlack_2020_Q1[i][dataOfBlack_2020_Q1[i].failure == 1]
    for j in dataOferrordisk.serial_number.values:
        k = 5
        while(k >= 0):
            listOfpositiveSample.append(dataOfBlack_2020_Q1[i - k][dataOfBlack_2020_Q1[i - k].serial_number == j])
            k = k - 1

positiveSample = pd.concat(listOfpositiveSample, ignore_index=True)

positiveSample.to_csv('E:/data/positiveSample.csv')