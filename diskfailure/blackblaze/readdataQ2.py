import os
import pandas as pd 
import numpy as np 

dataOfBlack_2020_Q2 = []

dataOfBlack_2020_Q2.append(pd.read_csv('E:/data/data_Q1_2018/2018-03-27.csv'))
dataOfBlack_2020_Q2.append(pd.read_csv('E:/data/data_Q1_2018/2018-03-28.csv'))
dataOfBlack_2020_Q2.append(pd.read_csv('E:/data/data_Q1_2018/2018-03-29.csv'))
dataOfBlack_2020_Q2.append(pd.read_csv('E:/data/data_Q1_2018/2018-03-30.csv'))
dataOfBlack_2020_Q2.append(pd.read_csv('E:/data/data_Q1_2018/2018-03-31.csv'))

dirs = os.listdir('E:/data/data_Q2_2018')

for i in dirs:
    dataOfBlack_2020_Q2.append(pd.read_csv('E:/data/data_Q2_2018/' + i))

listOfpositiveSample = []
for i in range(5, len(dataOfBlack_2020_Q2)):
    dataOferrordisk = dataOfBlack_2020_Q2[i][dataOfBlack_2020_Q2[i].failure == 1]
    for j in range(len(dataOferrordisk)):
        k = 5
        while(k >= 0):
            listOfpositiveSample.append(dataOfBlack_2020_Q2[i - k][dataOfBlack_2020_Q2[i - k].serial_number == dataOferrordisk.serial_number.values[j]])
            k = k - 1

positiveSample = pd.concat(listOfpositiveSample, ignore_index=True)

positiveSample.to_csv('E:/data/positiveSample_Q2.csv')