import os
import pandas as pd 
import numpy as np 

dataOfBlack_2020_Q3 = []

dataOfBlack_2020_Q3.append(pd.read_csv('E:/data/data_Q2_2018/2018-06-26.csv'))
dataOfBlack_2020_Q3.append(pd.read_csv('E:/data/data_Q2_2018/2018-06-27.csv'))
dataOfBlack_2020_Q3.append(pd.read_csv('E:/data/data_Q2_2018/2018-06-28.csv'))
dataOfBlack_2020_Q3.append(pd.read_csv('E:/data/data_Q2_2018/2018-06-29.csv'))
dataOfBlack_2020_Q3.append(pd.read_csv('E:/data/data_Q2_2018/2018-06-30.csv'))

dirs = os.listdir('E:/data/data_Q3_2018')

for i in dirs:
    dataOfBlack_2020_Q3.append(pd.read_csv('E:/data/data_Q3_2018/' + i))

listOfpositiveSample = []
for i in range(5, len(dataOfBlack_2020_Q3)):
    dataOferrordisk = dataOfBlack_2020_Q3[i][dataOfBlack_2020_Q3[i].failure == 1]
    for j in range(len(dataOferrordisk)):
        k = 5
        while(k >= 0):
            listOfpositiveSample.append(dataOfBlack_2020_Q3[i - k][dataOfBlack_2020_Q3[i - k].serial_number == dataOferrordisk.serial_number.values[j]])
            k = k - 1

positiveSample = pd.concat(listOfpositiveSample, ignore_index=True)

positiveSample.to_csv('E:/data/positiveSample_Q3.csv')