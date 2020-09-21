import os
import pandas as pd 
import numpy as np 

dataOfBlack_2020_Q1 = []
dirs = os.listdir('E:/data/data_Q1_2018')
for i in dirs:
    dataOfBlack_2020_Q1.append(pd.read_csv('E:/data/data_Q1_2018/' + i))

listOfnegativeSample = []
for i in range(5, len(dataOfBlack_2020_Q1)):
    dataOferrordisk = dataOfBlack_2020_Q1[i][dataOfBlack_2020_Q1[i].failure == 1]
    k = 5
    while(k >= 0):
        serial_number = list(dataOfBlack_2020_Q1[i - k].serial_number)
        serial_number = list(set(serial_number))
        for j in dataOferrordisk.serial_number.values:
            serial_number.remove(j)
        listOfnegativeSample.append(dataOfBlack_2020_Q1[i - k][dataOfBlack_2020_Q1[i - k].serial_number.isin(serial_number)])
        k = k - 1
        

negativeSample = pd.concat(listOfnegativeSample, ignore_index=True)

negativeSample.to_csv('E:/data/negativeSample_Q1.csv')